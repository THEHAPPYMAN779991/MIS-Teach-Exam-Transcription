# -*- coding: utf-8 -*-
"""
FINAL12 — LLM Prompt-Primary Exam Transcriber
(STRICT TEXT / OPTION / CODE / TABLE AUDIT + POST-CROP QC)
======================================================

設計原則
--------
1. Python 負責：
   - PDF -> 高解析 FULL_PAGE PNG
   - Google Gen AI SDK / Vertex AI API 呼叫與 retry
   - JSON Schema 結構化輸出
   - 非語意 schema / page / bbox / asset_ref 驗證
   - asset identifier namespace
   - 依 LLM bbox 裁切素材
   - 題幹、選項與子題的逐題雙重盲讀及一致性閘門
   - 候選隔離的全頁 block coverage 掃描、遺漏素材修復與最終零遺漏閘門
   - 程式碼裁切依截斷方向自動擴框、雙盲讀與第三次一致性仲裁
   - 表格候選區擴張、局部重新定位、空白格合法化與幻覺空欄排除
   - 表格／程式碼只在至少兩份獨立結果完全一致時產生結構化內容
   - atomic output、batch report、token/cost 統計

2. Gemini 2.5 Pro 負責：
   - 題界、題號、題幹、選項、子題、題型
   - 跨頁、題目去重
   - 表格 / 公式 / 程式碼 / 圖形辨識
   - asset 類型、歸屬、bbox、layout_blocks、render_strategy
   - Canonical JSON

3. 不使用 PDF 文字層、Markdown、OCR、PyMuPDF table/drawing/raster 語意偵測，
   也不使用 Python heuristic 推論題意。

4. Vertex AI 固定使用 Gemini 2.5 Pro。
"""

from __future__ import annotations

import argparse
import difflib
import html as html_lib
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz

try:
    from PIL import Image, ImageFilter, ImageOps
except ImportError as exc:
    raise RuntimeError("需要 Pillow：請執行 python -m pip install --upgrade Pillow") from exc


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


GEMINI_MODEL_NAME = "gemini-2.5-pro"
DEFAULT_GCP_LOCATION = "us-central1"
PAGE_LONG_SIDE_PX = 3600
MAX_OUTPUT_TOKENS = 65535

# Post-crop visual QC. The crop is actually rendered, then shown back to Gemini
# together with the original full page and question context. If Gemini detects
# truncation or unrelated content, it returns a corrected full-page bbox and
# the crop is regenerated for another inspection.
DEFAULT_MAX_POST_CROP_QC_ROUNDS = 3

# For short exam PDFs, the dedicated asset audit sees all pages so a wrong
# candidate page_number cannot prevent recovery. For longer PDFs, it sees the
# union of question source pages, candidate asset pages, and adjacent pages.
ASSET_AUDIT_SEND_ALL_PAGES_UP_TO = 12

# Table recovery runs after the ordinary post-crop QC. The candidate bbox is
# expanded deterministically, then Gemini works on the much larger local view.
# Geometry and cell content are verified separately; old HTML is never shown.
TABLE_CONTEXT_MIN_Y_PAD = 35.0
TABLE_CONTEXT_MIN_X_PAD = 55.0
TABLE_CONTEXT_PAD_RATIO = 0.65
TABLE_READ_MIN_LONG_SIDE_PX = 1800
TABLE_FINAL_SAFETY_PAD = 8.0

# Exact text audit re-reads every question twice without showing any candidate
# text. Code receives its own crop-based double read after geometry QC.
EXACT_TEXT_READ_MIN_LONG_SIDE_PX = 2400
CODE_READ_MIN_LONG_SIDE_PX = 2200

# A candidate-free coverage scan is deliberately independent from the normal
# extraction/review/arbitration chain.  The ordinary asset audit can only
# validate assets that already exist; this extra pass is what discovers an
# omitted second code block, one-line function stub, table, formula, or figure.
COVERAGE_GATE_MAX_ISSUES_PER_QUESTION = 20

# Printed tables and code must remain structured whenever their characters are
# readable.  A tight first bbox or one disagreeing reader is a recovery signal,
# not permission to silently replace the content with a screenshot.
STRUCTURED_ASSET_MAX_READ_ROUNDS = 3
STRUCTURED_ASSET_EDGE_EXPAND_UNITS = 35.0

# Vertex AI Gemini 2.5 Pro standard pay-as-you-go pricing tiers.
PRO_INPUT_PRICE_LE_200K = 1.25
PRO_OUTPUT_PRICE_LE_200K = 10.00
PRO_INPUT_PRICE_GT_200K = 2.50
PRO_OUTPUT_PRICE_GT_200K = 15.00

_CURRENT_PDF = ""
_STATS: Dict[str, Dict[str, Any]] = {}
_RUN_OVERHEAD: Dict[str, Any] = {
    "calls": 0,
    "input_tokens": 0,
    "candidate_tokens": 0,
    "thought_tokens": 0,
    "billable_output_tokens": 0,
    "cost_usd_total": 0.0,
}


# ============================================================
# Environment / Vertex configuration
# ============================================================
def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:
        print(f"WARNING: cannot read env file {path}: {exc}")


def load_local_env_files() -> None:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    seen: set[Path] = set()
    for folder in (cwd, script_dir):
        for name in (".env", "gcp.env", "api.env"):
            path = (folder / name).resolve()
            if path not in seen:
                seen.add(path)
                _load_env_file(path)


def _valid_project_id(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", value or ""))


def resolve_vertex_project(explicit: str = "") -> str:
    load_local_env_files()
    project = (
        explicit.strip()
        or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        or os.environ.get("GCP_PROJECT", "").strip()
    )
    if not project:
        try:
            import google.auth
            _, detected_project = google.auth.default()
            project = (detected_project or "").strip()
        except Exception:
            project = ""
    if not project:
        raise RuntimeError(
            "找不到 Google Cloud Project ID。請設定 GOOGLE_CLOUD_PROJECT，"
            "或使用 --gcp-project PROJECT_ID。"
        )
    if not _valid_project_id(project):
        raise RuntimeError(f"Google Cloud Project ID 格式不合法：{project!r}")
    return project


def resolve_vertex_location(explicit: str = "") -> str:
    load_local_env_files()
    return (
        explicit.strip()
        or os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip()
        or os.environ.get("GCP_LOCATION", "").strip()
        or DEFAULT_GCP_LOCATION
    )


def create_genai_client(project: str, location: str) -> Any:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "缺少 google-genai。請執行：python -m pip install --upgrade google-genai"
        ) from exc

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=types.HttpOptions(api_version="v1"),
    )


# ============================================================
# Usage / cost statistics
# ============================================================
def _empty_usage_bucket() -> Dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "candidate_tokens": 0,
        "thought_tokens": 0,
        "billable_output_tokens": 0,
        "cost_usd_total": 0.0,
        "usage_metadata_missing_calls": 0,
        "long_context_calls": 0,
    }


def stats_start(pdf_file: str) -> None:
    global _CURRENT_PDF
    _CURRENT_PDF = pdf_file
    bucket = _empty_usage_bucket()
    bucket.update({
        "model": GEMINI_MODEL_NAME,
        "started_at": now_iso(),
    })
    _STATS[pdf_file] = bucket


def _record_usage_into(bucket: Dict[str, Any], response: Any) -> None:
    bucket["calls"] = int(bucket.get("calls", 0)) + 1
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        bucket["usage_metadata_missing_calls"] = int(
            bucket.get("usage_metadata_missing_calls", 0)
        ) + 1
        return

    try:
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        candidate_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        thought_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    except Exception:
        bucket["usage_metadata_missing_calls"] = int(
            bucket.get("usage_metadata_missing_calls", 0)
        ) + 1
        return

    billable_output = candidate_tokens + thought_tokens
    is_long = prompt_tokens > 200_000
    input_rate = PRO_INPUT_PRICE_GT_200K if is_long else PRO_INPUT_PRICE_LE_200K
    output_rate = PRO_OUTPUT_PRICE_GT_200K if is_long else PRO_OUTPUT_PRICE_LE_200K

    request_cost = (
        prompt_tokens / 1_000_000 * input_rate
        + billable_output / 1_000_000 * output_rate
    )

    bucket["input_tokens"] = int(bucket.get("input_tokens", 0)) + prompt_tokens
    bucket["candidate_tokens"] = int(bucket.get("candidate_tokens", 0)) + candidate_tokens
    bucket["thought_tokens"] = int(bucket.get("thought_tokens", 0)) + thought_tokens
    bucket["billable_output_tokens"] = int(
        bucket.get("billable_output_tokens", 0)
    ) + billable_output
    bucket["cost_usd_total"] = round(
        float(bucket.get("cost_usd_total", 0.0)) + request_cost, 6
    )
    if is_long:
        bucket["long_context_calls"] = int(bucket.get("long_context_calls", 0)) + 1


def stats_record(response: Any) -> None:
    if not _CURRENT_PDF:
        _record_usage_into(_RUN_OVERHEAD, response)
        return
    bucket = _STATS.setdefault(_CURRENT_PDF, _empty_usage_bucket())
    _record_usage_into(bucket, response)


def stats_finish(pdf_file: str, question_count: int, status: str) -> None:
    bucket = _STATS.get(pdf_file)
    if not bucket:
        return
    bucket["ended_at"] = now_iso()
    bucket["question_count"] = int(question_count)
    bucket["status"] = status


def write_stats_report(output_dir: str) -> str:
    ensure_dir(output_dir)
    per_pdf_cost = round(
        sum(float(v.get("cost_usd_total", 0.0)) for v in _STATS.values()), 6
    )
    totals = {
        "pdfs_attempted": len(_STATS),
        "calls": sum(int(v.get("calls", 0)) for v in _STATS.values())
                 + int(_RUN_OVERHEAD.get("calls", 0)),
        "input_tokens": sum(int(v.get("input_tokens", 0)) for v in _STATS.values())
                        + int(_RUN_OVERHEAD.get("input_tokens", 0)),
        "candidate_tokens": sum(int(v.get("candidate_tokens", 0)) for v in _STATS.values())
                            + int(_RUN_OVERHEAD.get("candidate_tokens", 0)),
        "thought_tokens": sum(int(v.get("thought_tokens", 0)) for v in _STATS.values())
                          + int(_RUN_OVERHEAD.get("thought_tokens", 0)),
        "billable_output_tokens": sum(
            int(v.get("billable_output_tokens", 0)) for v in _STATS.values()
        ) + int(_RUN_OVERHEAD.get("billable_output_tokens", 0)),
        "cost_usd_total": round(
            per_pdf_cost + float(_RUN_OVERHEAD.get("cost_usd_total", 0.0)), 6
        ),
    }
    payload = {
        "generated_at": now_iso(),
        "model": GEMINI_MODEL_NAME,
        "architecture": "image_to_llm_prompt_to_canonical_json",
        "pricing_usd_per_million_tokens": {
            "input_le_200k": PRO_INPUT_PRICE_LE_200K,
            "output_le_200k": PRO_OUTPUT_PRICE_LE_200K,
            "input_gt_200k": PRO_INPUT_PRICE_GT_200K,
            "output_gt_200k": PRO_OUTPUT_PRICE_GT_200K,
            "output_includes_reasoning_tokens": True,
        },
        "run_overhead": _RUN_OVERHEAD,
        "totals": totals,
        "per_pdf": _STATS,
    }
    out = os.path.join(output_dir, "cost_report.json")
    atomic_write_json(out, payload)
    return out


# ============================================================
# PDF -> high-resolution FULL_PAGE PNG
# ============================================================
def pdf_to_page_images(
    pdf_path: str,
    image_dir: str,
    long_side_px: int = PAGE_LONG_SIDE_PX,
) -> List[str]:
    if long_side_px <= 0:
        raise ValueError("long_side_px must be > 0")
    ensure_dir(image_dir)
    paths: List[str] = []
    safe_stem = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", Path(pdf_path).stem)

    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            rect = page.rect
            longest = max(float(rect.width), float(rect.height))
            if longest <= 0:
                raise RuntimeError(f"PDF page {index + 1} has invalid dimensions")
            scale = float(long_side_px) / longest
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            out_path = os.path.join(image_dir, f"{safe_stem}_p{index + 1:03d}.png")
            pix.save(out_path)
            paths.append(out_path)

    return paths


def build_image_parts(
    page_images: List[str],
    reverse: bool = False,
    part_factory: Any = None,
    page_numbers: Optional[List[int]] = None,
) -> List[Any]:
    """Build multimodal parts while preserving the real PDF page number.

    page_numbers is used by focused audit passes that send only pages containing
    assets. This is bookkeeping only; Python does not infer document semantics.
    """
    if part_factory is None:
        from google.genai import types
        part_factory = lambda data: types.Part.from_bytes(
            data=data, mime_type="image/png"
        )

    if page_numbers is None:
        page_numbers = list(range(1, len(page_images) + 1))
    if len(page_numbers) != len(page_images):
        raise ValueError("page_numbers length must equal page_images length")

    indexed = list(zip(page_numbers, page_images))
    if reverse:
        indexed.reverse()

    parts: List[Any] = []
    for page_no, path in indexed:
        parts.append(f"FULL_PAGE page={page_no}")
        with open(path, "rb") as f:
            parts.append(part_factory(f.read()))
    return parts


# ============================================================
# Canonical output schema prompt
# ============================================================
QUESTION_TYPES = [
    "single-choice",
    "multiple-choice",
    "true-false",
    "fill-in-the-blank",
    "short-answer",
    "long-answer",
    "grouped-long-answer",
    "coding-answer",
    "draw-answer",
    "other",
]

ASSET_TYPES = [
    "table_simple",
    "table_complex",
    "formula",
    "formula_handwritten",
    "code",
    "plot_function",
    "plot_statistical",
    "figure_geometric",
    "figure_circuit",
    "figure_flowchart",
    "figure_tree",
    "figure_graph",
    "photo",
    "figure_other",
    "chemistry",
    "other",
]

RENDER_STRATEGIES = [
    "html_table",
    "latex_math",
    "code_block",
    "png_extracted",
    "description_only",
]

CANONICAL_SCHEMA_PROMPT = r"""
【唯一 Canonical Question JSON 格式】
你必須輸出「JSON 陣列」，每個元素就是一個 top-level printed question。
不得輸出 Markdown fence、說明文字或差異清單。

每題固定欄位如下：
{
  "school": "",
  "department": "",
  "exam_level": "",
  "year": "",
  "subject": "",
  "question_number": "",
  "printed_question_number": "",
  "question_text": "",
  "options": ["(A) ...", "(B) ..."],
  "type": "single-choice | multiple-choice | true-false | fill-in-the-blank | short-answer | long-answer | grouped-long-answer | coding-answer | draw-answer | other",
  "source_pages": [1],
  "grouped_subquestions": [
    {
      "label": "(a)",
      "printed_label": "(a)",
      "label_origin": "printed | synthetic",
      "question_text": "",
      "options": []
    }
  ],
  "continuation_note": "",
  "layout_blocks": [
    {
      "block_type": "paragraph | option_group | subquestion | table_ref | formula_ref | code_ref | figure_ref",
      "text": "",
      "asset_ref": "",
      "printed_label": ""
    }
  ],
  "latex_assets": [
    {
      "asset_id": "Q1-A1",
      "asset_ref": "Q1-A1",
      "asset_type": "table_simple | table_complex | formula | formula_handwritten | code | plot_function | plot_statistical | figure_geometric | figure_circuit | figure_flowchart | figure_tree | figure_graph | photo | figure_other | chemistry | other",
      "page_number": 1,
      "bbox_pct": [y_min, x_min, y_max, x_max],
      "latex": "",
      "description": "",
      "labels": [],
      "render_strategy": "html_table | latex_math | code_block | png_extracted | description_only",
      "needs_review": false,
      "confidence": 1.0,
      "notes": ""
    }
  ],
  "shared_asset_refs": [],
  "validation_issues": [],
  "needs_review": false
}

bbox_pct 規則：
- 使用 Gemini normalized coordinates 0..1000：左上 (0,0)，右下 (1000,1000)。
- 順序固定 [y_min, x_min, y_max, x_max]。這是 Gemini 官方 bounding-box 座標順序。
- 只框住素材本體與不可分割的標籤；不要框頁首、頁尾、其他題目。
- 純文字題 latex_assets=[]。

素材記錄規則：
- 一般純文字/數字表格：asset_type=table_simple/table_complex，latex 放完整精簡 HTML <table>，render_strategy=html_table。
- 公式：latex 放 LaTeX math，render_strategy=latex_math。
- 程式碼：latex 欄位存「原始程式碼文字」，必須保留縮排、換行、符號、大小寫；render_strategy=code_block。
- 圖形/統計圖/電路/流程/樹/graph/photo：latex 通常留空，保留 bbox，render_strategy=png_extracted。
- 印刷且清楚可讀的 `table_simple`、`table_complex`、`code` 絕對禁止使用
  render_strategy=png_extracted。裁切圖只能作為稽核證據與備援檔，不能取代 HTML／程式碼文字。
- 對表格或程式碼，初次輸出即使仍需後續覆核，也必須先提供逐字結構化候選：
  表格使用 HTML、程式碼使用原始 code text，並設定正確 render_strategy。
- 只有在專項放大、擴框重讀及多重獨立讀取全部完成後，仍存在實際不可辨識字元時，
  才允許 latex=""、needs_review=true；不得只因 bbox 太緊或兩次讀取不一致就改用 PNG。
- 題幹、選項或子題句子內的行內數學式不是獨立圖片素材，必須直接以
  `\(...\)` LaTeX 保存在原文字串中。例如原圖的 `y = f(x)` 必須保存為
  `\(y=f(x)\)`；沒有上橫線、箭頭或粗體的普通 `x` 絕對不得改成
  `\bar{x}`、`\vec{x}` 或其他具有額外語意的符號。
- 任何包含 LaTeX 指令的行內公式都不得裸露在一般文字中。`f(\bar{x})`、
  `\sigma^{2}`、`\Omega` 這類片段必須寫成 `\(f(\bar{x})\)`、
  `\(\sigma^{2}\)`、`\(\Omega\)`，否則前端可能無法渲染。
- 只有與一般句子分離、具有獨立視覺區塊的 displayed formula 才建立
  formula asset。不得因句中出現 `=`, `<`, `>`, `≥` 或函數記號就把整段公式跳過。

layout_blocks 規則：
- 完全依原卷視覺閱讀順序。
- 大型表格、程式碼、公式、圖片不要重複展開進 question_text；用 *_ref 指向 asset_id。
- options 已存在時，不要再把完整選項重複塞進 question_text。
- 只要 question_text、grouped_subquestions、options 任一非空，layout_blocks 就不得為空。
- 每個 grouped_subquestions 元素必須建立一個 block_type=subquestion；block.text 只能放
  該子題的 question_text 字串，禁止把整個子題 JSON object 塞進 text。

重要：Canonical JSON 是唯一語意來源。後續 Python 不會替你判斷題型、跨頁、去重、asset 類型或素材歸屬；Python 只做結構、識別碼、座標與檔案層驗證。
"""

TRANSCRIPTION_RULES_PROMPT = r"""
【文件理解與逐字轉錄規則】
1. 只依 FULL_PAGE 影像中實際可見內容轉錄；不得解題、翻譯、改寫、補常識或修正文法。
2. 題號、子題標籤、選項標籤、分數、數字、符號、百分號、括號、上下標、程式碼大小寫必須保留。
3. 每個 top-level printed question 只能輸出一次。跨頁題直接合成同一筆，source_pages 列出全部頁碼。
4. 不得用 Python/regex 的想像方式切題；以頁面視覺題界、題號與上下文判斷。
5. 選項必須是扁平字串陣列。從一個明確選項標籤到下一個標籤前的續行都屬於同一選項。
   options 陣列中的每一個元素都必須以原圖可見的選項 label 開頭，例如 `(A)`、
   `A)`、`(1)`、`1.`。沒有 label 的換行文字絕對不是新選項，必須併回前一個
   label 的 value。禁止輸出 `"the destination IP address"` 這種無 label 選項列。
6. statement-combination 題：A/B/C/D/E 敘述屬題幹；真正作答組合如 (1) AB 才是 options。
7. 子題只有原卷真的印有 label 才保留 printed_label；沒有印刷 label 時可用 item-1/item-2 作內部 label，並設 label_origin=synthetic。
8. type 以實際作答方式判斷，而不是依題目出現某個單字猜。
9. 程式碼逐字保留；不得自動修正語法。
10. 表格欄列與跨欄/跨列關係必須依視覺結構。印刷且清楚可讀的表格必須保留
    HTML 候選並進入表格專項覆核；不得在初次抽取階段直接改成 PNG。
11. 圖、流程圖、樹、電路、統計圖、幾何圖等由你判斷完整 bbox；bbox_pct 必須使用 [y_min, x_min, y_max, x_max]、0..1000，Python 只按 bbox 裁圖。
12. metadata 優先讀考卷頁首；檔名只能作頁首看不清時的弱提示。
13. 如果不確定，不要猜：保留最接近原圖內容、needs_review=true、validation_issues 說明。
14. 輸出前自己重新數一次：題數、題號、選項數、子題數、source_pages、asset 數、asset_ref。
15. 對每一題都要從題號開始沿閱讀順序掃描到下一題題號前；不得在找到第一個
    table/code/figure 後停止。題幹前、題幹中間或題幹最後的第二段素材都必須記錄。
16. 一行式函式宣告、函式骨架、prototype、`Position findFlag(Agent a) { ... }`
    之類的短程式碼仍是 code asset；不得因只有一行、位於長段落之後或靠近頁尾而省略。
17. 同一題若先有 code interface、後有說明文字、最後另有待完成的 function stub，
    必須建立兩個 code asset，layout_blocks 依 `code_ref -> paragraph -> code_ref` 保存。
"""


TEXT_OPTION_STRICT_PROMPT = r"""
【題幹、選項、子題逐字核對規則】

最高原則：FULL_PAGE 原圖中的可見字元是唯一最高權威。既有 question_text、
options、grouped_subquestions 與 layout_blocks.text 全部只能視為未驗證候選。

一、題幹與子題
1. 只轉錄原圖實際印出的文字；不得修正文法、換成同義詞、翻譯或補常識。
2. 保留題號、分數、子題標籤、項目符號、大小寫、連字號、引號及上下標。
3. `lock` 與 `Lock`、`i` 與 `I` 等識別字不得視為相同。
4. `>=`、`≥`、`<=`、`≤`、`<`、`>`、`==`、`!=`、`++`、`--` 都是不同的
   可見符號，禁止互相正規化。印刷排版的行內數學式必須用 `\(...\)` LaTeX
   保存；視覺上真正的上標 2 寫成 `^{2}`，原圖若印出字面 caret `^2` 則保留 caret。
5. 行內公式逐符號核對。普通 `x`、帶上橫線的 `\bar{x}`、帶箭頭的 `\vec{x}`、
   粗體 `\mathbf{x}` 是四個不同符號；不得根據「特徵向量」等題意自行加記號。

二、答案選項
1. 先確認原圖實際選項數量，再逐個分開讀取 label 與 value。
2. 從一個選項 label 到下一個 label 前的全部續行都屬於該選項。
3. statement-combination 題的 A/B/C/D/E 必須由左至右逐字元讀取。
4. 不得增加、刪除、交換或依常見格式補上任何字母。
5. 不得假設最後一個選項一定是 ABCDE；實際看到幾個字元就保存幾個。
6. 每個選項送出前，必須重查第一個字元、最後一個字元及字元總數。
7. options 與 layout_blocks.option_group 必須來自同一份已驗證選項矩陣。

三、候選隔離與不確定處理
1. 覆核時，候選文字若已被隱藏，不得根據題意或記憶回復舊值。
2. 若原圖與候選不同，使用原圖；不得因兩份候選相同就跳過原圖核對。
3. 任一字元無法確認時，needs_review=true，並在 validation_issues 記錄欄位與頁碼。
4. 不得以 confidence=1 掩蓋未逐字核對的欄位。
"""


CODE_EXACT_PROMPT = r"""
【程式碼逐行逐字核對規則】

1. 程式碼是需要複製的視覺資料，不是需要理解、改善或重構的程式。
2. 必須逐行轉錄；不得合併、拆分、簡化、格式化或改成語意等價寫法。
3. 特別逐字核對：數字初始值、比較運算子、`++`/`--`、識別字大小寫、
   引號、括號、大括號、分號，以及原圖中的行界。
4. 原圖分成兩行的指定與遞增敘述，禁止合併為同一行或複合運算式。
5. 原圖的迴圈起始值與邊界運算子禁止正規化成另一個等價迴圈。
6. 若程式碼排成左右兩欄，先完整讀完左欄，再完整讀完右欄；不得逐視覺行交錯兩欄。
7. 任何一行、任一字元或四邊完整性不確定時，needs_review=true；不得猜測。
8. 舊 latex 即使標成 UNTRUSTED 仍可能造成錨定；盲讀階段不得看到舊程式碼文字。
9. 印刷程式碼的 Canonical 表示必須是 asset_type=code、render_strategy=code_block，
   latex 保存原始程式碼文字。PNG crop 只能作稽核證據，不得作正常顯示格式。
10. 若裁切圖有任何一邊截斷，必須先依截斷方向擴大 bbox 並重新盲讀；
    禁止因第一次 crop_complete=false 就清空 latex 或改成 png_extracted。
11. 兩次讀取不一致時，必須進行第三次候選隔離讀取；至少兩次逐行完全一致才可通過。
12. 完成最多三輪擴框／重讀後仍有實際不可辨識字元時，保留最佳 code text 候選、
    render_strategy=code_block、needs_review=true，並把 crop 只當人工複核證據；
    不得把清楚印刷的程式碼改成正常顯示用 PNG。
"""


TABLE_STRICT_PROMPT = r"""
【表格專項嚴格視覺核對規則】

這些規則適用於 table_simple、table_complex，以及沒有外框、僅靠字元對齊、
空白或 `|` / `-` 分隔的文字表格。表格內容不得從題意、舊 JSON 或常識推測。

一、先做全頁表格清點，再建立 asset
1. 逐頁掃描所有表格候選；同一頁可能有多個 `Sample output:`、多個表頭或多張外觀相似的表格。
2. 對每個候選分別確認：page_number、頁面相對位置、最近的 printed question、表頭語意與題幹的對應。
3. 不得把同頁前一題或後一題的表格配給當前題目；不得把兩張表格的列、欄或數值混在一起。
4. 對於同頁上的 `Country | Count` 與 `Name | Count` 等相似表格，必須分別核對其
   垂直位置、最近題號、表頭及全部資料列，禁止沿用另一張表格的內容。

二、必須依序完成的視覺證據清點
對每張表格，在產生 HTML 前必須先從 FULL_PAGE 獨立確認：
1. 表頭列數。
2. 欄數，以及從最左欄到最右欄的所有欄名。
3. 資料列數（不包含表頭與分隔線）。
4. 第一筆資料列的每個 cell。
5. 最後一筆資料列的每個 cell。
6. 最右欄從第一筆到最後一筆的全部值。
7. 表格是否還向上、下、左、右延伸。

三、先建立視覺矩陣，再轉成 HTML
1. 先在內部逐格建立 headers 與 rows 矩陣，不得一邊猜測一邊直接寫 HTML。
2. HTML 中的 <th> 數量必須等於視覺欄數；<tbody> 中的 <tr> 數量必須等於視覺資料列數。
3. 每一列的 cell 數必須與表頭欄數一致，除非原圖明確有 rowspan 或 colspan。
4. 數字必須逐位讀取；不得因相鄰列、對齊偏差或舊結果增加、刪除、交換任何數位。
5. labels 必須依序存放原圖實際表頭文字。
6. notes 必須記錄可複核的簡短證據，格式為：
   `table_visual_check: columns=N; data_rows=M; last_row=...`
7. 原卷中視覺上確定為空白的表頭或儲存格是合法內容，必須以空字串 `""` 保存；
   「確定是空白」不得誤判為「無法辨識」。
8. 若某次讀取比完整定位結果多出最右側全空白欄，而且該欄表頭及所有列均為空，
   該欄是讀取代理幻覺，不是表格內容；必須排除後再做矩陣一致性比較。
9. 兩次矩陣不一致時必須執行第三次獨立讀取；至少兩份完整矩陣逐格一致才可建立 HTML。
10. 印刷且清楚可讀的表格不得因單一代理誤讀、空白作答格或局部 bbox 太緊而降級成 PNG。

四、bbox 與 `Sample output:` 的邊界定義
1. `Sample output:` 若已以 layout_blocks.table_ref.text 獨立保存，它可作為找到表格的定位線索，
   但不強制納入表格 bbox。
2. 表格 bbox 必須完整包含表頭、分隔線、所有資料列、最右欄及最後一列。
3. 不得為了包含 `Sample output:` 而同時納入其他題目；也不得為了排除標籤而截斷表頭。

五、禁止候選結果錨定
1. 看到舊 JSON 中的 HTML、bbox、description、labels、confidence 時，一律當作未驗證候選。
2. 即使初步、覆核 A、覆核 B 三份結果完全相同，也不能用一致性代替原圖核對。
3. 如果無法同時確認欄數、資料列數、最後一列與最右欄，禁止 confidence=1；
   必須 needs_review=true，並在 notes 說明不確定處。
"""


ASSET_STRICT_PROMPT = r"""
【非文字素材 / bbox 嚴格視覺定位鐵則】

最高原則：
- FULL_PAGE 原始頁面影像是唯一最高權威。
- 任何既有 asset_type、latex、description、page_number、bbox_pct 都只能當候選，
  不得因為先前 JSON 已有值就直接沿用。
- bbox 的目的不是「指出素材大概在哪裡」，而是產生一張可以直接交付使用、
  完整且不截字、不截線、不截列、不截欄的裁切圖。

一、座標格式必須絕對遵守
1. bbox_pct 固定為 [y_min, x_min, y_max, x_max]，範圍 0..1000。
2. y_min = 素材最上方；x_min = 素材最左方；y_max = 素材最下方；x_max = 素材最右方。
3. 例：素材位於頁面 x=10%~90%、y=20%~40%，正確輸出是
   [200, 100, 400, 900]，不是 [100, 200, 900, 400]。
4. 四個邊界必須各自從 FULL_PAGE 重新定位，禁止依文字長度、舊 bbox 或經驗推算。

二、完整邊界規則
1. bbox 必須包住該素材「最外側所有有意義的像素/文字/線條/框線」。
2. 任一文字字元、數字、括號、程式碼符號、表格框線、箭頭、節點、圖形線段，
   只要有任何部分落在 bbox 外，就視為錯誤。
3. 先找真實最外緣，再向外保留約 8~15 個 normalized units 的安全邊界；
   只有安全邊界會吃到另一題或無關區塊時才縮小。
4. 寧可稍微多留白，也絕對不可截斷素材。
5. 若素材本身有完整外框/矩形邊界，bbox 必須包含整個最外框，
   不得只框外框內其中一欄或其中一區。
6. 若同一個邏輯素材視覺上分成左右兩欄、上下兩區或多個相連區塊，
   bbox 必須取它們的 union rectangle，完整包住全部區塊。
7. 不要把題號、一般題幹、下一題、頁首、頁尾、水印當成素材；
   但與素材不可分割的標題、軸標籤、欄名、節點名必須包含。

三、程式碼 code
1. 必須先找到「整段程式碼」的最上、最下、最左、最右邊界。
2. 若同一程式被排成左右兩欄，兩欄仍是一個 code asset；
   bbox 必須一次包住左欄 + 右欄，不能只截左欄或只截到右欄一半。
3. 若程式碼有外框，bbox 包含完整外框。
4. latex 欄位必須逐字保留所有程式碼：縮排、換行、大小寫、括號、
   大括號、分號、運算子都不可自行修正。
5. 在送出前逐行對 FULL_PAGE：第一行、最後一行、最左字元、最右字元都必須存在。

四、表格 table_simple / table_complex
1. bbox 必須包含：完整表頭、所有欄、所有資料列、最右欄數值、最後一列以及分隔線。
2. 不得只框左半表格；不得遺漏右側數值欄；不得截斷任何 digit。
3. 先數「欄數」與「列數」，再逐 cell 從 FULL_PAGE 重讀。
4. HTML <table> 的每一個 cell 必須與原圖逐格一致。
5. 如果某一格無法確定，needs_review=true；禁止用其他列的值猜測。
6. 對 sample output 類型的文字表格，像 `Country | Count` 或 `Name | Count`，
   右側 Count 欄及所有數值是素材不可分割的一部分，bbox 必須完整包含。
7. table_simple 與可重建欄列的 table_complex 必須輸出 HTML；png_extracted 僅供
   真正無法以欄列表示的視覺圖形，不得用於一般印刷表格。

五、圖形 / tree / graph / flowchart / circuit / statistical plot
1. bbox 必須包含全部節點、全部邊/箭頭、全部標籤與最外側圖形。
2. 樹或 Pascal triangle 類圖形必須從最頂節點一直包含到最底一列；
   不能只裁上半部或前三列。
3. 流程圖不能截掉任何 connector；電路圖不能截掉任何導線或元件。
4. 有座標軸的圖必須包含完整座標軸、刻度、legend、資料線與必要標籤。

六、送出前強制做「四邊裁切模擬」
對每一個 asset 都必須在心中模擬 Python 按 bbox 裁切後的結果，逐項回答：
- TOP：最上方內容是否完整？有沒有切到文字/框線？
- BOTTOM：最後一列/最後一行/最底節點是否完整？
- LEFT：最左欄/最左字元/最左節點是否完整？
- RIGHT：最右欄數值/最右字元/最右節點是否完整？
任何一項答案不是 100% 肯定，就必須擴大 bbox 後重新檢查。

七、語意完整性檢查
1. 在決定 bbox 前，先理解題幹、選項、子題與圖形之間的關係，確認「這道題真正引用的是哪一個完整素材」。
2. 對 Pascal triangle、tree、graph、flowchart、circuit 等連續結構，不得因目前視野中某幾列/某幾個節點已形成完整小形狀就提前停止；必須沿連線、排列規律或同一視覺群組追到原卷中的真正終點。
3. 若素材上下或左右仍有與其連續、對齊、同框、同一結構的內容，視為素材仍在延伸，bbox 必須繼續擴張。
4. 若 bbox 會納入與素材無關的長水平線、頁面分隔線、前後題文字或其他題目的圖，應縮回該方向，但不得因此截掉素材本身。
5. 判斷完整性的依據是「題目語意 + FULL_PAGE 視覺結構」，不是單看局部形狀是否看起來像一張完整圖片。

八、禁止事項
- 禁止回傳「代表性小區塊」作為整個 asset 的 bbox。
- 禁止因舊 bbox 看起來合理就沿用。
- 禁止為了裁得緊而犧牲任何內容。
- 禁止只根據已轉錄的 latex/description 反推 bbox。
- 禁止把 confidence=1.0 當成預設值；只有逐邊核對無誤才能給高 confidence。
"""



QUESTION_TYPE_SET = set(QUESTION_TYPES)
ASSET_TYPE_SET = set(ASSET_TYPES)
RENDER_STRATEGY_SET = set(RENDER_STRATEGIES)
BLOCK_TYPES = {
    "paragraph", "option_group", "subquestion", "table_ref",
    "formula_ref", "code_ref", "figure_ref",
}

SUBQUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "label": {"type": "string"},
        "printed_label": {"type": "string"},
        "label_origin": {"type": "string", "enum": ["printed", "synthetic"]},
        "question_text": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["label", "printed_label", "label_origin", "question_text", "options"],
}

LAYOUT_BLOCK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "block_type": {"type": "string", "enum": sorted(BLOCK_TYPES)},
        "text": {"type": "string"},
        "asset_ref": {"type": "string"},
        "printed_label": {"type": "string"},
    },
    "required": ["block_type", "text", "asset_ref", "printed_label"],
}

ASSET_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "asset_id": {"type": "string"},
        "asset_ref": {"type": "string"},
        "asset_type": {"type": "string", "enum": ASSET_TYPES},
        "page_number": {"type": "integer", "minimum": 1},
        "bbox_pct": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1000},
            "minItems": 4,
            "maxItems": 4,
        },
        "latex": {"type": "string"},
        "description": {"type": "string"},
        "labels": {"type": "array", "items": {"type": "string"}},
        "render_strategy": {"type": "string", "enum": RENDER_STRATEGIES},
        "needs_review": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "notes": {"type": "string"},
    },
    "required": [
        "asset_id", "asset_ref", "asset_type", "page_number", "bbox_pct",
        "latex", "description", "labels", "render_strategy",
        "needs_review", "confidence", "notes",
    ],
}

POST_CROP_QC_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "asset_id": {"type": "string"},
            "status": {"type": "string", "enum": ["PASS", "RECROP", "REVIEW"]},
            "issues": {"type": "array", "items": {"type": "string"}},
            "corrected_page_number": {"type": "integer", "minimum": 1},
            "corrected_bbox_pct": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "notes": {"type": "string"},
            "observed_description": {"type": "string"},
            "observed_labels": {
                "type": "array", "items": {"type": "string"}
            },
        },
        "required": [
            "asset_id", "status", "issues", "corrected_page_number",
            "corrected_bbox_pct", "confidence", "notes",
            "observed_description", "observed_labels",
        ],
    },
}


TABLE_LOCALIZE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "asset_id": {"type": "string"},
            "status": {"type": "string", "enum": ["FOUND", "REVIEW"]},
            "issues": {"type": "array", "items": {"type": "string"}},
            "bbox_local_pct": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1000},
                "minItems": 4,
                "maxItems": 4,
            },
            "observed_headers": {
                "type": "array", "items": {"type": "string"}
            },
            "column_count": {"type": "integer", "minimum": 1},
            "data_row_count": {"type": "integer", "minimum": 0},
            "complete": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": [
            "asset_id", "status", "issues", "bbox_local_pct",
            "observed_headers", "column_count", "data_row_count",
            "complete", "notes",
        ],
    },
}


TABLE_READ_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "asset_id": {"type": "string"},
            "headers": {"type": "array", "items": {"type": "string"}},
            "rows": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "column_count": {"type": "integer", "minimum": 1},
            "data_row_count": {"type": "integer", "minimum": 0},
            "all_cells_readable": {"type": "boolean"},
            "crop_complete": {"type": "boolean"},
            "edge_issues": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": [
            "asset_id", "headers", "rows", "column_count",
            "data_row_count", "all_cells_readable", "crop_complete",
            "edge_issues", "notes",
        ],
    },
}


EXACT_TEXT_READ_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "audit_id": {"type": "string"},
            "printed_question_number": {"type": "string"},
            "paragraphs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "printed_label": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["printed_label", "text"],
                },
            },
            "options": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["label", "value"],
                },
            },
            "subquestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "printed_label": {"type": "string"},
                        "text": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "label": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                                "required": ["label", "value"],
                            },
                        },
                    },
                    "required": ["printed_label", "text", "options"],
                },
            },
            "all_characters_readable": {"type": "boolean"},
            "issues": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": [
            "audit_id", "printed_question_number", "paragraphs",
            "options", "subquestions", "all_characters_readable",
            "issues", "notes",
        ],
    },
}


CODE_READ_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "asset_id": {"type": "string"},
            "code_lines": {"type": "array", "items": {"type": "string"}},
            "all_characters_readable": {"type": "boolean"},
            "crop_complete": {"type": "boolean"},
            "edge_issues": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": [
            "asset_id", "code_lines", "all_characters_readable",
            "crop_complete", "edge_issues", "notes",
        ],
    },
}


FINAL_COVERAGE_GATE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question_number": {"type": "string"},
            "printed_question_number": {"type": "string"},
            "status": {"type": "string", "enum": ["PASS", "REVIEW"]},
            "observed_asset_count": {"type": "integer", "minimum": 0},
            "canonical_asset_count": {"type": "integer", "minimum": 0},
            "missing_asset_types": {
                "type": "array", "items": {"type": "string"}
            },
            "missing_visible_content": {
                "type": "array", "items": {"type": "string"}
            },
            "extra_canonical_content": {
                "type": "array", "items": {"type": "string"}
            },
            "symbol_mismatches": {
                "type": "array", "items": {"type": "string"}
            },
            "notes": {"type": "string"},
        },
        "required": [
            "question_number", "printed_question_number", "status",
            "observed_asset_count", "canonical_asset_count",
            "missing_asset_types", "missing_visible_content",
            "extra_canonical_content", "symbol_mismatches", "notes",
        ],
    },
}


QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "school": {"type": "string"},
        "department": {"type": "string"},
        "exam_level": {"type": "string"},
        "year": {"type": "string"},
        "subject": {"type": "string"},
        "question_number": {"type": "string"},
        "printed_question_number": {"type": "string"},
        "question_text": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
        "type": {"type": "string", "enum": QUESTION_TYPES},
        "source_pages": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 1,
        },
        "grouped_subquestions": {
            "type": "array",
            "items": SUBQUESTION_SCHEMA,
        },
        "continuation_note": {"type": "string"},
        "layout_blocks": {"type": "array", "items": LAYOUT_BLOCK_SCHEMA},
        "latex_assets": {"type": "array", "items": ASSET_SCHEMA},
        "shared_asset_refs": {"type": "array", "items": {"type": "string"}},
        "validation_issues": {"type": "array", "items": {"type": "string"}},
        "needs_review": {"type": "boolean"},
    },
    "required": [
        "school", "department", "exam_level", "year", "subject",
        "question_number", "printed_question_number", "question_text",
        "options", "type", "source_pages", "grouped_subquestions",
        "continuation_note", "layout_blocks", "latex_assets",
        "shared_asset_refs", "validation_issues", "needs_review",
    ],
}

CANONICAL_RESPONSE_JSON_SCHEMA = {
    "type": "array",
    "items": QUESTION_SCHEMA,
}


def make_generation_config() -> Any:
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_json_schema=CANONICAL_RESPONSE_JSON_SCHEMA,
    )


def make_asset_audit_config() -> Any:
    """Structured output for the dedicated final asset visual audit."""
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_json_schema={
            "type": "array",
            "items": ASSET_SCHEMA,
        },
    )


def make_post_crop_qc_config() -> Any:
    """Structured output for post-crop visual acceptance / recrop decisions."""
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=4096,
        response_mime_type="application/json",
        response_json_schema=POST_CROP_QC_SCHEMA,
    )


def make_table_localize_config() -> Any:
    """Structured output for locating one table inside an expanded context crop."""
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=4096,
        response_mime_type="application/json",
        response_json_schema=TABLE_LOCALIZE_SCHEMA,
    )


def make_table_read_config() -> Any:
    """Structured output for one blind table matrix transcription."""
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_json_schema=TABLE_READ_SCHEMA,
    )


def make_exact_text_read_config() -> Any:
    """Structured output for one candidate-free exact question text read."""
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=16384,
        response_mime_type="application/json",
        response_json_schema=EXACT_TEXT_READ_SCHEMA,
    )


def make_code_read_config() -> Any:
    """Structured output for one blind code crop transcription."""
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=16384,
        response_mime_type="application/json",
        response_json_schema=CODE_READ_SCHEMA,
    )


def make_final_coverage_gate_config() -> Any:
    """Structured output for the final non-mutating page coverage gate."""
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        max_output_tokens=16384,
        response_mime_type="application/json",
        response_json_schema=FINAL_COVERAGE_GATE_SCHEMA,
    )


def make_preflight_config() -> Any:
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=1024,
        response_mime_type="application/json",
        response_json_schema={"type": "array", "items": {"type": "object"}},
    )


# ============================================================
# JSON parsing / Gemini call
# ============================================================
def response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()
    try:
        parts: List[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(str(part_text))
        return "".join(parts).strip()
    except Exception:
        return ""


def extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    raw = (text or "").strip()
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, list):
        return None
    if any(not isinstance(item, dict) for item in obj):
        return None
    return obj


def _debug_attempt_path(base_path: Optional[str], attempt: int) -> Optional[str]:
    if not base_path:
        return None
    root, ext = os.path.splitext(base_path)
    return f"{root}_attempt{attempt}{ext or '.txt'}"


def call_model_json_array(
    client: Any,
    generation_config: Any,
    prompt: str,
    page_images: List[str],
    *,
    reverse_images: bool = False,
    retries: int = 3,
    debug_path: Optional[str] = None,
    part_factory: Any = None,
    page_numbers: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    contents: List[Any] = [prompt]
    contents.extend(
        build_image_parts(
            page_images,
            reverse=reverse_images,
            part_factory=part_factory,
            page_numbers=page_numbers,
        )
    )

    last_error = ""
    for attempt in range(1, max(1, retries) + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=contents,
                config=generation_config,
            )
            stats_record(response)
            text = response_text(response)

            attempt_path = _debug_attempt_path(debug_path, attempt)
            if attempt_path:
                ensure_dir(os.path.dirname(attempt_path) or ".")
                Path(attempt_path).write_text(text, encoding="utf-8")

            parsed = extract_json_array(text)
            if parsed is None:
                raise ValueError("模型回應不是完整合法的 JSON 陣列")

            if debug_path:
                ensure_dir(os.path.dirname(debug_path) or ".")
                Path(debug_path).write_text(text, encoding="utf-8")
            return parsed

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                wait = min(20, 2 ** attempt)
                print(
                    f"  WARNING Gemini attempt {attempt}/{retries} failed: "
                    f"{last_error[:220]}; retry in {wait}s"
                )
                time.sleep(wait)

    raise RuntimeError(f"Gemini 最終失敗：{last_error}")


# ============================================================
# Prompt-primary passes
# ============================================================
def llm_initial_extract(
    client: Any,
    generation_config: Any,
    pdf_file: str,
    page_images: List[str],
    debug_dir: str,
) -> List[Dict[str, Any]]:
    prompt = f"""
你是考卷多模態文件轉錄代理人。
你會收到完整考卷的所有 FULL_PAGE PNG，共 {len(page_images)} 頁。
來源檔名：{pdf_file}

你的任務不是先做 OCR 再分析，而是直接對頁面影像做視覺文件理解，
一次完成題界、跨頁、題型、選項、子題、表格、公式、程式碼、圖形、版面與 bbox 判斷，
輸出完整 Canonical Question JSON。

{CANONICAL_SCHEMA_PROMPT}

{TRANSCRIPTION_RULES_PROMPT}

{TEXT_OPTION_STRICT_PROMPT}

{CODE_EXACT_PROMPT}

{TABLE_STRICT_PROMPT}

{ASSET_STRICT_PROMPT}

只輸出 JSON 陣列.
"""
    return call_model_json_array(
        client, generation_config, prompt, page_images,
        retries=3,
        debug_path=os.path.join(debug_dir, "01_initial_raw.txt"),
    )


def _build_blind_table_review_context(
    initial: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Hide prior text and asset content so reviews are genuinely independent.

    Identity, order, source pages and asset ownership remain as navigation hints.
    Every transcribed value must be reconstructed from FULL_PAGE evidence.
    """
    context = json.loads(json.dumps(initial, ensure_ascii=False))
    for question in context:
        if not isinstance(question, dict):
            continue
        question["question_text"] = ""
        question["options"] = []
        question["continuation_note"] = ""
        question["validation_issues"] = []
        question["needs_review"] = False

        for sub in question.get("grouped_subquestions", []) or []:
            if not isinstance(sub, dict):
                continue
            sub["question_text"] = ""
            sub["options"] = []

        for block in question.get("layout_blocks", []) or []:
            if not isinstance(block, dict):
                continue
            block["text"] = ""

        for asset in question.get("latex_assets", []) or []:
            if not isinstance(asset, dict):
                continue
            for key in (
                "bbox_pct", "latex", "description", "labels",
                "confidence", "notes",
            ):
                asset.pop(key, None)
            asset["candidate_values_withheld_for_blind_review"] = True
            asset["review_instruction"] = (
                "Reconstruct page, bbox, type and every visible character "
                "independently from FULL_PAGE."
            )
        question["candidate_text_withheld_for_blind_review"] = True
    return context


def llm_visual_review(
    client: Any,
    generation_config: Any,
    pdf_file: str,
    page_images: List[str],
    initial: List[Dict[str, Any]],
    debug_dir: str,
    pass_name: str,
    reverse_images: bool,
) -> List[Dict[str, Any]]:
    review_context = _build_blind_table_review_context(initial)
    candidate = json.dumps(
        review_context, ensure_ascii=False, separators=(",", ":")
    )

    if str(pass_name).upper() == "A":
        stage_focus = r"""
【覆核 A：候選隔離的完整逐字重建】
1. 舊題幹、選項、子題、layout text、asset latex、bbox 與 description 都已隱藏。
2. 只使用 question_number、printed_question_number、source_pages、區塊與 asset 身分定位。
3. 必須從 FULL_PAGE 重新讀取每一題，不得根據題意或記憶補回候選文字。
4. 對答案選項分開建立 label/value 矩陣，逐字核對後才組成 options。
5. 對程式碼逐行重建，禁止語意等價改寫；對表格先建矩陣再產生 HTML。
6. 任一字元、cell 或邊界無法確認時，必須 needs_review=true，不得猜測。
"""
    else:
        stage_focus = r"""
【覆核 B：反證式字元、選項、程式碼與素材核對】
1. 舊內容已全部隱藏；只能依 FULL_PAGE 原圖獨立重建。
2. 專門尋找：選項少字/多字、最後選項截短、識別字大小寫錯誤、比較運算子被正規化、
   程式兩行被合併、跨頁續行遺失，以及表格漏最右欄/最後一列。
3. 每個答案選項至少核對兩次：第一次由左至右，第二次反向檢查最後字元與字元數。
   若 PDF 視覺換行導致一個選項分成多個印刷行，仍只能輸出成同一個 options 元素。
   不得把第二行、第三行做成新的 options 元素。
4. 每行程式碼重新核對數字、`<`/`<=`、`>`/`>=`、`++`/`--`、分號與大小寫。
5. 同頁有多個素材時，依題號、垂直位置、表頭與題目引用分別歸屬。
6. 只要任一字元、最後一列、最右欄或數字不確定，必須 needs_review=true。
"""

    prompt = f"""
你是獨立視覺覆核代理人 {pass_name}。
你會收到完整 FULL_PAGE 影像，以及第一次轉錄產生的覆核上下文。
來源檔名：{pdf_file}

第一次覆核上下文（所有舊文字、asset 內容與 bbox 已刻意隱藏）：
{candidate}

{stage_focus}

不要假設第一次 JSON 正確。重新從 FULL_PAGE 原圖獨立核對：
- 題目總數與 top-level 題界
- 題號與跨頁關係
- question_text 逐字內容
- options 數量、label、續行
- grouped_subquestions
- type
- source_pages
- layout_blocks
- 每個 asset 的存在性、類型、歸屬題目、page_number、bbox_pct
- table HTML / formula LaTeX / code 原始文字
- shared_asset_refs

你可以新增漏題、刪除誤抓題、拆分錯誤合併、合併真正跨頁題、修正任何欄位。
直接輸出一份完整、修正後、可取代第一次結果的 Canonical JSON 陣列。

{CANONICAL_SCHEMA_PROMPT}

{TRANSCRIPTION_RULES_PROMPT}

{TEXT_OPTION_STRICT_PROMPT}

{CODE_EXACT_PROMPT}

{TABLE_STRICT_PROMPT}

{ASSET_STRICT_PROMPT}

只輸出 JSON 陣列.
"""
    return call_model_json_array(
        client, generation_config, prompt, page_images,
        reverse_images=reverse_images,
        retries=3,
        debug_path=os.path.join(debug_dir, f"02_review_{pass_name}_raw.txt"),
    )


def llm_arbitrate(
    client: Any,
    generation_config: Any,
    pdf_file: str,
    page_images: List[str],
    initial: List[Dict[str, Any]],
    review_a: List[Dict[str, Any]],
    review_b: List[Dict[str, Any]],
    debug_dir: str,
) -> List[Dict[str, Any]]:
    initial_json = json.dumps(initial, ensure_ascii=False, separators=(",", ":"))
    a_json = json.dumps(review_a, ensure_ascii=False, separators=(",", ":"))
    b_json = json.dumps(review_b, ensure_ascii=False, separators=(",", ":"))
    prompt = f"""
你是最後視覺仲裁代理人。
你會收到完整考卷 FULL_PAGE PNG、第一次結果、覆核 A、覆核 B。
來源檔名：{pdf_file}

【第一次結果】
{initial_json}

【覆核 A】
{a_json}

【覆核 B】
{b_json}

不能用多數決，也不能因 A/B 相同就直接接受。
請重新查看 FULL_PAGE 原圖，以原圖為唯一最高權威來源，逐題仲裁所有差異。
如果三份結果都錯，直接改成原圖正確內容。

【文字欄位仲裁的強制順序】
1. 先列出 question_text、每個 options 元素、每個子題與每行程式碼的候選差異。
2. 在比較候選值之前，先忽略候選文字，只從 FULL_PAGE 原圖獨立讀取該欄位。
3. options 必須拆成 label/value，逐個字元核對，特別重查最後一個選項。
4. 程式碼必須逐行核對數字、比較運算子、遞增運算子、分號與識別字大小寫。
5. 完成原圖獨立讀取後才可比較三份候選；多數一致不能取代原圖證據。
6. 無法得到唯一可見結果時，保留最接近原圖的值並 needs_review=true，禁止猜測。

【表格仲裁的強制順序】
1. 在比較三份候選的表格 HTML 之前，先暫時忽略它們的所有表格值。
2. 只根據 FULL_PAGE，先獨立清點每頁所有表格，包括每張表格的相對位置、
   最近 printed question、表頭、欄數、資料列數、第一列、最後一列與最右欄所有數值。
3. 先在內部建立一份全新的 headers/rows 視覺矩陣，再回頭比較初步、A、B。
4. 三份文字結果一致不是正確證據；只要三份候選與獨立視覺矩陣不同，必須以原圖重建結果取代。
5. 同頁出現多張 `Sample output:` 時，必須對每張表格分別依垂直位置、最近題號與表頭語意歸屬，
   禁止將 `Country | Count` 與 `Name | Count` 等相鄰表格混合。
6. 任一表格的欄數、資料列數、最後一列、最右欄或數字位數無法從原圖獨立確認時，
   必須 needs_review=true，不得以三份候選的一致性作為猜測依據。

最終必須：
1. 每個 top-level printed question 恰好一筆。
2. 真正跨頁題直接合成一筆；不得留下 duplicate question records。
3. 題型、題界、asset 類型、asset 歸屬與 render_strategy 全部依原圖決定。
4. asset bbox_pct 一律使用 [y_min, x_min, y_max, x_max]、0..1000。
5. 表格 / 公式 / 程式碼可文字化時必須完整轉錄；視覺圖形保留 PNG bbox。
6. 不確定時標 needs_review，不得猜。
7. 直接輸出唯一的最終 Canonical JSON 陣列。

{CANONICAL_SCHEMA_PROMPT}

{TRANSCRIPTION_RULES_PROMPT}

{TEXT_OPTION_STRICT_PROMPT}

{CODE_EXACT_PROMPT}

{TABLE_STRICT_PROMPT}

{ASSET_STRICT_PROMPT}

只輸出 JSON 陣列，不要解釋仲裁過程.
"""
    return call_model_json_array(
        client, generation_config, prompt, page_images,
        retries=3,
        debug_path=os.path.join(debug_dir, "03_arbitrated_raw.txt"),
    )


# ============================================================
# Candidate-free full-page coverage reconciliation
# ============================================================
def llm_candidate_free_coverage_extract(
    client: Any,
    generation_config: Any,
    pdf_file: str,
    page_images: List[str],
    debug_dir: str,
) -> List[Dict[str, Any]]:
    """Build an independent complete block inventory without prior candidates.

    The normal review passes are deliberately supplied routing hints from the
    initial extraction.  That makes them good at correcting existing records,
    but it also means all reviewers can inherit the same omitted asset.  This
    pass sees no prior JSON and therefore can discover content that never
    received an asset_id in the first place.
    """
    prompt = fr"""
你是「候選隔離的全頁內容覆蓋盤點代理人」。你只會看到原始 FULL_PAGE 頁面，
完全看不到初次抽取、覆核或仲裁結果。來源檔名：{pdf_file}

請從第 1 頁到第 {len(page_images)} 頁重新建立一份完整 Canonical Question JSON。
這一輪的首要目標不是解題，而是保證每個題目區域內所有可見內容都被覆蓋。

輸出前必須在內部逐頁建立 block ledger：
1. 找出每個 top-level 題目從題號開始到下一個 top-level 題號前的完整範圍。
2. 依視覺順序清點 paragraph、option_group、subquestion、table、displayed formula、
   code、figure；頁首、頁尾、浮水印與作答備註不得列入題目內容。
3. 對每題分別記錄 table/code/formula/figure 的實際數量。找到第一個素材後仍須
   繼續掃描到題目真正結尾，禁止提早停止。
4. 一行式函式宣告、prototype、待完成的 function stub、sample call、輸出範例，
   只要是題目要求的一部分就必須保存。長段落後方或頁尾前的短 code 不得省略。
5. 同題中被說明文字隔開的兩段 code 是兩個 asset，layout_blocks 必須保留
   `code_ref -> paragraph -> code_ref` 等實際順序。
6. 可重建表格一律輸出 HTML；印刷程式碼一律輸出 code text。PNG 只給真正
   需要保留視覺外觀的 tree/graph/flowchart/circuit/plot/photo 等素材。
7. 句子內的行內公式留在文字原位置並以 `\(...\)` LaTeX 保存；不得替普通 x
   增加 bar、arrow、bold 或其他原圖沒有的數學記號。
8. 跨頁題合成一筆並列出所有 source_pages。輸出前重新核對題數、每題 block
   數及 asset 數，確認沒有任何非頁首／頁尾內容未歸屬。

asset_id 使用該份輸出內唯一的 `Q{{question_number}}-A{{sequence}}`；同一題依
視覺閱讀順序由 A1 開始。不得解題、翻譯、改寫或補常識。

{CANONICAL_SCHEMA_PROMPT}

{TRANSCRIPTION_RULES_PROMPT}

{TEXT_OPTION_STRICT_PROMPT}

{CODE_EXACT_PROMPT}

{TABLE_STRICT_PROMPT}

{ASSET_STRICT_PROMPT}

只輸出 JSON 陣列。
"""
    return call_model_json_array(
        client,
        generation_config,
        prompt,
        page_images,
        retries=3,
        debug_path=os.path.join(
            debug_dir, "03b_candidate_free_coverage_raw.txt"
        ),
    )


def llm_reconcile_full_page_coverage(
    client: Any,
    generation_config: Any,
    pdf_file: str,
    page_images: List[str],
    arbitrated: List[Dict[str, Any]],
    independent_coverage: List[Dict[str, Any]],
    debug_dir: str,
) -> List[Dict[str, Any]]:
    """Reconcile normal arbitration with an independent full-page inventory."""
    arbitrated_json = json.dumps(
        arbitrated, ensure_ascii=False, separators=(",", ":")
    )
    coverage_json = json.dumps(
        independent_coverage, ensure_ascii=False, separators=(",", ":")
    )
    prompt = fr"""
你是「全頁內容覆蓋仲裁代理人」。FULL_PAGE 原圖是唯一最高權威。
來源檔名：{pdf_file}

你會收到兩份互相獨立的完整候選：

【一般三代理仲裁結果；不可信】
{arbitrated_json}

【候選隔離全頁覆蓋盤點；不可信】
{coverage_json}

你的工作是直接重新查看 FULL_PAGE，輸出唯一完整 Canonical JSON。不得做多數決，
也不得把兩份候選做無條件聯集；只有原圖實際存在的內容才可保留。

強制覆蓋檢查：
1. 對每個 top-level 題目，從題號逐塊掃描到下一題題號前，建立完整閱讀順序。
2. 比對兩份候選的題數、paragraph、options、subquestions、assets 及 layout_blocks。
3. 若任一候選多出 table/code/formula/figure，必須回原圖確認；原圖存在就補入，
   不得因另一份候選沒有而刪除。
4. 特別尋找：第二段 code、長說明文字後的一行 function stub、跨頁開頭的 code、
   空白作答表格、獨立公式、題目最底端的圖形節點與最末資料列。
5. 同題多個素材必須各自建立 asset，並按原圖順序建立 *_ref。不得把兩個被
   prose 隔開的 code block 合併，也不得只留下較大的第一段 code。
6. 行內數學式逐符號核對並以 `\(...\)` 保存。普通 `x` 不得依題意改成
   `\bar{{x}}`、`\vec{{x}}` 或 `\mathbf{{x}}`。
7. 可重建表格必須保留 HTML 候選；清楚印刷程式碼必須保留 code text。
8. 輸出前做零遺漏宣告：每個非頁首／頁尾／浮水印的可見 block 必須恰好歸屬
   一題且恰好出現一次。無法確認時保留候選並 needs_review=true，不得靜默省略。

{CANONICAL_SCHEMA_PROMPT}

{TRANSCRIPTION_RULES_PROMPT}

{TEXT_OPTION_STRICT_PROMPT}

{CODE_EXACT_PROMPT}

{TABLE_STRICT_PROMPT}

{ASSET_STRICT_PROMPT}

只輸出 JSON 陣列。
"""
    return call_model_json_array(
        client,
        generation_config,
        prompt,
        page_images,
        retries=3,
        debug_path=os.path.join(debug_dir, "03c_coverage_reconciled_raw.txt"),
    )


# ============================================================
# Pass 4: dedicated asset geometry + transcription audit
# ============================================================
def _collect_asset_audit_candidates(
    questions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Create identity + question-context records; prior asset values are untrusted."""
    rows: List[Dict[str, Any]] = []
    for question in questions:
        qno = str(question.get("question_number", "") or "")
        printed = str(question.get("printed_question_number", "") or "")
        qtext = str(question.get("question_text", "") or "")
        options = [
            str(v) for v in (question.get("options", []) or [])
            if isinstance(v, str)
        ][:20]
        subquestions = []
        for sub in question.get("grouped_subquestions", []) or []:
            if not isinstance(sub, dict):
                continue
            subquestions.append({
                "label": str(sub.get("label", "") or ""),
                "printed_label": str(sub.get("printed_label", "") or ""),
                "question_text": str(sub.get("question_text", "") or "")[:1200],
                "options": [
                    str(v) for v in (sub.get("options", []) or [])
                    if isinstance(v, str)
                ][:12],
            })
        source_pages = [
            int(p) for p in (question.get("source_pages", []) or [])
            if isinstance(p, int)
        ]

        for asset in question.get("latex_assets", []) or []:
            if not isinstance(asset, dict):
                continue
            rows.append({
                "asset_id": str(asset.get("asset_id", "") or ""),
                "question_number": qno,
                "printed_question_number": printed,
                "question_context": {
                    "question_text": qtext[:3000],
                    "options": options,
                    "grouped_subquestions": subquestions[:20],
                    "source_pages": source_pages,
                },
                "current_asset_type_UNTRUSTED": str(asset.get("asset_type", "") or ""),
                "current_page_number_UNTRUSTED": asset.get("page_number"),
                "current_bbox_pct_UNTRUSTED": asset.get("bbox_pct"),
                "candidate_content_withheld": True,
            })
    return rows


def _validate_asset_audit_ids(
    audited_assets: Any,
    expected_asset_ids: List[str],
) -> List[str]:
    errors: List[str] = []
    if not isinstance(audited_assets, list):
        return ["asset audit root must be array"]

    seen: List[str] = []
    for i, asset in enumerate(audited_assets):
        if not isinstance(asset, dict):
            errors.append(f"asset audit [{i}] must be object")
            continue
        aid = str(asset.get("asset_id", "") or "").strip()
        if not aid:
            errors.append(f"asset audit [{i}] asset_id empty")
        seen.append(aid)
        if str(asset.get("asset_ref", "") or "").strip() != aid:
            errors.append(f"asset audit [{i}] asset_ref must equal asset_id")
        if not _valid_bbox_0_1000(asset.get("bbox_pct")):
            errors.append(f"asset audit [{i}] bbox_pct invalid")

    if len(seen) != len(set(seen)):
        errors.append("asset audit contains duplicate asset_id")

    expected = set(expected_asset_ids)
    actual = set(seen)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"asset audit missing ids: {missing}")
    if extra:
        errors.append(f"asset audit unexpected ids: {extra}")
    return errors


def llm_asset_strict_audit(
    client: Any,
    questions: List[Dict[str, Any]],
    page_images: List[str],
    debug_dir: str,
) -> List[Dict[str, Any]]:
    """Re-read every accepted asset from FULL_PAGE immediately before cropping."""
    candidates = _collect_asset_audit_candidates(questions)
    if not candidates:
        return []

    expected_ids = [row["asset_id"] for row in candidates]

    # Do not let an incorrect old page_number hide the real page from the audit.
    # Short exams: show all pages. Long exams: show source pages + candidate pages
    # + one neighboring page on each side.
    if len(page_images) <= ASSET_AUDIT_SEND_ALL_PAGES_UP_TO:
        page_numbers = list(range(1, len(page_images) + 1))
    else:
        seed_pages: set[int] = set()
        for row in candidates:
            current_page = row.get("current_page_number_UNTRUSTED")
            if isinstance(current_page, int) and 1 <= current_page <= len(page_images):
                seed_pages.add(current_page)
            context = row.get("question_context", {})
            if isinstance(context, dict):
                for p in context.get("source_pages", []) or []:
                    if isinstance(p, int) and 1 <= p <= len(page_images):
                        seed_pages.add(p)

        expanded_pages: set[int] = set()
        for p in seed_pages:
            for candidate_p in (p - 1, p, p + 1):
                if 1 <= candidate_p <= len(page_images):
                    expanded_pages.add(candidate_p)
        page_numbers = sorted(expanded_pages)

    if not page_numbers:
        raise RuntimeError("asset audit has assets but no usable page context")

    selected_images = [page_images[p - 1] for p in page_numbers]
    candidate_json = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))

    prompt = f"""
你是最後一道「Asset 專項視覺稽核代理人」。
這一輪只處理非文字素材，不重新改題號、題型、題幹、選項或子題。

你會收到：
1. 只包含目前有 asset 的 FULL_PAGE 原始頁面，頁碼標籤仍是原始 PDF 頁碼。
2. 一份 asset 候選清單。

【極重要】
候選清單中的 asset_id 是唯一必須保留的識別碼；
除此之外，current_asset_type_UNTRUSTED、current_page_number_UNTRUSTED、
current_bbox_pct_UNTRUSTED 全部視為「可能是錯的」，不得直接抄回。
舊 latex 與 description 已完全移除；不得依題意或記憶重建舊候選。

候選清單：
{candidate_json}

你的工作：
- 對每個 asset_id 恰好輸出一筆完整 asset record。
- 不可漏掉任何 asset_id，不可新增任何 asset_id。
- asset_ref 必須與 asset_id 完全相同。
- 先閱讀候選中的 question_context，理解題幹、選項、子題實際引用什麼素材，再從 FULL_PAGE 重新定位完整素材。
- 從 FULL_PAGE 重新找到素材本體，重新決定 asset_type、page_number、bbox_pct。
- 不可只因局部圖形「看起來已經完整」就停止；必須以題目語意與 FULL_PAGE 的連續結構確認真正邊界。
- 對 table / code / formula 重新從原圖逐字轉錄 latex。
- 對 figure 重新核對 description 與完整圖形範圍。
- 對 code 必須逐行建立全新內容，尤其核對初始值、比較運算子、`++`/`--`、
  分號、識別字大小寫與原始行界；禁止改成語意等價寫法。
- 尤其不要相信舊 bbox。舊 bbox 很可能只框到左半、上半或其中一欄。
- 如果同一程式碼在頁面上排成左右兩欄，必須視為同一 asset 的完整視覺區域，
  bbox 取兩欄 union rectangle。
- 如果是文字表格，必須看到最右欄與最後一列，並逐格核對數值。
- 對表格必須從 FULL_PAGE 獨立建立全新的 headers/rows 矩陣。
- 同頁存在多個 `Sample output:` 時，必須先清點每張表格的相對位置、最近題號、
  表頭、欄數、資料列數與最後一列，禁止把相鄰題目表格混合。
- 如果是樹、Pascal triangle、流程圖等，必須包含最頂到最底、最左到最右的全部元素。

{TABLE_STRICT_PROMPT}

{CODE_EXACT_PROMPT}

{ASSET_STRICT_PROMPT}

輸出格式：
- 只輸出 JSON 陣列。
- 陣列元素只能是完整 ASSET_SCHEMA 物件。
- 每個既有 asset_id 恰好一次。
- 不要輸出 question record，不要解釋。
"""
    audited = call_model_json_array(
        client,
        make_asset_audit_config(),
        prompt,
        selected_images,
        retries=3,
        debug_path=os.path.join(debug_dir, "05_asset_strict_audit_raw.txt"),
        page_numbers=page_numbers,
    )

    id_errors = _validate_asset_audit_ids(audited, expected_ids)
    if id_errors:
        atomic_write_json(
            os.path.join(debug_dir, "05_asset_strict_audit_errors.json"),
            id_errors,
        )
        raise RuntimeError(
            "Asset strict audit output invalid: " + "; ".join(id_errors[:8])
        )
    return audited


def apply_asset_strict_audit(
    questions: List[Dict[str, Any]],
    audited_assets: List[Dict[str, Any]],
) -> None:
    """Replace only asset records by exact asset_id; no semantic inference in Python."""
    audited_by_id = {
        str(asset.get("asset_id", "") or "").strip(): asset
        for asset in audited_assets
    }
    for question in questions:
        replaced: List[Dict[str, Any]] = []
        for asset in question.get("latex_assets", []) or []:
            aid = str(asset.get("asset_id", "") or "").strip()
            if aid not in audited_by_id:
                raise RuntimeError(f"Asset audit missing replacement for {aid}")
            replaced.append(dict(audited_by_id[aid]))
        question["latex_assets"] = replaced



# ============================================================
# Structural validation only
# ============================================================
REQUIRED_QUESTION_FIELDS = set(QUESTION_SCHEMA["required"])
REQUIRED_ASSET_FIELDS = set(ASSET_SCHEMA["required"])
REQUIRED_SUBQUESTION_FIELDS = set(SUBQUESTION_SCHEMA["required"])
REQUIRED_LAYOUT_FIELDS = set(LAYOUT_BLOCK_SCHEMA["required"])


def _valid_bbox_0_1000(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        y_min, x_min, y_max, x_max = [float(v) for v in value]
    except Exception:
        return False
    if not all(0 <= v <= 1000 for v in (y_min, x_min, y_max, x_max)):
        return False
    return y_max > y_min and x_max > x_min


def _pad_bbox_0_1000(bbox: Any, *, y_pad: float, x_pad: float) -> List[float]:
    if not _valid_bbox_0_1000(bbox):
        raise ValueError("invalid bbox_pct")
    y0, x0, y1, x1 = [float(value) for value in bbox]
    return [
        max(0.0, y0 - y_pad),
        max(0.0, x0 - x_pad),
        min(1000.0, y1 + y_pad),
        min(1000.0, x1 + x_pad),
    ]


def _all_strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


CHOICE_QUESTION_TYPES = {"single-choice", "multiple-choice"}

OPTION_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"\(([A-Ha-h])\)|"
    r"（([A-Ha-h])）|"
    r"([A-Ha-h])[\)\.、．:：]|"
    r"\(([1-9]|1[0-9]|20)\)|"
    r"（([1-9]|1[0-9]|20)）|"
    r"([1-9]|1[0-9]|20)[\)\.、．:：]"
    r")\s*"
)


def _option_label_match(value: Any) -> Optional[re.Match[str]]:
    return OPTION_LABEL_RE.match(str(value or ""))


def _option_label_key(value: Any) -> str:
    match = _option_label_match(value)
    if not match:
        return ""
    for group in match.groups():
        if group:
            return group.upper() if group.isalpha() else group
    return ""


def _option_label_style(key: str) -> str:
    if not key:
        return ""
    return "numeric" if key.isdigit() else "alpha"


def _option_value_after_label(value: Any) -> str:
    text = str(value or "").strip()
    match = _option_label_match(text)
    if not match:
        return text
    return text[match.end():].strip()


def _collapse_option_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_option_list_continuations(
    options: Any,
) -> Tuple[Any, bool]:
    """Merge visual wrap fragments back into the preceding labeled option."""
    if not _all_strings(options):
        return options, False

    has_labeled_option = any(_option_label_key(value) for value in options)
    if not has_labeled_option:
        return options, False

    merged: List[str] = []
    changed = False
    for raw in options:
        text = _collapse_option_text(raw)
        if not text:
            changed = True
            continue
        if _option_label_key(text):
            merged.append(text)
            continue
        if merged:
            merged[-1] = f"{merged[-1]} {text}".strip()
            changed = True
        else:
            merged.append(text)
            changed = True
    return merged, changed or merged != options


def _sync_option_layout_blocks(question: Dict[str, Any]) -> None:
    layout = question.get("layout_blocks", [])
    if not isinstance(layout, list):
        return
    option_blocks = [
        block for block in layout
        if isinstance(block, dict) and block.get("block_type") == "option_group"
    ]
    if len(option_blocks) == 1 and _all_strings(question.get("options", [])):
        option_blocks[0]["text"] = "\n".join(question.get("options", []))


LATEX_COMMAND_RE = (
    r"\\(?:bar|vec|hat|tilde|mathbf|mathrm|mathit|frac|sqrt|sum|prod|int|"
    r"mu|sigma|Omega|omega|epsilon|varepsilon|theta|lambda|leq|geq|neq|"
    r"times|cdot)\b"
)
BARE_LATEX_EXPR_RE = re.compile(
    r"(?:[A-Za-z]\s*=\s*)?(?:[A-Za-z]+\s*)?\([^()\n]*"
    + LATEX_COMMAND_RE
    + r"[^()\n]*\)"
    + r"|"
    + LATEX_COMMAND_RE
    + r"(?:\{[^{}\n]*\})?(?:\^\{?[^{}\s]+\}?)?"
)


def _inside_inline_math_delimiters(text: str, index: int) -> bool:
    before = text[:index]
    last_open = before.rfind(r"\(")
    last_close = before.rfind(r"\)")
    if last_open > last_close:
        return True
    dollar_count = len(re.findall(r"(?<!\\)\$", before))
    return bool(dollar_count % 2)


def _wrap_bare_inline_latex(text: Any) -> str:
    raw = str(text or "")
    if "\\" not in raw:
        return raw
    rebuilt: List[str] = []
    cursor = 0
    for match in BARE_LATEX_EXPR_RE.finditer(raw):
        start, end = match.span()
        expr = match.group(0)
        if _inside_inline_math_delimiters(raw, start):
            continue
        if raw[max(0, start - 2):start] in {r"\(", "$"}:
            continue
        if raw[end:end + 2] == r"\)" or raw[end:end + 1] == "$":
            continue
        rebuilt.append(raw[cursor:start])
        rebuilt.append(r"\(" + expr.strip() + r"\)")
        cursor = end
    if cursor == 0:
        return raw
    rebuilt.append(raw[cursor:])
    return "".join(rebuilt)


def normalize_inline_latex_delimiters(
    questions: List[Dict[str, Any]],
) -> None:
    for question in questions:
        if not isinstance(question, dict):
            continue
        question["question_text"] = _wrap_bare_inline_latex(
            question.get("question_text", "")
        )
        if _all_strings(question.get("options", [])):
            question["options"] = [
                _wrap_bare_inline_latex(value)
                for value in question.get("options", [])
            ]
            _sync_option_layout_blocks(question)

        for sub in question.get("grouped_subquestions", []) or []:
            if not isinstance(sub, dict):
                continue
            sub["question_text"] = _wrap_bare_inline_latex(
                sub.get("question_text", "")
            )
            if _all_strings(sub.get("options", [])):
                sub["options"] = [
                    _wrap_bare_inline_latex(value)
                    for value in sub.get("options", [])
                ]

        for block in question.get("layout_blocks", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("block_type") in {"paragraph", "subquestion", "option_group"}:
                block["text"] = _wrap_bare_inline_latex(block.get("text", ""))


def normalize_option_continuations(
    questions: List[Dict[str, Any]],
) -> None:
    """Deterministically repair a common LLM line-wrap failure.

    The model sometimes turns a visually wrapped option into separate JSON
    options.  This keeps the original text but folds any unlabeled continuation
    line into the previous labeled option before schema validation or output.
    """
    for question in questions:
        if not isinstance(question, dict):
            continue

        normalized, changed = _normalize_option_list_continuations(
            question.get("options", [])
        )
        if changed:
            question["options"] = normalized
            _sync_option_layout_blocks(question)

        for sub in question.get("grouped_subquestions", []) or []:
            if not isinstance(sub, dict):
                continue
            normalized, changed = _normalize_option_list_continuations(
                sub.get("options", [])
            )
            if changed:
                sub["options"] = normalized
    normalize_inline_latex_delimiters(questions)


def _expected_sequential_labels(keys: List[str]) -> List[str]:
    if not keys:
        return []
    style = _option_label_style(keys[0])
    if style == "alpha":
        start = ord(keys[0])
        return [chr(start + index) for index in range(len(keys))]
    if style == "numeric":
        start = int(keys[0])
        return [str(start + index) for index in range(len(keys))]
    return []


def _validate_option_list_semantics(
    options: Any,
    loc: str,
    *,
    require_labels: bool,
) -> List[str]:
    errors: List[str] = []
    if not _all_strings(options):
        return errors
    if not options:
        if require_labels:
            errors.append(f"{loc} empty for choice question")
        return errors

    keys = [_option_label_key(value) for value in options]
    has_label = any(keys)
    if require_labels and not has_label:
        errors.append(f"{loc} has no printed option labels")
        return errors
    if not has_label:
        return errors

    for index, key in enumerate(keys):
        if not key:
            errors.append(f"{loc}[{index}] missing printed option label")
        elif not _option_value_after_label(options[index]):
            errors.append(f"{loc}[{index}] has label but empty option text")

    labeled_keys = [key for key in keys if key]
    if len(labeled_keys) != len(set(labeled_keys)):
        errors.append(f"{loc} duplicate printed option labels")

    styles = {_option_label_style(key) for key in labeled_keys if key}
    styles.discard("")
    if len(styles) > 1:
        errors.append(f"{loc} mixes numeric and alphabetic option labels")
    elif labeled_keys and len(labeled_keys) == len(keys):
        if require_labels:
            first_key = labeled_keys[0]
            if _option_label_style(first_key) == "alpha" and first_key != "A":
                errors.append(f"{loc} starts at option {first_key}, expected A")
            if _option_label_style(first_key) == "numeric" and first_key != "1":
                errors.append(f"{loc} starts at option {first_key}, expected 1")
        expected = _expected_sequential_labels(labeled_keys)
        if expected and labeled_keys != expected:
            errors.append(
                f"{loc} option labels are not sequential: "
                f"{labeled_keys} != {expected}"
            )
    return errors


def validate_structure(
    questions: Any,
    page_count: int,
    *,
    require_nonempty: bool = True,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(questions, list):
        return ["root must be a JSON array"]
    if require_nonempty and page_count > 0 and not questions:
        errors.append("question array is empty for a non-empty exam PDF")

    asset_ids: set[str] = set()
    references: List[Tuple[int, str]] = []

    string_question_fields = {
        "school", "department", "exam_level", "year", "subject",
        "question_number", "printed_question_number", "question_text",
        "continuation_note",
    }

    for qi, question in enumerate(questions):
        loc = f"questions[{qi}]"
        if not isinstance(question, dict):
            errors.append(f"{loc} must be object")
            continue

        missing = sorted(REQUIRED_QUESTION_FIELDS - set(question))
        if missing:
            errors.append(f"{loc} missing fields: {missing}")

        for field in string_question_fields:
            if field in question and not isinstance(question.get(field), str):
                errors.append(f"{loc}.{field} must be string")

        qtype = question.get("type")
        if qtype not in QUESTION_TYPE_SET:
            errors.append(f"{loc}.type invalid: {qtype!r}")

        options_value = question.get("options", [])
        if not _all_strings(options_value):
            errors.append(f"{loc}.options must be list[str]")
        else:
            errors.extend(
                _validate_option_list_semantics(
                    options_value,
                    f"{loc}.options",
                    require_labels=qtype in CHOICE_QUESTION_TYPES,
                )
            )

        source_pages = question.get("source_pages", [])
        if not isinstance(source_pages, list) or not source_pages:
            errors.append(f"{loc}.source_pages must be a non-empty list")
            source_page_set: set[int] = set()
        else:
            source_page_set = set()
            for p in source_pages:
                if not isinstance(p, int) or not (1 <= p <= page_count):
                    errors.append(f"{loc}.source_pages invalid page: {p!r}")
                else:
                    source_page_set.add(p)
            if len(source_page_set) != len(source_pages):
                errors.append(f"{loc}.source_pages contains duplicates")

        subquestions = question.get("grouped_subquestions", [])
        if not isinstance(subquestions, list):
            errors.append(f"{loc}.grouped_subquestions must be list")
        else:
            for si, sub in enumerate(subquestions):
                sloc = f"{loc}.grouped_subquestions[{si}]"
                if not isinstance(sub, dict):
                    errors.append(f"{sloc} must be object")
                    continue
                smissing = sorted(REQUIRED_SUBQUESTION_FIELDS - set(sub))
                if smissing:
                    errors.append(f"{sloc} missing fields: {smissing}")
                for field in ("label", "printed_label", "question_text"):
                    if field in sub and not isinstance(sub.get(field), str):
                        errors.append(f"{sloc}.{field} must be string")
                if sub.get("label_origin") not in {"printed", "synthetic"}:
                    errors.append(f"{sloc}.label_origin invalid")
                sub_options = sub.get("options", [])
                if not _all_strings(sub_options):
                    errors.append(f"{sloc}.options must be list[str]")
                else:
                    errors.extend(
                        _validate_option_list_semantics(
                            sub_options,
                            f"{sloc}.options",
                            require_labels=bool(sub_options),
                        )
                    )

        layout = question.get("layout_blocks", [])
        if not isinstance(layout, list):
            errors.append(f"{loc}.layout_blocks must be list")
        else:
            has_renderable_content = bool(
                str(question.get("question_text", "") or "").strip()
                or (question.get("options", []) or [])
                or (question.get("grouped_subquestions", []) or [])
                or (question.get("latex_assets", []) or [])
            )
            if has_renderable_content and not layout:
                errors.append(
                    f"{loc}.layout_blocks cannot be empty when question content exists"
                )
            for bi, block in enumerate(layout):
                bloc = f"{loc}.layout_blocks[{bi}]"
                if not isinstance(block, dict):
                    errors.append(f"{bloc} must be object")
                    continue
                bmissing = sorted(REQUIRED_LAYOUT_FIELDS - set(block))
                if bmissing:
                    errors.append(f"{bloc} missing fields: {bmissing}")
                if block.get("block_type") not in BLOCK_TYPES:
                    errors.append(f"{bloc}.block_type invalid: {block.get('block_type')!r}")
                for field in ("text", "asset_ref", "printed_label"):
                    if field in block and not isinstance(block.get(field), str):
                        errors.append(f"{bloc}.{field} must be string")
                ref = block.get("asset_ref", "")
                if isinstance(ref, str) and ref.strip():
                    references.append((qi, ref.strip()))
            has_structured_asset_ref = any(
                isinstance(block, dict)
                and block.get("block_type") in {
                    "code_ref", "table_ref", "formula_ref", "figure_ref",
                }
                for block in layout
            )
            if (
                has_structured_asset_ref
                and _looks_like_structured_asset_paragraph(
                    question.get("question_text", "")
                )
            ):
                errors.append(
                    f"{loc}.question_text appears to duplicate structured asset content"
                )

        assets = question.get("latex_assets", [])
        if not isinstance(assets, list):
            errors.append(f"{loc}.latex_assets must be list")
            assets = []

        for ai, asset in enumerate(assets):
            aloc = f"{loc}.latex_assets[{ai}]"
            if not isinstance(asset, dict):
                errors.append(f"{aloc} must be object")
                continue

            amissing = sorted(REQUIRED_ASSET_FIELDS - set(asset))
            if amissing:
                errors.append(f"{aloc} missing fields: {amissing}")

            aid = asset.get("asset_id")
            aref = asset.get("asset_ref")
            if not isinstance(aid, str) or not aid.strip():
                errors.append(f"{aloc}.asset_id empty/invalid")
                aid_text = ""
            else:
                aid_text = aid.strip()
                if aid_text in asset_ids:
                    errors.append(f"duplicate asset_id: {aid_text}")
                else:
                    asset_ids.add(aid_text)

            if not isinstance(aref, str) or not aref.strip():
                errors.append(f"{aloc}.asset_ref empty/invalid")
            elif aid_text and aref.strip() != aid_text:
                errors.append(f"{aloc}.asset_ref must equal asset_id")

            if asset.get("asset_type") not in ASSET_TYPE_SET:
                errors.append(f"{aloc}.asset_type invalid: {asset.get('asset_type')!r}")
            if asset.get("render_strategy") not in RENDER_STRATEGY_SET:
                errors.append(
                    f"{aloc}.render_strategy invalid: {asset.get('render_strategy')!r}"
                )

            page_number = asset.get("page_number")
            if not isinstance(page_number, int) or not (1 <= page_number <= page_count):
                errors.append(f"{aloc}.page_number invalid: {page_number!r}")
            elif source_page_set and page_number not in source_page_set:
                errors.append(
                    f"{aloc}.page_number {page_number} not present in question source_pages"
                )

            if not _valid_bbox_0_1000(asset.get("bbox_pct")):
                errors.append(f"{aloc}.bbox_pct invalid")

            for field in ("latex", "description", "notes"):
                if field in asset and not isinstance(asset.get(field), str):
                    errors.append(f"{aloc}.{field} must be string")
            asset_type = str(asset.get("asset_type", "") or "")
            render_strategy = str(asset.get("render_strategy", "") or "")
            latex_text = str(asset.get("latex", "") or "")
            needs_asset_review = bool(asset.get("needs_review", False))
            if asset_type == "code":
                if render_strategy != "code_block":
                    errors.append(
                        f"{aloc} printed code must never render as PNG"
                    )
                if not needs_asset_review and not latex_text.strip():
                    errors.append(
                        f"{aloc} readable code must use non-empty code_block"
                    )
            if asset_type in {"table_simple", "table_complex"}:
                if render_strategy != "html_table":
                    errors.append(
                        f"{aloc} printed table must never render as PNG"
                    )
                if (
                    not needs_asset_review
                    and (
                        "<table" not in latex_text.lower()
                    or "</table>" not in latex_text.lower()
                    )
                ):
                    errors.append(
                        f"{aloc} readable table must use non-empty html_table"
                    )
            if not _all_strings(asset.get("labels", [])):
                errors.append(f"{aloc}.labels must be list[str]")
            if not isinstance(asset.get("needs_review"), bool):
                errors.append(f"{aloc}.needs_review must be boolean")

            try:
                confidence = float(asset.get("confidence"))
                if not 0.0 <= confidence <= 1.0:
                    errors.append(f"{aloc}.confidence out of range")
            except Exception:
                errors.append(f"{aloc}.confidence invalid")

        shared_refs = question.get("shared_asset_refs", [])
        if not _all_strings(shared_refs):
            errors.append(f"{loc}.shared_asset_refs must be list[str]")
        else:
            for ref in shared_refs:
                if ref.strip():
                    references.append((qi, ref.strip()))

        if not _all_strings(question.get("validation_issues", [])):
            errors.append(f"{loc}.validation_issues must be list[str]")
        if not isinstance(question.get("needs_review"), bool):
            errors.append(f"{loc}.needs_review must be boolean")

    for qi, ref in references:
        if ref not in asset_ids:
            errors.append(f"questions[{qi}] references unknown asset_id: {ref}")

    return errors


def llm_schema_repair(
    client: Any,
    generation_config: Any,
    pdf_file: str,
    page_images: List[str],
    candidate: List[Dict[str, Any]],
    errors: List[str],
    debug_dir: str,
    round_no: int,
) -> List[Dict[str, Any]]:
    candidate_json = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
    errors_json = json.dumps(errors, ensure_ascii=False)
    prompt = f"""
你是 Canonical JSON 結構修復代理人。
Python 只做機器可驗證的 schema / page / bbox / reference 檢查，發現：
{errors_json}

目前候選 JSON：
{candidate_json}

重新查看 FULL_PAGE 原圖後修正這些結構問題，同時維持原圖內容正確。
這些錯誤也可能包含選項語意檢查，例如：選擇題 options 空白、沒有印刷 label、
無 label 的續行被誤拆成新選項、label 重複、label 跳號或 `(A)`/`(1)` 混用。
遇到這類錯誤時，必須回到原圖確認每個選項 label，將沒有 label 的換行內容併回
前一個選項；不可為了通過檢查而虛構原圖不存在的選項。
不要為了通過 schema 虛構不存在的題目或 asset。
若原圖沒有 asset，移除錯誤 ref；若原圖確實有 asset，補正確 asset record 與 bbox。
bbox_pct 必須使用 [y_min, x_min, y_max, x_max]、0..1000。

{CANONICAL_SCHEMA_PROMPT}

{TRANSCRIPTION_RULES_PROMPT}

{TEXT_OPTION_STRICT_PROMPT}

{CODE_EXACT_PROMPT}

{TABLE_STRICT_PROMPT}

{ASSET_STRICT_PROMPT}

只輸出修正後的完整 JSON 陣列.
"""
    return call_model_json_array(
        client, generation_config, prompt, page_images,
        retries=3,
        debug_path=os.path.join(
            debug_dir, f"04_schema_repair_round{round_no}_raw.txt"
        ),
    )


# ============================================================
# Deterministic identifiers / crop execution
# ============================================================
def make_document_id(pdf_file: str) -> str:
    stem = re.sub(
        r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", Path(pdf_file).stem
    ).strip("_") or "document"
    digest = hashlib.sha1(pdf_file.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def namespace_asset_ids(
    questions: List[Dict[str, Any]],
    document_id: str,
) -> None:
    mapping: Dict[str, str] = {}
    for question in questions:
        for asset in question.get("latex_assets", []) or []:
            old = str(asset.get("asset_id", "") or "").strip()
            if old:
                mapping[old] = (
                    old if old.startswith(document_id + "__")
                    else f"{document_id}__{old}"
                )

    for question in questions:
        for asset in question.get("latex_assets", []) or []:
            old = str(asset.get("asset_id", "") or "").strip()
            if old in mapping:
                new = mapping[old]
                asset["asset_id"] = new
                asset["asset_ref"] = new

        for block in question.get("layout_blocks", []) or []:
            if not isinstance(block, dict):
                continue
            ref = str(block.get("asset_ref", "") or "").strip()
            if ref in mapping:
                block["asset_ref"] = mapping[ref]

        refs = question.get("shared_asset_refs", []) or []
        question["shared_asset_refs"] = [
            mapping.get(str(ref).strip(), str(ref).strip()) for ref in refs
        ]


def attach_execution_metadata(
    questions: List[Dict[str, Any]],
    pdf_file: str,
    document_id: str,
) -> None:
    for question in questions:
        question["source_pdf"] = pdf_file
        question["document_id"] = document_id


def validate_unique_audit_ids(
    questions: List[Dict[str, Any]],
) -> List[str]:
    """Return duplicate/missing final audit identifiers across documents."""
    seen: Dict[str, str] = {}
    errors: List[str] = []
    for index, question in enumerate(questions):
        verification = question.get("exact_text_verification", {})
        audit_id = (
            str(verification.get("audit_id", "") or "").strip()
            if isinstance(verification, dict) else ""
        )
        identity = (
            f"{question.get('document_id', '')}:"
            f"{question.get('question_number', index)}"
        )
        if not audit_id:
            errors.append(f"{identity}: missing exact_text audit_id")
            continue
        if audit_id in seen:
            errors.append(
                f"duplicate audit_id {audit_id}: {seen[audit_id]} and {identity}"
            )
        else:
            seen[audit_id] = identity
    return errors


def _relpath_posix(path: str, base_dir: str) -> str:
    return Path(os.path.relpath(path, base_dir)).as_posix()



def _render_bbox_crop(
    page_image: str,
    bbox: Any,
    out_path: str,
) -> Tuple[int, int, int, int]:
    """Render exactly the same bbox geometry used by the final crop step."""
    if not _valid_bbox_0_1000(bbox):
        raise ValueError("invalid bbox_pct")

    y_min, x_min, y_max, x_max = [float(v) / 1000.0 for v in bbox]
    ensure_dir(os.path.dirname(out_path) or ".")

    with Image.open(page_image) as img:
        width, height = img.size
        left = max(0, min(width - 1, int(round(x_min * width))))
        top = max(0, min(height - 1, int(round(y_min * height))))
        right = max(left + 1, min(width, int(round(x_max * width))))
        bottom = max(top + 1, min(height, int(round(y_max * height))))
        img.crop((left, top, right, bottom)).save(out_path)
    return left, top, right, bottom


def _append_asset_note(asset: Dict[str, Any], marker: str) -> None:
    prior = str(asset.get("notes", "") or "").strip()
    if marker and marker in [part.strip() for part in prior.split("|")]:
        return
    asset["notes"] = f"{prior} | {marker}".strip(" |")


def _question_semantic_context(question: Dict[str, Any]) -> Dict[str, Any]:
    subs: List[Dict[str, Any]] = []
    for sub in question.get("grouped_subquestions", []) or []:
        if not isinstance(sub, dict):
            continue
        subs.append({
            "label": str(sub.get("label", "") or ""),
            "printed_label": str(sub.get("printed_label", "") or ""),
            "question_text": str(sub.get("question_text", "") or "")[:1600],
            "options": [
                str(v) for v in (sub.get("options", []) or [])
                if isinstance(v, str)
            ][:12],
        })

    layout_context: List[Dict[str, str]] = []
    for block in question.get("layout_blocks", []) or []:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("block_type", "") or "")
        if block_type not in {
            "table_ref", "formula_ref", "code_ref", "figure_ref",
        }:
            continue
        layout_context.append({
            "block_type": block_type,
            "text": str(block.get("text", "") or "")[:500],
            "asset_ref": str(block.get("asset_ref", "") or ""),
        })

    return {
        "question_number": str(question.get("question_number", "") or ""),
        "printed_question_number": str(
            question.get("printed_question_number", "") or ""
        ),
        "question_text": str(question.get("question_text", "") or "")[:4000],
        "options": [
            str(v) for v in (question.get("options", []) or [])
            if isinstance(v, str)
        ][:24],
        "grouped_subquestions": subs[:24],
        "source_pages": [
            int(p) for p in (question.get("source_pages", []) or [])
            if isinstance(p, int)
        ],
        "asset_layout_context": layout_context[:24],
    }


def _qc_context_pages(
    question: Dict[str, Any],
    asset: Dict[str, Any],
    page_count: int,
) -> List[int]:
    pages: set[int] = set()
    current_page = asset.get("page_number")
    if isinstance(current_page, int) and 1 <= current_page <= page_count:
        pages.add(current_page)

    for p in question.get("source_pages", []) or []:
        if isinstance(p, int) and 1 <= p <= page_count:
            pages.add(p)

    # For a one-page question, also show the adjacent page(s). This helps recover
    # from a candidate page_number that is off by one without sending a whole
    # long document for every asset.
    if len(pages) <= 1:
        seeds = list(pages)
        for p in seeds:
            if p > 1:
                pages.add(p - 1)
            if p < page_count:
                pages.add(p + 1)

    return sorted(pages)


def _bbox_change_is_meaningful(old_bbox: Any, new_bbox: Any) -> bool:
    if not (_valid_bbox_0_1000(old_bbox) and _valid_bbox_0_1000(new_bbox)):
        return True
    old_values = [float(v) for v in old_bbox]
    new_values = [float(v) for v in new_bbox]
    return max(abs(a - b) for a, b in zip(old_values, new_values)) >= 1.0


def _is_visual_figure_asset(asset: Dict[str, Any]) -> bool:
    asset_type = str(asset.get("asset_type", "") or "")
    return asset_type in {
        "plot_function", "plot_statistical", "figure_geometric",
        "figure_circuit", "figure_flowchart", "figure_tree",
        "figure_graph", "photo", "figure_other", "chemistry", "other",
    }


def _apply_observed_visual_metadata(
    asset: Dict[str, Any],
    decision: Dict[str, Any],
) -> None:
    """Write figure metadata only from the crop that actually passed QC."""
    if not _is_visual_figure_asset(asset):
        return
    description = str(decision.get("observed_description", "") or "").strip()
    labels = [
        str(value) for value in (decision.get("observed_labels", []) or [])
        if isinstance(value, str)
    ]
    if description:
        asset["description"] = description
    asset["labels"] = labels
    asset["description_verification"] = {
        "status": "PASS",
        "source": "FINAL_PASSED_CROP_AND_FULL_PAGE",
        "description": description,
        "labels": labels,
    }
    asset["needs_review"] = False


def _normalized_label_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {
        re.sub(r"\s+", "", str(value)).strip().lower()
        for value in values
        if str(value).strip()
    }


def _apply_best_effort_visual_bbox_from_history(
    asset: Dict[str, Any],
    history: List[Dict[str, Any]],
    page_count: int,
    source_page_set: set[int],
) -> bool:
    """Prefer the cleanest unverified figure crop over a known dirty crop."""
    if not _is_visual_figure_asset(asset):
        return False
    expected_labels = _normalized_label_set(asset.get("labels", []))
    candidates: List[Tuple[float, float, int, List[float]]] = []
    for record in history:
        if not isinstance(record, dict):
            continue
        bbox = record.get("corrected_bbox_pct")
        page_no = record.get("corrected_page_number")
        if (
            not isinstance(page_no, int)
            or not (1 <= page_no <= page_count)
            or not _valid_bbox_0_1000(bbox)
            or (source_page_set and page_no not in source_page_set)
        ):
            continue

        observed_labels = _normalized_label_set(record.get("observed_labels", []))
        if expected_labels and not expected_labels.issubset(observed_labels):
            continue

        y0, x0, y1, x1 = [float(value) for value in bbox]
        area = (y1 - y0) * (x1 - x0)
        candidates.append((area, y1, page_no, [y0, x0, y1, x1]))

    if not candidates:
        return False

    _area, _y1, page_no, bbox = min(candidates, key=lambda item: (item[0], item[1]))
    current_page = asset.get("page_number")
    current_bbox = asset.get("bbox_pct")
    if (
        page_no == current_page
        and _valid_bbox_0_1000(current_bbox)
        and not _bbox_change_is_meaningful(current_bbox, bbox)
    ):
        return False

    asset["page_number"] = page_no
    asset["bbox_pct"] = bbox
    _append_asset_note(
        asset,
        "post_crop_qc_best_effort_smallest_labeled_visual_bbox_applied",
    )
    return True


def llm_post_crop_qc_single(
    client: Any,
    question: Dict[str, Any],
    asset: Dict[str, Any],
    page_images: List[str],
    crop_path: str,
    debug_path: str,
    round_no: int,
) -> Dict[str, Any]:
    """Inspect one actual crop against the original page(s) and question semantics."""
    from google.genai import types

    aid = str(asset.get("asset_id", "") or "").strip()
    current_page = asset.get("page_number")
    current_bbox = asset.get("bbox_pct")
    context_pages = _qc_context_pages(question, asset, len(page_images))
    if not context_pages:
        raise RuntimeError(f"{aid}: no page context available for post-crop QC")

    context = {
        "asset_id": aid,
        "asset_type": str(asset.get("asset_type", "") or ""),
        "candidate_description_withheld": True,
        "candidate_labels_withheld": True,
        "current_page_number": current_page,
        "current_bbox_pct": current_bbox,
        "question": _question_semantic_context(question),
    }
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))

    prompt = f"""
你是最後一道「實際裁切成果視覺驗收與自動重裁代理人」。

這不是 bbox 格式檢查。你必須真的比較：
1. 原始 FULL_PAGE 考卷頁面；
2. Python 依目前 bbox 真正裁出的 CURRENT_CROP；
3. 該素材所屬題目的題幹、選項、子題與 asset 語意。

目前是第 {round_no} 輪驗收。

【目前素材與題目脈絡】
{context_json}

【核心判定原則】
- 先理解題目在問什麼、題目引用哪一個完整素材，再判定裁切範圍。
- CURRENT_CROP 看起來「像一張完整小圖」不代表真的完整；必須回到 FULL_PAGE 檢查同一素材是否仍向上、下、左、右延伸。
- PASS 只允許在你能確認素材完整、四邊都未截斷、且沒有明顯無關內容時使用。
- 只要有一個方向不確定，禁止 PASS；應 RECROP 或 REVIEW。
- 如果 asset_layout_context 已將 `Sample output:` 保存為 table_ref.text，該文字只是表格定位線索，
  不強制納入表格 CURRENT_CROP。表格完整性以表頭、分隔線、所有欄、所有資料列、
  最右欄與最後一列為準；禁止因強行包入該標籤而裁到相鄰題目。
- 舊 description 與 labels 已刻意隱藏，禁止依先前描述判斷完整性。你必須直接
  從 FULL_PAGE 重新列出 observed_description 與 observed_labels。

【A. 四邊完整性】
逐邊比較 FULL_PAGE 與 CURRENT_CROP：
- TOP：素材是否還向上延伸？是否切到框線、文字、節點、線段？
- BOTTOM：最後一列、最後一行、最底節點、箭頭、導線是否真的結束？
- LEFT：最左欄、最左字元、節點、線段是否完整？
- RIGHT：最右欄、最右字元、節點、線段是否完整？

【B. 多餘內容】
如果 CURRENT_CROP 含有明顯不屬於素材的內容，例如：
- 前一題/下一題文字；
- 頁首、頁尾；
- 與圖形無關的長水平分隔線；
- 其他題目的圖；
則不能 PASS。應在不傷害素材本體的前提下縮回相應邊界。

【C. 特殊素材完整性】
- Pascal triangle：先在 FULL_PAGE 數清楚原卷中屬於該圖的完整列數，最頂列到最底列全部必須存在；不可只因前 3~4 列已形成三角形就判完整。
- tree / graph：所有屬於同一圖的節點、邊、方向與標籤都必須存在。
- flowchart：所有相連流程框與 connector/arrow 都必須存在。
- circuit：所有相連元件、導線與標籤都必須存在。
- plot：座標軸、刻度、legend、資料線與必要標籤必須完整。
- table：表頭、所有欄、所有資料列、最後一列與最右欄必須完整。
- code：第一行到最後一行、最左到最右字元必須完整。

【D. 圖形語意交叉檢查】
- 對 tree / graph / flowchart / circuit，observed_description 必須明確列出完整
  結構；tree 至少列出 root、每個 parent-child 關係及單側子節點，不得只列 labels。
- observed_labels 必須按原圖可讀順序列出全部標籤。FULL_PAGE 有而 CURRENT_CROP
  沒有的任何節點、標籤、連線或最底元素，都必須 RECROP，禁止 PASS。
- 對 table/code，observed_description 只需簡述視覺區塊，真正內容仍由後續專項
  雙讀決定；不得在這一輪根據舊 latex 補內容。

【E. RECROP 規則】
如果需要重裁：
- status="RECROP"。
- corrected_bbox_pct 必須是 FULL_PAGE 的 normalized [y_min,x_min,y_max,x_max]，0..1000，不是 CURRENT_CROP 的局部座標。
- corrected_page_number 必須是你看到該素材真正所在的原始 PDF 頁碼。
- 先找素材真正最外緣，再保留約 8~15 normalized units 的安全邊界；若安全邊界會吃到其他題目或分隔線，才縮小。
- 不可只修正一個被截斷方向而忽略其他三邊。
- corrected_bbox 應同時修正「缺內容」與「多餘內容」。

【F. REVIEW 規則】
如果 FULL_PAGE 仍無法可靠確認素材真正邊界，status="REVIEW"，不要猜。

【輸出】
只輸出一個元素的 JSON 陣列。
asset_id 必須原樣回傳。
PASS 時 corrected_page_number / corrected_bbox_pct 仍回傳目前值。
每種狀態都必須回傳 observed_description 與 observed_labels；只能寫原圖可見內容。
不要輸出解釋文字。
"""

    contents: List[Any] = [prompt]
    for p in context_pages:
        contents.append(f"FULL_PAGE page={p}")
        with open(page_images[p - 1], "rb") as f:
            contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))

    contents.append(
        "CURRENT_CROP: this image was actually rendered by Python from "
        f"page={current_page}, bbox_pct={current_bbox}"
    )
    with open(crop_path, "rb") as f:
        contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))

    last_error = ""
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=contents,
                config=make_post_crop_qc_config(),
            )
            stats_record(response)
            raw = response_text(response)

            attempt_path = _debug_attempt_path(debug_path, attempt)
            if attempt_path:
                ensure_dir(os.path.dirname(attempt_path) or ".")
                Path(attempt_path).write_text(raw, encoding="utf-8")

            parsed = extract_json_array(raw)
            if parsed is None or len(parsed) != 1:
                raise ValueError("post-crop QC must return exactly one JSON object")
            decision = parsed[0]
            if str(decision.get("asset_id", "") or "").strip() != aid:
                raise ValueError("post-crop QC asset_id mismatch")
            if decision.get("status") not in {"PASS", "RECROP", "REVIEW"}:
                raise ValueError("post-crop QC status invalid")
            if not _valid_bbox_0_1000(decision.get("corrected_bbox_pct")):
                raise ValueError("post-crop QC corrected_bbox_pct invalid")
            if not isinstance(decision.get("observed_description"), str):
                raise ValueError("post-crop QC observed_description invalid")
            if not _all_strings(decision.get("observed_labels", [])):
                raise ValueError("post-crop QC observed_labels invalid")
            if (
                decision.get("status") == "PASS"
                and _is_visual_figure_asset(asset)
                and not str(
                    decision.get("observed_description", "") or ""
                ).strip()
            ):
                raise ValueError(
                    "visual figure PASS requires observed_description"
                )

            page_no = decision.get("corrected_page_number")
            if not isinstance(page_no, int) or not (1 <= page_no <= len(page_images)):
                raise ValueError("post-crop QC corrected_page_number invalid")

            ensure_dir(os.path.dirname(debug_path) or ".")
            Path(debug_path).write_text(raw, encoding="utf-8")
            return decision

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 3:
                wait = min(20, 2 ** attempt)
                print(
                    f"  WARNING post-crop QC {aid} attempt {attempt}/3 failed: "
                    f"{last_error[:220]}; retry in {wait}s"
                )
                time.sleep(wait)

    raise RuntimeError(f"{aid}: post-crop QC failed: {last_error}")


def run_post_crop_visual_qc(
    client: Any,
    questions: List[Dict[str, Any]],
    page_images: List[str],
    debug_dir: str,
    max_rounds: int = DEFAULT_MAX_POST_CROP_QC_ROUNDS,
) -> List[Dict[str, Any]]:
    """Render -> visually inspect -> correct bbox -> re-render, per asset."""
    qc_root = os.path.join(debug_dir, "post_crop_qc")
    ensure_dir(qc_root)
    report: List[Dict[str, Any]] = []

    max_rounds = max(1, int(max_rounds))

    for question in questions:
        source_page_set = {
            int(p) for p in (question.get("source_pages", []) or [])
            if isinstance(p, int) and 1 <= p <= len(page_images)
        }

        for asset in question.get("latex_assets", []) or []:
            if not isinstance(asset, dict):
                continue

            aid = str(asset.get("asset_id", "") or "").strip() or "asset"
            safe_aid = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", aid)
            history: List[Dict[str, Any]] = []
            final_status = "REVIEW"

            for round_no in range(1, max_rounds + 1):
                page_no = asset.get("page_number")
                bbox_before = asset.get("bbox_pct")

                if (
                    not isinstance(page_no, int)
                    or not (1 <= page_no <= len(page_images))
                    or not _valid_bbox_0_1000(bbox_before)
                ):
                    asset["needs_review"] = True
                    _append_asset_note(asset, "post_crop_qc_invalid_page_or_bbox")
                    history.append({
                        "round": round_no,
                        "status": "REVIEW",
                        "issues": ["INVALID_PAGE_OR_BBOX"],
                        "page_number_before": page_no,
                        "bbox_before": bbox_before,
                    })
                    break

                crop_path = os.path.join(
                    qc_root, f"{safe_aid}_round{round_no}_crop.png"
                )
                pixels = _render_bbox_crop(
                    page_images[page_no - 1], bbox_before, crop_path
                )

                decision_debug = os.path.join(
                    qc_root, f"{safe_aid}_round{round_no}_decision.json"
                )
                try:
                    decision = llm_post_crop_qc_single(
                        client,
                        question,
                        asset,
                        page_images,
                        crop_path,
                        decision_debug,
                        round_no,
                    )
                except Exception as exc:
                    asset["needs_review"] = True
                    _append_asset_note(
                        asset,
                        f"post_crop_qc_call_failed:{type(exc).__name__}",
                    )
                    history.append({
                        "round": round_no,
                        "status": "REVIEW",
                        "issues": ["QC_CALL_FAILED"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "page_number_before": page_no,
                        "bbox_before": bbox_before,
                        "pixel_crop": list(pixels),
                    })
                    break

                status = str(decision.get("status", "") or "")
                corrected_page = decision.get("corrected_page_number")
                corrected_bbox = decision.get("corrected_bbox_pct")
                issues = [
                    str(v) for v in (decision.get("issues", []) or [])
                    if isinstance(v, str)
                ]
                history.append({
                    "round": round_no,
                    "status": status,
                    "issues": issues,
                    "confidence": decision.get("confidence"),
                    "notes": str(decision.get("notes", "") or ""),
                    "page_number_before": page_no,
                    "bbox_before": bbox_before,
                    "pixel_crop": list(pixels),
                    "corrected_page_number": corrected_page,
                    "corrected_bbox_pct": corrected_bbox,
                    "observed_description": str(
                        decision.get("observed_description", "") or ""
                    ),
                    "observed_labels": decision.get("observed_labels", []) or [],
                })

                if status == "PASS":
                    _apply_observed_visual_metadata(asset, decision)
                    final_status = "PASS"
                    break

                if status == "REVIEW":
                    asset["needs_review"] = True
                    _append_asset_note(
                        asset,
                        "post_crop_qc_review:" + (
                            ",".join(issues[:6]) if issues else "uncertain"
                        ),
                    )
                    final_status = "REVIEW"
                    break

                # RECROP
                if (
                    not isinstance(corrected_page, int)
                    or not (1 <= corrected_page <= len(page_images))
                    or not _valid_bbox_0_1000(corrected_bbox)
                ):
                    asset["needs_review"] = True
                    _append_asset_note(asset, "post_crop_qc_invalid_correction")
                    final_status = "REVIEW"
                    break

                # Asset page must remain consistent with the question's source pages.
                # If the model wants a page outside source_pages, do not silently alter
                # question semantics here; escalate it for manual review.
                if source_page_set and corrected_page not in source_page_set:
                    asset["needs_review"] = True
                    _append_asset_note(
                        asset,
                        f"post_crop_qc_page_outside_source_pages:{corrected_page}",
                    )
                    final_status = "REVIEW"
                    break

                if (
                    corrected_page == page_no
                    and not _bbox_change_is_meaningful(bbox_before, corrected_bbox)
                ):
                    asset["needs_review"] = True
                    _append_asset_note(asset, "post_crop_qc_recrop_without_change")
                    final_status = "REVIEW"
                    break

                asset["page_number"] = corrected_page
                asset["bbox_pct"] = [float(v) for v in corrected_bbox]
                final_status = "RECROP"

            # A correction made in the last allowed round has not yet been
            # inspected. Run one confirmation-only pass so RECROP can never be
            # mistaken for a verified final crop by a later content reader.
            if final_status == "RECROP":
                confirmation_round = max_rounds + 1
                page_no = asset.get("page_number")
                bbox_before = asset.get("bbox_pct")
                confirmation_crop = os.path.join(
                    qc_root, f"{safe_aid}_confirmation_crop.png"
                )
                try:
                    if (
                        not isinstance(page_no, int)
                        or not (1 <= page_no <= len(page_images))
                        or not _valid_bbox_0_1000(bbox_before)
                    ):
                        raise ValueError("invalid final page or bbox")
                    pixels = _render_bbox_crop(
                        page_images[page_no - 1], bbox_before, confirmation_crop
                    )
                    decision = llm_post_crop_qc_single(
                        client,
                        question,
                        asset,
                        page_images,
                        confirmation_crop,
                        os.path.join(
                            qc_root, f"{safe_aid}_confirmation_decision.json"
                        ),
                        confirmation_round,
                    )
                    confirmation_status = str(
                        decision.get("status", "") or ""
                    )
                    history.append({
                        "round": confirmation_round,
                        "confirmation_only": True,
                        "status": confirmation_status,
                        "issues": [
                            str(v) for v in (decision.get("issues", []) or [])
                            if isinstance(v, str)
                        ],
                        "confidence": decision.get("confidence"),
                        "notes": str(decision.get("notes", "") or ""),
                        "page_number_before": page_no,
                        "bbox_before": bbox_before,
                        "pixel_crop": list(pixels),
                        "corrected_page_number": decision.get(
                            "corrected_page_number"
                        ),
                        "corrected_bbox_pct": decision.get(
                            "corrected_bbox_pct"
                        ),
                        "observed_description": str(
                            decision.get("observed_description", "") or ""
                        ),
                        "observed_labels": (
                            decision.get("observed_labels", []) or []
                        ),
                    })
                    if confirmation_status == "PASS":
                        _apply_observed_visual_metadata(asset, decision)
                        final_status = "PASS"
                    elif confirmation_status == "RECROP":
                        # The previous version detected a better bbox in the
                        # confirmation call but discarded it.  Apply one final
                        # meaningful correction and verify the newly rendered
                        # crop once; never label an uninspected correction PASS.
                        final_page = decision.get("corrected_page_number")
                        final_bbox = decision.get("corrected_bbox_pct")
                        if (
                            isinstance(final_page, int)
                            and 1 <= final_page <= len(page_images)
                            and _valid_bbox_0_1000(final_bbox)
                            and (
                                not source_page_set
                                or final_page in source_page_set
                            )
                            and (
                                final_page != page_no
                                or _bbox_change_is_meaningful(
                                    bbox_before, final_bbox
                                )
                            )
                        ):
                            asset["page_number"] = final_page
                            asset["bbox_pct"] = [
                                float(value) for value in final_bbox
                            ]
                            last_crop = os.path.join(
                                qc_root,
                                f"{safe_aid}_final_correction_crop.png",
                            )
                            last_pixels = _render_bbox_crop(
                                page_images[final_page - 1],
                                asset["bbox_pct"],
                                last_crop,
                            )
                            last_decision = llm_post_crop_qc_single(
                                client,
                                question,
                                asset,
                                page_images,
                                last_crop,
                                os.path.join(
                                    qc_root,
                                    f"{safe_aid}_final_correction_decision.json",
                                ),
                                confirmation_round + 1,
                            )
                            last_status = str(
                                last_decision.get("status", "") or ""
                            )
                            history.append({
                                "round": confirmation_round + 1,
                                "final_correction_confirmation": True,
                                "status": last_status,
                                "issues": [
                                    str(value) for value in (
                                        last_decision.get("issues", []) or []
                                    ) if isinstance(value, str)
                                ],
                                "confidence": last_decision.get("confidence"),
                                "notes": str(
                                    last_decision.get("notes", "") or ""
                                ),
                                "page_number_before": final_page,
                                "bbox_before": asset["bbox_pct"],
                                "pixel_crop": list(last_pixels),
                                "corrected_page_number": last_decision.get(
                                    "corrected_page_number"
                                ),
                                "corrected_bbox_pct": last_decision.get(
                                    "corrected_bbox_pct"
                                ),
                                "observed_description": str(
                                    last_decision.get(
                                        "observed_description", ""
                                    ) or ""
                                ),
                                "observed_labels": (
                                    last_decision.get("observed_labels", []) or []
                                ),
                            })
                            if last_status == "PASS":
                                _apply_observed_visual_metadata(
                                    asset, last_decision
                                )
                                final_status = "PASS"
                            else:
                                if last_status == "RECROP":
                                    pending_page = last_decision.get(
                                        "corrected_page_number"
                                    )
                                    pending_bbox = last_decision.get(
                                        "corrected_bbox_pct"
                                    )
                                    if (
                                        isinstance(pending_page, int)
                                        and 1 <= pending_page <= len(page_images)
                                        and _valid_bbox_0_1000(pending_bbox)
                                        and (
                                            not source_page_set
                                            or pending_page in source_page_set
                                        )
                                        and (
                                            pending_page != final_page
                                            or _bbox_change_is_meaningful(
                                                asset.get("bbox_pct"),
                                                pending_bbox,
                                            )
                                        )
                                    ):
                                        asset["page_number"] = pending_page
                                        asset["bbox_pct"] = [
                                            float(value) for value in pending_bbox
                                        ]
                                        _append_asset_note(
                                            asset,
                                            "post_crop_qc_best_effort_unverified_correction_applied",
                                        )
                                final_status = "REVIEW"
                        else:
                            final_status = "REVIEW"
                        if final_status != "PASS":
                            asset["needs_review"] = True
                            _append_asset_note(
                                asset,
                                "post_crop_qc_final_correction_unverified",
                            )
                    else:
                        final_status = "REVIEW"
                        asset["needs_review"] = True
                        _append_asset_note(
                            asset,
                            "post_crop_qc_final_confirmation_failed:"
                            + (confirmation_status or "INVALID_STATUS"),
                        )
                except Exception as exc:
                    final_status = "REVIEW"
                    asset["needs_review"] = True
                    history.append({
                        "round": confirmation_round,
                        "confirmation_only": True,
                        "status": "REVIEW",
                        "issues": ["FINAL_CONFIRMATION_CALL_FAILED"],
                        "error": f"{type(exc).__name__}: {exc}",
                        "page_number_before": page_no,
                        "bbox_before": bbox_before,
                    })
                    _append_asset_note(
                        asset,
                        "post_crop_qc_final_confirmation_call_failed:"
                        f"{type(exc).__name__}",
                    )

            if final_status != "PASS":
                asset["needs_review"] = True
                _apply_best_effort_visual_bbox_from_history(
                    asset, history, len(page_images), source_page_set
                )
                if final_status == "RECROP":
                    _append_asset_note(
                        asset,
                        f"post_crop_qc_not_verified_after_{max_rounds}_rounds",
                    )

            asset["post_crop_qc"] = {
                "status": final_status,
                "rounds_used": len(history),
                "history": history,
            }
            report.append({
                "asset_id": aid,
                "status": final_status,
                "rounds_used": len(history),
                "final_page_number": asset.get("page_number"),
                "final_bbox_pct": asset.get("bbox_pct"),
                "needs_review": bool(asset.get("needs_review", False)),
                "history": history,
            })

    return report


def _is_table_asset(asset: Dict[str, Any]) -> bool:
    return (
        asset.get("asset_type") in {"table_simple", "table_complex"}
        or asset.get("render_strategy") == "html_table"
    )


def _expand_table_context_bbox(bbox: Any) -> List[float]:
    """Expand a near-table candidate without swallowing a vertically adjacent table."""
    if not _valid_bbox_0_1000(bbox):
        raise ValueError("invalid candidate table bbox")
    y0, x0, y1, x1 = [float(v) for v in bbox]
    height = y1 - y0
    width = x1 - x0
    y_pad = max(TABLE_CONTEXT_MIN_Y_PAD, height * TABLE_CONTEXT_PAD_RATIO)
    x_pad = max(TABLE_CONTEXT_MIN_X_PAD, width * TABLE_CONTEXT_PAD_RATIO)
    return [
        max(0.0, y0 - y_pad),
        max(0.0, x0 - x_pad),
        min(1000.0, y1 + y_pad),
        min(1000.0, x1 + x_pad),
    ]


def _local_bbox_to_full_page(
    context_bbox: Any,
    local_bbox: Any,
) -> List[float]:
    """Map normalized context-crop coordinates back to full-page coordinates."""
    if not _valid_bbox_0_1000(context_bbox):
        raise ValueError("invalid context bbox")
    if not _valid_bbox_0_1000(local_bbox):
        raise ValueError("invalid local bbox")
    cy0, cx0, cy1, cx1 = [float(v) for v in context_bbox]
    ly0, lx0, ly1, lx1 = [float(v) for v in local_bbox]
    ch = cy1 - cy0
    cw = cx1 - cx0
    mapped = [
        cy0 + (ly0 / 1000.0) * ch,
        cx0 + (lx0 / 1000.0) * cw,
        cy0 + (ly1 / 1000.0) * ch,
        cx0 + (lx1 / 1000.0) * cw,
    ]
    # Small full-page padding protects headers, dashed separators, rightmost
    # digits, and the last baseline from roundoff or an overly tight local bbox.
    mapped = [
        mapped[0] - TABLE_FINAL_SAFETY_PAD,
        mapped[1] - TABLE_FINAL_SAFETY_PAD,
        mapped[2] + TABLE_FINAL_SAFETY_PAD,
        mapped[3] + TABLE_FINAL_SAFETY_PAD,
    ]
    return [round(max(0.0, min(1000.0, v)), 3) for v in mapped]


def _resize_image_min_long_side(path: str, min_long_side: int) -> None:
    with Image.open(path) as source:
        image = source.convert("RGB")
        long_side = max(image.size)
        if long_side >= int(min_long_side):
            image.save(path)
            return
        scale = float(min_long_side) / max(1, long_side)
        size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        image.resize(size, Image.Resampling.LANCZOS).save(path)


def _render_enlarged_bbox_crop(
    page_image: str,
    bbox: Any,
    out_path: str,
    min_long_side: int = TABLE_READ_MIN_LONG_SIDE_PX,
) -> None:
    _render_bbox_crop(page_image, bbox, out_path)
    _resize_image_min_long_side(out_path, min_long_side)


def _make_enhanced_table_variant(source_path: str, out_path: str) -> None:
    """Create a second non-semantic visual view for an independent blind read."""
    ensure_dir(os.path.dirname(out_path) or ".")
    with Image.open(source_path) as source:
        gray = ImageOps.grayscale(source)
        enhanced = ImageOps.autocontrast(gray, cutoff=1)
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        enhanced.save(out_path)


def _normalized_verbatim_text(value: Any) -> str:
    """Normalize transport-only whitespace without changing visible characters."""
    text = unicodedata.normalize(
        "NFC", str(value if value is not None else "")
    ).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.strip().split("\n"))


def _normalized_compare_text(value: Any) -> str:
    """Normalize layout whitespace only; spelling, case and symbols stay exact."""
    return re.sub(r"\s+", " ", _normalized_verbatim_text(value)).strip()


def _printed_label_key(value: Any) -> str:
    """Identity form for locating labels; the verbatim label is still retained."""
    text = _normalized_compare_text(value).casefold()
    text = re.sub(r"^[\(\[\{]+", "", text)
    text = re.sub(r"[\)\]\}\.:：]+$", "", text)
    return text.strip()


def _compose_printed_text(label: Any, text: Any) -> str:
    """Combine a separately returned printed label with its visible text once."""
    raw_label = _normalized_verbatim_text(label)
    raw_text = _normalized_verbatim_text(text)
    if not raw_label:
        return raw_text
    if not raw_text:
        return raw_label
    label_cmp = _normalized_compare_text(raw_label).casefold()
    text_cmp = _normalized_compare_text(raw_text).casefold()
    if text_cmp == label_cmp or text_cmp.startswith(label_cmp + " "):
        return raw_text
    return f"{raw_label} {raw_text}".strip()


def _question_audit_id(
    question: Dict[str, Any],
    document_id: str,
) -> str:
    """Stable document-scoped routing id.

    Question number + printed number + source pages are not globally unique:
    two different exam years routinely have the same tuple.  The document id
    must participate in the digest or batch audit records can overwrite one
    another.
    """
    payload = {
        "document_id": str(document_id or ""),
        "question_number": str(question.get("question_number", "") or ""),
        "printed_question_number": str(
            question.get("printed_question_number", "") or ""
        ),
        "source_pages": [
            int(p) for p in (question.get("source_pages", []) or [])
            if isinstance(p, int)
        ],
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"EXACT-{digest}"


def _exact_text_read_prompt(
    question: Dict[str, Any],
    role: str,
    document_id: str,
) -> str:
    audit_id = _question_audit_id(question, document_id)
    qno = str(question.get("question_number", "") or "").strip()
    printed = str(question.get("printed_question_number", "") or "").strip()
    source_pages = [
        int(p) for p in (question.get("source_pages", []) or [])
        if isinstance(p, int)
    ]
    qtype = str(question.get("type", "") or "")
    grouped = [
        {
            "printed_label": str(sub.get("printed_label", "") or ""),
            "option_count_hint": len(sub.get("options", []) or []),
        }
        for sub in (question.get("grouped_subquestions", []) or [])
        if isinstance(sub, dict)
    ]
    return fr"""
你是單一考題的逐字盲讀代理人 {role}。你只會看到該題所在的 FULL_PAGE 原圖，
不會看到任何舊 question_text、options、子題文字或其他代理人的結果。

audit_id（只需原樣回傳，不是原卷題號）：{audit_id}
唯一目標 top-level printed_question_number：{printed}
source_pages：{json.dumps(source_pages, ensure_ascii=False)}
候選題型僅供定位：{qtype}
預期群組子題的印刷標籤與選項數定位提示：
{json.dumps(grouped, ensure_ascii=False)}

定位鐵則：
1. 只能以 top-level printed_question_number={printed!r} 定位父題。
2. 內部流水序號 {qno!r} 不是原卷題號，不得拿它尋找同名小題。
3. 若目標是 B-1、A-2、II-7 等群組父題，必須讀完整父題與其全部子題；
   不得改抓群組內的 `1.`、`2.`、`3.` 等局部小題。
4. paragraphs[0].printed_label 不得重複放 top-level 題號；top-level 題號只放在
   printed_question_number。

請在原圖找到這一題，輸出下列視覺矩陣：
1. paragraphs：一般題幹段落，依閱讀順序；不包含答案選項、子題、表格、程式碼、公式或圖片。
   statement-combination 題的 A)/B)/C)/D)/E) 敘述仍屬題幹，保留在 paragraph.text，
   每個 statement 以換行分隔；原圖的視覺自動換行要合併回同一敘述。
2. options：真正可勾選的答案列。label 與 value 分開輸出；value 不得包含 label。
   每一列都必須有 label。若原圖選項跨多個視覺行，後續行沒有 label 時必須併入
   前一列的 value，禁止建立 label="" 的新列。
3. subquestions：印刷子題。printed_label 只放原圖實際標籤或項目符號；text 不重複標籤。
   每個群組選擇題的答案列必須放在該 subquestion.options，label/value 分開；
   禁止把子題選項塞進 subquestion.text，也禁止省略群組選項。
4. 獨立成塊的程式碼、表格、displayed formula 與圖形只當作跳過的 asset，
   不得展開到 paragraphs；但句子內的行內公式必須以 `\(...\)` 原位轉錄，
   不得跳過、改寫或依題意增加 bar/vector/bold 等原圖不存在的記號。
5. 跨頁題必須依 source_pages 串接完整；不得加入同頁相鄰題目。
6. 每個字元逐字保留，尤其是選項末字、識別字大小寫、比較運算子、引號、`≥`/`≤`、
   上下標與印刷項目符號。不得翻譯、修正文法或正規化符號。
7. 若返回的題幹／選項／子題中任一字元不清楚，all_characters_readable=false
   並列出 issues；不得猜測。被明確跳過、交由 asset 專項處理的 code/table/figure
   不得拿來判定這一輪 all_characters_readable=false，也不得在 issues 抱怨其水印。
8. 不要輸出候選差異或解題結果，只輸出一個元素的 JSON 陣列。

{TEXT_OPTION_STRICT_PROMPT}
"""


def _exact_text_arbitration_prompt(
    question: Dict[str, Any],
    read_a: Dict[str, Any],
    read_b: Dict[str, Any],
    document_id: str,
) -> str:
    audit_id = _question_audit_id(question, document_id)
    printed = str(question.get("printed_question_number", "") or "").strip()
    source_pages = [
        int(p) for p in (question.get("source_pages", []) or [])
        if isinstance(p, int)
    ]
    return fr"""
你是逐字轉錄的局部視覺仲裁代理人。FULL_PAGE 原圖是唯一最高權威。
你會看到同一個 top-level 題目的兩份獨立盲讀候選；候選只能協助指出差異，
不能取代你重新查看原圖。

audit_id（原樣回傳）：{audit_id}
唯一目標 top-level printed_question_number：{printed}
source_pages：{json.dumps(source_pages, ensure_ascii=False)}

【盲讀 A】
{json.dumps(read_a, ensure_ascii=False, separators=(",", ":"))}

【盲讀 B】
{json.dumps(read_b, ensure_ascii=False, separators=(",", ":"))}

強制規則：
1. 先在 FULL_PAGE 找到 printed_question_number={printed!r} 的完整父題，再仲裁。
2. 不得改抓父題內具有相同數字的局部小題。
3. 逐字重建 paragraphs、top-level options、全部 subquestions，以及每個
   subquestion.options；不得使用多數決。
4. labels 與文字分欄輸出；不得把選項合併進題幹。
5. paragraphs 只能放一般題幹 prose。獨立成塊的 code/table/displayed formula/figure
   已由 layout_blocks 的 *_ref 與 asset 專項處理，禁止放入 paragraphs。
   若盲讀 A 或 B 的 notes 說某段 code/table 是 asset，仲裁時不得再把它補進 paragraphs。
6. 原卷文法或拼字即使錯誤也照錄，不得改寫。
7. 若仍有任何字元無法確認，all_characters_readable=false 並列出 issues。
8. 只輸出一個元素的 JSON 陣列。

{TEXT_OPTION_STRICT_PROMPT}
"""


def _normalize_exact_text_read(
    result: Any,
    expected_audit_id: str,
    expected_printed_question_number: str,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(result, dict):
        return None, ["exact text read is not object"]
    if str(result.get("audit_id", "") or "").strip() != expected_audit_id:
        errors.append("audit_id mismatch")
    actual_printed = _normalized_verbatim_text(
        result.get("printed_question_number", "")
    )
    if (
        expected_printed_question_number
        and _printed_label_key(actual_printed)
        != _printed_label_key(expected_printed_question_number)
    ):
        errors.append(
            "printed_question_number target mismatch: "
            f"{actual_printed!r} != {expected_printed_question_number!r}"
        )
    if not bool(result.get("all_characters_readable", False)):
        errors.append("not all characters readable")
    raw_issues = result.get("issues", [])
    if not isinstance(raw_issues, list):
        errors.append("issues must be list")
        issues: List[str] = []
    else:
        issues = [str(v).strip() for v in raw_issues if str(v).strip()]
    if issues:
        errors.extend(f"reader_issue:{value}" for value in issues)

    def normalize_options(value: Any, field_name: str) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        if not isinstance(value, list):
            errors.append(f"{field_name} must be list")
            return rows
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"{field_name}[{index}] must be object")
                continue
            label = _normalized_verbatim_text(item.get("label", ""))
            option_value = _normalized_verbatim_text(item.get("value", ""))
            if not label:
                errors.append(f"{field_name}[{index}] label blank")
            rows.append({"label": label, "value": option_value})
        labels = [_printed_label_key(row["label"]) for row in rows]
        if len(labels) != len(set(labels)):
            errors.append(f"{field_name} duplicate option labels")
        return rows

    def normalize_text_records(
        value: Any,
        field_name: str,
    ) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        if not isinstance(value, list):
            errors.append(f"{field_name} must be list")
            return rows
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                errors.append(f"{field_name}[{index}] must be object")
                continue
            row = {
                "printed_label": _normalized_verbatim_text(
                    item.get("printed_label", "")
                ),
                "text": _normalized_verbatim_text(item.get("text", "")),
            }
            if not row["text"]:
                errors.append(f"{field_name}[{index}] text blank")
            rows.append(row)
        return rows

    paragraphs = normalize_text_records(result.get("paragraphs", []), "paragraphs")
    options = normalize_options(result.get("options", []), "options")
    subquestions: List[Dict[str, Any]] = []
    raw_subquestions = result.get("subquestions", [])
    if not isinstance(raw_subquestions, list):
        errors.append("subquestions must be list")
    else:
        for index, item in enumerate(raw_subquestions):
            if not isinstance(item, dict):
                errors.append(f"subquestions[{index}] must be object")
                continue
            text = _normalized_verbatim_text(item.get("text", ""))
            if not text:
                errors.append(f"subquestions[{index}] text blank")
            subquestions.append({
                "printed_label": _normalized_verbatim_text(
                    item.get("printed_label", "")
                ),
                "text": text,
                "options": normalize_options(
                    item.get("options", []), f"subquestions[{index}].options"
                ),
            })

    # A reader may duplicate the top-level number as the first paragraph label.
    # It is a storage difference, not a visible text disagreement.
    if paragraphs and (
        _printed_label_key(paragraphs[0].get("printed_label", ""))
        == _printed_label_key(expected_printed_question_number)
    ):
        paragraphs[0]["printed_label"] = ""

    if not paragraphs:
        errors.append("question has no readable paragraph")

    return {
        "audit_id": expected_audit_id,
        "printed_question_number": actual_printed,
        "paragraphs": paragraphs,
        "options": options,
        "subquestions": subquestions,
    }, errors


def _exact_text_matrix_key(matrix: Dict[str, Any]) -> Tuple[Any, ...]:
    paragraph_text = _normalized_compare_text(" ".join(
        _compose_printed_text(row.get("printed_label", ""), row.get("text", ""))
        for row in (matrix.get("paragraphs", []) or [])
    ))
    options = tuple(
        (
            _printed_label_key(row.get("label", "")),
            _normalized_compare_text(row.get("value", "")),
        )
        for row in (matrix.get("options", []) or [])
    )
    subquestions = tuple(
        (
            _printed_label_key(row.get("printed_label", "")),
            _normalized_compare_text(row.get("text", "")),
            tuple(
                (
                    _printed_label_key(option.get("label", "")),
                    _normalized_compare_text(option.get("value", "")),
                )
                for option in (row.get("options", []) or [])
            ),
        )
        for row in (matrix.get("subquestions", []) or [])
    )
    return (
        _printed_label_key(matrix.get("printed_question_number", "")),
        paragraph_text,
        options,
        subquestions,
    )


def _question_text_snapshot(question: Dict[str, Any]) -> str:
    payload = {
        "printed_question_number": question.get("printed_question_number", ""),
        "question_text": question.get("question_text", ""),
        "options": question.get("options", []),
        "grouped_subquestions": [
            {
                "label": sub.get("label", ""),
                "printed_label": sub.get("printed_label", ""),
                "label_origin": sub.get("label_origin", ""),
                "question_text": sub.get("question_text", ""),
                "options": sub.get("options", []),
            }
            for sub in (question.get("grouped_subquestions", []) or [])
            if isinstance(sub, dict)
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _merge_verified_paragraphs_with_asset_refs(
    existing_text: Any,
    paragraphs: List[Dict[str, str]],
) -> str:
    """Replace visible text while retaining asset markers in their old order."""
    verified = [
        _compose_printed_text(row.get("printed_label", ""), row.get("text", ""))
        for row in paragraphs
    ]
    verified = [value for value in verified if value]
    current = _normalized_verbatim_text(existing_text)
    token_pattern = r"(\[asset_ref:[^\]]+\])"
    parts = re.split(token_pattern, current)
    tokens = parts[1::2]
    if not tokens:
        return "\n".join(verified).strip()

    segments = parts[0::2]
    segment_count = len(segments)
    paragraph_count = len(verified)
    # Partition the ordered verified paragraphs across the existing text
    # segments. Similarity is used only to preserve asset position, never to
    # choose or change transcribed characters.
    dp: List[List[Optional[Tuple[float, List[Tuple[int, int]]]]]] = [
        [None] * (paragraph_count + 1) for _ in range(segment_count + 1)
    ]
    dp[0][0] = (0.0, [])
    for segment_index in range(segment_count):
        for start in range(paragraph_count + 1):
            state = dp[segment_index][start]
            if state is None:
                continue
            for end in range(start, paragraph_count + 1):
                candidate = "\n".join(verified[start:end])
                left = _normalized_compare_text(segments[segment_index])
                right = _normalized_compare_text(candidate)
                if not left and not right:
                    cost = 0.0
                elif not left or not right:
                    cost = 1.25
                else:
                    cost = 1.0 - difflib.SequenceMatcher(
                        None, left, right, autojunk=False
                    ).ratio()
                total = state[0] + cost
                prior = dp[segment_index + 1][end]
                if prior is None or total < prior[0]:
                    dp[segment_index + 1][end] = (
                        total, state[1] + [(start, end)]
                    )

    final_state = dp[segment_count][paragraph_count]
    if final_state is None:
        raise ValueError("cannot preserve asset_ref positions while replacing text")
    rebuilt: List[str] = []
    for index, (start, end) in enumerate(final_state[1]):
        text_part = "\n".join(verified[start:end]).strip()
        if text_part:
            rebuilt.append(text_part)
        if index < len(tokens):
            rebuilt.append(tokens[index])
    return "\n\n".join(rebuilt).strip()


def _looks_like_structured_asset_paragraph(text: Any) -> bool:
    """Detect text that belongs in an asset block, not prose paragraphs."""
    raw = str(text or "").strip()
    if not raw:
        return False
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    lowered = raw.lower()
    code_hits = 0
    for line in lines:
        if re.search(
            r"\b(class|function|public|private|protected|static|void|int|bool|"
            r"return|if|else|for|while|switch|case|import|include)\b",
            line,
        ):
            code_hits += 1
        if any(token in line for token in ("{", "}", ";", "++", "--", "==", "!=")):
            code_hits += 1
    if code_hits >= 3:
        return True

    if "<table" in lowered or "</table>" in lowered:
        return True
    if len(lines) >= 3 and any("|" in line for line in lines):
        return True
    return False


def _filter_asset_paragraphs_for_question(
    question: Dict[str, Any],
    paragraphs: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    layout = question.get("layout_blocks", []) or []
    if not isinstance(layout, list):
        layout = []
    has_structured_asset_ref = any(
        isinstance(block, dict)
        and block.get("block_type") in {
            "code_ref", "table_ref", "formula_ref", "figure_ref",
        }
        for block in layout
    )
    if not has_structured_asset_ref:
        return paragraphs
    return [
        row for row in paragraphs
        if not _looks_like_structured_asset_paragraph(row.get("text", ""))
    ]


def _apply_exact_text_matrix(
    question: Dict[str, Any],
    matrix: Dict[str, Any],
) -> List[str]:
    """Apply verified text without treating optional layout_blocks as a gate."""
    errors: List[str] = []
    layout = question.get("layout_blocks", []) or []
    if not isinstance(layout, list):
        layout = []

    paragraphs = _filter_asset_paragraphs_for_question(
        question, matrix.get("paragraphs", []) or []
    )
    options = matrix.get("options", []) or []
    subquestions = matrix.get("subquestions", []) or []
    paragraph_blocks = [
        block for block in layout
        if isinstance(block, dict) and block.get("block_type") == "paragraph"
    ]
    option_blocks = [
        block for block in layout
        if isinstance(block, dict) and block.get("block_type") == "option_group"
    ]
    subquestion_blocks = [
        block for block in layout
        if isinstance(block, dict) and block.get("block_type") == "subquestion"
    ]
    grouped = question.get("grouped_subquestions", []) or []
    if not isinstance(grouped, list):
        errors.append("grouped_subquestions must be list")
        grouped = []

    if len(grouped) != len(subquestions):
        errors.append(
            f"grouped subquestion count differs: {len(grouped)} != {len(subquestions)}"
        )
        return errors

    merged_question_text = _merge_verified_paragraphs_with_asset_refs(
        question.get("question_text", ""), paragraphs
    )
    question["printed_question_number"] = matrix["printed_question_number"]
    question["question_text"] = merged_question_text
    if len(paragraph_blocks) == len(paragraphs):
        for block, row in zip(paragraph_blocks, paragraphs):
            block["text"] = row["text"]
            block["printed_label"] = row["printed_label"]

    composed_options = [
        row["label"] + ((" " + row["value"]) if row["value"] else "")
        for row in options
    ]
    question["options"] = composed_options
    if len(option_blocks) == 1:
        option_blocks[0]["text"] = "\n".join(composed_options)

    if len(grouped) == len(subquestions):
        for index, (sub, row) in enumerate(zip(grouped, subquestions)):
            marker = row["printed_label"]
            sub["question_text"] = row["text"]
            sub["printed_label"] = marker
            sub["options"] = [
                option["label"] + (
                    (" " + option["value"]) if option["value"] else ""
                )
                for option in (row.get("options", []) or [])
            ]
            if marker:
                sub["label_origin"] = "printed"
                if not str(sub.get("label", "") or "").strip():
                    sub["label"] = marker
            else:
                sub["label_origin"] = "synthetic"
                if not str(sub.get("label", "") or "").strip():
                    sub["label"] = f"item-{index + 1}"
            if len(subquestion_blocks) == len(subquestions):
                block = subquestion_blocks[index]
                block["text"] = row["text"]
                block["printed_label"] = marker

    return errors


def ensure_minimum_layout_blocks(
    questions: List[Dict[str, Any]],
) -> None:
    """Deterministically repair an empty layout for text-only questions.

    This does not infer semantics or reorder visual assets.  It only mirrors
    already verified question_text, grouped_subquestions and options into the
    canonical block objects expected by downstream renderers.  Questions with
    assets are left for the visual schema-repair pass because asset order must
    come from the page image.
    """
    for question in questions:
        if not isinstance(question, dict):
            continue
        layout = question.get("layout_blocks", [])
        if isinstance(layout, list) and layout:
            continue
        if question.get("latex_assets", []) or question.get("shared_asset_refs", []):
            continue

        blocks: List[Dict[str, str]] = []
        question_text = str(question.get("question_text", "") or "").strip()
        if question_text:
            blocks.append({
                "block_type": "paragraph",
                "text": question_text,
                "asset_ref": "",
                "printed_label": "",
            })

        for sub in question.get("grouped_subquestions", []) or []:
            if not isinstance(sub, dict):
                continue
            blocks.append({
                "block_type": "subquestion",
                "text": str(sub.get("question_text", "") or ""),
                "asset_ref": "",
                "printed_label": str(sub.get("printed_label", "") or ""),
            })

        options = question.get("options", []) or []
        if isinstance(options, list) and options:
            blocks.append({
                "block_type": "option_group",
                "text": "\n".join(str(value) for value in options),
                "asset_ref": "",
                "printed_label": "",
            })

        question["layout_blocks"] = blocks


def enforce_structured_asset_rendering_policy(
    questions: List[Dict[str, Any]],
) -> None:
    """Deterministically enforce representation, without inferring content.

    layout `code_ref` / `table_ref` and an already assigned asset_type are
    structural evidence.  They are sufficient to forbid a screenshot display
    strategy even while transcription is marked REVIEW.
    """
    for question in questions:
        if not isinstance(question, dict):
            continue
        code_refs = {
            str(block.get("asset_ref", "") or "").strip()
            for block in (question.get("layout_blocks", []) or [])
            if isinstance(block, dict) and block.get("block_type") == "code_ref"
        }
        table_refs = {
            str(block.get("asset_ref", "") or "").strip()
            for block in (question.get("layout_blocks", []) or [])
            if isinstance(block, dict) and block.get("block_type") == "table_ref"
        }
        for asset in question.get("latex_assets", []) or []:
            if not isinstance(asset, dict):
                continue
            aid = str(asset.get("asset_id", "") or "").strip()
            asset_type = str(asset.get("asset_type", "") or "")
            if aid in code_refs or asset_type == "code":
                asset["asset_type"] = "code"
                asset["render_strategy"] = "code_block"
            elif (
                aid in table_refs
                or asset_type in {"table_simple", "table_complex"}
            ):
                if asset_type not in {"table_simple", "table_complex"}:
                    asset["asset_type"] = "table_simple"
                asset["render_strategy"] = "html_table"


def _refresh_question_review_flag(question: Dict[str, Any]) -> None:
    issues = [
        value for value in (question.get("validation_issues", []) or [])
        if isinstance(value, str) and value.strip()
    ]
    asset_review = any(
        isinstance(asset, dict) and bool(asset.get("needs_review", False))
        for asset in (question.get("latex_assets", []) or [])
    )
    question["needs_review"] = bool(issues or asset_review)


def run_exact_text_double_read_verification(
    client: Any,
    questions: List[Dict[str, Any]],
    page_images: List[str],
    debug_dir: str,
    document_id: str,
) -> List[Dict[str, Any]]:
    """Blind-read every question twice; only exact matrices may replace text."""
    root = os.path.join(debug_dir, "exact_text_verification")
    ensure_dir(root)
    enhanced_by_page: Dict[int, str] = {}
    report: List[Dict[str, Any]] = []

    for question in questions:
        qno = str(question.get("question_number", "") or "").strip()
        audit_id = _question_audit_id(question, document_id)
        expected_printed = str(
            question.get("printed_question_number", "") or ""
        ).strip()
        safe_qno = re.sub(r"[^A-Za-z0-9_.-]+", "_", qno) or "question"
        record: Dict[str, Any] = {
            "question_number": qno,
            "audit_id": audit_id,
            "status": "REVIEW",
        }
        try:
            source_pages = [
                int(p) for p in (question.get("source_pages", []) or [])
                if isinstance(p, int) and 1 <= p <= len(page_images)
            ]
            source_pages = list(dict.fromkeys(source_pages))
            if not source_pages:
                raise ValueError("question has no usable source pages")
            original_images = [page_images[p - 1] for p in source_pages]
            enhanced_images: List[str] = []
            for page_no in source_pages:
                enhanced_path = enhanced_by_page.get(page_no)
                if not enhanced_path:
                    enhanced_path = os.path.join(root, f"page_{page_no:04d}_enhanced.png")
                    _make_enhanced_table_variant(page_images[page_no - 1], enhanced_path)
                    _resize_image_min_long_side(
                        enhanced_path, EXACT_TEXT_READ_MIN_LONG_SIDE_PX
                    )
                    enhanced_by_page[page_no] = enhanced_path
                enhanced_images.append(enhanced_path)

            reads: List[Dict[str, Any]] = []
            for role, images, reverse in (
                ("A_ORIGINAL", original_images, False),
                ("B_ENHANCED_REVERSED", enhanced_images, True),
            ):
                response = call_model_json_array(
                    client,
                    make_exact_text_read_config(),
                    _exact_text_read_prompt(question, role, document_id),
                    images,
                    reverse_images=reverse,
                    retries=3,
                    debug_path=os.path.join(
                        root, f"Q{safe_qno}_{role}_raw.txt"
                    ),
                    page_numbers=source_pages,
                )
                if len(response) != 1:
                    raise ValueError(f"exact text reader {role} must return one object")
                reads.append(response[0])

            matrix_a, errors_a = _normalize_exact_text_read(
                reads[0], audit_id, expected_printed
            )
            matrix_b, errors_b = _normalize_exact_text_read(
                reads[1], audit_id, expected_printed
            )
            matrices_equal = (
                matrix_a is not None
                and matrix_b is not None
                and _exact_text_matrix_key(matrix_a)
                == _exact_text_matrix_key(matrix_b)
            )
            reader_errors = list(errors_a) + list(errors_b)
            if matrix_a is not None and matrix_b is not None and not matrices_equal:
                reader_errors.append("independent exact text matrices differ")

            resolved: Optional[Dict[str, Any]] = None
            resolution = ""
            arbitration_result: Optional[Dict[str, Any]] = None
            final_errors: List[str] = []

            if matrices_equal and not reader_errors and matrix_a is not None:
                resolved = matrix_a
                resolution = "DOUBLE_READ_CONSENSUS"
            else:
                arbitration = call_model_json_array(
                    client,
                    make_exact_text_read_config(),
                    _exact_text_arbitration_prompt(
                        question, reads[0], reads[1], document_id
                    ),
                    original_images,
                    retries=3,
                    debug_path=os.path.join(
                        root, f"Q{safe_qno}_C_ARBITRATION_raw.txt"
                    ),
                    page_numbers=source_pages,
                )
                if len(arbitration) != 1:
                    raise ValueError("exact text arbitration must return one object")
                arbitration_result = arbitration[0]
                matrix_c, errors_c = _normalize_exact_text_read(
                    arbitration_result, audit_id, expected_printed
                )
                if matrix_c is not None and not errors_c:
                    resolved = matrix_c
                    resolution = "VISUAL_ARBITRATION"
                else:
                    final_errors = reader_errors + [
                        f"arbitration:{value}" for value in errors_c
                    ]

            before = _question_text_snapshot(question)
            if resolved is not None:
                apply_errors = _apply_exact_text_matrix(question, resolved)
                if apply_errors:
                    final_errors.extend(apply_errors)
                    resolved = None
            after = _question_text_snapshot(question)

            if resolved is None:
                marker = "exact_text_double_read_unverified"
                existing = [
                    str(v) for v in (question.get("validation_issues", []) or [])
                    if isinstance(v, str)
                ]
                if marker not in existing:
                    existing.append(marker)
                question["validation_issues"] = existing
                question["needs_review"] = True
                question["exact_text_verification"] = {
                    "status": "REVIEW",
                    "audit_id": audit_id,
                    "matrices_equal": matrices_equal,
                    "errors": final_errors or reader_errors,
                    "read_a": reads[0],
                    "read_b": reads[1],
                }
                if arbitration_result is not None:
                    question["exact_text_verification"]["arbitration"] = (
                        arbitration_result
                    )
                record.update({
                    "status": "REVIEW",
                    "errors": final_errors or reader_errors,
                    "read_a": reads[0],
                    "read_b": reads[1],
                })
            else:
                existing = [
                    str(v) for v in (question.get("validation_issues", []) or [])
                    if isinstance(v, str)
                    and v not in {
                        "exact_text_double_read_unverified",
                        "exact_text_verification_failed",
                    }
                ]
                question["validation_issues"] = existing
                status = "CORRECTED" if before != after else "PASS"
                question["exact_text_verification"] = {
                    "status": status,
                    "audit_id": audit_id,
                    "resolution": resolution,
                    "matrices_equal": matrices_equal,
                    "read_a": reads[0],
                    "read_b": reads[1],
                }
                if reader_errors:
                    question["exact_text_verification"]["reader_disagreements"] = (
                        reader_errors
                    )
                if arbitration_result is not None:
                    question["exact_text_verification"]["arbitration"] = (
                        arbitration_result
                    )
                _refresh_question_review_flag(question)
                record.update({
                    "status": status,
                    "resolution": resolution,
                    "paragraph_count": len(resolved["paragraphs"]),
                    "option_count": len(resolved["options"]),
                    "subquestion_count": len(resolved["subquestions"]),
                })
            report.append(record)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            marker = "exact_text_verification_failed"
            existing = [
                str(v) for v in (question.get("validation_issues", []) or [])
                if isinstance(v, str)
            ]
            if marker not in existing:
                existing.append(marker)
            question["validation_issues"] = existing
            question["needs_review"] = True
            question["exact_text_verification"] = {
                "status": "REVIEW",
                "audit_id": audit_id,
                "errors": [error],
            }
            record.update({"status": "REVIEW", "errors": [error]})
            report.append(record)
    return report


def _is_code_asset(asset: Dict[str, Any]) -> bool:
    return (
        str(asset.get("asset_type", "") or "") == "code"
        or str(asset.get("render_strategy", "") or "") == "code_block"
    )


def _code_read_prompt(
    question: Dict[str, Any],
    asset_id: str,
    role: str,
) -> str:
    context = _question_semantic_context(question)
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return f"""
你是程式碼逐行盲讀代理人 {role}。輸入影像是已完成幾何裁切與放大的單一
FINAL_CODE_CROP。你不會看到舊 latex 或其他代理人的程式碼。

asset_id: {asset_id}
題目脈絡（只供確認程式歸屬）：{context_json}

強制規則：
1. 只依 FINAL_CODE_CROP 實際可見內容轉錄；不得修正語法、重構或改成等價寫法。
2. code_lines 每個元素恰好對應原圖一個印刷程式行，不包含 Markdown fence。
3. 保留每行前導縮排；移除只因頁面排版產生的右側空白。
4. 左右雙欄程式碼：先完整輸出左欄所有行，再輸出右欄所有行。
5. 逐字核對數字、`<`/`<=`、`>`/`>=`、`==`/`!=`、`++`/`--`、分號、
   識別字大小寫、引號、圓括號與大括號。
6. 原圖分成兩行的敘述不得合併；不得將迴圈邊界正規化成等價形式。
7. 任一行或字元不清楚時 all_characters_readable=false；任一邊截斷時
   crop_complete=false 並列出 edge_issues。不得猜測。
8. 只輸出一個元素的 JSON 陣列。

{CODE_EXACT_PROMPT}
"""


def _normalize_code_read(
    result: Any,
    expected_asset_id: str,
) -> Tuple[Optional[List[str]], List[str]]:
    errors: List[str] = []
    if not isinstance(result, dict):
        return None, ["code read is not object"]
    if str(result.get("asset_id", "") or "").strip() != expected_asset_id:
        errors.append("asset_id mismatch")
    raw_lines = result.get("code_lines", [])
    if not isinstance(raw_lines, list) or not all(isinstance(v, str) for v in raw_lines):
        errors.append("code_lines must be list[str]")
        lines: List[str] = []
    else:
        lines = [str(v).replace("\r", "").rstrip() for v in raw_lines]
        if not lines or not any(line.strip() for line in lines):
            errors.append("code_lines empty")
    if not bool(result.get("all_characters_readable", False)):
        errors.append("not all code characters readable")
    if not bool(result.get("crop_complete", False)):
        errors.append("code crop not complete")
    edge_issues = result.get("edge_issues", [])
    if not isinstance(edge_issues, list):
        errors.append("edge_issues must be list")
    else:
        errors.extend(
            f"edge:{str(value).strip()}"
            for value in edge_issues if str(value).strip()
        )
    if errors:
        return None, errors
    return lines, []


def _mark_code_fallback(
    asset: Dict[str, Any],
    marker: str,
    details: Optional[List[str]] = None,
) -> None:
    # A failed verification is not permission to turn clearly printed source
    # code into a screenshot.  Preserve the best structured candidate, mark it
    # unverified, and keep the crop only as audit evidence.  This is fail-closed:
    # downstream consumers see REVIEW instead of silently rendering a PNG.
    asset["content_state"] = "UNVERIFIED_CODE_STRUCTURED_REVIEW"
    asset["latex_valid"] = False
    asset["latex"] = str(asset.get("latex", "") or "")
    asset["render_strategy"] = "code_block"
    asset["asset_type"] = "code"
    asset["needs_review"] = True
    asset["confidence"] = 0.0
    suffix = marker
    if details:
        suffix += ":" + ",".join(str(v) for v in details[:8])
    _append_asset_note(asset, suffix)


def _read_reports_incomplete_geometry(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    if not bool(result.get("crop_complete", False)):
        return True
    edge_issues = result.get("edge_issues", [])
    return bool(isinstance(edge_issues, list) and any(
        str(value).strip() for value in edge_issues
    ))


def _expand_bbox_for_read_issues(
    bbox: Any,
    reads: List[Dict[str, Any]],
    pad: float = STRUCTURED_ASSET_EDGE_EXPAND_UNITS,
) -> List[float]:
    """Expand only the edges that blind readers report as truncated."""
    if not _valid_bbox_0_1000(bbox):
        raise ValueError("invalid bbox for structured-asset recovery")
    y0, x0, y1, x1 = [float(value) for value in bbox]
    issue_texts: List[str] = []
    incomplete_without_named_edge = False
    for read in reads:
        if not isinstance(read, dict):
            incomplete_without_named_edge = True
            continue
        if not bool(read.get("crop_complete", False)):
            incomplete_without_named_edge = True
        raw_edges = read.get("edge_issues", [])
        if isinstance(raw_edges, list):
            issue_texts.extend(str(value).upper() for value in raw_edges)

    joined = " ".join(issue_texts)
    top = "TOP" in joined
    bottom = "BOTTOM" in joined
    left = "LEFT" in joined
    right = "RIGHT" in joined
    if not any((top, bottom, left, right)) and incomplete_without_named_edge:
        top = bottom = left = right = True

    if top:
        y0 = max(0.0, y0 - pad)
    if bottom:
        y1 = min(1000.0, y1 + pad)
    if left:
        x0 = max(0.0, x0 - pad)
    if right:
        x1 = min(1000.0, x1 + pad)
    return [y0, x0, y1, x1]


def _majority_code_lines(
    candidates: List[Optional[List[str]]],
) -> Tuple[Optional[List[str]], int]:
    counts: Dict[str, Tuple[List[str], int]] = {}
    for lines in candidates:
        if lines is None:
            continue
        key = json.dumps(lines, ensure_ascii=False, separators=(",", ":"))
        previous = counts.get(key)
        counts[key] = (lines, 1 if previous is None else previous[1] + 1)
    if not counts:
        return None, 0
    winner, count = max(counts.values(), key=lambda item: item[1])
    if count >= 2:
        return winner, count

    normalized_counts: Dict[str, Tuple[List[List[str]], int]] = {}
    for lines in candidates:
        if lines is None:
            continue
        key = json.dumps(
            [_normalize_code_line_for_consensus(line) for line in lines],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        previous = normalized_counts.get(key)
        bucket = [] if previous is None else previous[0]
        bucket.append(lines)
        normalized_counts[key] = (
            bucket,
            1 if previous is None else previous[1] + 1,
        )
    if not normalized_counts:
        return None, 0
    bucket, normalized_count = max(
        normalized_counts.values(), key=lambda item: item[1]
    )
    if normalized_count < 2:
        return None, max(count, normalized_count)
    best_layout = max(bucket, key=lambda lines: sum(len(line) for line in lines))
    return best_layout, normalized_count


def _normalize_code_line_for_consensus(line: Any) -> str:
    """Ignore only presentation spacing when comparing code OCR candidates."""
    text = str(line or "").strip()
    rebuilt: List[str] = []
    quote = ""
    escaping = False
    pending_space = False
    for char in text:
        if quote:
            rebuilt.append(char)
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            if pending_space and rebuilt:
                rebuilt.append(" ")
                pending_space = False
            quote = char
            rebuilt.append(char)
            continue
        if char.isspace():
            pending_space = True
            continue
        if pending_space and rebuilt and char not in ")}],;":
            rebuilt.append(" ")
        pending_space = False
        rebuilt.append(char)
    return "".join(rebuilt)


def run_code_crop_content_verification(
    client: Any,
    questions: List[Dict[str, Any]],
    page_images: List[str],
    debug_dir: str,
) -> List[Dict[str, Any]]:
    """Recover tight code crops, then require two-of-three exact line reads."""
    root = os.path.join(debug_dir, "code_crop_verification")
    ensure_dir(root)
    report: List[Dict[str, Any]] = []

    for question in questions:
        code_refs = {
            str(block.get("asset_ref", "") or "").strip()
            for block in (question.get("layout_blocks", []) or [])
            if isinstance(block, dict) and block.get("block_type") == "code_ref"
        }
        for asset in question.get("latex_assets", []) or []:
            if not isinstance(asset, dict):
                continue
            candidate_id = str(asset.get("asset_id", "") or "").strip()
            if not _is_code_asset(asset) and candidate_id not in code_refs:
                continue
            # A code_ref is canonical structural evidence even if an earlier
            # model mislabeled the asset type.
            if candidate_id in code_refs:
                asset["asset_type"] = "code"
            aid = str(asset.get("asset_id", "") or "").strip() or "code_asset"
            safe_aid = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", aid)
            record: Dict[str, Any] = {"asset_id": aid, "status": "REVIEW"}
            try:
                page_no = asset.get("page_number")
                bbox = asset.get("bbox_pct")
                if (
                    not isinstance(page_no, int)
                    or not (1 <= page_no <= len(page_images))
                    or not _valid_bbox_0_1000(bbox)
                ):
                    raise ValueError("invalid code page or bbox")

                original_bbox = [float(value) for value in bbox]
                # Start from the last visual-QC bbox.  The previous version
                # expanded every side unconditionally, which reintroduced page
                # headers and the next question, then incorrectly overwrote the
                # geometry status with PASS.  Expansion now happens only when a
                # blind reader identifies a truncated edge.
                current_bbox = list(original_bbox)
                attempts: List[Dict[str, Any]] = []
                verified_lines: Optional[List[str]] = None
                verified_reads: List[Dict[str, Any]] = []
                consensus_count = 0
                geometry_verified = False
                final_errors: List[str] = []

                for recovery_round in range(1, STRUCTURED_ASSET_MAX_READ_ROUNDS + 1):
                    color_path = os.path.join(
                        root, f"{safe_aid}_r{recovery_round}_color.png"
                    )
                    enhanced_path = os.path.join(
                        root, f"{safe_aid}_r{recovery_round}_enhanced.png"
                    )
                    _render_enlarged_bbox_crop(
                        page_images[page_no - 1],
                        current_bbox,
                        color_path,
                        min_long_side=CODE_READ_MIN_LONG_SIDE_PX,
                    )
                    _make_enhanced_table_variant(color_path, enhanced_path)

                    reads: List[Dict[str, Any]] = []
                    normalized: List[Optional[List[str]]] = []
                    read_errors: List[str] = []
                    for role, image_path in (
                        ("A_ORIGINAL_COLOR", color_path),
                        ("B_ENHANCED_GRAYSCALE", enhanced_path),
                    ):
                        response = call_model_json_array(
                            client,
                            make_code_read_config(),
                            _code_read_prompt(question, aid, role),
                            [image_path],
                            retries=3,
                            debug_path=os.path.join(
                                root,
                                f"{safe_aid}_r{recovery_round}_{role}_raw.txt",
                            ),
                            page_numbers=[page_no],
                        )
                        if len(response) != 1:
                            raise ValueError(
                                f"code reader {role} must return one object"
                            )
                        reads.append(response[0])
                        lines, errors = _normalize_code_read(response[0], aid)
                        normalized.append(lines)
                        read_errors.extend(errors)

                    geometry_incomplete = any(
                        _read_reports_incomplete_geometry(read) for read in reads
                    )
                    if geometry_incomplete:
                        expanded_bbox = _expand_bbox_for_read_issues(
                            current_bbox, reads
                        )
                        attempts.append({
                            "round": recovery_round,
                            "status": "RECROP",
                            "bbox_before": list(current_bbox),
                            "bbox_after": list(expanded_bbox),
                            "errors": list(dict.fromkeys(read_errors)),
                            "reads": reads,
                        })
                        final_errors = list(dict.fromkeys(read_errors))
                        if (
                            recovery_round < STRUCTURED_ASSET_MAX_READ_ROUNDS
                            and expanded_bbox != current_bbox
                        ):
                            current_bbox = expanded_bbox
                            continue
                        break

                    winner, consensus_count = _majority_code_lines(normalized)
                    if winner is None:
                        role = "C_TIE_BREAK_ORIGINAL_COLOR"
                        response = call_model_json_array(
                            client,
                            make_code_read_config(),
                            _code_read_prompt(question, aid, role),
                            [color_path],
                            retries=3,
                            debug_path=os.path.join(
                                root,
                                f"{safe_aid}_r{recovery_round}_{role}_raw.txt",
                            ),
                            page_numbers=[page_no],
                        )
                        if len(response) != 1:
                            raise ValueError(
                                "code tie-break reader must return one object"
                            )
                        reads.append(response[0])
                        lines_c, errors_c = _normalize_code_read(response[0], aid)
                        normalized.append(lines_c)
                        read_errors.extend(errors_c)
                        winner, consensus_count = _majority_code_lines(normalized)

                    if winner is not None and consensus_count >= 2:
                        # Exact lines and crop geometry are separate gates.  A
                        # crop may contain all code lines yet also include a
                        # page header, prose, or the next question.  Reuse the
                        # full-page semantic crop checker before declaring the
                        # code asset PASS.
                        geometry_decision = llm_post_crop_qc_single(
                            client,
                            question,
                            asset,
                            page_images,
                            color_path,
                            os.path.join(
                                root,
                                f"{safe_aid}_r{recovery_round}_geometry_raw.json",
                            ),
                            recovery_round,
                        )
                        geometry_status = str(
                            geometry_decision.get("status", "") or ""
                        )
                        corrected_page = geometry_decision.get(
                            "corrected_page_number"
                        )
                        corrected_bbox = geometry_decision.get(
                            "corrected_bbox_pct"
                        )
                        if geometry_status == "PASS":
                            verified_lines = winner
                            verified_reads = reads
                            geometry_verified = True
                            attempts.append({
                                "round": recovery_round,
                                "status": "PASS",
                                "bbox": list(current_bbox),
                                "consensus_count": consensus_count,
                                "geometry_decision": geometry_decision,
                                "reads": reads,
                            })
                            break

                        if (
                            geometry_status == "RECROP"
                            and corrected_page == page_no
                            and _valid_bbox_0_1000(corrected_bbox)
                            and _bbox_change_is_meaningful(
                                current_bbox, corrected_bbox
                            )
                            and recovery_round
                            < STRUCTURED_ASSET_MAX_READ_ROUNDS
                        ):
                            attempts.append({
                                "round": recovery_round,
                                "status": "RECROP_AFTER_CONTENT_CONSENSUS",
                                "bbox_before": list(current_bbox),
                                "bbox_after": [
                                    float(value) for value in corrected_bbox
                                ],
                                "consensus_count": consensus_count,
                                "geometry_decision": geometry_decision,
                                "reads": reads,
                            })
                            current_bbox = [
                                float(value) for value in corrected_bbox
                            ]
                            continue

                        # Keep verified code text but do not lie about crop
                        # geometry when no confirmed correction remains.
                        verified_lines = winner
                        verified_reads = reads
                        geometry_verified = False
                        final_errors = [
                            "code geometry not independently verified",
                            f"geometry_status:{geometry_status or 'INVALID'}",
                        ]
                        attempts.append({
                            "round": recovery_round,
                            "status": "PASS_WITH_GEOMETRY_REVIEW",
                            "bbox": list(current_bbox),
                            "consensus_count": consensus_count,
                            "geometry_decision": geometry_decision,
                            "reads": reads,
                        })
                        break

                    final_errors = list(dict.fromkeys(
                        read_errors + ["no two exact code reads agree"]
                    ))
                    attempts.append({
                        "round": recovery_round,
                        "status": "REVIEW",
                        "bbox": list(current_bbox),
                        "errors": final_errors,
                        "reads": reads,
                    })
                    break

                if verified_lines is None:
                    if _valid_bbox_0_1000(current_bbox):
                        asset["page_number"] = page_no
                        asset["bbox_pct"] = _pad_bbox_0_1000(
                            current_bbox, y_pad=10.0, x_pad=35.0
                        )
                        _append_asset_note(
                            asset,
                            "code_review_crop_safety_padding_applied",
                        )
                    _mark_code_fallback(
                        asset, "code_content_verification_failed", final_errors
                    )
                    asset["code_verification"] = {
                        "status": "REVIEW",
                        "content_status": "REVIEW",
                        "geometry_status": "REVIEW",
                        "errors": final_errors,
                        "attempts": attempts,
                    }
                    record.update({
                        "status": "REVIEW",
                        "errors": final_errors,
                        "attempts": attempts,
                    })
                else:
                    asset["page_number"] = page_no
                    asset["bbox_pct"] = list(current_bbox)
                    if not geometry_verified:
                        asset["bbox_pct"] = _pad_bbox_0_1000(
                            asset["bbox_pct"], y_pad=10.0, x_pad=35.0
                        )
                        _append_asset_note(
                            asset,
                            "code_review_crop_safety_padding_applied",
                        )
                    asset["latex"] = "\n".join(verified_lines)
                    asset["render_strategy"] = "code_block"
                    asset["needs_review"] = not geometry_verified
                    asset["confidence"] = 1.0
                    asset["content_state"] = (
                        "VERIFIED_FROM_FINAL_CODE_CROP"
                        if geometry_verified
                        else "VERIFIED_CODE_CONTENT_GEOMETRY_REVIEW"
                    )
                    asset["latex_valid"] = True
                    _append_asset_note(
                        asset,
                        f"code_verified_from_final_crop: lines={len(verified_lines)}; "
                        f"consensus={consensus_count}; recovery_rounds={len(attempts)}",
                    )
                    prior_qc = asset.get("post_crop_qc", {})
                    history = (
                        prior_qc.get("history", [])
                        if isinstance(prior_qc, dict) else []
                    )
                    asset["post_crop_qc"] = {
                        "status": "PASS" if geometry_verified else "REVIEW",
                        "code_recovery_status": (
                            "PASS" if geometry_verified
                            else "PASS_WITH_GEOMETRY_REVIEW"
                        ),
                        "bbox_before_code_recovery": original_bbox,
                        "final_bbox_pct": list(current_bbox),
                        "history": history,
                    }
                    asset["code_verification"] = {
                        "status": (
                            "PASS" if geometry_verified
                            else "PASS_WITH_GEOMETRY_REVIEW"
                        ),
                        "content_status": "PASS",
                        "geometry_status": (
                            "PASS" if geometry_verified else "REVIEW"
                        ),
                        "post_crop_qc_status": (
                            "PASS" if geometry_verified else "REVIEW"
                        ),
                        "lines_equal": True,
                        "consensus_count": consensus_count,
                        "line_count": len(verified_lines),
                        "reads": verified_reads,
                        "attempts": attempts,
                    }
                    record.update({
                        "status": (
                            "PASS" if geometry_verified
                            else "PASS_WITH_GEOMETRY_REVIEW"
                        ),
                        "line_count": len(verified_lines),
                        "consensus_count": consensus_count,
                        "final_bbox_pct": list(current_bbox),
                        "attempts": attempts,
                    })
                report.append(record)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                _mark_code_fallback(asset, "code_recovery_failed", [error])
                asset["code_verification"] = {
                    "status": "REVIEW", "errors": [error]
                }
                record.update({"status": "REVIEW", "errors": [error]})
                report.append(record)
    return report


def _table_html(headers: List[str], rows: List[List[str]]) -> str:
    head = "".join(
        f"<th>{html_lib.escape(value, quote=False)}</th>" for value in headers
    )
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html_lib.escape(value, quote=False)}</td>" for value in row
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _normalize_table_read(
    result: Any,
    expected_asset_id: str,
) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    """Separate transcription failures from crop-geometry warnings.

    A table matrix can be content-valid even when a decorative separator,
    border, or safety margin is tight.  Only content errors invalidate the
    matrix; geometry issues remain review metadata and must not erase a
    verified HTML table.
    """
    content_errors: List[str] = []
    geometry_issues: List[str] = []
    if not isinstance(result, dict):
        return None, ["table read is not object"], []
    if str(result.get("asset_id", "") or "").strip() != expected_asset_id:
        content_errors.append("asset_id mismatch")

    raw_headers = result.get("headers", [])
    raw_rows = result.get("rows", [])
    if not isinstance(raw_headers, list) or not all(
        isinstance(v, str) for v in raw_headers
    ):
        content_errors.append("headers must be list[str]")
        headers: List[str] = []
    else:
        headers = [v.strip() for v in raw_headers]
        if not headers:
            content_errors.append("table has no headers")
        # A visually empty corner header is valid (for example a worksheet
        # whose first column contains row labels).  Empty is data, not an OCR
        # failure, when the reader explicitly reports all cells readable.

    rows: List[List[str]] = []
    if not isinstance(raw_rows, list):
        content_errors.append("rows must be list")
    else:
        for index, row in enumerate(raw_rows):
            if not isinstance(row, list) or not all(isinstance(v, str) for v in row):
                content_errors.append(f"row {index} must be list[str]")
                continue
            normalized_row = [v.strip() for v in row]
            rows.append(normalized_row)

    column_count = result.get("column_count")
    data_row_count = result.get("data_row_count")
    if not isinstance(column_count, int) or column_count != len(headers):
        content_errors.append("column_count does not match headers")
    if not isinstance(data_row_count, int) or data_row_count != len(rows):
        content_errors.append("data_row_count does not match rows")
    if headers and any(len(row) != len(headers) for row in rows):
        content_errors.append("row cell count does not match headers")
    if not bool(result.get("all_cells_readable", False)):
        content_errors.append("not all cells readable")

    # Geometry is tracked independently.  It may require a better archival
    # crop, but does not invalidate two matching, fully readable matrices.
    if not bool(result.get("crop_complete", False)):
        geometry_issues.append("crop not complete")
    edge_issues = result.get("edge_issues", [])
    if not isinstance(edge_issues, list):
        content_errors.append("edge_issues must be list")
    else:
        geometry_issues.extend(
            f"edge:{str(value).strip()}"
            for value in edge_issues
            if str(value).strip()
        )

    if content_errors:
        return None, content_errors, geometry_issues
    return {"headers": headers, "rows": rows}, [], geometry_issues


def _trim_provably_phantom_trailing_columns(
    result: Dict[str, Any],
    expected_columns: int,
) -> Tuple[Dict[str, Any], List[str]]:
    """Remove only extra rightmost columns that are empty in every position.

    The complete localization pass supplies the expected visible column count.
    No non-empty header or cell is ever removed.
    """
    cleaned = dict(result)
    notes: List[str] = []
    raw_headers = result.get("headers", [])
    raw_rows = result.get("rows", [])
    if not isinstance(raw_headers, list) or not isinstance(raw_rows, list):
        return cleaned, notes
    if len(raw_headers) <= expected_columns:
        return cleaned, notes
    if any(str(value).strip() for value in raw_headers[expected_columns:]):
        return cleaned, notes
    if any(
        not isinstance(row, list)
        or len(row) < len(raw_headers)
        or any(str(value).strip() for value in row[expected_columns:])
        for row in raw_rows
    ):
        return cleaned, notes

    removed = len(raw_headers) - expected_columns
    cleaned["headers"] = list(raw_headers[:expected_columns])
    cleaned["rows"] = [list(row[:expected_columns]) for row in raw_rows]
    cleaned["column_count"] = expected_columns
    notes.append(
        f"removed {removed} provably empty trailing phantom column(s)"
    )
    return cleaned, notes


def _majority_table_matrix(
    candidates: List[Optional[Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Any]], int]:
    counts: Dict[str, Tuple[Dict[str, Any], int]] = {}
    for matrix in candidates:
        if matrix is None:
            continue
        key = json.dumps(matrix, ensure_ascii=False, sort_keys=True)
        previous = counts.get(key)
        counts[key] = (matrix, 1 if previous is None else previous[1] + 1)
    if not counts:
        return None, 0
    winner, count = max(counts.values(), key=lambda item: item[1])
    return (winner, count) if count >= 2 else (None, count)


def _table_read_geometry_complete(result: Any) -> bool:
    if not isinstance(result, dict) or not bool(result.get("crop_complete", False)):
        return False
    edge_issues = result.get("edge_issues", [])
    return bool(
        isinstance(edge_issues, list)
        and not any(str(value).strip() for value in edge_issues)
    )


def _table_localize_prompt(
    question: Dict[str, Any],
    asset: Dict[str, Any],
) -> str:
    context = _question_semantic_context(question)
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    aid = str(asset.get("asset_id", "") or "").strip()
    return f"""
你是單一表格的局部幾何定位代理人。輸入影像是從完整考卷頁面擴大裁出的
TABLE_CONTEXT_CROP，不是完整頁面，也不是最後裁切結果。

asset_id: {aid}
題目脈絡：{context_json}

這一輪只負責在 TABLE_CONTEXT_CROP 內找到該題實際引用的完整表格。
你完全看不到舊 HTML 與舊數值；不得根據題意虛構表格內容。

強制規則：
1. 若影像中有 `Sample output:`，它只是定位線索，不必納入 bbox。
2. bbox_local_pct 使用 TABLE_CONTEXT_CROP 的 [y_min,x_min,y_max,x_max]、0..1000。
3. bbox 必須包含表頭、分隔線、所有欄、所有資料列、最右欄及最後一列。
4. 先辨識表頭、欄數及資料列數，以確認真正邊界；這一輪不需要轉錄每個數值。
5. 若同一局部圖仍有兩張表格，以題目脈絡、表頭及相對位置決定歸屬。
6. 任何邊界或歸屬無法確認時 status=REVIEW、complete=false，不得猜測。
7. FOUND 只允許在四邊完整且 observed_headers、column_count、data_row_count
   都能從影像確認時使用。

只輸出一個元素的 JSON 陣列。
"""


def _table_read_prompt(
    question: Dict[str, Any],
    asset_id: str,
    role: str,
) -> str:
    context = _question_semantic_context(question)
    context_json = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    return f"""
你是表格逐格盲目轉錄代理人 {role}。輸入影像是已重新定位並放大的單一
FINAL_TABLE_CROP。你不會收到舊 HTML、舊 bbox、其他代理人的數值或候選答案。

asset_id: {asset_id}
題目脈絡：{context_json}

強制規則：
1. 只依 FINAL_TABLE_CROP 實際可見內容轉錄，不得依題意、常識或相鄰表格補值。
2. 先判斷四邊是否完整；表頭、分隔線、最右欄及最後一列任一被截斷時，
   crop_complete=false，並列出 edge_issues。
3. 先建立 headers，再由上到下逐列、由左到右逐 cell 建立 rows。
   `Sample output:` 是表格外標籤，不得當作 header 或資料 cell。
4. 每個數字逐位讀取，不得增加、刪除、交換或從其他列複製數位。
5. column_count 必須等於 headers 長度；data_row_count 必須等於 rows 長度；
   每列 cell 數必須等於 column_count。
6. 視覺上確定為空白的表頭或儲存格，必須輸出空字串 `""`；這種情況仍可設
   all_cells_readable=true。只有無法判斷「是空白還是被遮住」時才設 false。
7. 不得因右側留白而新增沒有框線／分隔線的空白欄；最右欄以實際垂直框線、
   對齊或分隔符號為準。
8. 不要輸出 HTML；只輸出矩陣，HTML 將由 Python 產生。

只輸出一個元素的 JSON 陣列。
"""


def _mark_table_fallback(
    asset: Dict[str, Any],
    marker: str,
    details: Optional[List[str]] = None,
) -> None:
    # A localization/count disagreement is not evidence that a printed table
    # is inherently graphical.  Keep the best HTML candidate and fail closed
    # with REVIEW; the crop remains audit evidence only.  This prevents a
    # clear 3x7 worksheet from silently becoming a screenshot.
    existing_html = str(asset.get("latex", "") or "")
    has_html_candidate = (
        "<table" in existing_html.lower()
        and "</table>" in existing_html.lower()
    )
    asset["content_state"] = (
        "UNVERIFIED_TABLE_HTML_CANDIDATE"
        if has_html_candidate
        else "UNVERIFIED_TABLE_STRUCTURED_REVIEW"
    )
    asset["latex_valid"] = False
    asset["latex"] = existing_html
    asset["render_strategy"] = "html_table"
    if str(asset.get("asset_type", "") or "") not in {
        "table_simple", "table_complex"
    }:
        asset["asset_type"] = "table_simple"
    asset["needs_review"] = True
    asset["confidence"] = 0.0
    suffix = marker
    if details:
        suffix += ":" + ",".join(str(v) for v in details[:8])
    _append_asset_note(asset, suffix)


def run_table_crop_content_verification(
    client: Any,
    questions: List[Dict[str, Any]],
    page_images: List[str],
    debug_dir: str,
) -> List[Dict[str, Any]]:
    """Recover table geometry locally, then accept content only after two exact reads."""
    table_root = os.path.join(debug_dir, "table_crop_verification")
    ensure_dir(table_root)
    report: List[Dict[str, Any]] = []

    for question in questions:
        table_refs = {
            str(block.get("asset_ref", "") or "").strip()
            for block in (question.get("layout_blocks", []) or [])
            if isinstance(block, dict) and block.get("block_type") == "table_ref"
        }
        for asset in question.get("latex_assets", []) or []:
            if not isinstance(asset, dict):
                continue
            candidate_id = str(asset.get("asset_id", "") or "").strip()
            if not _is_table_asset(asset) and candidate_id not in table_refs:
                continue
            # A table_ref is canonical structural evidence.  If the earlier
            # asset classifier used `other`, route it through strict table
            # recovery instead of silently leaving a screenshot.
            if candidate_id in table_refs and not _is_table_asset(asset):
                asset["asset_type"] = "table_simple"

            aid = str(asset.get("asset_id", "") or "").strip() or "table_asset"
            safe_aid = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", aid)
            record: Dict[str, Any] = {"asset_id": aid, "status": "REVIEW"}

            try:
                page_no = asset.get("page_number")
                candidate_bbox = asset.get("bbox_pct")
                if (
                    not isinstance(page_no, int)
                    or not (1 <= page_no <= len(page_images))
                    or not _valid_bbox_0_1000(candidate_bbox)
                ):
                    raise ValueError("invalid table page or candidate bbox")
                prior_qc = asset.get("post_crop_qc", {})
                prior_status = (
                    str(prior_qc.get("status", "") or "")
                    if isinstance(prior_qc, dict) else ""
                )

                context_bbox = _expand_table_context_bbox(candidate_bbox)
                context_path = os.path.join(table_root, f"{safe_aid}_context.png")
                _render_enlarged_bbox_crop(
                    page_images[page_no - 1], context_bbox, context_path
                )

                localization = call_model_json_array(
                    client,
                    make_table_localize_config(),
                    _table_localize_prompt(question, asset),
                    [context_path],
                    retries=3,
                    debug_path=os.path.join(
                        table_root, f"{safe_aid}_01_localize_raw.txt"
                    ),
                    page_numbers=[page_no],
                )
                if len(localization) != 1:
                    raise ValueError("table localization must return one object")
                located = localization[0]
                if str(located.get("asset_id", "") or "").strip() != aid:
                    raise ValueError("table localization asset_id mismatch")
                if located.get("status") != "FOUND" or not bool(
                    located.get("complete", False)
                ):
                    issues = [str(v) for v in located.get("issues", []) or []]
                    _mark_table_fallback(
                        asset, "table_localization_review", issues
                    )
                    record.update({
                        "status": "REVIEW",
                        "stage": "LOCALIZATION",
                        "context_bbox_pct": context_bbox,
                        "localization": located,
                    })
                    report.append(record)
                    continue

                local_bbox = located.get("bbox_local_pct")
                if not _valid_bbox_0_1000(local_bbox):
                    raise ValueError("localized table bbox invalid")
                expected_columns = located.get("column_count")
                expected_rows = located.get("data_row_count")
                observed_headers = located.get("observed_headers", [])
                localization_warnings: List[str] = []
                if not isinstance(observed_headers, list):
                    observed_headers = []
                    localization_warnings.append(
                        "localization observed_headers is not a list"
                    )
                observed_headers = [str(value) for value in observed_headers]
                if not isinstance(expected_columns, int) or expected_columns < 1:
                    expected_columns = len(observed_headers)
                    localization_warnings.append(
                        "localization column_count invalid; use reader consensus"
                    )
                if not isinstance(expected_rows, int) or expected_rows < 0:
                    expected_rows = -1
                    localization_warnings.append(
                        "localization data_row_count invalid; use reader consensus"
                    )
                if expected_columns != len(observed_headers):
                    # A blank upper-left header is frequently omitted by a
                    # localizer even though it correctly sees the vertical
                    # column boundary.  Treat localizer counts as geometry
                    # hints; two exact full-matrix readers remain authoritative.
                    localization_warnings.append(
                        "localization header/count mismatch; use reader consensus"
                    )

                final_bbox = _local_bbox_to_full_page(context_bbox, local_bbox)
                if not _valid_bbox_0_1000(final_bbox):
                    raise ValueError("mapped full-page table bbox invalid")

                # Do not overwrite the last generally verified bbox yet. The
                # localized crop must pass both independent geometry reads first.
                asset["content_state"] = "STALE_AFTER_BBOX_CHANGE"
                asset["latex_valid"] = False

                final_color = os.path.join(
                    table_root, f"{safe_aid}_final_color.png"
                )
                final_enhanced = os.path.join(
                    table_root, f"{safe_aid}_final_enhanced.png"
                )
                _render_enlarged_bbox_crop(
                    page_images[page_no - 1], final_bbox, final_color
                )
                _make_enhanced_table_variant(final_color, final_enhanced)

                reads: List[Dict[str, Any]] = []
                for role, image_path in (
                    ("A_ORIGINAL_COLOR", final_color),
                    ("B_ENHANCED_GRAYSCALE", final_enhanced),
                ):
                    response = call_model_json_array(
                        client,
                        make_table_read_config(),
                        _table_read_prompt(question, aid, role),
                        [image_path],
                        retries=3,
                        debug_path=os.path.join(
                            table_root, f"{safe_aid}_02_read_{role}_raw.txt"
                        ),
                        page_numbers=[page_no],
                    )
                    if len(response) != 1:
                        raise ValueError(f"table reader {role} must return one object")
                    reads.append(response[0])

                processed_reads: List[Dict[str, Any]] = []
                normalization_notes: List[List[str]] = []
                matrices: List[Optional[Dict[str, Any]]] = []
                per_read_content_errors: List[List[str]] = []
                per_read_geometry: List[List[str]] = []
                for read in reads:
                    cleaned, cleanup_notes = (
                        _trim_provably_phantom_trailing_columns(
                            read, expected_columns
                        )
                    )
                    matrix, read_errors, geometry = _normalize_table_read(
                        cleaned, aid
                    )
                    processed_reads.append(cleaned)
                    normalization_notes.append(cleanup_notes)
                    matrices.append(matrix)
                    per_read_content_errors.append(read_errors)
                    per_read_geometry.append(geometry)

                consensus_matrix, consensus_count = _majority_table_matrix(matrices)
                geometry_votes = [
                    _table_read_geometry_complete(read) for read in reads
                ]
                need_geometry_tie_break = (
                    len(geometry_votes) == 2
                    and geometry_votes[0] != geometry_votes[1]
                )
                if consensus_matrix is None or need_geometry_tie_break:
                    role = "C_TIE_BREAK_ORIGINAL_COLOR"
                    response = call_model_json_array(
                        client,
                        make_table_read_config(),
                        _table_read_prompt(question, aid, role),
                        [final_color],
                        retries=3,
                        debug_path=os.path.join(
                            table_root, f"{safe_aid}_02_read_{role}_raw.txt"
                        ),
                        page_numbers=[page_no],
                    )
                    if len(response) != 1:
                        raise ValueError(
                            "table tie-break reader must return one object"
                        )
                    reads.append(response[0])
                    cleaned, cleanup_notes = (
                        _trim_provably_phantom_trailing_columns(
                            response[0], expected_columns
                        )
                    )
                    matrix, read_errors, geometry = _normalize_table_read(
                        cleaned, aid
                    )
                    processed_reads.append(cleaned)
                    normalization_notes.append(cleanup_notes)
                    matrices.append(matrix)
                    per_read_content_errors.append(read_errors)
                    per_read_geometry.append(geometry)
                    consensus_matrix, consensus_count = _majority_table_matrix(
                        matrices
                    )

                geometry_votes = [
                    _table_read_geometry_complete(read) for read in reads
                ]
                if sum(1 for vote in geometry_votes if vote) >= 2:
                    geometry_issues: List[str] = []
                else:
                    geometry_issues = list(dict.fromkeys(
                        issue
                        for issues in per_read_geometry
                        for issue in issues
                    ))
                    if not geometry_issues:
                        geometry_issues.append(
                            "fewer than two readers confirmed complete geometry"
                        )
                content_errors: List[str] = []
                if consensus_matrix is None or consensus_count < 2:
                    content_errors.extend(
                        issue
                        for issues in per_read_content_errors
                        for issue in issues
                    )
                    content_errors.append("no two exact table matrices agree")
                else:
                    normalized_observed_headers = [
                        str(v).strip() for v in observed_headers
                    ]
                    if len(consensus_matrix["headers"]) != expected_columns:
                        localization_warnings.append(
                            "consensus columns differ from localization"
                        )
                    if (
                        expected_rows >= 0
                        and len(consensus_matrix["rows"]) != expected_rows
                    ):
                        localization_warnings.append(
                            "consensus rows differ from localization"
                        )
                    if (
                        len(normalized_observed_headers)
                        == len(consensus_matrix["headers"])
                        and consensus_matrix["headers"]
                        != normalized_observed_headers
                    ):
                        localization_warnings.append(
                            "consensus headers differ from localization"
                        )
                content_errors = list(dict.fromkeys(content_errors))

                if content_errors or consensus_matrix is None:
                    _mark_table_fallback(
                        asset,
                        "table_content_verification_failed",
                        content_errors,
                    )
                    asset["table_verification"] = {
                        "status": "REVIEW",
                        "content_status": "REVIEW",
                        "geometry_status": (
                            "REVIEW" if geometry_issues else "UNKNOWN"
                        ),
                        "context_bbox_pct": context_bbox,
                        "final_bbox_pct": final_bbox,
                        "localization": located,
                        "reads": reads,
                        "processed_reads": processed_reads,
                        "normalization_notes": normalization_notes,
                        "matrices_equal": (
                            len(matrices) >= 2
                            and matrices[0] is not None
                            and matrices[0] == matrices[1]
                        ),
                        "consensus_count": consensus_count,
                        "content_errors": content_errors,
                        "geometry_issues": geometry_issues,
                        "localization_warnings": localization_warnings,
                    }
                    record.update({
                        "status": "REVIEW",
                        "stage": "CONTENT_VERIFICATION",
                        "context_bbox_pct": context_bbox,
                        "final_bbox_pct": final_bbox,
                        "localization": located,
                        "reads": reads,
                        "processed_reads": processed_reads,
                        "content_errors": content_errors,
                        "geometry_issues": geometry_issues,
                        "localization_warnings": localization_warnings,
                    })
                    report.append(record)
                    continue

                headers = consensus_matrix["headers"]
                rows = consensus_matrix["rows"]
                localized_geometry_issues = list(geometry_issues)
                bbox_reverted_to_general_qc = bool(
                    geometry_issues and prior_status == "PASS"
                )
                if bbox_reverted_to_general_qc:
                    accepted_bbox = [float(v) for v in candidate_bbox]
                    geometry_issues = []
                else:
                    accepted_bbox = [float(v) for v in final_bbox]
                asset["page_number"] = page_no
                asset["bbox_pct"] = accepted_bbox
                asset["latex"] = _table_html(headers, rows)
                asset["labels"] = list(headers)
                asset["render_strategy"] = "html_table"
                asset["needs_review"] = bool(geometry_issues)
                asset["confidence"] = 1.0
                asset["content_state"] = (
                    "VERIFIED_CONTENT_GEOMETRY_REVIEW"
                    if geometry_issues
                    else "VERIFIED_FROM_FINAL_CROP"
                )
                asset["latex_valid"] = True
                last_row = " | ".join(rows[-1]) if rows else ""
                asset["notes"] = (
                    f"table_verified_from_final_crop: columns={len(headers)}; "
                    f"data_rows={len(rows)}; last_row={last_row}; "
                    f"consensus={consensus_count}"
                )
                if geometry_issues:
                    _append_asset_note(
                        asset,
                        "table_geometry_review:" + ",".join(geometry_issues[:8]),
                    )
                if bbox_reverted_to_general_qc:
                    _append_asset_note(
                        asset,
                        "table_localized_bbox_rejected_keep_general_qc_bbox:"
                        + ",".join(localized_geometry_issues[:8]),
                    )
                asset["table_verification"] = {
                    "status": (
                        "PASS_WITH_GEOMETRY_REVIEW"
                        if geometry_issues else "PASS"
                    ),
                    "content_status": "PASS",
                    "geometry_status": (
                        "REVIEW" if geometry_issues else "PASS"
                    ),
                    "context_bbox_pct": context_bbox,
                    "localized_bbox_pct": final_bbox,
                    "final_bbox_pct": accepted_bbox,
                    "bbox_reverted_to_general_qc": (
                        bbox_reverted_to_general_qc
                    ),
                    "localization": located,
                    "reads": reads,
                    "processed_reads": processed_reads,
                    "normalization_notes": normalization_notes,
                    "matrices_equal": (
                        len(matrices) >= 2
                        and matrices[0] is not None
                        and matrices[0] == matrices[1]
                    ),
                    "consensus_count": consensus_count,
                    "content_errors": [],
                    "geometry_issues": geometry_issues,
                    "localized_geometry_issues": localized_geometry_issues,
                    "localization_warnings": localization_warnings,
                }
                asset["post_crop_qc"] = {
                    "status": "REVIEW" if geometry_issues else "PASS",
                    "general_qc_status_before_table_recovery": prior_status,
                    "table_recovery_status": (
                        "PASS_WITH_GEOMETRY_REVIEW"
                        if geometry_issues else "PASS"
                    ),
                    "localized_bbox_pct": final_bbox,
                    "final_bbox_pct": accepted_bbox,
                    "history": (
                        prior_qc.get("history", [])
                        if isinstance(prior_qc, dict) else []
                    ),
                }
                record.update({
                    "status": (
                        "PASS_WITH_GEOMETRY_REVIEW"
                        if geometry_issues else "PASS"
                    ),
                    "stage": "CONTENT_VERIFIED",
                    "context_bbox_pct": context_bbox,
                    "localized_bbox_pct": final_bbox,
                    "final_bbox_pct": accepted_bbox,
                    "headers": headers,
                    "rows": rows,
                    "consensus_count": consensus_count,
                    "normalization_notes": normalization_notes,
                    "geometry_issues": geometry_issues,
                    "localized_geometry_issues": localized_geometry_issues,
                    "localization_warnings": localization_warnings,
                    "bbox_reverted_to_general_qc": (
                        bbox_reverted_to_general_qc
                    ),
                })
                report.append(record)

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                _mark_table_fallback(asset, "table_recovery_failed", [error])
                asset["table_verification"] = {
                    "status": "REVIEW",
                    "errors": [error],
                }
                record.update({
                    "status": "REVIEW",
                    "stage": "EXCEPTION",
                    "errors": [error],
                })
                report.append(record)

    return report


def run_final_full_page_coverage_gate(
    client: Any,
    questions: List[Dict[str, Any]],
    page_images: List[str],
    debug_dir: str,
) -> List[Dict[str, Any]]:
    """Final non-mutating gate: no visible question block may be unrepresented.

    Earlier coverage reconciliation repairs omissions.  This last pass is a
    fail-closed safeguard after code/table recovery: it cannot silently invent
    an asset or alter verified text, but it makes any remaining omission visible
    in validation_issues and batch review counts.
    """
    canonical = json.dumps(
        questions, ensure_ascii=False, separators=(",", ":")
    )
    prompt = fr"""
你是最終「FULL_PAGE 零遺漏覆蓋閘門」。你只能稽核，不得解題或改寫 JSON。
你會收到完整原始考卷頁面，以及已完成文字、程式碼、表格與裁切驗證的 Canonical JSON。

【待稽核 Canonical JSON】
{canonical}

逐題執行以下檢查：
1. 從 top-level printed question number 開始，沿原圖閱讀順序掃描到下一題前。
2. 忽略頁首、頁尾、浮水印與全卷作答備註；其餘每一個 paragraph、option、
   subquestion、table、displayed formula、code、figure 都必須能在 Canonical 找到。
3. 特別檢查同一題是否有第二段素材：長段落後的一行函式骨架、prototype、
   sample output、頁尾前 code、跨頁頁首 code、表格後的小圖或樹的最底節點。
4. observed_asset_count 是原圖實際獨立素材區塊數；canonical_asset_count 是該題
   latex_assets 數。兩者不同必須 REVIEW，並在 missing_asset_types 說明。
5. 檢查 layout_blocks 是否把每個 asset_ref 放在正確閱讀位置；被 prose 分隔的
   兩段 code 不得被錯誤合併成一段。
6. 逐符號檢查行內數學式，尤其普通 x 與 bar/vector/bold x、`<`/`≤`、`>`/`≥`。
   任何原圖不存在的附加記號列入 symbol_mismatches。
7. 可重建表格若 render_strategy 不是 html_table，或印刷程式碼不是 code_block，
   必須 REVIEW；crop_path 只是稽核證據，不能當正常顯示格式。
8. PASS 僅允許在四個 issue 陣列全空、資產數相等且你已實際查看原圖時使用。

每個 Canonical question_number 恰好輸出一筆；不要新增或省略題目。
只輸出 JSON 陣列。
"""

    expected = {
        str(question.get("question_number", "") or ""): question
        for question in questions if isinstance(question, dict)
    }
    try:
        results = call_model_json_array(
            client,
            make_final_coverage_gate_config(),
            prompt,
            page_images,
            retries=3,
            debug_path=os.path.join(
                debug_dir, "08_final_full_page_coverage_gate_raw.txt"
            ),
        )
        actual: Dict[str, Dict[str, Any]] = {}
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("coverage gate record is not object")
            qno = str(result.get("question_number", "") or "").strip()
            if not qno or qno in actual:
                raise ValueError("coverage gate duplicate/blank question_number")
            actual[qno] = result
        if set(actual) != set(expected):
            raise ValueError(
                "coverage gate question set mismatch: "
                f"missing={sorted(set(expected) - set(actual))}; "
                f"extra={sorted(set(actual) - set(expected))}"
            )

        report: List[Dict[str, Any]] = []
        issue_fields = (
            "missing_asset_types", "missing_visible_content",
            "extra_canonical_content", "symbol_mismatches",
        )
        for qno, question in expected.items():
            result = dict(actual[qno])
            issue_values = [
                str(value).strip()
                for field in issue_fields
                for value in (result.get(field, []) or [])
                if str(value).strip()
            ][:COVERAGE_GATE_MAX_ISSUES_PER_QUESTION]
            observed_count = int(result.get("observed_asset_count", 0) or 0)
            canonical_count = len(question.get("latex_assets", []) or [])
            result["canonical_asset_count"] = canonical_count
            if observed_count != canonical_count:
                issue_values.append(
                    f"asset_count:{observed_count}!={canonical_count}"
                )
            status = (
                "PASS"
                if result.get("status") == "PASS" and not issue_values
                else "REVIEW"
            )
            result["status"] = status
            question["coverage_verification"] = result
            existing = [
                str(value)
                for value in (question.get("validation_issues", []) or [])
                if isinstance(value, str)
                and not value.startswith("coverage_unverified:")
            ]
            if status != "PASS":
                detail = ",".join(dict.fromkeys(issue_values)) or "uncertain"
                existing.append(f"coverage_unverified:{detail[:800]}")
            question["validation_issues"] = list(dict.fromkeys(existing))
            _refresh_question_review_flag(question)
            report.append(result)
        return report

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report = []
        for qno, question in expected.items():
            result = {
                "question_number": qno,
                "printed_question_number": str(
                    question.get("printed_question_number", "") or ""
                ),
                "status": "REVIEW",
                "observed_asset_count": 0,
                "canonical_asset_count": len(
                    question.get("latex_assets", []) or []
                ),
                "missing_asset_types": [],
                "missing_visible_content": [],
                "extra_canonical_content": [],
                "symbol_mismatches": [],
                "notes": error,
            }
            question["coverage_verification"] = result
            existing = [
                str(value)
                for value in (question.get("validation_issues", []) or [])
                if isinstance(value, str)
                and not value.startswith("coverage_unverified:")
            ]
            existing.append(f"coverage_unverified:gate_failed:{error[:500]}")
            question["validation_issues"] = list(dict.fromkeys(existing))
            _refresh_question_review_flag(question)
            report.append(result)
        return report


def propagate_asset_review_to_questions(
    questions: List[Dict[str, Any]],
) -> None:
    """Make asset failures visible at question and batch level."""
    for question in questions:
        if not isinstance(question, dict):
            continue
        existing = [
            str(v) for v in (question.get("validation_issues", []) or [])
            if isinstance(v, str)
            and not v.startswith("asset_review_required:")
            and not v.startswith("table_content_unverified:")
        ]
        review_assets: List[Dict[str, Any]] = []
        for asset in question.get("latex_assets", []) or []:
            if isinstance(asset, dict) and bool(asset.get("needs_review", False)):
                review_assets.append(asset)

        for asset in review_assets:
            aid = str(asset.get("asset_id", "") or "asset")
            state = str(asset.get("content_state", "") or "asset_needs_review")
            marker = f"asset_review_required:{aid}:{state}"
            if marker not in existing:
                existing.append(marker)
            verification = asset.get("table_verification", {})
            table_content_verified = (
                isinstance(verification, dict)
                and verification.get("content_status") == "PASS"
                and (
                    bool(verification.get("matrices_equal", False))
                    or int(verification.get("consensus_count", 0) or 0) >= 2
                )
            )
            if _is_table_asset(asset) and not table_content_verified:
                table_marker = f"table_content_unverified:{aid}"
                if table_marker not in existing:
                    existing.append(table_marker)

        question["validation_issues"] = existing
        # Recompute instead of only ever setting True. A later verified repair
        # is allowed to clear an obsolete asset-review marker.
        question["needs_review"] = bool(existing or review_assets)


def enforce_final_qc_invariants(
    questions: List[Dict[str, Any]],
) -> None:
    """Prevent content verification from hiding unresolved crop geometry."""
    for question in questions:
        if not isinstance(question, dict):
            continue
        for asset in question.get("latex_assets", []) or []:
            if not isinstance(asset, dict):
                continue
            post_qc = asset.get("post_crop_qc", {})
            post_status = (
                str(post_qc.get("status", "") or "")
                if isinstance(post_qc, dict) else ""
            )
            if post_status and post_status != "PASS":
                asset["needs_review"] = True
                _append_asset_note(
                    asset, f"final_invariant_unverified_geometry:{post_status}"
                )

            code_verification = asset.get("code_verification", {})
            if (
                isinstance(code_verification, dict)
                and code_verification.get("content_status") == "PASS"
                and post_status != "PASS"
            ):
                code_verification["status"] = "PASS_WITH_GEOMETRY_REVIEW"
                code_verification["geometry_status"] = "REVIEW"

            table_verification = asset.get("table_verification", {})
            if (
                isinstance(table_verification, dict)
                and table_verification.get("content_status") == "PASS"
                and post_status != "PASS"
            ):
                table_verification["status"] = "PASS_WITH_GEOMETRY_REVIEW"
                table_verification["geometry_status"] = "REVIEW"

        _refresh_question_review_flag(question)


def crop_assets_from_llm_bbox(
    questions: List[Dict[str, Any]],
    page_images: List[str],
    asset_dir: str,
    run_dir: str,
    pdf_file: str,
    document_id: str,
) -> List[Dict[str, Any]]:
    ensure_dir(asset_dir)
    flat_assets: List[Dict[str, Any]] = []

    for question in questions:
        for asset in question.get("latex_assets", []) or []:
            page_no = asset.get("page_number")
            bbox = asset.get("bbox_pct")
            aid = str(asset.get("asset_id", "asset") or "asset")
            safe_aid = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", aid)
            crop_path = ""

            try:
                if (
                    not isinstance(page_no, int)
                    or not (1 <= page_no <= len(page_images))
                    or not _valid_bbox_0_1000(bbox)
                ):
                    raise ValueError("invalid page_number or bbox_pct")

                out_path = os.path.join(asset_dir, f"{safe_aid}.png")
                _render_bbox_crop(
                    page_images[page_no - 1], bbox, out_path
                )
                crop_path = _relpath_posix(out_path, run_dir)

            except Exception as exc:
                asset["needs_review"] = True
                prior = str(asset.get("notes", "") or "").strip()
                marker = f"crop_failed:{type(exc).__name__}"
                asset["notes"] = f"{prior} | {marker}".strip(" |")

            asset["crop_path"] = crop_path
            asset["source_page_image_path"] = (
                _relpath_posix(page_images[page_no - 1], run_dir)
                if isinstance(page_no, int) and 1 <= page_no <= len(page_images)
                else ""
            )
            asset["source_pdf"] = pdf_file
            asset["document_id"] = document_id
            flat_assets.append(dict(asset))

    return flat_assets


# ============================================================
# Atomic output
# ============================================================
def atomic_write_json(path: str, data: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        with open(tmp, "r", encoding="utf-8") as f:
            json.load(f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ============================================================
# One PDF
# ============================================================
def analyze_pdf_prompt_primary(
    client: Any,
    generation_config: Any,
    pdf_path: str,
    run_dir: str,
    max_schema_repairs: int = 2,
    max_post_crop_qc_rounds: int = DEFAULT_MAX_POST_CROP_QC_ROUNDS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pdf_file = os.path.basename(pdf_path)
    document_id = make_document_id(pdf_file)
    pdf_work_dir = os.path.join(run_dir, "work", document_id)
    page_dir = os.path.join(pdf_work_dir, "pages")
    debug_dir = os.path.join(pdf_work_dir, "llm_debug")
    asset_dir = os.path.join(run_dir, "assets", document_id)
    ensure_dir(debug_dir)

    print(f"PDF -> high-resolution PNG: {pdf_file}")
    page_images = pdf_to_page_images(pdf_path, page_dir)
    if not page_images:
        raise RuntimeError("PDF 未產生任何頁面 PNG")
    print(f"   pages: {len(page_images)}")

    print("Pass 1: Gemini canonical extraction")
    initial = llm_initial_extract(
        client, generation_config, pdf_file, page_images, debug_dir
    )
    normalize_option_continuations(initial)
    atomic_write_json(os.path.join(debug_dir, "01_initial.json"), initial)

    print("Pass 2A: independent visual review")
    review_a = llm_visual_review(
        client, generation_config, pdf_file, page_images, initial, debug_dir,
        pass_name="A", reverse_images=False,
    )
    normalize_option_continuations(review_a)
    atomic_write_json(os.path.join(debug_dir, "02_review_A.json"), review_a)

    print("Pass 2B: reversed-evidence visual review")
    review_b = llm_visual_review(
        client, generation_config, pdf_file, page_images, initial, debug_dir,
        pass_name="B", reverse_images=True,
    )
    normalize_option_continuations(review_b)
    atomic_write_json(os.path.join(debug_dir, "02_review_B.json"), review_b)

    print("Pass 3: visual arbitration")
    final = llm_arbitrate(
        client, generation_config, pdf_file, page_images,
        initial, review_a, review_b, debug_dir,
    )
    normalize_option_continuations(final)
    ensure_minimum_layout_blocks(final)
    enforce_structured_asset_rendering_policy(final)
    atomic_write_json(os.path.join(debug_dir, "03_arbitrated.json"), final)

    errors = validate_structure(final, len(page_images), require_nonempty=True)
    repair_round = 0
    while errors and repair_round < max_schema_repairs:
        repair_round += 1
        print(
            f"Schema repair {repair_round}: {len(errors)} structural error(s)"
        )
        atomic_write_json(
            os.path.join(debug_dir, f"04_schema_errors_{repair_round}.json"),
            errors,
        )
        final = llm_schema_repair(
            client, generation_config, pdf_file, page_images, final, errors,
            debug_dir, repair_round,
        )
        normalize_option_continuations(final)
        ensure_minimum_layout_blocks(final)
        enforce_structured_asset_rendering_policy(final)
        atomic_write_json(
            os.path.join(debug_dir, f"04_schema_repaired_{repair_round}.json"),
            final,
        )
        errors = validate_structure(
            final, len(page_images), require_nonempty=True
        )

    if errors:
        atomic_write_json(
            os.path.join(debug_dir, "FINAL_SCHEMA_ERRORS.json"), errors
        )
        raise RuntimeError(
            f"Gemini schema repair 後仍有 {len(errors)} 個結構錯誤。"
        )

    # The three normal agents can inherit the same omission because the two
    # reviews receive routing hints derived from the initial extraction.  Run a
    # fully candidate-free page inventory, then reconcile it against the
    # arbitrated result using the original pages.  This does not replace the
    # existing flow; it closes its shared-blind-spot before exact text and asset
    # verification begin.
    print("Pass 3B: candidate-free full-page content coverage scan")
    independent_coverage = llm_candidate_free_coverage_extract(
        client, generation_config, pdf_file, page_images, debug_dir
    )
    normalize_option_continuations(independent_coverage)
    ensure_minimum_layout_blocks(independent_coverage)
    enforce_structured_asset_rendering_policy(independent_coverage)
    atomic_write_json(
        os.path.join(debug_dir, "03b_candidate_free_coverage.json"),
        independent_coverage,
    )

    print("Pass 3C: reconcile arbitration with independent coverage")
    final = llm_reconcile_full_page_coverage(
        client,
        generation_config,
        pdf_file,
        page_images,
        final,
        independent_coverage,
        debug_dir,
    )
    normalize_option_continuations(final)
    ensure_minimum_layout_blocks(final)
    enforce_structured_asset_rendering_policy(final)
    atomic_write_json(
        os.path.join(debug_dir, "03c_coverage_reconciled.json"), final
    )

    coverage_errors = validate_structure(
        final, len(page_images), require_nonempty=True
    )
    coverage_repair_round = 0
    while (
        coverage_errors
        and coverage_repair_round < max_schema_repairs
    ):
        coverage_repair_round += 1
        atomic_write_json(
            os.path.join(
                debug_dir,
                f"03d_coverage_schema_errors_{coverage_repair_round}.json",
            ),
            coverage_errors,
        )
        final = llm_schema_repair(
            client,
            generation_config,
            pdf_file,
            page_images,
            final,
            coverage_errors,
            debug_dir,
            100 + coverage_repair_round,
        )
        normalize_option_continuations(final)
        ensure_minimum_layout_blocks(final)
        enforce_structured_asset_rendering_policy(final)
        coverage_errors = validate_structure(
            final, len(page_images), require_nonempty=True
        )

    if coverage_errors:
        atomic_write_json(
            os.path.join(debug_dir, "FINAL_COVERAGE_SCHEMA_ERRORS.json"),
            coverage_errors,
        )
        raise RuntimeError(
            "全頁覆蓋修復後結構仍不合法："
            + "; ".join(coverage_errors[:8])
        )

    # Every question is re-read twice without candidate text. Only two exact
    # visual matrices may update question_text/options/subquestions.
    print("Pass 4: candidate-free exact text/option/subquestion double read")
    exact_text_report = run_exact_text_double_read_verification(
        client, final, page_images, debug_dir, document_id
    )
    normalize_option_continuations(final)
    ensure_minimum_layout_blocks(final)
    enforce_structured_asset_rendering_policy(final)
    atomic_write_json(
        os.path.join(debug_dir, "04_exact_text_verification_report.json"),
        exact_text_report,
    )
    exact_text_errors = validate_structure(
        final, len(page_images), require_nonempty=True
    )
    exact_repair_round = 0
    while exact_text_errors and exact_repair_round < max_schema_repairs:
        exact_repair_round += 1
        atomic_write_json(
            os.path.join(
                debug_dir,
                f"04b_exact_text_schema_errors_{exact_repair_round}.json",
            ),
            exact_text_errors,
        )
        final = llm_schema_repair(
            client,
            generation_config,
            pdf_file,
            page_images,
            final,
            exact_text_errors,
            debug_dir,
            200 + exact_repair_round,
        )
        normalize_option_continuations(final)
        ensure_minimum_layout_blocks(final)
        enforce_structured_asset_rendering_policy(final)
        exact_text_errors = validate_structure(
            final, len(page_images), require_nonempty=True
        )

    if exact_text_errors:
        atomic_write_json(
            os.path.join(debug_dir, "04_exact_text_structure_errors.json"),
            exact_text_errors,
        )
        raise RuntimeError(
            "Exact text verification 後結構驗證失敗："
            + "; ".join(exact_text_errors[:8])
        )

    # Dedicated final visual audit for every asset immediately before cropping.
    # This pass exists because general question review is not sufficiently
    # specialized for exact four-edge localization.
    if any((q.get("latex_assets") or []) for q in final):
        print("Pass 5: strict asset geometry/content audit")
        audited_assets = llm_asset_strict_audit(
            client, final, page_images, debug_dir
        )
        atomic_write_json(
            os.path.join(debug_dir, "05_asset_strict_audit.json"),
            audited_assets,
        )
        apply_asset_strict_audit(final, audited_assets)
        enforce_structured_asset_rendering_policy(final)

        asset_audit_errors = validate_structure(
            final, len(page_images), require_nonempty=True
        )
        if asset_audit_errors:
            atomic_write_json(
                os.path.join(debug_dir, "05_asset_post_audit_errors.json"),
                asset_audit_errors,
            )
            raise RuntimeError(
                "Asset strict audit 後結構驗證失敗："
                + "; ".join(asset_audit_errors[:8])
            )

        print("Pass 6: render crop -> visual QC -> automatic recrop")
        post_crop_qc_report = run_post_crop_visual_qc(
            client,
            final,
            page_images,
            debug_dir,
            max_rounds=max_post_crop_qc_rounds,
        )
        enforce_structured_asset_rendering_policy(final)
        atomic_write_json(
            os.path.join(debug_dir, "06_post_crop_qc_report.json"),
            post_crop_qc_report,
        )

        # Post-crop QC may update only asset page_number/bbox_pct plus execution
        # review metadata. Canonical structural constraints must still hold.
        post_crop_errors = validate_structure(
            final, len(page_images), require_nonempty=True
        )
        if post_crop_errors:
            atomic_write_json(
                os.path.join(debug_dir, "06_post_crop_qc_structure_errors.json"),
                post_crop_errors,
            )
            raise RuntimeError(
                "Post-crop QC 後結構驗證失敗："
                + "; ".join(post_crop_errors[:8])
            )

        print("Pass 7: code crop -> edge-aware recrop -> 2-of-3 exact line read")
        code_verification_report = run_code_crop_content_verification(
            client,
            final,
            page_images,
            debug_dir,
        )
        enforce_structured_asset_rendering_policy(final)
        atomic_write_json(
            os.path.join(debug_dir, "06b_code_crop_verification_report.json"),
            code_verification_report,
        )
        propagate_asset_review_to_questions(final)

        code_stage_errors = validate_structure(
            final, len(page_images), require_nonempty=True
        )
        if code_stage_errors:
            atomic_write_json(
                os.path.join(debug_dir, "06b_code_stage_structure_errors.json"),
                code_stage_errors,
            )
            raise RuntimeError(
                "Code crop verification 後結構驗證失敗："
                + "; ".join(code_stage_errors[:8])
            )

        print("Pass 8: table localize -> phantom-column guard -> 2-of-3 read")
        table_verification_report = run_table_crop_content_verification(
            client,
            final,
            page_images,
            debug_dir,
        )
        enforce_structured_asset_rendering_policy(final)
        atomic_write_json(
            os.path.join(debug_dir, "07_table_crop_verification_report.json"),
            table_verification_report,
        )

        # A table is accepted only after local geometry recovery and two exact
        # matrix reads. Unverified printed tables remain structured REVIEW
        # candidates; their crops are audit evidence, never silent PNG output.
        propagate_asset_review_to_questions(final)

        table_stage_errors = validate_structure(
            final, len(page_images), require_nonempty=True
        )
        if table_stage_errors:
            atomic_write_json(
                os.path.join(debug_dir, "07_table_stage_structure_errors.json"),
                table_stage_errors,
            )
            raise RuntimeError(
                "Table crop verification 後結構驗證失敗："
                + "; ".join(table_stage_errors[:8])
            )

    print("Pass 9: final FULL_PAGE zero-omission coverage gate")
    normalize_option_continuations(final)
    final_coverage_report = run_final_full_page_coverage_gate(
        client, final, page_images, debug_dir
    )
    enforce_structured_asset_rendering_policy(final)
    atomic_write_json(
        os.path.join(debug_dir, "08_final_full_page_coverage_gate.json"),
        final_coverage_report,
    )

    # Also propagates non-table review flags when a document has no table.
    normalize_option_continuations(final)
    enforce_final_qc_invariants(final)
    propagate_asset_review_to_questions(final)

    # Namespace only changes identifiers, never document semantics.
    namespace_asset_ids(final, document_id)
    post_namespace_errors = validate_structure(
        final, len(page_images), require_nonempty=True
    )
    if post_namespace_errors:
        raise RuntimeError(
            f"asset namespace 後結構失敗：{post_namespace_errors[:5]}"
        )

    attach_execution_metadata(final, pdf_file, document_id)
    audit_id_errors = validate_unique_audit_ids(final)
    if audit_id_errors:
        atomic_write_json(
            os.path.join(debug_dir, "FINAL_AUDIT_ID_ERRORS.json"),
            audit_id_errors,
        )
        raise RuntimeError(
            "exact_text audit_id 不唯一或缺失："
            + "; ".join(audit_id_errors[:8])
        )

    print("Crop assets from Gemini bbox")
    flat_assets = crop_assets_from_llm_bbox(
        final, page_images, asset_dir, run_dir, pdf_file, document_id
    )

    # Final file-level crop failures can add needs_review after table recovery.
    enforce_final_qc_invariants(final)
    propagate_asset_review_to_questions(final)

    # Crop may add execution fields only. Canonical fields must remain valid.
    final_errors = validate_structure(
        final, len(page_images), require_nonempty=True
    )
    if final_errors:
        raise RuntimeError(f"裁圖後結構驗證失敗：{final_errors[:5]}")

    return final, flat_assets


# ============================================================
# Batch output
# ============================================================
def write_conversion_summary(
    run_dir: str,
    project: str,
    location: str,
    summary: List[Dict[str, Any]],
    question_count: int,
    asset_count: int,
) -> str:
    success_count = sum(
        1 for row in summary
        if row.get("status") in {"SUCCESS", "SUCCESS_WITH_REVIEW"}
    )
    review_count = sum(
        1 for row in summary if row.get("status") == "SUCCESS_WITH_REVIEW"
    )
    corrected_question_count = sum(
        int(row.get("corrected_question_count", 0) or 0) for row in summary
    )
    fail_count = sum(1 for row in summary if row.get("status") == "FAILED")
    lines = [
        "# FINAL12 Prompt-Primary Conversion Summary\n\n",
        f"- Generated: {now_iso()}\n",
        f"- Model: `{GEMINI_MODEL_NAME}`\n",
        f"- Vertex project: `{project}`\n",
        f"- Vertex location: `{location}`\n",
        f"- PDF success: **{success_count}**\n",
        f"- PDF success with review required: **{review_count}**\n",
        f"- PDF failed: **{fail_count}**\n",
        f"- Questions: **{question_count}**\n",
        f"- Questions automatically corrected and verified: **{corrected_question_count}**\n",
        f"- Assets: **{asset_count}**\n",
        "- Main output: `new_exam_output.json`\n",
        "- Asset output: `new_exam_assets.json`\n",
        "- Batch report: `batch_summary.json`\n",
        "- Cost report: `cost_report.json`\n",
        "- Failed list: `failed_files.txt`\n",
    ]
    path = os.path.join(run_dir, "conversion_summary.md")
    Path(path).write_text("".join(lines), encoding="utf-8", newline="\n")
    return path


def _make_unique_run_dir(output_root: str) -> str:
    ensure_dir(output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_root, f"run_{stamp}")
    if not os.path.exists(base):
        return base
    for index in range(1, 1000):
        candidate = f"{base}_{index:02d}"
        if not os.path.exists(candidate):
            return candidate
    raise RuntimeError("無法建立唯一 run directory")


def process_pdf_folder(
    client: Any,
    generation_config: Any,
    *,
    project: str,
    location: str,
    pdf_folder: str = "exam_img",
    output_root: str = "output_json",
    fresh_run: bool = True,
    max_schema_repairs: int = 2,
    max_post_crop_qc_rounds: int = DEFAULT_MAX_POST_CROP_QC_ROUNDS,
) -> str:
    if not os.path.isdir(pdf_folder):
        raise FileNotFoundError(f"找不到 PDF 資料夾：{pdf_folder}")

    pdf_files = sorted(
        name for name in os.listdir(pdf_folder)
        if name.lower().endswith(".pdf")
    )
    if not pdf_files:
        raise RuntimeError(f"PDF 資料夾內沒有 .pdf：{pdf_folder}")

    run_dir = _make_unique_run_dir(output_root) if fresh_run else output_root
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "results"))

    output_file = os.path.join(run_dir, "new_exam_output.json")
    asset_output_file = os.path.join(run_dir, "new_exam_assets.json")
    failed_file = os.path.join(run_dir, "failed_files.txt")
    batch_summary_file = os.path.join(run_dir, "batch_summary.json")
    manifest_file = os.path.join(run_dir, "run_manifest.json")

    manifest = {
        "generated_at": now_iso(),
        "model": GEMINI_MODEL_NAME,
        "project": project,
        "location": location,
        "pdf_folder": os.path.abspath(pdf_folder),
        "output_root": os.path.abspath(output_root),
        "architecture": (
            "prompt_primary_with_candidate_free_coverage_exact_text_"
            "structured_assets_and_post_crop_qc"
        ),
        "bbox_convention": "[y_min,x_min,y_max,x_max] normalized 0..1000",
        "max_post_crop_qc_rounds": int(max_post_crop_qc_rounds),
    }
    atomic_write_json(manifest_file, manifest)

    all_questions: List[Dict[str, Any]] = []
    all_assets: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    failures: List[str] = []

    print("=" * 70)
    print("FINAL12 STRICT TEXT / OPTION / CODE / TABLE AUDIT + POST-CROP QC")
    print(f"Model   : {GEMINI_MODEL_NAME}")
    print(f"Project : {project}")
    print(f"Location: {location}")
    print(f"PDFs    : {len(pdf_files)}")
    print("=" * 70)

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"\nPDF: {filename}")
        stats_start(filename)
        started = time.time()
        status = "SUCCESS"
        reason = ""
        q_count = 0
        a_count = 0
        review_question_count = 0
        corrected_question_count = 0

        try:
            questions, assets = analyze_pdf_prompt_primary(
                client, generation_config, pdf_path, run_dir,
                max_schema_repairs=max_schema_repairs,
                max_post_crop_qc_rounds=max_post_crop_qc_rounds,
            )

            # Persist a single per-PDF bundle before committing it to aggregates.
            document_id = make_document_id(filename)
            bundle_path = os.path.join(
                run_dir, "results", f"{document_id}.json"
            )
            atomic_write_json(
                bundle_path,
                {
                    "source_pdf": filename,
                    "document_id": document_id,
                    "questions": questions,
                    "assets": assets,
                },
            )

            all_questions.extend(questions)
            all_assets.extend(assets)
            q_count = len(questions)
            a_count = len(assets)
            review_question_count = sum(
                1 for question in questions
                if bool(question.get("needs_review", False))
            )
            corrected_question_count = sum(
                1 for question in questions
                if isinstance(question.get("exact_text_verification"), dict)
                and question["exact_text_verification"].get("status")
                == "CORRECTED"
            )
            if review_question_count:
                status = "SUCCESS_WITH_REVIEW"
                reason = f"{review_question_count} question(s) require review"
                print(
                    f"SUCCESS_WITH_REVIEW {filename}: {q_count} questions / "
                    f"{a_count} assets / {review_question_count} review / "
                    f"{corrected_question_count} corrected"
                )
            else:
                print(
                    f"SUCCESS {filename}: {q_count} questions / {a_count} assets / "
                    f"{corrected_question_count} corrected"
                )

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            status = "FAILED"
            reason = f"{type(exc).__name__}: {exc}"
            failures.append(f"{filename}\t{reason}")
            print(f"FAILED {filename}: {reason}")

        stats_finish(filename, q_count, status)
        summary.append({
            "filename": filename,
            "status": status,
            "reason": reason,
            "question_count": q_count,
            "asset_count": a_count,
            "review_question_count": review_question_count,
            "corrected_question_count": corrected_question_count,
            "duration_sec": round(time.time() - started, 2),
            "model": GEMINI_MODEL_NAME,
        })
        atomic_write_json(batch_summary_file, summary)
        write_stats_report(run_dir)

    # Aggregates are written once from successful per-PDF results only.  The
    # document-scoped digest must also remain unique across the whole batch.
    aggregate_audit_id_errors = validate_unique_audit_ids(all_questions)
    if aggregate_audit_id_errors:
        atomic_write_json(
            os.path.join(run_dir, "FINAL_AUDIT_ID_ERRORS.json"),
            aggregate_audit_id_errors,
        )
        raise RuntimeError(
            "批次 exact_text audit_id 發生碰撞："
            + "; ".join(aggregate_audit_id_errors[:8])
        )

    atomic_write_json(output_file, all_questions)
    atomic_write_json(asset_output_file, all_assets)
    Path(failed_file).write_text(
        ("\n".join(failures) + "\n") if failures else "",
        encoding="utf-8",
        newline="\n",
    )
    atomic_write_json(batch_summary_file, summary)
    write_stats_report(run_dir)
    write_conversion_summary(
        run_dir, project, location, summary,
        len(all_questions), len(all_assets),
    )

    print("\n" + "=" * 70)
    print(f"Output : {output_file}")
    print(f"Assets : {asset_output_file}")
    print(f"Run dir: {run_dir}")
    print("=" * 70)

    if failures:
        raise RuntimeError(
            f"批次完成但有 {len(failures)} 份 PDF 失敗；"
            f"成功結果已保留於 {run_dir}"
        )

    return run_dir


# ============================================================
# Preflight
# ============================================================
def preflight(client: Any) -> Tuple[bool, str]:
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents="Return the empty JSON array [] and nothing else.",
            config=make_preflight_config(),
        )
        _record_usage_into(_RUN_OVERHEAD, response)
        text = response_text(response)
        parsed = extract_json_array(text)
        if parsed is None:
            if not text:
                return True, "Gemini 2.5 Pro 可連線；preflight 未回傳文字，略過 JSON 格式檢查"
            return False, f"模型可連線，但 preflight JSON 不合法：{text[:120]!r}"
        return True, f"Gemini 2.5 Pro 可連線；response={text[:80]!r}"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        low = msg.lower()
        if "default credentials" in low or "could not automatically determine credentials" in low:
            msg += "；請執行 gcloud auth application-default login"
        return False, msg


# ============================================================
# CLI
# ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDF -> high-res PNG -> Gemini 2.5 Pro -> canonical exam JSON"
    )
    parser.add_argument("--pdf-folder", default="exam_img")
    parser.add_argument("--output-root", default="output_json")
    parser.add_argument("--max-schema-repairs", type=int, default=2)
    parser.add_argument(
        "--max-post-crop-qc-rounds",
        type=int,
        default=DEFAULT_MAX_POST_CROP_QC_ROUNDS,
        help="每個 asset 實際裁切後的視覺驗收/自動重裁最多輪數（預設 3）",
    )
    parser.add_argument("--no-fresh-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--gcp-project", default="")
    parser.add_argument("--gcp-location", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    project = resolve_vertex_project(args.gcp_project)
    location = resolve_vertex_location(args.gcp_location)
    client = create_genai_client(project, location)
    generation_config = make_generation_config()

    print(
        f"Vertex AI: project={project}, location={location}, "
        f"model={GEMINI_MODEL_NAME}"
    )

    if args.preflight:
        ok, message = preflight(client)
        print(("OK " if ok else "ERROR ") + message)
        if not ok:
            raise SystemExit(2)

    process_pdf_folder(
        client,
        generation_config,
        project=project,
        location=location,
        pdf_folder=args.pdf_folder,
        output_root=args.output_root,
        fresh_run=not args.no_fresh_run,
        max_schema_repairs=max(0, args.max_schema_repairs),
        max_post_crop_qc_rounds=max(1, args.max_post_crop_qc_rounds),
    )


if __name__ == "__main__":
    main()
