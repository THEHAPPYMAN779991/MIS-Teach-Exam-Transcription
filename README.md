# Exam transcription

This package converts a permitted PDF collection into structured examination JSON. It renders
pages, uses Gemini through Vertex AI to identify question structure and assets, and writes
intermediate quality reports alongside the generated result.

## Public-source mapping

`src/pipeline.py` is an unchanged functional copy of the latest local runner, whose original
filename contained an implementation-history suffix. It was renamed only to give the public
package a stable entry-point name; its processing logic, prompts, model choice, and output
contract were not changed during packaging.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GOOGLE_CLOUD_PROJECT` (or `GCP_PROJECT`) in the local `.env`. Vertex AI authentication is
provided by your own Google Cloud credentials; no API key, PDF, image, or generated question
data is included in this repository.

## Run

```powershell
python .\scripts\run_pipeline.py --preflight
python .\scripts\run_pipeline.py --pdf-folder .\input --output-root .\generated
```

`input/` and `generated/` are intentionally ignored. Only process documents that you have the
right to use and distribute.

## Public-package check

```powershell
python -m unittest discover -s tests
```
