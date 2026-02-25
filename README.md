# Duke Breast MRI Gemini Evaluation Pipeline

This repo runs DUKE Breast MRI evaluation with:
- Label: `Nottingham grade` (1/2/3)
- Image preprocessing: RGB fusion from MRI (`R=pre`, `G=post1`, `B=post1-pre`)
- Split: fixed validation patient list

## Prerequisites

- Python 3.10+
- Local DUKE data under `/mnt/e/BreastCancerData`
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

- Set Gemini API key for online steps:

```bash
export GEMINI_API_KEY=YOUR_KEY
```

## Main Workflow

1. Prepare feature subset and RGB images

```bash
python scripts/duke_gemini_pipeline.py prepare-nottingham-rgb
```

Key outputs:
- `data/intermediate/validation_patients_nottingham.csv`
- `data/intermediate/clinical_features_validation_nottingham.csv`
- `data/intermediate/nottingham_rgb_image_manifest_256crop.csv`
- `data/images_rgb_nottingham_256crop/{PatientID}/slice_XXX_rgb.png`

2. Build upload manifest

```bash
python scripts/duke_gemini_pipeline.py seed-upload-manifest
```

Output:
- `data/intermediate/upload_manifest_nottingham.csv`

3. Build batch JSONL

Option A (inline image bytes, fastest to start):

```bash
python scripts/duke_gemini_pipeline.py build-jsonl --use-inline-data-from-local

# optional: control Gemini reasoning strength in request generation config
python scripts/duke_gemini_pipeline.py build-jsonl \
  --use-inline-data-from-local \
  --reasoning-strength medium
```

Option B (recommended for larger runs, upload files then reference URIs):

```bash
python scripts/duke_gemini_pipeline.py upload-files-from-manifest \
  --manifest-in data/intermediate/upload_manifest_nottingham.csv \
  --manifest-out data/intermediate/upload_manifest_nottingham_uploaded.csv

python scripts/duke_gemini_pipeline.py build-jsonl-nottingham \
  --upload-manifest data/intermediate/upload_manifest_nottingham_uploaded.csv
```

Main output JSONL:
- `data/gemini/batch_requests_validation_nottingham.jsonl`

4. (Optional) Create a 3-row low-cost test batch

```bash
python scripts/duke_gemini_pipeline.py build-test3-jsonl
```

Outputs:
- `data/gemini/batch_requests_validation_nottingham_test3.jsonl`
- `data/intermediate/test3_nottingham_selected_keys.txt`

5. Submit / monitor / fetch

```bash
python scripts/duke_gemini_pipeline.py submit-batch \
  --model gemini-3-flash-preview \
  --jsonl-path data/gemini/batch_requests_validation_nottingham_test3.jsonl \
  --display-name duke-nottingham-test3

python scripts/duke_gemini_pipeline.py poll-batch

python scripts/duke_gemini_pipeline.py fetch-batch-results \
  --job-meta data/gemini/batch_job.json \
  --output-path data/gemini/nottingham_test3_results.jsonl
```

6. Evaluate against Nottingham grade labels

```bash
python scripts/duke_gemini_pipeline.py evaluate \
  --results-jsonl data/gemini/nottingham_test3_results.jsonl
```

Outputs:
- `data/results/nottingham_eval.csv`
- `data/results/nottingham_metrics.json`

## Useful Commands

- List recent batch jobs:

```bash
python scripts/duke_gemini_pipeline.py list-batches --limit 20
```

- Poll a specific job:

```bash
python scripts/duke_gemini_pipeline.py poll-batch --job-name batches/XXXX
```

- Fetch results for a specific job:

```bash
python scripts/duke_gemini_pipeline.py fetch-batch-results --job-name batches/XXXX
```

## Troubleshooting

- `Missing GEMINI_API_KEY`: set env var in the same shell.
- Batch stuck in `PENDING`: wait, then retry with small test job or stable model.
- `poll-batch` JSON serialization error: fixed in current script version.
