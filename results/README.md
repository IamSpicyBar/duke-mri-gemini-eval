# Results

Tracked evaluation artifacts for quick review on GitHub.

Current files:
- `nottingham_metrics.json`: aggregate metrics for Nottingham-grade evaluation.
- `nottingham_eval.csv`: per-patient predictions vs gold labels.

Source pipeline command:
```bash
python scripts/duke_gemini_pipeline.py evaluate --results-jsonl <batch_output_jsonl>
```
