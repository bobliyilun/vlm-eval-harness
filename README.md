# vlm-eval-harness

A small, model-agnostic evaluation harness for vision-language model outputs. It scores JSONL prediction exports, tolerates common free-form answer wrappers, and reports overall and per-category accuracy.

## Current scope

- multiple-choice answer normalization
- invalid-output accounting
- per-category metrics
- deterministic JSON reports

The repository evaluates existing predictions; model inference adapters remain work in progress.

## Run

```bash
python3 evaluate.py data/sample_predictions.jsonl
python3 -m unittest -v
```

See [ROADMAP.md](ROADMAP.md) for the next adapters and evaluation slices.

