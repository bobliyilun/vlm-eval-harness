# vlm-eval-harness

A small, model-agnostic evaluation harness for vision-language model outputs. It scores JSONL prediction exports, tolerates common free-form answer wrappers, and reports overall and per-category accuracy.

## Current scope

- multiple-choice and exact-match text scoring
- invalid-output accounting
- per-category metrics
- deterministic JSON reports

The repository evaluates existing predictions; model inference adapters remain work in progress.

## Run

```bash
python3 evaluate.py data/sample_predictions.jsonl
python3 evaluate.py --ignore-case --ignore-punctuation data/sample_predictions.jsonl
python3 evaluate.py --numeric-tolerance 0.05 predictions.jsonl
python3 -m unittest -v
```

Text labels are exact-match by default. Use `--ignore-case` and/or
`--ignore-punctuation` when a dataset treats those differences as equivalent.
For direct numeric labels and predictions, `--numeric-tolerance` enables an
absolute tolerance (for example, `0.05` accepts `2.54` for a label of `2.5`).
Wrapped numeric prose is reported as invalid rather than guessed.

See [ROADMAP.md](ROADMAP.md) for the next adapters and evaluation slices.
