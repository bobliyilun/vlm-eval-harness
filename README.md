# vlm-eval-harness

A small, model-agnostic evaluation harness for vision-language model outputs. It scores JSONL prediction exports, tolerates common free-form answer wrappers, and reports overall and per-category accuracy.

## Current scope

- multiple-choice and exact-match text scoring
- invalid-output accounting
- per-category metrics
- deterministic JSON reports
- optional confidence coverage, mean confidence, and calibration error
- optional latency and input/output token-usage totals
- optional top-k candidate-list scoring
- macro-average accuracy and invalid-rate metrics across categories
- optional deterministic bootstrap confidence intervals for accuracy and invalid rate

The repository evaluates existing predictions; model inference adapters remain work in progress.

## HTTP inference adapters

`HTTPInferenceAdapter` provides the small common layer for JSON-speaking model
endpoints. Supply the endpoint URL plus functions that map a dataset record to
the endpoint's JSON request and its JSON response to a prediction:

```python
from inference import HTTPInferenceAdapter

adapter = HTTPInferenceAdapter(
    "http://localhost:8000/infer",
    lambda record: {"prompt": record["prompt"]},
    lambda response: response["answer"],
)
prediction = adapter.infer({"prompt": "Which letter is correct?"})
```

The adapter uses a JSON `POST`, has a 30-second default timeout, and leaves
provider-specific request and response shapes to those two mapping functions.

## Run

```bash
python3 evaluate.py data/sample_predictions.jsonl
python3 evaluate.py --ignore-case --ignore-punctuation data/sample_predictions.jsonl
python3 evaluate.py --numeric-tolerance 0.05 predictions.jsonl
python3 evaluate.py --bootstrap-samples 1000 predictions.jsonl
python3 evaluate.py baseline.jsonl --compare-with candidate.jsonl
python3 -m unittest -v
```

Text labels are exact-match by default. Use `--ignore-case` and/or
`--ignore-punctuation` when a dataset treats those differences as equivalent.
For direct numeric labels and predictions, `--numeric-tolerance` enables an
absolute tolerance (for example, `0.05` accepts `2.54` for a label of `2.5`).
Wrapped numeric prose is reported as invalid rather than guessed.

Predictions may include an optional `confidence` field from 0 to 1. Metrics
with at least one such field report coverage, mean confidence, and a 10-bin
expected calibration error; records without confidence remain supported.

To measure candidate-list accuracy, add a non-empty ordered `top_k` JSON array
to any record. The report retains top-1 `accuracy` and adds `top_k_count` plus
`top_k_accuracy` over records with that field.

The overall report also includes `macro_accuracy` and `macro_invalid_rate`:
each is the unweighted mean of the respective per-category metric.

Every record must have a non-empty string `id`; duplicate or missing IDs are
rejected before scoring.

## Dataset schema

Each non-blank JSONL line must be a JSON object with `id`, `label`, and
`prediction` fields. `id` and an optional `category` must be non-empty strings;
an optional `image_path` must be a non-empty local file path. Relative image
paths are resolved from the JSONL file's directory and checked before scoring.
The scorer validates label, prediction, and optional metric values according to
the selected scoring mode. Unknown fields are retained for dataset metadata and
future adapters. Validation errors name the one-based record number and
offending field before scoring, so a malformed export can be fixed without
interpreting a partial report.

Records may optionally include non-negative `latency_ms`, `input_tokens`, and
`output_tokens`. Each report group with those fields includes latency coverage,
mean and total latency, plus summed input and output token counts. Token counts
must be whole numbers.

For uncertainty estimates, `--bootstrap-samples` adds deterministic percentile
95% confidence intervals for accuracy and invalid rate to every reported group.
Use `--bootstrap-seed` to change the resampling sequence (the default is `0`).

See [ROADMAP.md](ROADMAP.md) for the next adapters and evaluation slices.

`--compare-with` requires the same unique record IDs and labels in both files.
It reports each model's accuracy, the paired accuracy difference, discordant-pair
counts, and a two-sided exact paired p-value.
