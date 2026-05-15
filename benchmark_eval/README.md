# Benchmark Evaluation

This directory evaluates:

- **Enabling contribution generation**
- **Grounding**
- **End-to-end generation + grounding**

The code supports the public claim-level files in `../data/`.

## Files

| Path | Purpose |
| --- | --- |
| `scripts/generate_enabling_contributions.py` | Generate enabling contribution predictions. |
| `scripts/judge_enabling_contributions.py` | Judge generated enabling contributions against `dev.json` or `training.json`. |
| `scripts/run_grounding.py` | Run grounding retrieval and ranking. |
| `scripts/run_end_to_end.py` | Run generation, judging, and grounding in sequence. |
| `modules/` | Shared loading, generation, judging, retrieval, and metric code. |

## Setup

```bash
cd benchmark_eval
pip install -r requirements.txt
cp .env.example .env
```

Set the API keys you need:

```text
GEMINI_API_KEY=<Gemini API key>
SEMANTIC_SCHOLAR_API_KEY=<optional Semantic Scholar API key>
```

## Enabling Contribution Generation

Generate predictions on the public development split:

```bash
python scripts/generate_enabling_contributions.py \
  --source gold \
  --gold-file ../data/dev.json \
  --gold-claim-field target_contribution \
  --setting 1 \
  --generator-model gemini/gemini-3.1-pro-preview \
  --concurrency 4 \
  --output results/dev_generation_predictions.json \
  --resume
```

Judge the predictions:

```bash
python scripts/judge_enabling_contributions.py \
  --predictions-file results/dev_generation_predictions.json \
  --source gold \
  --gold-file ../data/dev.json \
  --gold-claim-field target_contribution \
  --judge-model gemini/gemini-3.1-pro-preview \
  --concurrency 8 \
  --output results/dev_generation_judged.json \
  --resume
```

Use `--limit 5` for a smoke test.

## Grounding

Run grounding on the public development split with gold enabling contributions:

```bash
python scripts/run_grounding.py \
  --gold-file ../data/dev.json \
  --gold-claim-field target_contribution \
  --mode B1 B3 \
  --model gemini/gemini-3.1-pro-preview \
  --n-queries 5 \
  --results-per-query 20 \
  --max-candidates 100 \
  --llm-rerank-candidates 30 \
  --llm-score-batch-size 5 \
  --candidate-snippets tldr-or-abstract \
  --ranker deterministic llm_score \
  --budget 5 10 \
  --concurrency 3 \
  --s2-concurrency 2 \
  --fallback-repair-model gemini/gemini-3.1-pro-preview \
  --output results/dev_grounding.json
```

To evaluate grounding with predicted enabling contributions, pass judged generation output and include `B4` or `B4_MATCHED`:

```bash
python scripts/run_grounding.py \
  --gold-file ../data/dev.json \
  --gold-claim-field target_contribution \
  --predicted-results results/dev_generation_judged.json \
  --mode B1 B3 B4 B4_MATCHED \
  --model gemini/gemini-3.1-pro-preview \
  --ranker deterministic llm_score \
  --budget 5 \
  --output results/dev_grounding_with_predictions.json
```

## End-to-End

Run enabling contribution generation, judging, and grounding in one command:

```bash
python scripts/run_end_to_end.py \
  --gold-file ../data/dev.json \
  --output-dir results/end_to_end_dev \
  --generator-model gemini/gemini-3.1-pro-preview \
  --judge-model gemini/gemini-3.1-pro-preview \
  --grounding-model gemini/gemini-3.1-pro-preview
```

Use `--limit 5` for a smoke test.

## Outputs

Outputs are JSON files under `results/`. Grounding outputs include an aggregate `metrics` field and per-example retrieval details.
