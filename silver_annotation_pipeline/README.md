# Silver Annotation Pipeline

This directory runs the automatic SciPaths annotation pipeline for arXiv papers.

It starts from a target paper, finds downstream citation usage, derives target contributions from those usage clusters, and decomposes each target contribution into enabling contributions and groundings.

Outputs are saved locally under `runs/`.

## Setup

```bash
cd silver_annotation_pipeline
pip install -r requirements.txt
cp .env.example .env
```

Set:

```text
GEMINI_API_KEY=<Gemini API key>
SEMANTIC_SCHOLAR_API_KEY=<optional Semantic Scholar API key>
```

## Run One Paper End-to-End

```bash
python scripts/run_end_to_end.py \
  --paper https://arxiv.org/abs/2505.17978 \
  --output-root runs
```

Each run creates a directory like:

```text
runs/run_YYYYMMDD_HHMMSS_xxxxxxxx/
```

The run directory contains:

- `processed_papers/`: intermediate files for the target paper.
- `logs/`: stdout/stderr log for each step.
- `two_pass_outputs/`: final target-contribution annotations.
- `summary.txt`: run status.

If a paper has no usable citations, no verified USES/EXTENDS contexts, no extractable paragraphs, or no surviving usage clusters, the pipeline stops cleanly and leaves the partial run in `runs/`.

## Run a Batch Step by Step

Create a JSON file with arXiv IDs:

```json
[
  {"id": "2505.17978", "title": "", "id_type": "ArXiv"}
]
```

Then run the steps:

```bash
mkdir -p runs/batch_example

python scripts/run_step.py --step 1 --ids examples/input_ids.json --run-dir runs/batch_example
python scripts/run_step.py --step 2 --run-dir runs/batch_example
python scripts/run_step.py --step 3 --run-dir runs/batch_example
python scripts/run_step.py --step 4 --run-dir runs/batch_example
python scripts/run_step.py --step 5 --run-dir runs/batch_example
python scripts/run_step.py --step 6 --run-dir runs/batch_example
python scripts/run_step.py --step 7 --run-dir runs/batch_example
python scripts/run_step.py --step 8 --run-dir runs/batch_example
```

## Steps

| Step | Name | Output |
| ---: | --- | --- |
| 1 | Fetch metadata + LaTeX for input papers | Paper metadata, citation metadata, merged LaTeX |
| 2 | Add citation markers | LaTeX with normalized citation markers |
| 3 | Build usage contexts | Citation-context snippets from the target paper |
| 4 | Label citation functions | Citation-function labels from the classifier |
| 5 | Verify USES/EXTENDS | LLM-verified downstream usage contexts |
| 6 | Extract arXiv paragraphs | Paragraphs from downstream citing papers |
| 7 | Extract target contributions and refine clusters | Refined downstream usage clusters |
| 8 | Annotate target contributions and enabling contributions | Final silver annotations with groundings |

## Notes

Step 4 uses the bundled Deep-Citation classifier checkpoint. The checkpoint is tracked with Git LFS.

Step 5, step 7, and step 8 use LLM calls. The default provider is Gemini.

