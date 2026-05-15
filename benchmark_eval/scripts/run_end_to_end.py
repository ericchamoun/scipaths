#!/usr/bin/env python3
"""Run enabling contribution generation, judging, and grounding in sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require_generation_success(path: Path) -> None:
    summary = load_json(path).get("summary", {})
    if summary.get("n_errors", 0):
        raise SystemExit(f"Generation failed for {summary['n_errors']} claim(s). See {path}.")


def require_judging_success(path: Path) -> None:
    summary = load_json(path).get("summary", {})
    n_gen_errors = summary.get("n_gen_errors", 0)
    n_judge_errors = summary.get("n_judge_errors", 0)
    if n_gen_errors or n_judge_errors:
        raise SystemExit(
            f"Judging output contains errors: generation={n_gen_errors}, judging={n_judge_errors}. See {path}."
        )


def require_grounding_success(path: Path) -> None:
    data = load_json(path)
    failures = []
    for block in data.get("results") or []:
        mode = block.get("mode_name") or block.get("mode")
        for claim in block.get("claims") or []:
            claim_id = f"{claim.get('paper_id')}:{claim.get('claim_idx')}"
            if claim.get("error"):
                failures.append(f"{mode} {claim_id}: {claim['error']}")
            if claim.get("query_error"):
                failures.append(f"{mode} {claim_id}: query_error={claim['query_error']}")
            search_errors = [err for err in claim.get("search_errors") or [] if err]
            if search_errors:
                failures.append(f"{mode} {claim_id}: search_errors={search_errors[:2]}")
    if failures:
        preview = "\n".join(failures[:5])
        raise SystemExit(f"Grounding output contains errors. See {path}.\n{preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SciPaths enabling contribution generation and grounding end-to-end."
    )
    parser.add_argument("--gold-file", default="../data/dev.json")
    parser.add_argument("--output-dir", default="results/end_to_end_dev")
    parser.add_argument("--generator-model", default="gemini/gemini-3.1-pro-preview")
    parser.add_argument("--judge-model", default="gemini/gemini-3.1-pro-preview")
    parser.add_argument("--grounding-model", default="gemini/gemini-3.1-pro-preview")
    parser.add_argument("--fallback-repair-model", default="gemini/gemini-3.1-pro-preview")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--s2-concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    predictions = output_dir / "generation_predictions.json"
    judged = output_dir / "generation_judged.json"
    grounding = output_dir / "grounding.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    limit_args = ["--limit", str(args.limit)] if args.limit else []

    run([
        sys.executable,
        "scripts/generate_enabling_contributions.py",
        "--source", "gold",
        "--gold-file", args.gold_file,
        "--gold-claim-field", "target_contribution",
        "--setting", "1",
        "--generator-model", args.generator_model,
        "--concurrency", str(args.concurrency),
        "--output", str(predictions),
        "--resume",
        *limit_args,
    ])
    require_generation_success(predictions)

    run([
        sys.executable,
        "scripts/judge_enabling_contributions.py",
        "--predictions-file", str(predictions),
        "--source", "gold",
        "--gold-file", args.gold_file,
        "--gold-claim-field", "target_contribution",
        "--judge-model", args.judge_model,
        "--concurrency", str(args.concurrency),
        "--output", str(judged),
        "--resume",
        *limit_args,
    ])
    require_judging_success(judged)

    run([
        sys.executable,
        "scripts/run_grounding.py",
        "--gold-file", args.gold_file,
        "--gold-claim-field", "target_contribution",
        "--predicted-results", str(judged),
        "--mode", "B1", "B3", "B4", "B4_MATCHED",
        "--model", args.grounding_model,
        "--n-queries", "5",
        "--results-per-query", "20",
        "--max-candidates", "100",
        "--llm-rerank-candidates", "30",
        "--llm-score-batch-size", "5",
        "--candidate-snippets", "tldr-or-abstract",
        "--ranker", "deterministic", "llm_score",
        "--budget", "5", "10",
        "--concurrency", str(args.concurrency),
        "--s2-concurrency", str(args.s2_concurrency),
        "--fallback-repair-model", args.fallback_repair_model,
        "--output", str(grounding),
        *limit_args,
    ])
    require_grounding_success(grounding)


if __name__ == "__main__":
    main()
