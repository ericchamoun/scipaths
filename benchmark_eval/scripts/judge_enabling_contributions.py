#!/usr/bin/env python3
"""Judge cached enabling contribution prediction files.

Input:
  results/enabling_contribution_predictions/*.json produced by
  generate_enabling_contributions.py

Output:
  claim-level judged prediction JSON:
    {"summary": {...}, "papers": [...]}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from tqdm.asyncio import tqdm as atqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.judge import judge_ingredient_matching
from modules.loader import (
    DEFAULT_GOLD_FILE,
    get_gold_ingredients_for_claim,
    get_gold_paper_title,
    get_paper_title,
    get_silver_ingredients_for_claim,
)
from modules.metrics import (
    aggregate_task_a_results,
    compute_matching_precision,
    compute_matching_recall,
    compute_paper_f1,
    maximum_full_matching,
)

load_dotenv()
console = Console()


def _model_slug(model: str) -> str:
    return (
        model.replace("/", "-")
        .replace(":", "-")
        .replace("_", "-")
        .replace(".", "-")
    )


def _load_predictions(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary") or {}, data.get("predictions") or []


def _roles(ingredient: dict) -> str:
    annotation = ingredient.get("canonical_annotation") or {}
    if isinstance(annotation.get("roles"), list):
        return ", ".join(annotation["roles"])
    return annotation.get("role", "")


def _has_judge_error(paper: dict) -> bool:
    return any("error:" in j.get("reasoning", "") for j in paper.get("recall_judgments", []))


def _load_existing(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    papers = data.get("papers") or []
    return {
        (paper["paper_id"], paper.get("claim_idx", 0)): paper
        for paper in papers
        if paper.get("error") is None and paper.get("recall") is not None and not _has_judge_error(paper)
    }


async def _judge_one(
    semaphore: asyncio.Semaphore,
    row: dict,
    judge_model: str,
    source: str,
    gold_file: str | Path,
) -> dict | None:
    async with semaphore:
        paper_id = row["paper_id"]
        claim_idx = row.get("claim_idx", 0)
        claim = row.get("claim", "")

        if source == "gold":
            reference_ingredients = get_gold_ingredients_for_claim(paper_id, claim_idx, gold_file)
            title = get_gold_paper_title(paper_id, gold_file)
        else:
            reference_ingredients = get_silver_ingredients_for_claim(paper_id, claim_idx)
            title = get_paper_title(paper_id)

        if not reference_ingredients:
            return None

        predicted = row.get("predicted_ingredients") or []
        generation_error = row.get("error")

        if generation_error:
            return {
                "paper_id": paper_id,
                "claim_idx": claim_idx,
                "paper_title": title,
                "claim": claim,
                "source": source,
                "n_reference": len(reference_ingredients),
                "n_predicted": len(predicted),
                "has_related_work": row.get("has_related_work"),
                "related_work_chars": row.get("related_work_chars"),
                "generation_metadata": row.get("generation_metadata"),
                "predicted_ingredients": predicted,
                "recall_judgments": [],
                "recall": None,
                "precision": None,
                "f1": None,
                "error": generation_error,
            }

        try:
            matching_judgment = await judge_ingredient_matching(
                judge_model,
                claim,
                reference_ingredients,
                predicted,
            )
            candidate_matches = matching_judgment.get("matches", [])
            judge_error = None
        except Exception as exc:
            candidate_matches = []
            judge_error = str(exc)

        full_matches = [m for m in candidate_matches if m.get("match") == "full"]
        partial_matches = [m for m in candidate_matches if m.get("match") == "partial"]
        matched_pairs = maximum_full_matching(
            candidate_matches,
            len(reference_ingredients),
            len(predicted),
        )
        matched_by_ref = {m["reference_idx"]: m for m in matched_pairs}
        partial_by_ref: dict[int, list[dict]] = {}
        for match in partial_matches:
            partial_by_ref.setdefault(match["reference_idx"], []).append(match)

        recall_judgments = []
        for idx, reference in enumerate(reference_ingredients, 1):
            matched = matched_by_ref.get(idx)
            partials = partial_by_ref.get(idx, [])
            if matched:
                pred_idx = matched["predicted_idx"]
                reasoning = matched.get("reasoning", "")
                best_match = predicted[pred_idx - 1]
            else:
                pred_idx = None
                reasoning = f"error: {judge_error}" if judge_error else ""
                best_match = None
            recall_judgments.append({
                "reference_idx": idx,
                "reference_ingredient": reference.get("ingredient") or reference.get("enabling_contribution") or "",
                "reference_role": _roles(reference),
                "covered": matched is not None,
                "best_match_idx": pred_idx,
                "reasoning": reasoning,
                "best_match": best_match,
                "partial_matches": partials,
            })

        recall = compute_matching_recall(len(matched_pairs), len(reference_ingredients))
        precision = compute_matching_precision(len(matched_pairs), len(predicted))
        return {
            "paper_id": paper_id,
            "claim_idx": claim_idx,
            "paper_title": title,
            "claim": claim,
            "source": source,
            "n_reference": len(reference_ingredients),
            "n_predicted": len(predicted),
            "has_related_work": row.get("has_related_work"),
            "related_work_chars": row.get("related_work_chars"),
            "generation_metadata": row.get("generation_metadata"),
            "predicted_ingredients": predicted,
            "candidate_matches": candidate_matches,
            "full_candidate_matches": full_matches,
            "partial_matches": partial_matches,
            "matched_pairs": matched_pairs,
            "judge_error": judge_error,
            "recall_judgments": recall_judgments,
            "recall": recall,
            "precision": precision,
            "f1": compute_paper_f1(recall, precision),
        }


def _print_results_table(summary: dict, prediction_file: Path, judge_model: str, output_path: Path) -> None:
    console.rule("[bold]Enabling Contribution Judge Results")
    t = Table(show_header=True, header_style="bold cyan")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_row("Prediction file", str(prediction_file.name))
    t.add_row("Papers evaluated", str(summary["n_papers"]))
    t.add_row("Claims evaluated", str(summary["n_claims"]))
    t.add_row("Reference source", summary.get("source", "gold"))
    if summary.get("gold_claim_field"):
        t.add_row("Gold claim field", summary["gold_claim_field"])
    if summary.get("conference_year"):
        t.add_row("Conference year", str(summary["conference_year"]))
    t.add_row("Setting", str(summary["setting"]))
    t.add_row("Generator model", summary["generator_model"])
    t.add_row("Judge model", judge_model)
    t.add_row("Generator errors", str(summary["n_gen_errors"]))
    t.add_row("Judge errors", str(summary["n_judge_errors"]))
    t.add_row("Mean reference ingredients / claim", str(summary.get("mean_n_reference", summary["mean_n_silver"])))
    t.add_row("Mean predicted ingredients / claim", str(summary["mean_n_predicted"]))
    if summary["mean_recall"] is not None:
        t.add_row("Mean recall", f"{summary['mean_recall']:.3f}")
        t.add_row("Mean precision", f"{summary['mean_precision']:.3f}" if summary["mean_precision"] is not None else "n/a")
        t.add_row("Mean F1", f"{summary['mean_f1']:.3f}" if summary["mean_f1"] is not None else "n/a")
    console.print(t)
    console.print(f"\n[green]Results written to: {output_path}[/green]")


async def _run(args: argparse.Namespace) -> None:
    summary_in, rows = _load_predictions(args.predictions_file)
    source = args.source or summary_in.get("source", "gold")
    gold_file = args.gold_file or summary_in.get("gold_file") or str(DEFAULT_GOLD_FILE)
    gold_claim_field = args.gold_claim_field or summary_in.get("gold_claim_field")
    conference_year = args.conference_year if args.conference_year is not None else summary_in.get("conference_year")
    setting = args.setting if args.setting is not None else summary_in.get("setting", 1)
    generator_model = summary_in.get("generator_model", "unknown")

    output_path = args.output or (
        args.output_dir / f"{args.predictions_file.stem}__judge_{_model_slug(args.judge_model)}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _load_existing(output_path) if args.resume else {}
    pending = [
        row for row in rows
        if (row["paper_id"], row.get("claim_idx", 0)) not in existing
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    console.print(f"Predictions file: {args.predictions_file}")
    console.print(f"Judge model: {args.judge_model}")
    console.print(f"Rows total: {len(rows)}")
    if args.resume:
        console.print(f"Already complete: {len(existing)}")
        console.print(f"Pending: {len(pending)}")

    by_key = dict(existing)
    semaphore = asyncio.Semaphore(args.concurrency)
    coros = [
        _judge_one(semaphore, row, args.judge_model, source, gold_file)
        for row in pending
    ]
    for fut in atqdm.as_completed(coros, desc="Judging cached predictions"):
        result = await fut
        if result is None:
            continue
        by_key[(result["paper_id"], result.get("claim_idx", 0))] = result
        ordered = [
            by_key[(row["paper_id"], row.get("claim_idx", 0))]
            for row in rows
            if (row["paper_id"], row.get("claim_idx", 0)) in by_key
        ]
        summary = aggregate_task_a_results(ordered)
        summary["generator_model"] = generator_model
        summary["judge_model"] = args.judge_model
        summary["setting"] = setting
        summary["source"] = source
        if source == "gold":
            summary["gold_file"] = str(gold_file)
            summary["gold_claim_field"] = gold_claim_field
            summary["conference_year"] = conference_year
        summary["n_claims"] = len(ordered)
        summary["n_papers"] = len({paper["paper_id"] for paper in ordered})
        output_path.write_text(
            json.dumps({"summary": summary, "papers": ordered}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    final_data = json.loads(output_path.read_text(encoding="utf-8"))
    _print_results_table(final_data["summary"], args.predictions_file, args.judge_model, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge cached enabling contribution prediction files.")
    parser.add_argument("--predictions-file", type=Path, required=True, help="Cached predictions JSON from results/enabling_contribution_predictions.")
    parser.add_argument("--judge-model", default="gemini/gemini-3-flash-preview")
    parser.add_argument("--source", choices=["gold", "silver"], default=None, help="Override source from predictions summary.")
    parser.add_argument("--gold-file", default=None, help="Override gold file from predictions summary.")
    parser.add_argument(
        "--gold-claim-field",
        choices=["target_contribution", "rewritten_capability", "rewritten_claim"],
        default=None,
        help="Override gold claim field from predictions summary.",
    )
    parser.add_argument("--conference-year", type=int, choices=[2023, 2024, 2025], default=None)
    parser.add_argument("--setting", type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "enabling_contribution_judged")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
