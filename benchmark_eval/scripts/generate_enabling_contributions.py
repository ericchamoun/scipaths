#!/usr/bin/env python3
"""Generate enabling contribution predictions without judging.

This is a cache-building script: it runs the generator over each target
contribution and writes predicted enabling contributions. A separate scoring step can judge these cached
predictions later for any dev/test/year split.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.generator import generate_ingredients
from modules.loader import (
    get_gold_claim_tasks,
    get_gold_paper_title,
    get_overlapping_ids,
    get_paper_title,
    get_references,
    get_related_work,
    get_updated_claims,
)

load_dotenv()

DEFAULT_FINAL_GOLD_FILE = Path("../data/dev.json")


def _model_slug(model: str) -> str:
    return (
        model.replace("/", "-")
        .replace(":", "-")
        .replace("_", "-")
        .replace(".", "-")
    )


def _claim_field_slug(claim_field: str) -> str:
    return {
        "rewritten_capability": "capability",
        "rewritten_claim": "claim",
    }.get(claim_field, claim_field)


def _build_silver_tasks(limit: int | None) -> tuple[list[str], list[tuple[str, int, str]]]:
    paper_ids = get_overlapping_ids()
    if limit:
        paper_ids = paper_ids[:limit]
    tasks = [
        (paper_id, claim_idx, claim)
        for paper_id in paper_ids
        for claim_idx, claim in enumerate(get_updated_claims(paper_id))
    ]
    return paper_ids, tasks


def _build_gold_tasks(
    gold_file: str | Path,
    gold_claim_field: str,
    conference_year: int | None,
    limit: int | None,
) -> tuple[list[str], list[tuple[str, int, str]]]:
    tasks = get_gold_claim_tasks(
        gold_file,
        claim_field=gold_claim_field,
        conference_year=conference_year,
    )
    if limit:
        paper_ids = []
        limited_tasks = []
        for task in tasks:
            if task[0] not in paper_ids:
                if len(paper_ids) >= limit:
                    continue
                paper_ids.append(task[0])
            limited_tasks.append(task)
        return paper_ids, limited_tasks
    return sorted({paper_id for paper_id, _, _ in tasks}), tasks


def _load_existing(path: Path) -> dict[tuple[str, int], dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("predictions") or []

    def is_valid(row: dict) -> bool:
        if row.get("error") is not None:
            return False
        ingredients = row.get("predicted_ingredients") or []
        if not ingredients:
            return False
        for ingredient in ingredients:
            if not isinstance(ingredient, dict):
                return False
            values = [
                str(ingredient.get("description", "")).strip().lower(),
                str(ingredient.get("role", "")).strip().lower(),
                str(ingredient.get("rationale", "")).strip().lower(),
            ]
            if any(value in {"", "...", "…", "null", "none", "n/a"} for value in values):
                return False
        return True

    return {
        (row["paper_id"], row["claim_idx"]): row
        for row in rows
        if is_valid(row)
    }


def _ingredient_role(ingredient: dict) -> str:
    annotation = ingredient.get("canonical_annotation") or {}
    roles = annotation.get("roles")
    if isinstance(roles, list) and roles:
        return str(roles[0])
    if annotation.get("role"):
        return str(annotation["role"])
    if ingredient.get("role"):
        return str(ingredient["role"])
    return ""


def _ingredient_rationale(ingredient: dict) -> str:
    annotation = ingredient.get("canonical_annotation") or {}
    return str(
        annotation.get("rationale")
        or annotation.get("contribution")
        or ingredient.get("rationale")
        or ""
    )


def _format_few_shot_ingredients(ingredients: list[dict]) -> list[dict]:
    formatted = []
    for ingredient in ingredients:
        description = str(
            ingredient.get("ingredient")
            or ingredient.get("enabling_contribution")
            or ingredient.get("description")
            or ""
        ).strip()
        role = _ingredient_role(ingredient).strip()
        rationale = _ingredient_rationale(ingredient).strip()
        if description and role and rationale:
            formatted.append({
                "description": description,
                "role": role,
                "rationale": rationale,
            })
    return formatted


def _load_few_shot_examples(
    few_shot_file: str | Path | None,
    claim_field: str,
    n: int,
    ids: list[str] | None = None,
) -> list[dict]:
    if not few_shot_file or n <= 0:
        return []
    fallback_fields = [
        field
        for field in ["target_contribution", "rewritten_capability", "rewritten_claim"]
        if field != claim_field
    ]
    requested = _parse_few_shot_ids(ids or [])
    rows = []
    data = json.loads(Path(few_shot_file).read_text(encoding="utf-8"))
    if data and "target_contribution" in data[0] and "claims" not in data[0]:
        for row in data:
            paper_id = str(row.get("target_paper_id") or row.get("idx") or "")
            claim_idx = int(row.get("claim_idx", row.get("idx", 0)) or 0)
            if requested and (paper_id, claim_idx) not in requested:
                continue
            claim_text = row.get(claim_field) or next((row.get(field) for field in fallback_fields if row.get(field)), "")
            ingredients = _format_few_shot_ingredients(row.get("enabling_contributions") or [])
            if paper_id and claim_text and ingredients:
                rows.append({
                    "paper_id": paper_id,
                    "claim_idx": claim_idx,
                    "claim": claim_text,
                    "ingredients": ingredients,
                })
        return rows

    for paper in data:
        paper_id = paper.get("target_paper_id") or ""
        for claim_idx, claim in enumerate(paper.get("claims") or []):
            if requested and (paper_id, claim_idx) not in requested:
                continue
            claim_text = claim.get(claim_field) or next((claim.get(field) for field in fallback_fields if claim.get(field)), "")
            ingredients = _format_few_shot_ingredients(claim.get("ingredients") or [])
            if paper_id and claim_text and ingredients:
                rows.append({
                    "paper_id": paper_id,
                    "claim_idx": claim_idx,
                    "claim": claim_text,
                    "ingredients": ingredients,
                })
    return rows


def _parse_few_shot_ids(ids: list[str]) -> set[tuple[str, int]]:
    """Parse paper IDs or paper_id:claim_idx selectors.

    Bare paper IDs select claim_idx=0 because examples are claim-level.
    claim_idx is 0-based to match the JSON rows and output files.
    """
    parsed: set[tuple[str, int]] = set()
    for raw in ids:
        value = raw.strip()
        if not value:
            continue
        paper_id = value
        claim_idx = 0
        if ":" in value:
            maybe_paper_id, maybe_claim_idx = value.rsplit(":", 1)
            if maybe_claim_idx.isdigit():
                paper_id = maybe_paper_id
                claim_idx = int(maybe_claim_idx)
        parsed.add((paper_id, claim_idx))
    return parsed


def _few_shots_for_task(
    examples: list[dict],
    paper_id: str,
    claim_idx: int,
    n: int,
) -> list[dict]:
    if not examples or n <= 0:
        return []
    selected = [
        example
        for example in examples
        if not (
            example.get("paper_id") == paper_id
            and example.get("claim_idx") == claim_idx
        )
    ]
    return selected[:n]


async def _predict_one(
    semaphore: asyncio.Semaphore,
    paper_id: str,
    claim_idx: int,
    claim: str,
    generator_model: str,
    setting: int,
    source: str,
    gold_file: str | Path,
    few_shot_examples: list[dict] | None = None,
    few_shot_n: int = 0,
) -> dict:
    async with semaphore:
        title = (
            get_gold_paper_title(paper_id, gold_file)
            if source == "gold"
            else get_paper_title(paper_id)
        )
        references = None
        related_work = None
        if setting == 2:
            all_refs = get_references(paper_id)
            influential = [r for r in all_refs if r["is_influential"]]
            references = influential if len(influential) >= 5 else all_refs[:30]
        elif setting == 3:
            related_work = get_related_work(paper_id)

        task_few_shots = _few_shots_for_task(
            few_shot_examples or [],
            paper_id,
            claim_idx,
            few_shot_n,
        )
        try:
            predicted, generation_metadata = await generate_ingredients(
                generator_model,
                claim,
                references=references,
                related_work=related_work,
                few_shot_examples=task_few_shots,
                repair_model=None,
            )
            error = None
        except Exception as exc:
            predicted = []
            generation_metadata = None
            error = str(exc)

        return {
            "paper_id": paper_id,
            "claim_idx": claim_idx,
            "paper_title": title,
            "claim": claim,
            "source": source,
            "setting": setting,
            "generator_model": generator_model,
            "has_related_work": bool(related_work) if setting == 3 else None,
            "related_work_chars": len(related_work or "") if setting == 3 else None,
            "few_shot_n": len(task_few_shots) if few_shot_examples else 0,
            "n_predicted": len(predicted),
            "predicted_ingredients": predicted,
            "generation_metadata": generation_metadata,
            "error": error,
        }


async def _run(args: argparse.Namespace) -> None:
    output = Path(args.output) if args.output else _default_output(args)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "gold":
        paper_ids, tasks = _build_gold_tasks(
            args.gold_file,
            args.gold_claim_field,
            args.conference_year,
            args.limit,
        )
    else:
        paper_ids, tasks = _build_silver_tasks(args.limit)

    if args.claim_limit is not None:
        tasks = tasks[:args.claim_limit]
        paper_ids = sorted({paper_id for paper_id, _, _ in tasks})

    few_shot_examples = _load_few_shot_examples(
        args.few_shot_file,
        args.few_shot_claim_field or args.gold_claim_field,
        args.few_shot_n,
        args.few_shot_ids,
    )

    existing = _load_existing(output) if args.resume else {}
    pending = [task for task in tasks if (task[0], task[1]) not in existing]

    print(f"Output: {output}")
    print(f"Generator: {args.generator_model}")
    print(f"Setting: {args.setting}")
    if args.few_shot_n:
        print(f"Few-shot examples: {len(few_shot_examples)} from {args.few_shot_file}")
        if args.few_shot_ids:
            print(f"Few-shot IDs: {', '.join(args.few_shot_ids)}")
    print(f"Claims total: {len(tasks)}")
    if args.resume:
        print(f"Already complete: {len(existing)}")
        print(f"Pending: {len(pending)}")

    rows_by_key = dict(existing)

    def write_checkpoint() -> dict:
        rows = [
            rows_by_key[(paper_id, claim_idx)]
            for paper_id, claim_idx, _ in tasks
            if (paper_id, claim_idx) in rows_by_key
        ]
        summary = {
            "timestamp": datetime.now().isoformat(),
            "source": args.source,
            "gold_file": str(args.gold_file) if args.source == "gold" else None,
            "gold_claim_field": args.gold_claim_field if args.source == "gold" else None,
            "conference_year": args.conference_year if args.source == "gold" else None,
            "setting": args.setting,
            "generator_model": args.generator_model,
            "few_shot_file": str(args.few_shot_file) if args.few_shot_file else None,
            "few_shot_n": args.few_shot_n,
            "few_shot_ids": args.few_shot_ids,
            "few_shot_claim_field": args.few_shot_claim_field or args.gold_claim_field,
            "n_papers": len(paper_ids),
            "n_claims": len(rows),
            "n_total_claims": len(tasks),
            "n_errors": sum(1 for row in rows if row.get("error")),
            "mean_n_predicted": (
                round(sum(row["n_predicted"] for row in rows) / len(rows), 4)
                if rows else None
            ),
        }
        output.write_text(
            json.dumps({"summary": summary, "predictions": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary

    if pending:
        semaphore = asyncio.Semaphore(args.concurrency)
        coros = [
            _predict_one(
                semaphore,
                paper_id,
                claim_idx,
                claim,
                args.generator_model,
                args.setting,
                args.source,
                args.gold_file,
                few_shot_examples,
                args.few_shot_n,
            )
            for paper_id, claim_idx, claim in pending
        ]
        for fut in atqdm.as_completed(coros, desc="Generating predictions"):
            row = await fut
            rows_by_key[(row["paper_id"], row["claim_idx"])] = row
            write_checkpoint()

    summary = write_checkpoint()
    print(json.dumps(summary, indent=2))


def _default_output(args: argparse.Namespace) -> Path:
    out_dir = Path("results") / "enabling_contribution_predictions"
    field_part = (
        f"_{_claim_field_slug(args.gold_claim_field)}"
        if args.source == "gold"
        else ""
    )
    year_part = (
        f"_y{args.conference_year}"
        if args.source == "gold" and args.conference_year
        else ""
    )
    few_shot_part = f"_fs{args.few_shot_n}" if getattr(args, "few_shot_n", 0) else ""
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    name = (
        f"predictions_{args.source}{field_part}{year_part}_s{args.setting}{few_shot_part}_"
        f"{_model_slug(args.generator_model)}_{ts}.json"
    )
    return out_dir / name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enabling contribution predictions without judge calls.")
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--source", choices=["gold", "silver"], default="gold")
    parser.add_argument("--gold-file", default=str(DEFAULT_FINAL_GOLD_FILE))
    parser.add_argument(
        "--gold-claim-field",
        choices=["target_contribution", "rewritten_capability", "rewritten_claim"],
        default="target_contribution",
    )
    parser.add_argument("--conference-year", type=int, choices=[2023, 2024, 2025], default=None)
    parser.add_argument("--setting", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N papers.")
    parser.add_argument("--claim-limit", type=int, default=None, help="Limit to first N claim examples.")
    parser.add_argument(
        "--few-shot-file",
        default=None,
        help="Gold annotation file to draw fixed few-shot examples from.",
    )
    parser.add_argument("--few-shot-n", type=int, default=0, help="Number of examples to prepend.")
    parser.add_argument(
        "--few-shot-ids",
        nargs="*",
        default=None,
        help=(
            "Optional explicit few-shot selectors. Use paper_id for claim_idx=0, "
            "or paper_id:claim_idx for a specific 0-based claim."
        ),
    )
    parser.add_argument(
        "--few-shot-claim-field",
        choices=["target_contribution", "rewritten_capability", "rewritten_claim"],
        default=None,
        help="Claim text field to use in few-shot examples. Defaults to --gold-claim-field.",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse successful rows already in --output.")
    return parser.parse_args()


def main() -> int:
    asyncio.run(_run(_parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
