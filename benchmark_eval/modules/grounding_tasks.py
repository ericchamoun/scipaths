import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from modules.loader import (
    DEFAULT_GOLD_FILE,
    get_gold_claim_tasks,
    get_gold_ingredients_for_claim,
    get_gold_paper_title,
    get_overlapping_ids,
    get_rewritten_claim,
    get_silver_ingredients,
    get_paper_title,
)

_S2_ID_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class IngredientTask:
    paper_id: str
    paper_title: str
    desc: str                           # ingredient / description text
    role: str
    claim: str                          # the paper's scientific claim (context for search)
    source: str                         # "silver" or "predicted"
    canonical_id: Optional[str]         # primary expected S2 paper_id (may be None)
    all_acceptable_ids: list[str]       # union of all valid S2 paper_ids
    title_leaked: bool                  # description shares words with canonical ref title
    extra: dict = field(default_factory=dict)  # source-specific metadata


def extract_groundings(ingredient: dict) -> tuple[Optional[str], list[str]]:
    """Returns (canonical_s2_id_or_None, all_acceptable_s2_ids) from an ingredient."""
    canonical = ingredient.get("canonical_grounding") or {}
    canonical_id = canonical.get("paper_id") or ""
    if not _S2_ID_RE.match(canonical_id):
        canonical_id = None

    all_ids: list[str] = [canonical_id] if canonical_id else []
    for g in ingredient.get("additional_groundings") or []:
        gid = g.get("paper_id") or ""
        if _S2_ID_RE.match(gid) and gid not in all_ids:
            all_ids.append(gid)

    return canonical_id, all_ids


def title_leaked(desc: str, ref_title: str) -> bool:
    """True if the description shares content words (>4 chars) with the ref title.

    Flags potential data leakage: Gemini (and Setting-2 generators) saw reference
    titles when writing descriptions, making those ingredients easier to search for.
    Stratify on this flag rather than ignoring it — see README for details.
    """
    desc_lower = desc.lower()
    content_words = [w for w in ref_title.lower().split() if len(w) > 4]
    return bool(content_words) and any(w in desc_lower for w in content_words)


def recall_at_k(result_ids: list[str], targets: set[str], k: int) -> bool:
    return any(rid in targets for rid in result_ids[:k])


def load_silver_tasks(paper_ids: list[str]) -> list[IngredientTask]:
    """Load silver ingredients from Gemini annotations as grounding tasks."""
    tasks: list[IngredientTask] = []
    for pid in paper_ids:
        pt = get_paper_title(pid)
        claim_text = get_rewritten_claim(pid) or ""
        for ing in get_silver_ingredients(pid):
            desc = ing.get("ingredient", "")
            role = (ing.get("canonical_annotation") or {}).get("role", "")
            canonical_id, all_ids = extract_groundings(ing)
            ref_title = (ing.get("canonical_grounding") or {}).get("ref_title") or ""
            tasks.append(IngredientTask(
                paper_id=pid,
                paper_title=pt,
                desc=desc,
                role=role,
                claim=claim_text,
                source="silver",
                canonical_id=canonical_id,
                all_acceptable_ids=all_ids,
                title_leaked=title_leaked(desc, ref_title),
                extra={"ref_title": ref_title},
            ))
    return tasks


def _role(ingredient: dict) -> str:
    annotation = ingredient.get("canonical_annotation") or {}
    roles = annotation.get("roles")
    if isinstance(roles, list):
        return ", ".join(str(r) for r in roles)
    return annotation.get("role", "")


def _ref_title(ingredient: dict) -> str:
    return (ingredient.get("canonical_grounding") or {}).get("ref_title") or ""


def _paper_metadata(gold_file: str | Path) -> dict[str, dict]:
    data = json.loads(Path(gold_file).read_text(encoding="utf-8"))
    if data and "target_contribution" in data[0] and "claims" not in data[0]:
        return {
            str(row.get("target_paper_id") or row.get("idx")): row
            for row in data
            if row.get("target_paper_id") or row.get("idx") is not None
        }
    return {
        paper.get("target_paper_id"): paper
        for paper in data
        if paper.get("target_paper_id")
    }


def _claim_keys_from_gold(gold_file: str | Path, limit: Optional[int] = None) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    data = json.loads(Path(gold_file).read_text(encoding="utf-8"))
    if data and "target_contribution" in data[0] and "claims" not in data[0]:
        for row in data[:limit]:
            pid = str(row.get("target_paper_id") or row.get("idx") or "")
            if not pid:
                continue
            claim_idx = int(row.get("claim_idx", row.get("idx", 0)) or 0)
            keys.add((pid, claim_idx))
        return keys

    paper_count = 0
    for paper in data:
        pid = paper.get("target_paper_id")
        if not pid:
            continue
        if limit and paper_count >= limit:
            break
        paper_count += 1
        for claim_idx, claim in enumerate(paper.get("claims") or []):
            keys.add((pid, claim_idx))
    return keys


def load_gold_tasks(
    gold_file: str | Path = DEFAULT_GOLD_FILE,
    limit: Optional[int] = None,
    claim_field: str = "target_contribution",
) -> list[IngredientTask]:
    """Load manually annotated gold ingredients as grounding tasks."""
    claim_tasks = get_gold_claim_tasks(gold_file, claim_field=claim_field)
    metadata_by_id = _paper_metadata(gold_file)
    if limit:
        paper_ids = []
        limited = []
        for task in claim_tasks:
            if task[0] not in paper_ids:
                if len(paper_ids) >= limit:
                    continue
                paper_ids.append(task[0])
            limited.append(task)
        claim_tasks = limited

    tasks: list[IngredientTask] = []
    for pid, claim_idx, claim_text in claim_tasks:
        pt = get_gold_paper_title(pid, gold_file)
        paper_meta = metadata_by_id.get(pid, {})
        for ing in get_gold_ingredients_for_claim(pid, claim_idx, gold_file):
            desc = ing.get("ingredient") or ing.get("enabling_contribution") or ""
            canonical_id, all_ids = extract_groundings(ing)
            ref_title = _ref_title(ing)
            tasks.append(IngredientTask(
                paper_id=pid,
                paper_title=pt,
                desc=desc,
                role=_role(ing),
                claim=claim_text,
                source="gold",
                canonical_id=canonical_id,
                all_acceptable_ids=all_ids,
                title_leaked=title_leaked(desc, ref_title),
                extra={
                    "claim_idx": claim_idx,
                    "ingredient_id": ing.get("ingredient_id") or ing.get("enabling_contribution_id"),
                    "ref_title": ref_title,
                    "conference_year": paper_meta.get("conference_year"),
                    "target_year": paper_meta.get("target_year"),
                },
            ))
    return tasks


def load_predicted_tasks(
    results_path: str,
    limit: Optional[int] = None,
    restrict_gold_file: str | Path | None = None,
) -> list[IngredientTask]:
    """Load predicted ingredients from an eval_ingredient_recall.py output file.

    Ground truth is inferred by linking matched predictions back to the reference
    ingredients' canonical_groundings:
      matched_pairs[j].reference_idx  →  reference ingredient in gold/silver data
        → ingredient dict  →  canonical_grounding.paper_id

    For Setting 2 results (generator saw reference titles), the leakage flag checks
    whether the predicted description contains words from the covered silver's ref_title.
    """
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    setting: int = data["summary"].get("setting", 1)
    source: str = data.get("summary", {}).get("source", "silver")
    gold_file = data.get("summary", {}).get("gold_file", str(DEFAULT_GOLD_FILE))
    claim_field = data.get("summary", {}).get("gold_claim_field", "rewritten_capability")
    if restrict_gold_file is not None:
        gold_file = str(restrict_gold_file)
    metadata_by_id = _paper_metadata(gold_file) if source == "gold" else {}
    allowed_keys = _claim_keys_from_gold(gold_file, limit=limit) if restrict_gold_file is not None else None

    tasks: list[IngredientTask] = []
    papers = data["papers"]
    if limit and allowed_keys is None:
        papers = papers[:limit]

    for paper in papers:
        pid = paper["paper_id"]
        claim_idx = paper.get("claim_idx", 0)
        if allowed_keys is not None and (pid, claim_idx) not in allowed_keys:
            continue
        predicted = paper.get("predicted_ingredients") or []
        judgments = paper.get("recall_judgments") or []
        if not predicted:
            continue

        pt = paper.get("paper_title") or get_paper_title(pid)
        paper_meta = metadata_by_id.get(pid, {})
        if source == "gold":
            reference_ings = get_gold_ingredients_for_claim(pid, claim_idx, gold_file)
            claim_text = paper.get("claim") or ""
        else:
            reference_ings = get_silver_ingredients(pid)
            claim_text = get_rewritten_claim(pid) or ""
        reference_by_idx = {i + 1: ing for i, ing in enumerate(reference_ings)}
        reference_by_text = {
            (s.get("ingredient") or s.get("enabling_contribution") or ""): s
            for s in reference_ings
        }

        # Build: 1-based predicted index → list of covered reference ingredient dicts.
        covered_by: dict[int, list[dict]] = {}
        matched_pairs = paper.get("matched_pairs") or []
        if matched_pairs:
            for match in matched_pairs:
                pred_idx = match.get("predicted_idx")
                ref_idx = match.get("reference_idx")
                reference = reference_by_idx.get(ref_idx)
                if isinstance(pred_idx, int) and reference:
                    covered_by.setdefault(pred_idx, []).append(reference)
        else:
            # Backward compatibility for old per-reference result files.
            for j in judgments:
                if j.get("covered") and isinstance(j.get("best_match_idx"), int):
                    reference_text = j.get("reference_ingredient") or j.get("silver_ingredient")
                    reference = reference_by_text.get(reference_text)
                    if reference:
                        covered_by.setdefault(j["best_match_idx"], []).append(reference)

        for i, pred in enumerate(predicted, 1):  # 1-based to match best_match_idx
            covered_references = covered_by.get(i, [])
            desc = pred.get("description", "")
            role = pred.get("role", "")

            all_ids: list[str] = []
            canonical_id: Optional[str] = None
            ref_titles: list[str] = []
            for reference in covered_references:
                _, s_ids = extract_groundings(reference)
                for sid in s_ids:
                    if sid not in all_ids:
                        all_ids.append(sid)
                if canonical_id is None and s_ids:
                    canonical_id = s_ids[0]
                rt = _ref_title(reference)
                if rt:
                    ref_titles.append(rt)

            # Leakage: Setting-2 generator received reference titles as input.
            leaked = setting == 2 and any(title_leaked(desc, rt) for rt in ref_titles)

            tasks.append(IngredientTask(
                paper_id=pid,
                paper_title=pt,
                desc=desc,
                role=role,
                claim=claim_text,
                source="predicted",
                canonical_id=canonical_id,
                all_acceptable_ids=all_ids,
                title_leaked=leaked,
                extra={
                    "setting": setting,
                    "task_a_source": source,
                    "ref_title_in_pred": pred.get("ref_title"),
                    "covered_reference_ingredients": [
                        s.get("ingredient") or s.get("enabling_contribution")
                        for s in covered_references
                    ],
                    "covered_reference_ref_titles": ref_titles,
                    "claim_idx": claim_idx,
                    "conference_year": paper_meta.get("conference_year"),
                    "target_year": paper_meta.get("target_year"),
                },
            ))

    return tasks
