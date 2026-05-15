#!/usr/bin/env python3
"""Claim-level grounding retrieval under equal final budget.

All modes answer the same final question:

    target contribution -> ranked list of at most K prior papers

Only the input evidence differs:
  B1: target contribution only
  B3: target contribution + gold enabling contributions
  B4: target contribution + predicted enabling contributions
  B4_MATCHED: target contribution + predicted enabling contributions that matched gold

The internal pipeline is fixed across modes:
  evidence -> query generation -> Semantic Scholar retrieval -> candidate rerank
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

import litellm
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.grounding_tasks import extract_groundings
from modules.loader import (
    DEFAULT_GOLD_FILE,
    get_gold_claim_tasks,
    get_gold_ingredients_for_claim,
    get_gold_paper_title,
    get_semantic_scholar_paper_id,
)
from modules.semantic_scholar import s2_search

load_dotenv()


MODE_NAMES = {
    "B1": "Claim only",
    "B3": "Gold enabling contributions",
    "B4": "Predicted enabling contributions",
    "B4_MATCHED": "Matched predicted enabling contributions",
}


COMMON_CRITERIA = """\
You are identifying prior papers that enabled a target scientific discovery.

A prior paper is relevant only if it realizes a concrete enabling contribution, such as a dataset, benchmark, model, tool, metric, protocol, method, or conceptual framework.

Do not include generic background, comparison-only baselines, papers merely about the same topic, or the target paper itself.
"""


QUERY_PROMPT = """\
{criteria}

Generate focused Semantic Scholar search queries for the target contribution.

{evidence}

Write {n_queries} diverse search queries that can retrieve prior papers necessary for realizing the discovery pathway.

Rules:
- Use only the evidence provided for this condition.
- Prefer distinctive dataset, benchmark, model, tool, metric, method, or framework names when inferable.
- Use broad reformulations for likely enabling methods/components.
- Do not include explanations.

Respond with JSON only:
{{"queries": ["...", "..."]}}
"""


SCORE_RERANK_PROMPT = """\
{criteria}

You are scoring candidate prior papers for a target scientific discovery.

For each candidate, rate whether it realizes an enabling contribution for the target contribution.

Scoring rubric:
5 = Exact grounding: directly realizes an enabling contribution in the same functional role.
4 = Strong grounding: clearly provides the required capability, but differs slightly in scope/version/form.
3 = Partial grounding: provides an important subcomponent or close predecessor, but does not fully realize the contribution.
2 = Related support: relevant background, motivation, baseline, or comparison, but not a functional grounding.
1 = Topically related only.
0 = Irrelevant or wrong paper.

This is a scoring task, not a selection task. Score every candidate listed below.

{coverage_rule}

{evidence}

Candidate prior papers retrieved from Semantic Scholar:
{candidates}

Rules:
- Return one score object for every candidate paper ID above.
- Use integer scores only: 0, 1, 2, 3, 4, or 5.
- For B3/B4-style inputs with explicit enabling contributions, grounds should list 1-based ingredient indices that the paper realizes. Use [] if none.
- For claim-only inputs, grounds may be [].

Respond with JSON only:
{{
  "scores": [
    {{"paper_id": "<candidate paper_id>", "score": 5, "grounds": [1], "reason": "<short reason>"}}
  ]
}}
"""


REPAIR_PROMPT = """\
Convert the following model output into valid JSON matching exactly this schema:
{schema}

Rules:
- Preserve only fields already present in the model output.
- Do not invent new paper IDs, scores, notes, reasons, or scientific content.
- If no valid items are recoverable, return an empty list for the schema's main array.
- Respond with JSON only.

Model output:
{raw}
"""


@dataclass
class ClaimTask:
    paper_id: str
    paper_title: str
    claim_idx: int
    claim: str
    conference_year: int | None
    target_ids: list[str]
    target_titles: dict[str, str]
    target_ingredient_groups: list[list[str]]
    gold_ingredients: list[dict]
    predicted_ingredients: list[dict]


def _strip_latex(text: str) -> str:
    text = re.sub(r"%.*", " ", text)
    text = re.sub(r"\\cite[t|p|alp|author|year]*\*?(?:\[[^\]]*\])*\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    text = re.sub(r"[{}$]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _score_items(data) -> list[dict]:
    if isinstance(data, dict):
        items = data.get("scores") or []
    elif isinstance(data, list):
        items = data
    else:
        return []
    return [item for item in items if isinstance(item, dict)]


def _is_transient_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = [
        "503",
        "serviceunavailable",
        "service unavailable",
        "unavailable",
        "temporarily",
        "timeout",
        "timed out",
        "rate limit",
        "ratelimit",
        "429",
        "500",
        "502",
        "504",
    ]
    return any(marker in text for marker in transient_markers)


async def _acompletion_with_retry(
    *,
    model: str,
    max_tokens: int,
    messages: list[dict],
    retries: int = 5,
    base_sleep: float = 2.0,
    max_sleep: float = 60.0,
) -> object:
    for attempt in range(retries + 1):
        try:
            return await litellm.acompletion(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
        except Exception as exc:
            if attempt >= retries or not _is_transient_llm_error(exc):
                raise
            sleep_s = min(max_sleep, base_sleep * (2 ** attempt))
            sleep_s += random.uniform(0, min(1.0, sleep_s * 0.1))
            await asyncio.sleep(sleep_s)


async def _complete_json(
    model: str,
    prompt: str,
    max_tokens: int,
    schema: str,
    fallback_repair_model: str | None = None,
) -> tuple[dict, str, str | None, bool]:
    """Call an LLM and parse JSON, with repair passes on malformed output."""
    msg = await _acompletion_with_retry(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.choices[0].message.content or ""
    try:
        return _parse_json(raw), raw, None, False
    except Exception as parse_exc:
        repair_msg = await _acompletion_with_retry(
            model=model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": REPAIR_PROMPT.format(schema=schema, raw=raw),
            }],
        )
        repaired = repair_msg.choices[0].message.content or ""
        try:
            return _parse_json(repaired), raw, None, True
        except Exception as repair_exc:
            if fallback_repair_model and fallback_repair_model != model:
                strong_repair_msg = await _acompletion_with_retry(
                    model=fallback_repair_model,
                    max_tokens=max_tokens,
                    messages=[{
                        "role": "user",
                        "content": REPAIR_PROMPT.format(schema=schema, raw=raw),
                    }],
                )
                strong_repaired = strong_repair_msg.choices[0].message.content or ""
                try:
                    return _parse_json(strong_repaired), raw, None, True
                except Exception as strong_repair_exc:
                    return (
                        {},
                        raw,
                        (
                            f"{parse_exc}; repair_error: {repair_exc}; "
                            f"fallback_repair_model={fallback_repair_model}; "
                            f"fallback_repair_error: {strong_repair_exc}"
                        ),
                        True,
                    )
            return {}, raw, f"{parse_exc}; repair_error: {repair_exc}", True


def _roles(ingredient: dict) -> str:
    annotation = ingredient.get("canonical_annotation") or {}
    roles = annotation.get("roles")
    if isinstance(roles, list):
        return ", ".join(str(r) for r in roles)
    return annotation.get("role") or ingredient.get("role") or ""


def _target_title(ingredient: dict, paper_id: str) -> str:
    canonical = ingredient.get("canonical_grounding") or {}
    if canonical.get("paper_id") == paper_id:
        return canonical.get("ref_title") or ""
    for grounding in ingredient.get("additional_groundings") or []:
        if grounding.get("paper_id") == paper_id:
            return grounding.get("ref_title") or ""
    return ""


def _matched_prediction_indices(row: dict) -> set[int]:
    """Return 0-based predicted ingredient indices with full gold matches."""
    matched = set()
    for pair in row.get("matched_pairs") or []:
        if str(pair.get("match", "")).lower() != "full":
            continue
        try:
            matched.add(int(pair["predicted_idx"]) - 1)
        except Exception:
            pass
    if matched:
        return matched

    for judgment in row.get("recall_judgments") or []:
        if not judgment.get("covered"):
            continue
        try:
            matched.add(int(judgment["best_match_idx"]) - 1)
        except Exception:
            pass
    return matched


def _load_predicted_by_key(
    path: str | Path | None,
    *,
    matched_only: bool = False,
) -> dict[tuple[str, int], list[dict]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("papers") or data.get("predictions") or []
    out: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        key = (row.get("paper_id"), row.get("claim_idx", 0))
        ingredients = row.get("predicted_ingredients") or []
        if matched_only:
            matched = _matched_prediction_indices(row)
            ingredients = [
                ingredient
                for idx, ingredient in enumerate(ingredients)
                if idx in matched
            ]
        out[key] = ingredients
    return out


def _result_slug(path: str | Path) -> str:
    stem = Path(path).stem
    for prefix in ("predictions_final_capability_s1_", "predictions_final_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    if "__judge_" in stem:
        stem = stem.split("__judge_", 1)[0]
    return (
        stem.replace("/", "-")
        .replace(":", "-")
        .replace("_", "-")
        .replace(".", "-")
    )


def _load_claim_tasks(
    gold_file: str | Path,
    claim_field: str,
    predicted_file: str | Path | None,
    matched_predicted_only: bool,
    conference_year: int | None,
    limit: int | None,
) -> list[ClaimTask]:
    predicted_by_key = _load_predicted_by_key(
        predicted_file,
        matched_only=matched_predicted_only,
    )
    claim_tasks = get_gold_claim_tasks(
        gold_file,
        claim_field=claim_field,
        conference_year=conference_year,
    )
    metadata = {
        paper.get("target_paper_id"): paper
        for paper in json.loads(Path(gold_file).read_text(encoding="utf-8"))
    }
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

    tasks = []
    for paper_id, claim_idx, claim in claim_tasks:
        ingredients = get_gold_ingredients_for_claim(paper_id, claim_idx, gold_file)
        target_ids = []
        target_titles = {}
        target_ingredient_groups = []
        for ingredient in ingredients:
            _, ids = extract_groundings(ingredient)
            if ids:
                target_ingredient_groups.append(ids)
            target_ids.extend(ids)
            for sid in ids:
                target_titles[sid] = _target_title(ingredient, sid)
        unique_ids = []
        seen = set()
        for sid in target_ids:
            if sid and sid not in seen:
                seen.add(sid)
                unique_ids.append(sid)
        paper_meta = metadata.get(paper_id, {})
        tasks.append(ClaimTask(
            paper_id=paper_id,
            paper_title=get_gold_paper_title(paper_id, gold_file),
            claim_idx=claim_idx,
            claim=claim,
            conference_year=paper_meta.get("conference_year"),
            target_ids=unique_ids,
            target_titles=target_titles,
            target_ingredient_groups=target_ingredient_groups,
            gold_ingredients=ingredients,
            predicted_ingredients=predicted_by_key.get((paper_id, claim_idx), []),
        ))
    return tasks


def _format_ingredients(ingredients: list[dict], *, gold: bool) -> str:
    lines = []
    for i, ingredient in enumerate(ingredients, 1):
        if gold:
            desc = ingredient.get("ingredient") or ingredient.get("enabling_contribution") or ""
            role = _roles(ingredient)
            rationale = (ingredient.get("canonical_annotation") or {}).get("rationale", "")
        else:
            desc = ingredient.get("description") or ingredient.get("ingredient") or ""
            role = ingredient.get("role") or ""
            rationale = ingredient.get("rationale") or ""
        line = f"{i}. {desc}"
        if role:
            line += f"\n   Role: {role}"
        if rationale:
            line += f"\n   Rationale: {rationale}"
        lines.append(line)
    return "\n".join(lines) if lines else "(No enabling contributions provided.)"


def _evidence_for_mode(mode: str, task: ClaimTask) -> str:
    if mode == "B1":
        return (
            f"Target discovery claim:\n{task.claim}\n\n"
            "Available evidence:\nOnly the target claim is available. Infer likely enabling contributions from the claim."
        )
    if mode == "B3":
        return (
            f"Target discovery claim:\n{task.claim}\n\n"
            "Known enabling contributions:\n"
            f"{_format_ingredients(task.gold_ingredients, gold=True)}\n\n"
            "Available evidence:\nUse these human-annotated enabling contributions to identify prior papers."
        )
    if mode in {"B4", "B4_MATCHED"}:
        label = (
            "Matched predicted enabling contributions"
            if mode == "B4_MATCHED"
            else "Predicted enabling contributions"
        )
        note = (
            "Use these model-predicted enabling contributions that were judged to match human annotations to identify prior papers."
            if mode == "B4_MATCHED"
            else "Use these model-predicted enabling contributions to identify prior papers. Some predictions may be noisy or unmapped."
        )
        return (
            f"Target discovery claim:\n{task.claim}\n\n"
            f"{label}:\n"
            f"{_format_ingredients(task.predicted_ingredients, gold=False)}\n\n"
            f"Available evidence:\n{note}"
        )
    raise ValueError(f"Unknown mode: {mode}")


def _coverage_rule(mode: str) -> str:
    if mode in {"B3", "B4", "B4_MATCHED"}:
        return (
            "Select papers that collectively cover as many enabling contributions as possible. "
            "Avoid selecting multiple papers for the same contribution unless they realize distinct components."
        )
    return (
        "Infer likely enabling contributions and select papers that collectively cover the discovery pathway."
    )


async def _generate_queries(
    model: str,
    mode: str,
    task: ClaimTask,
    n_queries: int,
    llm_sem: asyncio.Semaphore,
    fallback_repair_model: str | None,
) -> tuple[list[str], str | None]:
    prompt = QUERY_PROMPT.format(
        criteria=COMMON_CRITERIA,
        evidence=_evidence_for_mode(mode, task),
        n_queries=n_queries,
    )
    async with llm_sem:
        try:
            data, _, error, repaired = await _complete_json(
                model,
                prompt,
                1024,
                '{"queries": ["...", "..."]}',
                fallback_repair_model=fallback_repair_model,
            )
            if error:
                return _fallback_queries(mode, task, n_queries), error
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
            if queries:
                return queries[:n_queries], None
        except Exception as exc:
            return _fallback_queries(mode, task, n_queries), str(exc)
    return _fallback_queries(mode, task, n_queries), "empty query generation"


def _fallback_queries(mode: str, task: ClaimTask, n_queries: int) -> list[str]:
    claim = " ".join(task.claim.split()[:18])
    if mode == "B3" and task.gold_ingredients:
        query_basis = [
            ingredient.get("ingredient") or ingredient.get("enabling_contribution") or ""
            for ingredient in task.gold_ingredients[:n_queries]
        ]
    elif mode in {"B4", "B4_MATCHED"} and task.predicted_ingredients:
        query_basis = [
            ingredient.get("description") or ingredient.get("ingredient") or ""
            for ingredient in task.predicted_ingredients[:n_queries]
        ]
    else:
        query_basis = [claim]
    queries = [q for q in query_basis if q]
    queries.append(claim)
    return queries[:n_queries]


def _snippet_for_candidate(paper: dict, mode: str, max_chars: int = 300) -> str:
    if mode == "none":
        return ""
    tldr = (paper.get("tldr") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    if mode == "tldr":
        text = tldr
    elif mode == "abstract":
        text = abstract
    elif mode == "tldr-or-abstract":
        text = tldr or abstract
    else:
        text = ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + "..."
    return text


def _dedupe_candidates(
    search_results: list[list[dict]],
    queries: list[str],
    max_candidates: int,
    candidate_snippets: str,
) -> list[dict]:
    by_id: dict[str, dict] = {}
    for search_idx, papers in enumerate(search_results, 1):
        for search_rank, paper in enumerate(papers, 1):
            paper_id = paper.get("paperId")
            if not paper_id:
                continue
            query = queries[search_idx - 1] if search_idx - 1 < len(queries) else ""
            if paper_id not in by_id:
                by_id[paper_id] = {
                    "paper_id": paper_id,
                    "paper_title": paper.get("title"),
                    "year": paper.get("year"),
                    "authors": paper.get("authors") or [],
                    "best_rank": search_rank,
                    "frequency": 0,
                    "retrieved_by": [],
                    "snippet": _snippet_for_candidate(paper, candidate_snippets),
                }
            candidate = by_id[paper_id]
            candidate["frequency"] += 1
            if search_rank < candidate["best_rank"]:
                candidate["best_rank"] = search_rank
            candidate["retrieved_by"].append({
                "query_idx": search_idx,
                "query": query,
                "rank": search_rank,
            })
    candidates = []
    for candidate in by_id.values():
        best_rank = max(1, int(candidate["best_rank"]))
        frequency = int(candidate["frequency"])
        candidate["deterministic_score"] = round(frequency + 1.0 / best_rank, 6)
        candidates.append(candidate)
    candidates.sort(key=lambda c: (-c["deterministic_score"], c["best_rank"], c.get("paper_title") or ""))
    return candidates[:max_candidates]


def _candidate_for_llm(candidate: dict) -> dict:
    out = {
        "paper_id": candidate["paper_id"],
        "title": candidate.get("paper_title"),
        "year": candidate.get("year"),
        "deterministic_score": candidate.get("deterministic_score"),
        "best_rank": candidate.get("best_rank"),
        "frequency": candidate.get("frequency"),
    }
    if candidate.get("snippet"):
        out["snippet"] = candidate["snippet"]
    return out


def _deterministic_rank(candidates: list[dict]) -> list[dict]:
    ranked = []
    for idx, candidate in enumerate(candidates, 1):
        ranked.append({
            "paper_id": candidate["paper_id"],
            "paper_title": candidate.get("paper_title"),
            "rank": idx,
            "best_rank": candidate.get("best_rank"),
            "frequency": candidate.get("frequency"),
            "deterministic_score": candidate.get("deterministic_score"),
            "ranker": "deterministic",
        })
    return ranked


async def _score_rerank_candidates(
    model: str,
    mode: str,
    task: ClaimTask,
    candidates: list[dict],
    rerank_candidates: int,
    score_batch_size: int,
    verbose: bool,
    llm_sem: asyncio.Semaphore,
    fallback_repair_model: str | None,
) -> tuple[list[dict], list[dict], str | None, int]:
    if not candidates:
        return [], [], None, 0
    allowed = {c["paper_id"]: c for c in candidates}
    to_score = candidates[:rerank_candidates]
    scored = {}
    score_rows = []
    errors = []

    for start in range(0, len(to_score), score_batch_size):
        batch = to_score[start:start + score_batch_size]
        if verbose:
            print(
                f"[score] {task.paper_id}:{task.claim_idx} {mode} "
                f"batch {start // score_batch_size + 1} "
                f"candidates {start + 1}-{start + len(batch)} / {len(to_score)}",
                flush=True,
            )
        prompt = SCORE_RERANK_PROMPT.format(
            criteria=COMMON_CRITERIA,
            coverage_rule=_coverage_rule(mode),
            evidence=_evidence_for_mode(mode, task),
            candidates=json.dumps([_candidate_for_llm(c) for c in batch], ensure_ascii=False, indent=2),
        )
        async with llm_sem:
            data, raw, error, repaired = await _complete_json(
                model,
                prompt,
                4096,
                (
                    '{"scores": [{"paper_id": "<candidate paper_id>", '
                    '"score": 5, "grounds": [1], "reason": "<short reason>"}]}'
                ),
                fallback_repair_model=fallback_repair_model,
            )
            if error:
                if verbose:
                    print(f"[score:error] {task.paper_id}:{task.claim_idx} {mode} {error}", flush=True)
                errors.append(error)
                continue
            before = len(score_rows)
            for item in _score_items(data):
                paper_id = str(item.get("paper_id") or "").strip()
                if paper_id not in allowed or paper_id in scored:
                    continue
                try:
                    score = int(item.get("score"))
                except Exception:
                    score = 0
                score = max(0, min(5, score))
                grounds = item.get("grounds") or []
                if not isinstance(grounds, list):
                    grounds = []
                scored[paper_id] = {
                    "llm_score": score,
                    "llm_grounds": grounds,
                    "llm_reason": item.get("reason") or "",
                }
                score_rows.append({"paper_id": paper_id, **scored[paper_id]})
            if verbose:
                print(
                    f"[score:done] {task.paper_id}:{task.claim_idx} {mode} "
                    f"batch_scored={len(score_rows) - before}",
                    flush=True,
                )

    missing_scored = max(0, len(to_score) - len(scored))
    ranked_candidates = []
    for candidate in candidates:
        paper_id = candidate["paper_id"]
        llm = scored.get(paper_id, {
            "llm_score": 0,
            "llm_grounds": [],
            "llm_reason": "Not scored by LLM; assigned 0 for reranking and retained by deterministic tie-breaker.",
            "missing_llm_score": True,
        })
        ranked_candidates.append((candidate, llm))
    ranked_candidates.sort(
        key=lambda pair: (
            -pair[1]["llm_score"],
            -pair[0]["deterministic_score"],
            pair[0]["best_rank"],
            pair[0].get("paper_title") or "",
        )
    )
    ranked = []
    for idx, (candidate, llm) in enumerate(ranked_candidates, 1):
        ranked.append({
            "paper_id": candidate["paper_id"],
            "paper_title": candidate.get("paper_title"),
            "rank": idx,
            "best_rank": candidate.get("best_rank"),
            "frequency": candidate.get("frequency"),
            "deterministic_score": candidate.get("deterministic_score"),
            "llm_score": llm["llm_score"],
            "llm_grounds": llm["llm_grounds"],
            "reason": llm["llm_reason"],
            "missing_llm_score": llm.get("missing_llm_score", False),
            "ranker": "llm_score",
        })
    return ranked, score_rows, "; ".join(errors) if errors else None, missing_scored


async def _process_claim(
    mode: str,
    task: ClaimTask,
    query_model: str,
    rerank_model: str,
    llm_sem: asyncio.Semaphore,
    s2_sem: asyncio.Semaphore,
    n_queries: int,
    results_per_query: int,
    max_candidates: int,
    llm_rerank_candidates: int,
    llm_score_batch_size: int,
    candidate_snippets: str,
    run_llm_score: bool,
    verbose: bool,
    fallback_repair_model: str | None,
) -> dict:
    if verbose:
        print(f"[claim:start] {task.paper_id}:{task.claim_idx} {mode}", flush=True)
    queries, query_error = await _generate_queries(
        query_model,
        mode,
        task,
        n_queries,
        llm_sem,
        fallback_repair_model,
    )
    if verbose:
        print(
            f"[queries] {task.paper_id}:{task.claim_idx} {mode} "
            f"n={len(queries)} error={query_error!r} queries={queries}",
            flush=True,
        )
    source_ids = {pid for pid in [task.paper_id, get_semantic_scholar_paper_id(task.paper_id)] if pid}
    search_results = []
    search_errors = []
    for query in queries:
        try:
            if verbose:
                print(f"[s2:start] {task.paper_id}:{task.claim_idx} {mode} query={query}", flush=True)
            papers = await s2_search(
                query,
                s2_sem,
                limit=results_per_query,
                max_year=task.conference_year,
                include_snippets=candidate_snippets != "none",
            )
            search_results.append([p for p in papers if p.get("paperId") not in source_ids])
            if verbose:
                print(
                    f"[s2:done] {task.paper_id}:{task.claim_idx} {mode} "
                    f"hits={len(papers)} kept={len(search_results[-1])}",
                    flush=True,
                )
        except Exception as exc:
            search_results.append([])
            search_errors.append(str(exc))
            if verbose:
                print(f"[s2:error] {task.paper_id}:{task.claim_idx} {mode} {exc}", flush=True)
    candidates = _dedupe_candidates(search_results, queries, max_candidates, candidate_snippets)
    if verbose:
        print(
            f"[candidates] {task.paper_id}:{task.claim_idx} {mode} "
            f"n={len(candidates)} run_llm_score={run_llm_score}",
            flush=True,
        )
    deterministic_ranked = _deterministic_rank(candidates)
    if run_llm_score:
        llm_ranked, llm_scores, rerank_error, n_missing_llm_scores = await _score_rerank_candidates(
            rerank_model,
            mode,
            task,
            candidates,
            llm_rerank_candidates,
            llm_score_batch_size,
            verbose,
            llm_sem,
            fallback_repair_model,
        )
    else:
        llm_ranked, llm_scores, rerank_error, n_missing_llm_scores = deterministic_ranked, [], None, 0
    if verbose:
        print(
            f"[claim:done] {task.paper_id}:{task.claim_idx} {mode} "
            f"llm_scored={len(llm_scores)} missing={n_missing_llm_scores} error={rerank_error!r}",
            flush=True,
        )
    return {
        "paper_id": task.paper_id,
        "paper_title": task.paper_title,
        "claim_idx": task.claim_idx,
        "claim": task.claim,
        "mode": mode,
        "mode_name": MODE_NAMES[mode],
        "target_ids": task.target_ids,
        "target_titles": task.target_titles,
        "target_ingredient_groups": task.target_ingredient_groups,
        "n_target_ingredients": len(task.target_ingredient_groups),
        "n_targets": len(task.target_ids),
        "queries": queries,
        "query_error": query_error,
        "search_errors": search_errors,
        "candidate_papers": candidates,
        "n_candidates": len(candidates),
        "rankings": {
            "deterministic": deterministic_ranked,
            "llm_score": llm_ranked,
        },
        "ranked_papers": llm_ranked,
        "ranked_ids": [paper["paper_id"] for paper in llm_ranked],
        "llm_scores": llm_scores,
        "n_llm_scored": len(llm_scores),
        "n_missing_llm_scores": n_missing_llm_scores,
        "error": rerank_error,
    }


def _paper_scores(selected: list[str], gold: set[str]) -> tuple[float, float, float, float]:
    hits = len(set(selected) & gold)
    recall = hits / len(gold) if gold else 0.0
    precision = hits / len(selected) if selected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    any_hit = 1.0 if hits else 0.0
    return recall, precision, f1, any_hit


def _ingredient_coverage(selected: list[str], groups: list[list[str]]) -> float:
    if not groups:
        return 0.0
    selected_set = set(selected)
    covered = sum(1 for group in groups if selected_set & set(group))
    return covered / len(groups)


def _metrics(rows: list[dict], budgets: list[int], ranker: str) -> list[dict]:
    groundable = [row for row in rows if row["target_ids"]]
    candidate_recalls = []
    candidate_precisions = []
    candidate_f1s = []
    candidate_any_hits = []
    candidate_coverages = []
    for row in groundable:
        candidates = [p["paper_id"] for p in row["candidate_papers"]]
        gold = set(row["target_ids"])
        recall, precision, f1, any_hit = _paper_scores(candidates, gold)
        candidate_recalls.append(recall)
        candidate_precisions.append(precision)
        candidate_f1s.append(f1)
        candidate_any_hits.append(any_hit)
        candidate_coverages.append(_ingredient_coverage(candidates, row.get("target_ingredient_groups") or []))

    out = []
    for budget in budgets:
        recalls = []
        precisions = []
        f1s = []
        any_hits = []
        coverages = []
        for row in groundable:
            ranking = (row.get("rankings") or {}).get(ranker) or []
            selected = [p["paper_id"] for p in ranking[:budget]]
            gold = set(row["target_ids"])
            recall, precision, f1, any_hit = _paper_scores(selected, gold)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            any_hits.append(any_hit)
            coverages.append(_ingredient_coverage(selected, row.get("target_ingredient_groups") or []))
        out.append({
            "budget": budget,
            "claims": len(groundable),
            "candidate_pool_recall": round(mean(candidate_recalls), 4) if candidate_recalls else 0.0,
            "candidate_pool_precision": round(mean(candidate_precisions), 4) if candidate_precisions else 0.0,
            "candidate_pool_f1": round(mean(candidate_f1s), 4) if candidate_f1s else 0.0,
            "candidate_pool_any_hit": round(mean(candidate_any_hits), 4) if candidate_any_hits else 0.0,
            "candidate_pool_ingredient_coverage": round(mean(candidate_coverages), 4) if candidate_coverages else 0.0,
            "mean_recall": round(mean(recalls), 4) if recalls else 0.0,
            "mean_precision": round(mean(precisions), 4) if precisions else 0.0,
            "mean_f1": round(mean(f1s), 4) if f1s else 0.0,
            "any_hit_rate": round(mean(any_hits), 4) if any_hits else 0.0,
            "ingredient_coverage": round(mean(coverages), 4) if coverages else 0.0,
        })
    return out


async def _run(args: argparse.Namespace) -> None:
    modes = args.mode
    predicted_paths = args.predicted_results or []
    if any(mode in {"B4", "B4_MATCHED"} for mode in modes) and not predicted_paths:
        raise SystemExit("--predicted-results is required when --mode includes B4 or B4_MATCHED")

    llm_sem = asyncio.Semaphore(args.concurrency)
    s2_sem = asyncio.Semaphore(args.s2_concurrency)
    all_results = []
    metric_rows = []
    outpath = Path(args.output) if args.output else Path("results") / "task_b_claim_level" / f"claim_level_{datetime.now().strftime('%Y%m%d-%H%M')}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    csv_path = outpath.with_suffix(".csv")

    def write_checkpoint() -> None:
        output = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "gold_file": args.gold_file,
                "gold_claim_field": args.gold_claim_field,
                "predicted_results": predicted_paths,
                "model": args.model,
                "query_model": args.query_model or args.model,
                "fallback_repair_model": args.fallback_repair_model,
                "modes": modes,
                "n_queries": args.n_queries,
                "results_per_query": args.results_per_query,
                "max_candidates": args.max_candidates,
                "llm_rerank_candidates": args.llm_rerank_candidates,
                "llm_score_batch_size": args.llm_score_batch_size,
                "candidate_snippets": args.candidate_snippets,
                "rankers": args.ranker,
                "verbose": args.verbose,
                "budgets": args.budget,
                "conference_year": args.conference_year,
                "limit": args.limit,
                "completed_blocks": len(all_results),
            },
            "metrics": metric_rows,
            "results": all_results,
        }
        outpath.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        if metric_rows:
            fieldnames = list(metric_rows[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metric_rows)

    for mode in modes:
        mode_predicted_paths = predicted_paths if mode in {"B4", "B4_MATCHED"} else [None]
        for predicted_path in mode_predicted_paths:
            tasks = _load_claim_tasks(
                args.gold_file,
                args.gold_claim_field,
                predicted_path,
                mode == "B4_MATCHED",
                args.conference_year,
                args.limit,
            )
            result_label = _result_slug(predicted_path) if predicted_path else None
            display_name = (
                f"{MODE_NAMES[mode]} ({result_label})"
                if result_label else MODE_NAMES[mode]
            )
            coros = [
                _process_claim(
                    mode,
                    task,
                    args.query_model or args.model,
                    args.model,
                    llm_sem,
                    s2_sem,
                    args.n_queries,
                    args.results_per_query,
                    args.max_candidates,
                    args.llm_rerank_candidates,
                    args.llm_score_batch_size,
                    args.candidate_snippets,
                    "llm_score" in args.ranker,
                    args.verbose,
                    args.fallback_repair_model,
                )
                for task in tasks
            ]
            rows = []
            for fut in atqdm(asyncio.as_completed(coros), total=len(coros), desc=f"{mode} {display_name}"):
                rows.append(await fut)
            rows.sort(key=lambda r: (r["paper_id"], r["claim_idx"]))
            for ranker in args.ranker:
                metrics = _metrics(rows, args.budget, ranker)
                for metric in metrics:
                    metric_rows.append({
                        "mode": mode,
                        "mode_name": MODE_NAMES[mode],
                        "result_label": result_label,
                        "display_name": display_name,
                        "ranker": ranker,
                        "predicted_results": predicted_path,
                        "model": args.model,
                        "query_model": args.query_model or args.model,
                        "fallback_repair_model": args.fallback_repair_model,
                        "llm_rerank_candidates": args.llm_rerank_candidates,
                        "candidate_snippets": args.candidate_snippets,
                        "n_claims_total": len(rows),
                        "n_errors": sum(1 for row in rows if row.get("error")),
                        "mean_candidates": round(mean([row["n_candidates"] for row in rows]), 4) if rows else 0.0,
                        "mean_ranked": round(mean([len((row.get("rankings") or {}).get(ranker) or []) for row in rows]), 4) if rows else 0.0,
                        "mean_llm_scored": round(mean([row.get("n_llm_scored", 0) for row in rows]), 4) if rows and ranker == "llm_score" else "",
                        "mean_missing_llm_scores": round(mean([row.get("n_missing_llm_scores", 0) for row in rows]), 4) if rows and ranker == "llm_score" else "",
                        **metric,
                    })
            all_results.append({
                "mode": mode,
                "mode_name": MODE_NAMES[mode],
                "result_label": result_label,
                "predicted_results": predicted_path,
                "claims": rows,
            })
            write_checkpoint()

    print("| Mode | Ranker | K | Coverage | Recall | Precision | F1 | Any-hit | Cand recall | Cand coverage | Mean candidates | Mean ranked | LLM scored | Missing scores | Errors |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in metric_rows:
        print(
            f"| {row['display_name']} | {row['ranker']} | {row['budget']} | {row['ingredient_coverage']:.3f} | "
            f"{row['mean_recall']:.3f} | "
            f"{row['mean_precision']:.3f} | {row['mean_f1']:.3f} | "
            f"{row['any_hit_rate']:.3f} | {row['candidate_pool_recall']:.3f} | "
            f"{row['candidate_pool_ingredient_coverage']:.3f} | {row['mean_candidates']:.2f} | "
            f"{row['mean_ranked']:.2f} | {row['mean_llm_scored']} | "
            f"{row['mean_missing_llm_scores']} | {row['n_errors']} |"
        )
    print(f"\nWrote {outpath} and {csv_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Claim-level grounding retrieval under equal top-K budget.")
    parser.add_argument("--gold-file", default=str(DEFAULT_GOLD_FILE))
    parser.add_argument("--gold-claim-field", choices=["target_contribution", "rewritten_capability", "rewritten_claim"], default="target_contribution")
    parser.add_argument(
        "--predicted-results",
        nargs="+",
        default=None,
        help="One or more judged/prediction JSON files for B4.",
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["B1", "B3", "B4", "B4_MATCHED"],
        default=["B1", "B3", "B4"],
    )
    parser.add_argument("--model", default="gemini/gemini-3-flash-preview", help="LLM used for final candidate reranking.")
    parser.add_argument("--query-model", default=None, help="LLM used for query generation. Defaults to --model.")
    parser.add_argument(
        "--fallback-repair-model",
        default="gemini/gemini-3.1-pro-preview",
        help="Stronger model used only to repair malformed JSON after same-model repair fails. Set to '' to disable.",
    )
    parser.add_argument("--n-queries", type=int, default=5)
    parser.add_argument("--results-per-query", type=int, default=20)
    parser.add_argument("--max-candidates", type=int, default=100)
    parser.add_argument("--max-papers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--llm-rerank-candidates",
        type=int,
        default=30,
        help="Number of deterministic top candidates to score with the LLM reranker.",
    )
    parser.add_argument(
        "--llm-score-batch-size",
        type=int,
        default=10,
        help="Number of candidates scored per LLM call.",
    )
    parser.add_argument(
        "--candidate-snippets",
        choices=["none", "tldr", "abstract", "tldr-or-abstract"],
        default="tldr-or-abstract",
        help="Candidate text shown to the LLM scorer. Snippets are requested only when not 'none'.",
    )
    parser.add_argument(
        "--ranker",
        nargs="+",
        choices=["deterministic", "llm_score"],
        default=["deterministic", "llm_score"],
        help="Which ranking variants to report.",
    )
    parser.add_argument("--budget", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--s2-concurrency", type=int, default=2)
    parser.add_argument("--conference-year", type=int, choices=[2023, 2024, 2025], default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N papers for smoke tests.")
    parser.add_argument("--verbose", action="store_true", help="Print per-claim query, search, and scoring progress.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.fallback_repair_model == "":
        args.fallback_repair_model = None
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
