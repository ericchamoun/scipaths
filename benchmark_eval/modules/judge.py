import json
import re

import litellm

# ---------------------------------------------------------------------------
# Claim comparison judge (used by llm_as_a_judge.py)
# ---------------------------------------------------------------------------

CLAIM_ALIGNMENT_PROMPT = """\
You are evaluating whether two descriptions of a scientific paper's main contribution refer to the same contribution.

Paper title: {title}

Description A:
{claim_a}

Description B:
{claim_b}

Score the alignment on a 1–3 scale:
3 = Strong alignment: same type, same key artifact/mechanism/dataset named, same scope.
2 = Partial alignment: same general area but different emphasis, scope, or key artifacts named.
1 = Weak alignment: describe fundamentally different contributions.

Respond with JSON only (no markdown): {{"score": <1|2|3>, "reasoning": "<one sentence>"}}\
"""


async def judge_claim_alignment(
    model: str,
    claim_a: str,
    claim_b: str,
    paper_title: str,
) -> dict:
    prompt = CLAIM_ALIGNMENT_PROMPT.format(
        title=paper_title, claim_a=claim_a, claim_b=claim_b
    )
    msg = await litellm.acompletion(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'"score"\s*:\s*(\d)', raw)
        return {"score": int(m.group(1)) if m else 0, "reasoning": raw}


# ---------------------------------------------------------------------------
# Recall judge
# ---------------------------------------------------------------------------

MATCHING_JUDGE_PROMPT = """\
You are evaluating whether predicted enabling contributions match reference enabling contributions for a scientific discovery.

A full match requires that the predicted item express the same functional requirement needed to realize the discovery. It may use different wording, but it must preserve the key resource, method, process, role, and level of specificity.

A partial match means the predicted item is related but too broad, too narrow, missing an important role/detail, or merges multiple distinct reference requirements.

Do not mark broad umbrella items as full matches for multiple distinct reference contributions.

Discovery claim:
{claim}

Reference enabling contributions:
{references}

Predicted enabling contributions:
{predicted}

Return candidate matches only where match is "full" or "partial". Omit unrelated pairs.

Respond with JSON only (no markdown):
{{"matches": [{{"reference_idx": <1-based index>, "predicted_idx": <1-based index>, "match": "full"|"partial", "reasoning": "<one sentence>"}}]}}\
"""

RECALL_JUDGE_PROMPT = """\
You are evaluating whether one predicted enabling contribution semantically matches one reference enabling contribution.

A match requires that the predicted item express the same functional requirement needed to realize the discovery. It may use different wording, but it must preserve the key resource, method, process, role, and level of specificity.

Do not mark as covered if the predicted item is only a broader category, a related neighboring concept, a partial subcomponent, or merely useful background.

Discovery claim:
{claim}

Reference enabling contribution:
{silver}

Predicted enabling contributions:
{predicted}

Choose at most one predicted item. If multiple predicted items are needed together to cover the reference contribution, mark covered=false.

Respond with JSON only (no markdown): {{"covered": <true|false>, "best_match_idx": <1-based index of the best matching item, or null if none>, "reasoning": "<one sentence>"}}\
"""


def _format_reference_ingredient(silver_ingredient: dict) -> str:
    annotation = silver_ingredient.get("canonical_annotation") or {}
    roles = annotation.get("roles")
    if isinstance(roles, list):
        role_text = ", ".join(str(r) for r in roles)
    else:
        role_text = annotation.get("role", "")

    description = silver_ingredient.get("ingredient") or silver_ingredient.get("enabling_contribution") or ""
    lines = [f"Enabling contribution: {description}"]
    if role_text:
        lines.append(f"Role: {role_text}")
    if annotation.get("contribution"):
        lines.append(f"Contribution: {annotation['contribution']}")
    if annotation.get("rationale"):
        lines.append(f"Rationale: {annotation['rationale']}")
    if annotation.get("evidence_span"):
        lines.append(f"Evidence: {annotation['evidence_span']}")
    return "\n".join(lines)


def _format_reference_ingredient_indexed(idx: int, ingredient: dict) -> str:
    return f"[{idx}] " + _format_reference_ingredient(ingredient).replace("\n", "\n    ")


def _format_predicted_ingredient(idx: int, ingredient: dict) -> str:
    lines = [f"[{idx}] {ingredient.get('description', '')}"]
    if ingredient.get("role"):
        lines.append(f"    Role: {ingredient['role']}")
    if ingredient.get("rationale"):
        lines.append(f"    Rationale: {ingredient['rationale']}")
    if ingredient.get("evidence"):
        lines.append(f"    Evidence: {ingredient['evidence']}")
    return "\n".join(lines)


def _parse_matching_judgment(raw: str, n_references: int, n_predicted: int) -> dict | None:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    raw_matches = data.get("matches")
    if not isinstance(raw_matches, list):
        return None

    matches = []
    for match in raw_matches:
        if not isinstance(match, dict):
            continue
        ref_idx = match.get("reference_idx", match.get("gold_idx"))
        pred_idx = match.get("predicted_idx", match.get("pred_idx"))
        label = str(match.get("match", "")).lower()
        if label not in {"full", "partial"}:
            continue
        if not isinstance(ref_idx, int) or not isinstance(pred_idx, int):
            continue
        if not (1 <= ref_idx <= n_references and 1 <= pred_idx <= n_predicted):
            continue
        matches.append({
            "reference_idx": ref_idx,
            "predicted_idx": pred_idx,
            "match": label,
            "reasoning": str(match.get("reasoning", "")),
        })
    return {"matches": matches}


async def judge_ingredient_matching(
    model: str,
    claim: str,
    reference_ingredients: list[dict],
    predicted_ingredients: list[dict],
    max_retries: int = 3,
) -> dict:
    """Judge all reference/predicted ingredients for one claim in one call."""
    references_text = "\n\n".join(
        _format_reference_ingredient_indexed(i + 1, ing)
        for i, ing in enumerate(reference_ingredients)
    ) or "(none)"
    predicted_text = "\n\n".join(
        _format_predicted_ingredient(i + 1, ing)
        for i, ing in enumerate(predicted_ingredients)
    ) or "(none)"

    prompt = MATCHING_JUDGE_PROMPT.format(
        claim=claim,
        references=references_text,
        predicted=predicted_text,
    )

    for attempt in range(max_retries):
        msg = await litellm.acompletion(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.choices[0].message.content.strip()
        result = _parse_matching_judgment(raw, len(reference_ingredients), len(predicted_ingredients))
        if result is not None:
            return result

    raise ValueError(f"Failed to parse valid matching judgment after {max_retries} attempts")


def _parse_recall_judgment(raw: str, predicted_ingredients: list[dict]) -> dict | None:
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'"covered"\s*:\s*(true|false)', raw, re.IGNORECASE)
        if not m:
            return None
        covered = m.group(1).lower() == "true"
        result = {"covered": covered, "best_match_idx": None, "reasoning": raw}

    if "covered" not in result:
        return None

    idx = result.get("best_match_idx")
    if isinstance(idx, int) and 1 <= idx <= len(predicted_ingredients):
        result["best_match"] = predicted_ingredients[idx - 1]
    else:
        result["best_match"] = None
        result["best_match_idx"] = None
    return result


async def judge_ingredient_recall(
    model: str,
    claim: str,
    silver_ingredient: dict,
    predicted_ingredients: list[dict],
    max_retries: int = 3,
) -> dict:
    silver_text = _format_reference_ingredient(silver_ingredient)
    predicted_text = "\n".join(
        _format_predicted_ingredient(i + 1, ing)
        for i, ing in enumerate(predicted_ingredients)
    ) or "(none)"

    prompt = RECALL_JUDGE_PROMPT.format(
        claim=claim, silver=silver_text, predicted=predicted_text
    )

    for attempt in range(max_retries):
        msg = await litellm.acompletion(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.choices[0].message.content.strip()
        result = _parse_recall_judgment(raw, predicted_ingredients)
        if result is not None:
            return result

    raise ValueError(f"Failed to parse valid judgment after {max_retries} attempts")
