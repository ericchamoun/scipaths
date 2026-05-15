import json
import re
import asyncio

import litellm

from modules.local_hf import generate_local_hf

# Setting 1: claim only
# GENERATION_PROMPT_S1 = """\
# Given the following scientific discovery claim, identify the enabling contributions \
# (prior works, datasets, tools, or methods) that were necessary to realize it.

# Discovery claim:
# {claim}

# For each enabling contribution provide:
# - "description": a brief functional description of what it is
# - "role": one of DATA_SOURCE | EVALUATION_PROTOCOL | IMPLEMENTATION_TOOLING | TRAINING_DATA | CONCEPTUAL_FRAMEWORK | MODEL_INITIALIZATION | CORE_METHOD
# - "rationale": one sentence on why this contribution was necessary

# Respond with JSON only (no markdown):
# {{"ingredients": [{{"description": "...", "role": "...", "rationale": "..."}}]}}\
# """

# # Setting 2: claim + reference list
# GENERATION_PROMPT_S2 = """\
# Given the following scientific discovery claim and the list of papers it cites, \
# identify which cited works are enabling contributions — prior works, datasets, tools, \
# or methods that were necessary to realize the discovery.

# Discovery claim:
# {claim}

# Cited papers:
# {references}

# For each enabling contribution provide:
# - "description": a brief functional description of what it contributes
# - "role": one of DATA_SOURCE | EVALUATION_PROTOCOL | IMPLEMENTATION_TOOLING | BASELINE_METHOD | THEORETICAL_FOUNDATION
# - "rationale": one sentence on why this contribution was necessary
# - "ref_title": the title of the cited paper it corresponds to (or null if not directly tied to one reference)

# Only include contributions that are genuinely enabling — not every cited paper qualifies.

# Respond with JSON only (no markdown):
# {{"ingredients": [{{"description": "...", "role": "...", "rationale": "...", "ref_title": "..."}}]}}\
# """

GENERATION_PROMPT_S1 = """\
Given the following scientific discovery we would like to realize, forecast the enabling contributions required to make it possible.

An enabling contribution is a functional component whose absence would prevent the discovery from being realized in its claimed form. It may come from prior work, or it may be newly introduced / paper-specific (e.g., a data construction pipeline, annotation procedure, or integration step).

Discovery claim:
{claim}

Use the following role definitions:
- CORE_METHOD: algorithm, architecture, training objective, optimization/inference procedure, or other method needed to implement the discovery.
- CONCEPTUAL_FRAMEWORK: task definition, problem formulation, representation, theory, or empirical phenomenon the discovery builds on.
- DATA_SOURCE: dataset or corpus used as a source to construct another dataset/resource.
- TRAINING_DATA: dataset or labeled resource used directly for training or supervision.
- MODEL_INITIALIZATION: pretrained model, checkpoint, encoder, decoder, or initialization essential to the result.
- EVALUATION_PROTOCOL: benchmark, metric, annotation scheme, or evaluation setup central to the discovery.
- IMPLEMENTATION_TOOLING: software, infrastructure, or tooling explicitly required to implement the discovery.

Only include necessary enabling contributions. Exclude generic background, motivation, baselines, broad related work, and evaluation datasets used only for comparison.

For each enabling contribution, provide:
- "description": a concise functional description
- "role": one of the roles above
- "rationale": one sentence explaining why it is necessary

Respond with JSON only. Use concrete strings from your analysis; never copy placeholder values such as "...":
{{"ingredients": [{{"description": "...", "role": "...", "rationale": "..."}}]}}\
"""

GENERATION_PROMPT_S2 = """\
Given the following scientific discovery we would like to realize, and a list of cited papers from the target paper, forecast the enabling contributions required to make the discovery possible.

The cited papers are context about available prior work. Do not simply list citations: not every cited paper is enabling, and some enabling contributions may be newly introduced or paper-specific.

An enabling contribution is a functional component whose absence would prevent the discovery from being realized in its claimed form. It may come from prior work, or it may be newly introduced / paper-specific.

Discovery claim:
{claim}

Cited papers:
{references}

Use the following role definitions:
- CORE_METHOD: algorithm, architecture, training objective, optimization/inference procedure, or other method needed to implement the discovery.
- CONCEPTUAL_FRAMEWORK: task definition, problem formulation, representation, theory, or empirical phenomenon the discovery builds on.
- DATA_SOURCE: dataset or corpus used as a source to construct another dataset/resource.
- TRAINING_DATA: dataset or labeled resource used directly for training or supervision.
- MODEL_INITIALIZATION: pretrained model, checkpoint, encoder, decoder, or initialization essential to the result.
- EVALUATION_PROTOCOL: benchmark, metric, annotation scheme, or evaluation setup central to the discovery.
- IMPLEMENTATION_TOOLING: software, infrastructure, or tooling explicitly required to implement the discovery.

Only include necessary enabling contributions. Exclude generic background, motivation, baselines, broad related work, and evaluation datasets used only for comparison.

For each enabling contribution, provide:
- "description": a concise functional description
- "role": one of the roles above
- "rationale": one sentence explaining why it is necessary
- "ref_title": cited paper title if the contribution is realized by one of the cited papers, otherwise null

Respond with JSON only. Use concrete strings from your analysis; never copy placeholder values such as "...":
{{"ingredients": [{{"description": "...", "role": "...", "rationale": "...", "ref_title": "..."}}]}}\
"""

GENERATION_PROMPT_S3 = """\
Given the following scientific discovery we would like to realize, and the Related Work section from the target paper, forecast the enabling contributions required to make the discovery possible.

The Related Work section is context about relevant prior work and problem framing. Do not simply list related work: not every mentioned paper is enabling, and some enabling contributions may be newly introduced or paper-specific.

An enabling contribution is a functional component whose absence would prevent the discovery from being realized in its claimed form. It may come from prior work, or it may be newly introduced / paper-specific.

Discovery claim:
{claim}

Related Work section:
{related_work}

Use the following role definitions:
- CORE_METHOD: algorithm, architecture, training objective, optimization/inference procedure, or other method needed to implement the discovery.
- CONCEPTUAL_FRAMEWORK: task definition, problem formulation, representation, theory, or empirical phenomenon the discovery builds on.
- DATA_SOURCE: dataset or corpus used as a source to construct another dataset/resource.
- TRAINING_DATA: dataset or labeled resource used directly for training or supervision.
- MODEL_INITIALIZATION: pretrained model, checkpoint, encoder, decoder, or initialization essential to the result.
- EVALUATION_PROTOCOL: benchmark, metric, annotation scheme, or evaluation setup central to the discovery.
- IMPLEMENTATION_TOOLING: software, infrastructure, or tooling explicitly required to implement the discovery.

Only include necessary enabling contributions. Exclude generic background, motivation, baselines, broad related work, and evaluation datasets used only for comparison.

For each enabling contribution, provide:
- "description": a concise functional description
- "role": one of the roles above
- "rationale": one sentence explaining why it is necessary

Respond with JSON only. Use concrete strings from your analysis; never copy placeholder values such as "...":
{{"ingredients": [{{"description": "...", "role": "...", "rationale": "..."}}]}}\
"""


def _format_references(references: list[dict]) -> str:
    lines = []
    for i, ref in enumerate(references, 1):
        line = f"[{i}] {ref['title']}"
        if ref.get("contexts"):
            # Include first citation context for semantic grounding
            line += f"\n    Context: \"{ref['contexts'][0][:200]}\""
        lines.append(line)
    return "\n".join(lines)


def _format_few_shot_examples(examples: list[dict] | None) -> str:
    if not examples:
        return ""
    blocks = []
    for i, example in enumerate(examples, 1):
        blocks.append(
            f"Example {i}\n"
            f"Discovery claim:\n{example['claim']}\n\n"
            "Gold enabling contributions:\n"
            f"{json.dumps({'ingredients': example['ingredients']}, ensure_ascii=False)}"
        )
    return (
        "Use these annotated examples only to learn the expected level of abstraction, "
        "role labels, and JSON style. Do not copy their content into the new answer.\n\n"
        + "\n\n".join(blocks)
        + "\n\nNow answer the new instance.\n\n"
    )


def _clean_ingredients(parsed: object) -> list[dict]:
    if not isinstance(parsed, dict):
        return []
    ingredients = parsed.get("ingredients", [])
    if not isinstance(ingredients, list):
        return []
    cleaned = []
    placeholder_values = {"", "...", "…", "null", "none", "n/a"}
    for ingredient in ingredients:
        if not isinstance(ingredient, dict):
            continue
        description = str(ingredient.get("description", "")).strip()
        role = str(ingredient.get("role", "")).strip()
        rationale = str(ingredient.get("rationale", "")).strip()
        if (
            description.lower() in placeholder_values
            or role.lower() in placeholder_values
            or rationale.lower() in placeholder_values
        ):
            continue
        cleaned.append(ingredient)
    return cleaned


def _parse_ingredients(raw: str) -> list[dict]:
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    try:
        return _clean_ingredients(json.loads(raw))
    except json.JSONDecodeError:
        match = re.search(r'"ingredients"\s*:\s*(\[.*?\])', raw, re.DOTALL)
        if match:
            try:
                return _clean_ingredients({"ingredients": json.loads(match.group(1))})
            except json.JSONDecodeError:
                pass
        return []


REPAIR_PROMPT = """\
Convert the following model output into valid JSON matching exactly this schema:
{{"ingredients": [{{"description": "...", "role": "...", "rationale": "..."}}]}}

Rules:
- Preserve only ingredients that are present in the model output.
- Do not add new ingredients.
- Do not improve, correct, or infer scientific content.
- If no recoverable ingredients are present, return {{"ingredients": []}}.
- Respond with JSON only, no markdown.

Model output:
{raw}
"""


async def _repair_ingredients(raw: str, repair_model: str) -> list[dict]:
    msg = await litellm.acompletion(
        model=repair_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": REPAIR_PROMPT.format(raw=raw)}],
    )
    repaired = msg.choices[0].message.content.strip()
    return _parse_ingredients(repaired)


def _build_prompt(
    claim: str,
    references: list[dict] | None = None,
    related_work: str | None = None,
    few_shot_examples: list[dict] | None = None,
) -> str:
    few_shot_prefix = _format_few_shot_examples(few_shot_examples)
    if related_work is not None:
        return few_shot_prefix + GENERATION_PROMPT_S3.format(
            claim=claim,
            related_work=related_work or "(Related Work section unavailable.)",
        )
    if references:
        return few_shot_prefix + GENERATION_PROMPT_S2.format(
            claim=claim,
            references=_format_references(references),
        )
    return few_shot_prefix + GENERATION_PROMPT_S1.format(claim=claim)


async def generate_ingredients(
    model: str,
    claim: str,
    references: list[dict] | None = None,
    related_work: str | None = None,
    few_shot_examples: list[dict] | None = None,
    repair_model: str | None = None,
    max_retries: int = 3,
) -> tuple[list[dict], dict]:
    """Predict enabling contributions for a discovery claim.

    Setting 1 (references=None, related_work=None): claim only.
    Setting 2 (references provided): claim + cited paper list.
    Setting 3 (related_work provided): claim + target paper Related Work section.
    Retries up to max_retries times if the response is not valid JSON.
    """
    prompt = _build_prompt(claim, references, related_work, few_shot_examples=few_shot_examples)

    last_err = None
    repair_attempted = False
    repair_succeeded = False
    for attempt in range(max_retries):
        if model.startswith(("local_hf/", "hf_local/", "local_hf_lora/")):
            raw = await asyncio.to_thread(generate_local_hf, model, prompt)
        else:
            msg = await litellm.acompletion(
                model=model,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.choices[0].message.content.strip()
        result = _parse_ingredients(raw)
        if result:
            return result, {
                "raw_parse_failed": False,
                "repaired": False,
                "repair_model": None,
                "attempts": attempt + 1,
            }
        if repair_model:
            repair_attempted = True
            repaired = await _repair_ingredients(raw, repair_model)
            if repaired:
                repair_succeeded = True
                return repaired, {
                    "raw_parse_failed": True,
                    "repaired": True,
                    "repair_model": repair_model,
                    "attempts": attempt + 1,
                }
        last_err = raw

    raise ValueError(
        f"Failed to parse valid ingredients after {max_retries} attempts "
        f"(repair_attempted={repair_attempted}, repair_succeeded={repair_succeeded}). "
        f"Last response: {last_err[:200]}"
    )
