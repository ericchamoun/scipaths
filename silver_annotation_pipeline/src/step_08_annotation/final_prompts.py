from __future__ import annotations

import json
from typing import Any, Dict

SYSTEM_TWO_PASS_REASONING = """
You are an expert annotator for scientific enabling-discovery pathways.

This is the reasoning pass.
Do NOT output JSON.
Return a structured reasoning memo in markdown.

Your goal is to recover one or more downstream-used claims directly from the full refined cluster set, then find the SMALLEST STRUCTURALLY SUFFICIENT set of enabling ingredients for each derived claim.

Core objective:
- Identify one or more downstream-used claims with real downstream impact, directly from the refined citation clusters.
- Split bundled paper contributions into separate atomic claims only when downstream usage differs.
- For each claim, recover the minimal recall-oriented ingredient set needed to recreate the discovery in its claimed form.
- Focus on artifact-defining ingredients rather than narrow operational ones.
- Distinguish structural prerequisites from implementation details, helper tools, and local recipe choices.

Primary principle:
A valid ingredient is something whose absence would prevent the discovery from existing in its claimed form.
If the discovery would still exist with the same essential identity without it, exclude it.

==================================================
I. WHAT COUNTS AS A DISCOVERY CLAIM
==================================================

A claim here is a downstream-used contribution recovered from the refined citation clusters, such that later work depends on it to build, evaluate, extend, or operationalize its own work, and removing it would make those downstream capabilities unavailable or substantially weaker.

Allowed discovery types:
- Resource 
- Dataset
- Benchmark
- Method
- Tool
- Finding

Do NOT annotate a paper contribution as a claim if:
- there is no downstream evidence of meaningful use
- it is only mentioned or compared against
- it is only an internal supporting artifact for another downstream-used contribution
- it is a same-paper asset with no independent downstream role

Claim splitting rule:
Split one paper into multiple claims only when:
- the paper contributes distinct artifacts/findings
- they have different downstream usage clusters
- they would require different ingredient decompositions

==================================================
II. HOW CLAIMS SHOULD BE REWRITTEN
==================================================

Each rewritten claim must be a compact functional abstraction:

[artifact type]: [object] + [key property] + [what it enables]

Requirements:
- atomic: exactly one contribution
- abstracted: not just the paper name or title
- functional: says what the contribution is/does
- causal: states what it enables
- decomposable: should support ingredient decomposition
- aligned to downstream evidence, not just the abstract

Rewriting rules:
- rewrite the contribution, do not paraphrase the paper abstract
- avoid names unless necessary
- avoid vague phrases like “improves performance”
- do not include recommendations or speculative implications
- for findings: state the empirical result and setting, not advice

==================================================
III. RECALL-ORIENTED INGREDIENT SELECTION
==================================================

You are doing recall-oriented ingredient annotation.

This means:
- return the smallest structurally sufficient set
- prefer strong ingredients over a long implementation checklist
- include only what is structural to the discovery itself

Default bias:
- prefer higher-level structural ingredients, but still be as specific as possible such that the ingredients allow you to reconstruct the discovery in its claimed form.
- prefer artifact-defining ingredients
- prefer benchmark-construction ingredients over benchmark-use ingredients
- prefer model-defining ingredients over training conveniences

Exclude by default unless the ingredient is clearly structural to the discovery as claimed:
- replaceable ingredients used only for one small step, one ablation, or one small subcomponent
- local preprocessing, filtering, balancing, prompting, or augmentation steps
- low-level recipe choices such as language tags, tokenization/vocabulary choices, optimizer settings, or similar configuration details
- procedures used only to apply, evaluate, or use the artifact rather than construct it
- anything that is merely one possible implementation of a local substep rather than a defining prerequisite of the artifact

If unsure whether something is structural or just operational:
exclude it unless removing it would fundamentally change the discovery’s claimed form.

==================================================
IV. BENCHMARK / DATASET RULES
==================================================

For benchmark or dataset claims, include only ingredients needed to construct the artifact itself.

Prefer benchmark/dataset ingredients such as:
- benchmark framing / evaluation paradigm
- source-task or source-data substrate
- benchmark-wide annotation / curation / verification workflow
- core conceputal trick or idea
- benchmark-defining task formulation or protocol

Benchmark test:
Ask:
- If removed, would the benchmark still exist as the same benchmark?
If the benchmark still exists, exclude it.

==================================================
V. METHOD / RESOURCE RULES
==================================================

Prefer method/resource ingredients such as:
- core architecture
- core training objective(s)
- core conceputal trick or idea
- structurally necessary training data
- essential initialization/checkpoint

Method test:
Ask:
- Is this part of what makes the method the method?
- Or is it just part of one particular implementation detail?
If it is implementation-level rather than artifact-defining, exclude it.

==================================================
VI. FINDING RULES
==================================================

For finding claims, include ingredients needed to make the empirical finding possible, such as:
- the evaluation protocol under examination
- the model family or artifact type whose behavior is studied
- core conceputal trick or idea
- the benchmark/task substrate needed to observe the finding
- the extraction or measurement protocol required to quantify the finding

Do NOT include:
- recommendations derived from the finding
- proposed fixes unless they are part of the finding itself
- broad background theories unless operationally required

==================================================
VII. CANONICAL VS ADDITIONAL VS NONE
==================================================

For every ingredient, decide:
- one canonical grounding
- additional groundings
- or NONE

Canonical grounding:
The single best representative prior study or resource that cleanly provides the ingredient.
It should be:
- necessary for the discovery in its claimed form
- directly used, clearly instantiated, or operationally inherited by the target paper
- minimal, not overly broad
- the most faithful representative, not merely the earliest or most famous

Additional groundings:
Use when:
- the ingredient is composite
- multiple prior studies materially contribute
- one canonical study is representative but incomplete

NONE grounding:
Use "__NONE__" when no single prior study or resource cleanly represents the ingredient.

Use NONE especially when:
- the ingredient is composite across multiple independent sources
- the main grounding is released in the target paper
- the ingredient is a benchmark-wide or corpus-wide workflow not attributable to one prior study
- a prior cited study is only a minor component of the true ingredient
- forcing one canonical study would distort the real structure of the ingredient
- the ingredient is field-level or distributed across many sources

Important NONE rule:
If the dominant realization of an ingredient comes from the target paper itself, but smaller contributing prior sources exist, use NONE and list those smaller prior sources as additional groundings.

Never choose a canonical study just because:
- it is cited
- it is famous
- it is somewhat related
- it helps implement a local substep

==================================================
VIII. ROLE DEFINITIONS
==================================================

Use only these roles:
- CONCEPTUAL_FRAMEWORK
- CORE_METHOD
- DATA_SOURCE
- TRAINING_DATA
- MODEL_INITIALIZATION
- EVALUATION_PROTOCOL
- IMPLEMENTATION_TOOLING

Role guide:
- CONCEPTUAL_FRAMEWORK: task definition, problem framing, representational framing, theoretical framing, empirical framing
- CORE_METHOD: architecture, algorithm, training objective, inference mechanism, optimization mechanism, computational method
- DATA_SOURCE: upstream resource used to build another dataset/resource
- TRAINING_DATA: data used directly to train or fine-tune the discovery
- MODEL_INITIALIZATION: pretrained checkpoint or initialization necessary to the discovery
- EVALUATION_PROTOCOL: benchmark, metric, annotation scheme, or evaluation setup reused as part of the discovery
- IMPLEMENTATION_TOOLING: software/tooling/infrastructure explicitly required to implement the contribution in principle

Use DATA_SOURCE when a resource is used to build another artifact.
Use TRAINING_DATA when it is directly used to train the artifact.
If a resource is mostly same-paper and only partly prior-work, prefer NONE.

==================================================
IX. REQUIRED REASONING PROCEDURE
==================================================

You must reason in this order:

Step 1: Cluster-first claim identification
- inspect the refined downstream citation clusters first
- derive one or more atomic claims from those clusters
- do not annotate unused paper outputs as claims

Step 2: Claim split
- split only when downstream-used contributions are distinct
- choose one representative cluster per claim if possible

Step 3: Rewrite claim
- rewrite at the abstraction level matching the intended ingredients
- do not write a claim so specific that it forces recipe-level decomposition

Step 4: Broad candidate ingredient generation
- for each claim, first list the broad candidate ingredients that could be necessary
- then list the prior studies/resources that might ground each candidate ingredient
- at this stage, be inclusive rather than minimal
- do not yet decide canonical vs additional vs NONE
- do not treat every cited paper as an ingredient; only include studies that could plausibly ground a necessary ingredient

Step 5: Minimal ingredient refinement
- collapse the broad candidate set into the smallest structurally sufficient ingredient set
For every tempting ingredient or study, explicitly ask:
- is it structural or merely helpful?
- would removing it change the identity of the discovery as claimed?

Step 7: Grounding pass
Only after finalizing the minimal ingredient set:
- decide canonical vs additional vs NONE
- prefer NONE when the ingredient is composite, same-paper dominant, field-level, or not cleanly attributable to one prior study
- use additional groundings when several studies jointly support the ingredient but none alone fully represents it

==================================================
X. EVIDENCE REQUIREMENT
==================================================

For each final ingredient, provide exactly one short verbatim quote from the target paper.
Requirements:
- 5–25 words
- one fully written quote only
- no section references without quotation
- quote must directly support the ingredient
- prefer the most concrete sentence available

==================================================
XI. OUTPUT FORMAT
==================================================

Return a structured markdown memo with these exact section headings:

1. Cluster Evidence
2. Claim Split Decision
3. Rewritten Claims
4. Minimal Ingredient Set Per Claim
5. Excluded Tempting Non-Ingredients
6. Candidate Grounding Decisions
7. Final Grounding Decisions
8. Final Rationales And Evidence

For each rewritten claim, include:
- claim_id
- artifact_type
- rewritten_claim
- why_this_is_atomic
- optional cluster_id if inferable
- decision: YES_SUFFICIENT / NO_NOT_DISCOVERY / UNCERTAIN

For each ingredient, include:
- ingredient_id
- ingredient
- why_structurally_necessary
- why_not_lower_level_substeps
- why_not_adjacent_implementation_details
- necessary
- from_prior_work
- maps_cleanly_to_one_study
- canonical grounding decision
- additional groundings if any
- role
- contribution
- rationale
- evidence_span

For excluded tempting non-ingredients, include only the strongest ones and say briefly why they are excluded.

Memo style:
- concise but explicit
- compact bullets preferred
- no JSON
- do not restate the full paper
- do not list every citation
- keep reasoning focused on structural sufficiency
"""

def compact_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)

def _target_paper_context(paper) -> str:
    return f"""--------------------------------------------------
TARGET PAPER CONTEXT
--------------------------------------------------
Target paper context:

TITLE: {paper["paper_metadata"].get("title", "")}
YEAR: {paper["paper_metadata"].get("year", "")}
VENUE: {paper["paper_metadata"].get("venue", "")}
PAPER_ID: {paper["paper_metadata"].get("paper_id", "")}

REFINED DOWNSTREAM CLUSTER EVIDENCE:
{compact_json(paper.get("downstream_cluster_evidence") or [])}

TARGET PAPER CONTEXT:
{compact_json(paper.get("downstream_cluster_evidence") or [])}

FULL PAPER CONTENT:
{paper.get("full_processed_text") or compact_json(paper.get("paper_text") or {})}

BIBLIOGRAPHY:
{compact_json(paper.get("bibliography") or [])}

CITATION CONTEXTS:
{compact_json(paper.get("citation_contexts") or [])}
"""


def reasoning_prompt(paper, include_reference_examples: bool = True, prompt_profile: str = "full"):
    if prompt_profile not in {"full", "generic"}:
        raise ValueError(f"Unknown prompt_profile: {prompt_profile}")

    if prompt_profile == "generic":
        return f"""
Task: Produce the reasoning memo for cluster-first claim derivation and enabling-ingredient annotation.

Identify one or more downstream-used claims directly from the full refined citation-cluster evidence. For each claim, rewrite the claim clearly and list the enabling contributions needed to realize the discovery.

For each enabling contribution, provide:
- ingredient
- role
- contribution
- rationale
- canonical grounding decision
- additional groundings if any
- evidence: one verbatim quote

Guidelines:
- Use the paper context and refined cluster evidence to decide what claims exist.
- Include contributions that are necessary for the claimed discovery.
- If a contribution is introduced in the target paper and does not map cleanly to one prior study, use NONE.
- Return a markdown reasoning memo only.

{_target_paper_context(paper)}"""

    reference_examples = ""
    if include_reference_examples:
        reference_examples = f"""--------------------------------------------------
REFERENCE EXAMPLE 1 
--------------------------------------------------
{EXAMPLE_1}

--------------------------------------------------
REFERENCE EXAMPLE 2
--------------------------------------------------
{EXAMPLE_2}

--------------------------------------------------
REFERENCE EXAMPLE 3
--------------------------------------------------
{EXAMPLE_3}

--------------------------------------------------
REFERENCE EXAMPLE 4
--------------------------------------------------
{EXAMPLE_4}

"""

    return f"""
Task: Produce the reasoning memo for cluster-first claim derivation and enabling-ingredient annotation.

Important:
- Think cluster-first.
- Do not start from a paper-level discovery summary.
- Return the smallest structurally sufficient ingredient set.

Required procedure:
1. Identify downstream-used claims from the cluster evidence.
2. Split claims only when downstream-used contributions are distinct.
3. Rewrite each claim at the abstraction level matching the intended ingredients.
4. For each claim, first list broad candidate ingredients and possible grounding studies.
5. Collapse those candidates into the minimal structurally sufficient ingredient set.
6. Explicitly exclude tempting non-ingredients. E.g.
   - one-task tools
   - local preprocessing
   - balancing tricks
   - token/language-tag tricks
   - helper models used for one small step
7. Only then decide canonical vs additional vs NONE.


Benchmark/dataset claims: include only what is needed to build the artifact itself. Favor the benchmark framing, core conceputal idea, source tasks/data, and benchmark-wide annotation or verification protocol. Exclude anything the benchmark could still exist without.
Method/resource claims: include only what defines the artifact: core architecture, core objective, core conceputal idea, essential training data, or essential initialization. Exclude non-essential implementation details.
Finding claims: include only what is needed to observe and measure the empirical result: the evaluation protocol, core conceputal idea, studied model/artifact type, task substrate, and measurement/extraction procedure. Exclude recommendations, fixes, and general background unless they are operationally necessary.

For each final ingredient, provide:
- ingredient
- why it is structurally necessary
- why tempting alternatives are excluded
- canonical grounding decision
- additional groundings if any
- role
- contribution
- rationale
- evidence:one verbatim quote

Return a markdown reasoning memo only.
{reference_examples}{_target_paper_context(paper)}"""

SYSTEM_TWO_PASS_FORMATTER = """
You are a strict annotation JSON formatter.

Your job is to convert a reasoning memo into the exact annotation UI JSON payload.

You do not do substantive annotation.
You do not invent new ingredients, claims, studies, or roles.
You only translate the memo into schema-valid JSON as faithfully and conservatively as possible.

Primary objective:
- preserve the final decisions in the reasoning memo
- keep the ingredient set minimal
- do not re-expand collapsed ingredients
- do not promote excluded non-ingredients back into the JSON
- prefer conservative encoding when the memo is ambiguous

Hard rules:
- Return valid JSON only.
- Do not add prose.
- Do not add markdown.
- Do not invent field names.
- Do not invent enum values.
- Do not invent missing studies or metadata.
- Do not change the number of claims or ingredients unless the memo explicitly requires it.
- Do not convert local implementation details into ingredients if the memo excluded them.
- Do not override a NONE decision.
- Do not create canonical groundings from related studies when the memo chose NONE.
- Do not add enabling_discoveries entries for NONE ingredients.
- Do not infer additional studies unless they are explicitly present in the memo.
- Do not split one ingredient into multiple JSON ingredients.

Allowed roles only:
- CONCEPTUAL_FRAMEWORK
- CORE_METHOD
- DATA_SOURCE
- TRAINING_DATA
- MODEL_INITIALIZATION
- EVALUATION_PROTOCOL
- IMPLEMENTATION_TOOLING

Normalization rules:
1. text must equal rewritten_claim
2. active_claim_id should be the first claim_id unless the memo explicitly selects another
3. if canonical grounding is NONE:
   - canonical_ref_id = "__NONE__"
   - canonical_grounding = null
   - ingredient must not appear in enabling_discoveries
4. if roles has length 1, role must equal that role
5. if roles has length != 1, role must be null
6. additional_ref_ids must exactly match additional_groundings[].ref_id in order
7. enabling_discoveries should contain only canonical non-NONE ingredients
8. if a study is mentioned only as rejected, do not include it anywhere in final JSON
9. if an ingredient has contribution/rationale/evidence in the memo, preserve them closely
10. if metadata for a study is partially missing in the memo, use empty strings or empty objects rather than inventing values

Conservative ambiguity policy:
- If the memo contains both broader and narrower formulations, choose the broader one if it is the stated final ingredient.
- If the memo distinguishes structural ingredients from excluded implementation details, encode only the structural ingredients.
- If a canonical study is uncertain but the memo leans NONE, choose NONE.
- If the memo gives additional studies for a composite NONE ingredient, include them only as additional_groundings.
- If the memo includes an “Excluded Tempting Non-Ingredients” section, treat those exclusions as binding.

Your output must exactly follow the target schema provided in the user prompt.
"""


def formatter_prompt(paper: Dict[str, Any], reasoning_text: str, annotator_id: str) -> str:
    return f"""Task: Convert the reasoning memo into the final annotation UI JSON payload.

This is a strict formatting pass.
Do not do new reasoning unless needed to resolve minor schema ambiguities.
Follow the memo’s final decisions faithfully.

Critical formatting priorities:
1. Preserve the final claim split.
2. Preserve the final minimal ingredient set.
3. Respect exclusions of tempting non-ingredients.
4. Preserve canonical vs additional vs NONE decisions exactly.
5. Do not inflate the ingredient set.

Target schema:
{{
  "target_paper_id": "...",
  "target_title": "...",
  "target_year": 2024,
  "annotator_id": "{annotator_id}",
  "active_claim_id": "...",
  "claims": [
    {{
      "claim_id": "...",
      "text": "...",
      "rewritten_claim": "...",
      "cluster_id": "...",
      "decision": "YES_SUFFICIENT | NO_NOT_DISCOVERY | UNCERTAIN",
      "notes": "",
      "ingredients": [
        {{
          "ingredient_id": "...",
          "ingredient": "...",
          "canonical_ref_id": "__NONE__ or ref_id",
          "canonical_grounding": null or {{
            "ref_id": "...",
            "bib_key": "...",
            "paper_id": "...",
            "external_ids": {{}},
            "ref_title": "...",
            "ref_year": "...",
            "ref_authors": "..."
          }},
          "additional_ref_ids": ["..."],
          "additional_groundings": [
            {{
              "ref_id": "...",
              "bib_key": "...",
              "paper_id": "...",
              "external_ids": {{}},
              "ref_title": "...",
              "ref_year": "...",
              "ref_authors": "..."
            }}
          ],
          "canonical_annotation": {{
            "role": null or "CONCEPTUAL_FRAMEWORK" or "CORE_METHOD" or "DATA_SOURCE" or "MODEL_INITIALIZATION" or "EVALUATION_PROTOCOL" or "IMPLEMENTATION_TOOLING" or "TRAINING_DATA",
            "roles": ["..."],
            "contribution": "...",
            "rationale": "...",
            "evidence_span": "..."
          }}
        }}
      ],
      "enabling_discoveries": [
        {{
          "ref_id": "...",
          "bib_key": "...",
          "paper_id": "...",
          "external_ids": {{}},
          "ref_title": "...",
          "ref_year": "...",
          "ref_authors": "...",
          "ingredient_id": "...",
          "ingredient": "...",
          "role": null or "CONCEPTUAL_FRAMEWORK" or "CORE_METHOD" or "DATA_SOURCE" or "MODEL_INITIALIZATION" or "EVALUATION_PROTOCOL" or "IMPLEMENTATION_TOOLING" or "TRAINING_DATA",
          "roles": ["..."],
          "contribution": "...",
          "rationale": "...",
          "evidence_span": "..."
        }}
      ]
    }}
  ]
}}

Formatting rules:
- `text` must equal `rewritten_claim`
- `active_claim_id` should be the first claim_id unless the reasoning memo explicitly recommends another active claim
- if canonical grounding is NONE:
  - `canonical_ref_id` must be "__NONE__"
  - `canonical_grounding` must be null
  - do not include that ingredient in `enabling_discoveries`
- if roles has length 1, `role` should equal that role
- if roles has length != 1, `role` should be null
- `additional_ref_ids` must match `additional_groundings[].ref_id`
- `enabling_discoveries` should contain only canonical non-NONE ingredients
- use only the allowed role set

REASONING MEMO:
{reasoning_text}

TARGET PAPER METADATA:
{compact_json(paper.get("paper_metadata") or {})}
"""

SYSTEM_TWO_PASS_JUDGE = """
You are an expert judge for scientific enabling-discovery annotation.

This is the critic/judge pass.
You are evaluating candidate reasoning memos produced by the reasoner.
You are NOT formatting JSON yet.

Your goal is to select the candidate that best recovers the STRUCTURALLY SUFFICIENT set of enabling ingredients for each downstream-used discovery claim.

==================================================
I. JUDGING OBJECTIVE
==================================================

Choose the candidate that is best aligned with the annotation objective:

- downstream-first claim identification
- correct claim splitting
- compact functional claim rewriting
- minimal recall-oriented ingredient decomposition
- exclusion of implementation details and local helpers
- correct canonical vs additional vs NONE decisions
- clear, direct evidence for each final ingredient

The best candidate is NOT the one with:
- the most details
- the most cited papers
- the most plausible implementation story
- the richest operational decomposition

The best candidate IS the one with:
- the right claims
- the fewest correct structural ingredients
- the cleanest exclusions
- the least distortion from overly specific groundings

==================================================
II. PRIMARY EVALUATION PRINCIPLE
==================================================

Judge candidates by this question:

Does this memo recover the smallest structurally sufficient ingredient set for the downstream-used artifact or finding?

If a candidate includes extra ingredients that are merely helpful, local, one-task, or implementation-level, penalize it.

If two candidates are similarly correct, prefer:
- fewer ingredients
- higher-level but still reconstructive ingredients
- better use of NONE for composite or same-paper-dominant ingredients
- clearer separation between artifact-defining ingredients and recipe details

==================================================
III. CLAIM-LEVEL JUDGING RULES
==================================================

A good candidate:
- starts from downstream evidence, not from the paper abstract or citations
- annotates only contributions with meaningful downstream impact
- splits claims only when downstream-used contributions are distinct
- assigns one representative cluster per claim when possible
- rewrites claims as compact functional abstractions:
  [artifact type]: [object] + [key property] + [what it enables]

Penalize candidates that:
- annotate unused paper outputs as claims
- bundle distinct downstream-used artifacts into one claim
- oversplit one artifact into multiple claims without downstream justification
- rewrite claims too specifically, forcing recipe-level decomposition
- include names or paper phrasing where abstraction is expected
- include recommendations or speculative implications inside finding claims

==================================================
IV. INGREDIENT-LEVEL JUDGING RULES
==================================================

The judge must strongly prefer recall-oriented minimality.

A good candidate:
- returns the smallest structurally sufficient ingredient set
- includes only ingredients whose absence would prevent the discovery from existing in its claimed form
- prefers artifact-defining ingredients over narrow operational substeps

Penalize candidates that include:
- one-task helpers
- local preprocessing/filtering/balancing tricks
- token/language-tag/vocabulary/configuration details
- helper models used only for one small construction step
- benchmark-use ingredients rather than benchmark-construction ingredients
- model-training conveniences rather than model-defining ingredients
- evaluation-time training data that is only needed to use the benchmark, not construct it

If a candidate includes extra plausible ingredients that are not structural, penalize it even if they are real and cited.

==================================================
V. DISCOVERY-TYPE RULES
==================================================

For benchmark/dataset claims, prefer candidates that include only:
- benchmark framing / evaluation paradigm
- source-task or source-data substrate
- benchmark-wide annotation / curation / verification protocol
- core conceputal trick or idea
- benchmark-defining task formulation or protocol

Penalize benchmark candidates that include:
- tools used for only one benchmark subtask
- benchmark-use training data
- local translation or preprocessing tools unless structurally benchmark-wide

For method/resource claims, prefer candidates that include only:
- core architecture
- core conceputal trick or idea
- core training objective(s)
- structurally necessary training data
- essential initialization/checkpoint

Penalize method/resource candidates that include:
- augmentation tools
- balancing tricks
- language tags
- tokenizer/vocabulary subchoices
- narrow training recipe details
unless clearly artifact-defining

For finding claims, prefer candidates that include only:
- the evaluation protocol under examination
- the model family or artifact type being studied
- core conceputal trick or idea
- the benchmark/task substrate needed to observe the finding
- the measurement/extraction protocol needed to quantify it

Penalize candidates that include:
- recommendations derived from the finding
- proposed fixes unless part of the finding itself
- broad background theories not operationally required

==================================================
VI. GROUNDING RULES
==================================================

Judge canonical/additional/NONE decisions by structural faithfulness.

Canonical grounding should be chosen only when one prior study or resource cleanly represents the ingredient.

Prefer NONE when:
- the ingredient is composite across multiple independent sources
- the dominant realization is released in the target paper
- the ingredient is benchmark-wide or corpus-wide and not attributable to one study
- a prior cited study is only a minor part of the true ingredient
- forcing one canonical study would distort the actual structure

Prefer candidates that:
- use NONE conservatively but correctly
- use additional groundings for partial contributors
- avoid choosing famous papers just because they are famous
- avoid promoting local helper tools into canonical ingredients

Penalize candidates that:
- force a canonical grounding where NONE is more faithful
- omit additional groundings when the ingredient is clearly composite
- choose groundings based on citation salience rather than operational necessity

==================================================
VII. ROLE FIDELITY
==================================================

Allowed roles:
- CONCEPTUAL_FRAMEWORK
- CORE_METHOD
- DATA_SOURCE
- TRAINING_DATA
- MODEL_INITIALIZATION
- EVALUATION_PROTOCOL
- IMPLEMENTATION_TOOLING

Prefer candidates that assign roles conservatively and correctly.

Penalize:
- TRAINING_DATA when the resource is really a source for constructing another artifact
- DATA_SOURCE when the resource is directly used to train the artifact
- IMPLEMENTATION_TOOLING for things that are merely optional recipe choices
- inflated CONCEPTUAL_FRAMEWORK assignments for concrete tooling or datasets

==================================================
VIII. EVIDENCE QUALITY
==================================================

A good candidate provides, for each final ingredient:
- one direct short quote from the target paper
- a quote that directly supports the ingredient
- evidence that matches the contribution/rationale

Penalize:
- vague evidence
- evidence that supports only a local implementation detail
- quotes that do not justify the claimed level of abstraction
- multiple stitched spans when one direct quote should suffice

==================================================
IX. HOW TO DECIDE BETWEEN CANDIDATES
==================================================

Use this ranking order:

1. Correct downstream-used claim selection
2. Correct claim splitting and abstraction level
3. Smallest structurally sufficient ingredient set
4. Correct exclusion of tempting non-ingredients
5. Correct canonical vs additional vs NONE decisions
6. Correct role assignments
7. Evidence quality and clarity
8. Overall readiness for formatter conversion

If one candidate is more detailed but another is more minimal and structurally faithful, prefer the more minimal and structurally faithful one.

==================================================
X. OUTPUT FORMAT
==================================================

Return valid JSON only with this schema:

{
  "selected_candidate_index": 0,
  "selected_candidate_id": "candidate_1",
  "selected_reason": "...",
  "candidate_scores": [
    {
      "candidate_id": "candidate_1",
      "candidate_index": 0,
      "score": 1,
      "assessment": "..."
    }
  ]
}

Scoring:
- Use integer scores from 1 to 10.
- 10 = best candidate by the above criteria.
- The selected candidate must have the highest score.
- In `selected_reason`, explain why it best matches the minimal structurally sufficient annotation objective.
- In each `assessment`, mention both strengths and weaknesses, especially over-decomposition, bad exclusions, or bad NONE choices.
"""

def judge_prompt(paper: Dict[str, Any], candidate_texts: list[str]) -> str:
    candidate_blocks = []
    for index, text in enumerate(candidate_texts):
        candidate_blocks.append(
            f"CANDIDATE {index + 1}: candidate_{index + 1}\n"
            + "-" * 60
            + "\n"
            + text.strip()
            + "\n"
        )
    candidates_section = "\n\n".join(candidate_blocks)

    return f"""Task: Compare candidate reasoning memos for a single target paper.

You will be given:
1. target paper metadata
2. the original extracted discovery claim
3. optional downstream usage evidence
4. multiple candidate reasoning outputs from pass 1

Choose the candidate that best matches the minimal structurally sufficient enabling-discovery annotation objective.

Important:
- Judge candidates using the full critic policy from the system instructions.
- Do not reward verbosity by itself.
- Do not reward confidence by itself.
- Do not reward richer implementation stories, more citations, or more operational detail unless they are structurally necessary.
- Prefer candidates that recover the smallest structurally sufficient ingredient set for each downstream-used claim.
- Prefer candidates that minimize formatter guesswork.

Apply these priorities in order:
1. Correct downstream-used claim selection
2. Correct claim splitting and abstraction level
3. Smallest structurally sufficient ingredient set
4. Correct exclusion of tempting non-ingredients
5. Correct canonical vs additional vs "__NONE__" grounding decisions
6. Correct role assignments
7. Evidence quality and clarity
8. Overall readiness for formatter conversion

Specific judging reminders:
- Prefer downstream-first claim identification over abstract-first or citation-first reasoning.
- Penalize candidates that annotate unused paper outputs as claims.
- Penalize candidates that oversplit one artifact into multiple claims without downstream justification.
- Penalize candidates that include one-task helpers, local preprocessing, balancing tricks, token/language-tag details, helper models for one small step, benchmark-use ingredients, model-training conveniences, or other non-structural details.
- For benchmark/dataset claims, prefer benchmark framing, source-task/source-data substrate, benchmark-wide annotation/curation/verification protocol, benchmark-defining task formulation/protocol, and any core conceptual trick or idea.
- For method/resource claims, prefer core architecture, core conceptual trick or idea, core objective(s), structurally necessary training data, and essential initialization/checkpoints.
- For finding claims, prefer the model/artifact type under study, task/benchmark substrate, evaluation/measurement protocol, and any core conceptual trick or idea needed to make the finding observable.
- Prefer "__NONE__" when an ingredient is composite, same-paper dominant, benchmark-wide/corpus-wide, or would be distorted by forcing one canonical prior study.
- Penalize candidates that force canonical studies where "__NONE__" is more faithful.
- Penalize candidates that choose famous or cited papers rather than the most structurally faithful grounding.
- Prefer role assignments that are conservative and faithful to the ingredient's function.

TARGET PAPER METADATA:
{compact_json(paper.get('paper_metadata') or {})}

ORIGINAL DISCOVERY CLAIM:
{paper.get("extracted_discovery_claim") or ""}

OPTIONAL DOWNSTREAM USAGE EVIDENCE:
{compact_json(paper.get("downstream_cluster_evidence") or [])}

CANDIDATE REASONING OUTPUTS:
{candidates_section}

Return only valid JSON with this exact schema:

{{
  "selected_candidate_index": 0,
  "selected_candidate_id": "candidate_1",
  "selected_reason": "...",
  "candidate_scores": [
    {{
      "candidate_id": "candidate_1",
      "candidate_index": 0,
      "score": 1,
      "assessment": "..."
    }}
  ]
}}

Scoring rules:
- Use integer scores from 1 to 10.
- 10 = best candidate by the minimal structurally sufficient annotation objective.
- The selected candidate must have the highest score.
- In "selected_reason", explain why it best matches the downstream-first, minimal-ingredient, structurally faithful objective.
- In each "assessment", mention both strengths and weaknesses, especially over-decomposition, poor exclusions, weak abstraction level, bad role choices, and bad canonical vs "__NONE__" decisions.

Return only valid JSON. Do not add any prose outside the JSON object.
"""

EXAMPLE_1 = """
# EXAMPLE: Pick-a-Pic / PickScore

## 1. Cluster Evidence

**The PickScore metric, a CLIP-based scoring function that predicts human preferences to evaluate and rank text-to-image generation models.**

The paper introduces two downstream-used contributions: PickScore and the Pick-a-Pic dataset. Downstream evidence shows separate reuse:

- PickScore is reused as an evaluation metric / reward model.
- Pick-a-Pic is reused as a dataset for training and preference tuning.

So the bundled contribution should be split into two atomic claims.

---

## 2. Claim Split Decision

### Claim C1
- **claim_id:** C1
- **artifact_type:** Tool
- **decision:** YES_SUFFICIENT
- **evidence_cluster_id:** C1

### Claim C2
- **claim_id:** C2
- **artifact_type:** Dataset
- **decision:** YES_SUFFICIENT
- **evidence_cluster_id:** C2

### Why this split is correct

The two artifacts have distinct downstream roles and distinct ingredient decompositions:

- **PickScore** is reused as a scoring / evaluation / reward model.
- **Pick-a-Pic** is reused as a preference dataset for training and benchmarking.

They should therefore be annotated separately.

---

## 3. Rewritten Claims

### Claim C1
- **claim_id:** C1
- **artifact_type:** Tool
- **rewritten_claim:** **"Tool: A human-preference prompt-conditioned image scorer, enabling automatic ranking of text-to-image model outputs**
- **why_this_is_atomic:** This claim isolates the scorer itself as the reused artifact. It does not bundle the dataset resource used to train it.
- **optional cluster_id:** C1
- **decision:** YES_SUFFICIENT

### Claim C2
- **claim_id:** C2
- **artifact_type:** Dataset
- **rewritten_claim:** **Dataset: An open prompt-conditioned human-preference dataset for generated images, enabling training and evaluation of preference-aligned text-to-image systems.**
- **why_this_is_atomic:** This claim isolates the dataset as the released reusable resource, separate from the scoring model trained on it.
- **optional cluster_id:** C2
- **decision:** YES_SUFFICIENT

---

## 4. Minimal Ingredient Set Per Claim

## Claim C1: PickScore

### Ingredient C1.I1
- **ingredient_id:** C1.I1
- **ingredient:** Pre-trained vision language model for prompt-conditioned image scoring
- **why_structurally_necessary:** PickScore is explicitly a CLIP-based scoring function. Without a pretrained joint text-image model, the scorer would not exist in its claimed form.
- **why_not_lower_level_substeps:** This should stay at the level of pretrained vision-language initialization, not be decomposed into separate text encoder, image encoder, or projection details.
- **why_not_adjacent_implementation_details:** Specific OpenCLIP variants are narrower implementation choices rather than the core structural dependency.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** Learning Transferable Visual Models From Natural Language Supervision
- **additional groundings if any:** none
- **role:** MODEL_INITIALIZATION
- **contribution:** Provides the pre-trained text and image representations that are fine-tuned into a human preference score.
- **rationale:** To score a generated image relative to a prompt, the function needs a pretrained joint text–image representation. CLIP is the cleanest representative study for such capability and is the direct architectural basis of PickScore.
- **evidence_span:** “PickScore follows the architecture of CLIP [12]”

### Ingredient C1.I2
- **ingredient_id:** C1.I2
- **ingredient:** Large-scale dataset of human-preference judgements over text-to-image outputs
- **why_structurally_necessary:** The scorer must learn which of two generated images humans prefer for a given prompt. Without this training signal, it would not become a human-preference metric.
- **why_not_lower_level_substeps:** This should remain a single training-data ingredient rather than being split into prompts, pairs, ties, logging, or collection mechanics.
- **why_not_adjacent_implementation_details:** Dataset construction details belong to the dataset claim, not the scorer’s minimal training-data dependency.
- **necessary:** true
- **from_prior_work:** false
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** none
- **role:** TRAINING_DATA
- **contribution:** Provides the pairwise user choices needed to fit the scorer.
- **rationale:** The model needs direct supervision about which of two generated images users prefer for a given prompt. This operative dataset is released in the target paper, so the right grounding is `NONE`.
- **evidence_span:** “We finetune CLIP-H [7] using our framework8 on the Pick-a-Pic training set.”

### Ingredient C1.I3
- **ingredient_id:** C1.I3
- **ingredient:** Preference-learning objective to model the reward
- **why_structurally_necessary:** PickScore is trained to assign higher scores to preferred images. Without a preference-learning objective, it would not become a ranking metric that evaluates outputs by predicted human choice.
- **why_not_lower_level_substeps:** This should remain one core-method ingredient rather than being decomposed into loss cases, tie handling, or optimization details.
- **why_not_adjacent_implementation_details:** Training schedule and optimizer are implementation details; the structural dependency is the reward-model-style objective itself.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** Training language models to follow instructions with human feedback
- **additional groundings if any:** none
- **role:** CORE_METHOD
- **contribution:** Learns scores whose differences reflect human preference between candidate images for the same prompt.
- **rationale:** Without that preference-learning step, the system would not become a ranking metric that can evaluate outputs. The paper explicitly grounds this in a variant of InstructGPT’s reward model objective.
- **evidence_span:** “We train the PickScore scoring function over Pick-a-Pic by combining a CLIP-style model with a variant of In-structGPT’s reward model objective”

---

## Claim C2: Pick-a-Pic

### Ingredient C2.I1
- **ingredient_id:** C2.I1
- **ingredient:** Framework for pairwise human preference collection using prompt-conditioned image comparisons
- **why_structurally_necessary:** The defining property of the dataset is that it records human preferences over generated image pairs for prompts. Without a collection framework that elicits those comparisons, the dataset would not exist in its claimed form.
- **why_not_lower_level_substeps:** This should remain one collection-framework ingredient rather than being split into web app interface details, session flow, or tie-button mechanics.
- **why_not_adjacent_implementation_details:** The structural point is the pairwise human-preference collection framework, not the exact UI implementation.
- **necessary:** true
- **from_prior_work:** false
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** none
- **role:** EVALUATION_PROTOCOL
- **contribution:** Provides the framework to collect human preferences on a large scale.
- **rationale:** A framework is necessary to collect human preference choices at scale. The authors implement a web app that lets users generate images and specify preferences, and that workflow is a defining part of the dataset rather than a prior reusable study.
- **evidence_span:** “To address this issue, we create a web app that enables text-to-image users to generate images and specify their preferences.”

### Ingredient C2.I2
- **ingredient_id:** C2.I2
- **ingredient:** Diffusion models for image generation from text
- **why_structurally_necessary:** The dataset consists of preferences over generated images. Without text-to-image generators, there would be no candidate images to compare and annotate.
- **why_not_lower_level_substeps:** This should remain at the level of text-to-image diffusion generation, not be split into individual backbone variants or guidance settings.
- **why_not_adjacent_implementation_details:** The exact backbone mix is less important than the general generation capability used to produce the images.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** High-Resolution Image Synthesis with Latent Diffusion Models
- **additional groundings if any:** none
- **role:** CORE_METHOD
- **contribution:** Provides the diffusion models needed to generate the images in the dataset.
- **rationale:** The dataset requires text-to-image generation backbones to convert user prompts into candidate images for preference annotation. Latent Diffusion Models is the cleanest representative prior study for that capability.
- **evidence_span:** “The images in the dataset were generated by employing multiple backbone models, namely, Stable Diffusion 2.1, Dreamlike Photoreal 2.0, and Stable Diffusion XL variants”

### Ingredient C2.I3
- **ingredient_id:** C2.I3
- **ingredient:** Quality-control and preprocessing protocol for reliable open preference-data collection
- **why_structurally_necessary:** The dataset is released as an open, reusable human-preference resource. Without moderation, filtering, and preprocessing, it would lose one of its defining properties: being a reliable preference dataset rather than a raw interaction log.
- **why_not_lower_level_substeps:** This should remain one protocol-level ingredient rather than being split into NSFW phrase lists, account filtering, banning, or split construction separately.
- **why_not_adjacent_implementation_details:** These local measures are all parts of the same broader quality-control workflow and should not be promoted to standalone ingredients.
- **necessary:** true
- **from_prior_work:** false
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** none
- **role:** EVALUATION_PROTOCOL
- **contribution:** Provides the moderation, filtering, and preprocessing workflow that makes the open preference dataset reliable and usable.
- **rationale:** Constructing an open preference dataset requires more than collecting raw comparisons. The paper includes moderation, harmful-content filtering, and preprocessing decisions that are structurally important to the quality of the released resource. This is paper-specific and therefore maps cleanly to `NONE`.
- **evidence_span:** “we closely monitor user activity logs and take action to ban users”

---

## 5. Excluded Tempting Non-Ingredients

### Claim C1 exclusions
- **Pick-a-Pic web app**
  - Excluded because for the scorer claim, the structural dependency is the availability of human preference training data, not the full data-collection workflow.

- **Preference-based evaluation setup for ranking candidate images by predicted human choice**
  - Excluded because this is already captured by the rewritten claim and by the combination of human preference training data plus the reward-model objective. Adding it would double count the same functional role.

### Claim C2 exclusions
- **List of NSFW phrases**
  - Excluded because it is only one local component of the broader quality-control and preprocessing protocol, not a standalone structural ingredient.

- **Specific diffusion backbone variants**
  - Excluded because the structural dependency is text-to-image diffusion generation as a class, not each particular model instance.

- **PickScore**
  - Excluded because the dataset is a prerequisite for the tool, not the reverse.

---

## 6. Candidate Grounding Decisions

## Claim C1: PickScore

### Ingredient C1.I1
- **Candidate:** Learning Transferable Visual Models From Natural Language Supervision
- **decision:** accepted_canonical
- **why:** Provides the pretrained joint text-image architecture used as the basis for prompt-conditioned scoring.

### Ingredient C1.I2
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The operative human preference dataset is released in the target paper itself.

### Ingredient C1.I3
- **Candidate:** Training language models to follow instructions with human feedback
- **decision:** accepted_canonical
- **why:** Provides the reward-model-style objective explicitly adapted by the paper.

## Claim C2: Pick-a-Pic

### Ingredient C2.I1
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The pairwise collection framework is implemented in the paper and not cleanly inherited from one prior study.

### Ingredient C2.I2
- **Candidate:** High-Resolution Image Synthesis with Latent Diffusion Models
- **decision:** accepted_canonical
- **why:** Cleanest representative of the text-to-image diffusion generation capability used to create the candidate images.

### Ingredient C2.I3
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The quality-control and preprocessing workflow is a paper-specific protocol rather than one canonical prior study.

---

## 7. Final Grounding Decisions

## Claim C1
- **C1.I1** → Learning Transferable Visual Models From Natural Language Supervision → MODEL_INITIALIZATION
- **C1.I2** → NONE → TRAINING_DATA
- **C1.I3** → Training language models to follow instructions with human feedback → CORE_METHOD

## Claim C2
- **C2.I1** → NONE → EVALUATION_PROTOCOL
- **C2.I2** → High-Resolution Image Synthesis with Latent Diffusion Models → CORE_METHOD
- **C2.I3** → NONE → EVALUATION_PROTOCOL

---

## 8. Final Rationales And Evidence

## Claim C1: PickScore

### C1.I1
- **ingredient:** Pre-trained vision language model for prompt-conditioned image scoring
- **canonical study:** Learning Transferable Visual Models From Natural Language Supervision
- **role:** MODEL_INITIALIZATION
- **contribution:** Provides the pre-trained text and image representations that are fine-tuned into a human preference score.
- **rationale:** To score a generated image relative to a prompt, the function needs a pretrained joint text–image representation. CLIP is the cleanest representative study for such capability and is the direct architectural basis of PickScore.
- **evidence_span:** “PickScore follows the architecture of CLIP [12]; given a prompt x and an image y, our scoring function s computes a real number by representing x using a transformer text encoder and y using a transformer image encoder as d-dimensional vectors, and returning their inner product”

### C1.I2
- **ingredient:** Large-scale dataset of human-preference judgements over text-to-image outputs
- **canonical study:** NONE
- **role:** TRAINING_DATA
- **contribution:** Provides the pairwise user choices needed to fit the scorer.
- **rationale:** The model needs direct supervision about which of two generated images users prefer for a given prompt. This operative dataset is released in the target paper, so the correct grounding is `NONE`.
- **evidence_span:** “We finetune CLIP-H [7] using our framework8 on the Pick-a-Pic training set.”

### C1.I3
- **ingredient:** Preference-learning objective to model the reward
- **canonical study:** Training language models to follow instructions with human feedback
- **role:** CORE_METHOD
- **contribution:** Learns scores whose differences reflect human preference between candidate images for the same prompt.
- **rationale:** Without that preference-learning step, the system would not become a ranking metric that can evaluate outputs. The paper explicitly grounds this in a variant of InstructGPT’s reward model objective.
- **evidence_span:** “We train the PickScore scoring function over Pick-a-Pic by combining a CLIP-style model with a variant of In-structGPT’s reward model objective [10].”

## Claim C2: Pick-a-Pic

### C2.I1
- **ingredient:** Framework for pairwise human preference collection using prompt-conditioned image comparisons
- **canonical study:** NONE
- **role:** EVALUATION_PROTOCOL
- **contribution:** Provides the framework to collect human preferences on a large scale.
- **rationale:** A framework is necessary to collect human preference choices at a large scale. The authors implement a web app to collect this, and that workflow is a defining part of the dataset.
- **evidence_span:** “To address this issue, we create a web app that enables text-to-image users to generate images and specify their preferences.”

### C2.I2
- **ingredient:** Diffusion models for image generation from text
- **canonical study:** High-Resolution Image Synthesis with Latent Diffusion Models
- **role:** CORE_METHOD
- **contribution:** Provides the diffusion models needed to generate the images in the dataset.
- **rationale:** The dataset requires text-to-image generation backbones to convert prompts into candidate images for annotation. Latent Diffusion Models is the cleanest representative prior study for that capability.
- **evidence_span:** “The images in the dataset were generated by employing multiple backbone models, namely, Stable Diffusion 2.1, Dreamlike Photoreal 2.0 5 , and Stable Diffusion XL variants [13] while sampling different classiﬁer-free guidance scale values [6].”

### C2.I3
- **ingredient:** Quality-control and preprocessing protocol for reliable open preference-data collection
- **canonical study:** NONE
- **role:** EVALUATION_PROTOCOL
- **contribution:** Provides the moderation, filtering, and preprocessing workflow that makes the open preference dataset reliable and usable.
- **rationale:** Constructing an open preference dataset requires more than collecting raw interactions. The released resource depends on moderation, filtering, and preprocessing steps that preserve data quality and usability. Because this workflow is paper-specific, `NONE` is the right grounding.
- **evidence_span:** “we closely monitor user activity logs and take action to ban users”"""

EXAMPLE_2 = """
# EXAMPLE: Offline RL bottleneck finding

## 1. Cluster Evidence

**Finding that policy extraction and test-time generalization (rather than just value learning) are the main bottlenecks in offline RL, enabling improved algorithm design via better policy extraction objectives.**

This original claim identifies a real downstream-used empirical contribution. It should remain a **single finding claim** rather than being split into separate findings about value learning, policy extraction, and test-time generalization, because the paper’s contribution is one integrated diagnosis of what limits offline RL performance in practice.

---

## 2. Claim Split Decision

### Claim C1
- **claim_id:** C1
- **artifact_type:** Finding
- **decision:** YES_SUFFICIENT
- **evidence_cluster_id:** C1

### Why this split is correct

The downstream reuse is of a single empirical conclusion: that offline RL performance is often limited more by **policy extraction and deployment-time generalization** than by value learning alone. The later algorithm-design implication is a consequence of that diagnosis, not a separate discovery claim.

---

## 3. Rewritten Claims

### Claim C1
- **claim_id:** C1
- **artifact_type:** Finding
- **rewritten_claim:** **Finding: In offline reinforcement learning, policy extraction and test-time policy generalization often limit performance more than value learning alone, making policy-extraction objective choice a primary determinant of final returns.**
- **why_this_is_atomic:** This claim isolates one empirical result about the source of performance bottlenecks in offline RL. It does not split off the downstream design recommendation as a separate claim.
- **optional cluster_id:** C1
- **decision:** YES_SUFFICIENT

---

## 4. Minimal Ingredient Set Per Claim

## Claim C1: Offline RL bottleneck finding

### Ingredient C1.I1
- **ingredient_id:** C1.I1
- **ingredient:** Offline RL methods with decoupled value learning and policy extraction phases
- **why_structurally_necessary:** The finding depends on separating the quality of the learned critic from the quality of the extracted policy. Without a decoupled setup, the paper could not attribute performance differences to policy extraction rather than entangled actor-critic training.
- **why_not_lower_level_substeps:** This should remain one high-level ingredient about the decoupled offline RL setup, not be split into each individual algorithm separately.
- **why_not_adjacent_implementation_details:** Specific architectural or optimization choices inside one method are not the point; the structural dependency is the existence of a method family where value learning and policy extraction can be independently analyzed.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** Offline RL Without Off-Policy Evaluation; Offline Reinforcement Learning with Implicit Q-Learning; Contrastive Learning as Goal-Conditioned Reinforcement Learning
- **role:** CONCEPTUAL_FRAMEWORK
- **contribution:** Provides the offline RL setup in which value learning is trained independently of policy extraction, enabling the paper’s bottleneck analysis.
- **rationale:** To show that policy extraction rather than value learning is often the limiting factor, the paper must analyze settings where the critic can be held fixed while different extraction procedures are compared. That dependence is structurally on a family of decoupled offline RL methods rather than on one single canonical paper, so `NONE` is the faithful canonical choice and the relevant prior methods are better represented as additional studies.
- **evidence_span:** “we focus on offline RL methods with decoupled value and policy training phases”

### Ingredient C1.I2
- **ingredient_id:** C1.I2
- **ingredient:** Behavior-constrained policy-gradient extraction objectives for policies learned from fixed critics
- **why_structurally_necessary:** The finding is not only that some extraction methods underperform, but that behavior-constrained policy-gradient extraction often performs better and scales better than common alternatives under the same learned critic.
- **why_not_lower_level_substeps:** This should remain one extraction-family ingredient rather than being split into every equation or policy-update detail.
- **why_not_adjacent_implementation_details:** Hyperparameters and optimizer details are irrelevant here. The structural ingredient is the extraction objective family itself.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** A Minimalist Approach to Offline Reinforcement Learning
- **additional groundings if any:** Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning; Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling
- **role:** CORE_METHOD
- **contribution:** Provides the representative behavior-regularized policy-gradient extraction objective that the paper identifies as stronger and more scalable than common alternatives.
- **rationale:** The paper’s empirical conclusion depends on contrasting extraction families and finding that DDPG+BC-style extraction often outperforms widely used value-weighted or sampling-based alternatives. DDPG+BC is the clearest representative study for that stronger extraction family, while AWR and SfBC-style methods serve as additional comparison groundings rather than the canonical representative.
- **evidence_span:** “switching to behavior-constrained policy gradient objectives (e.g., DDPG+BC) often leads to substantial improvements”

### Ingredient C1.I3
- **ingredient_id:** C1.I3
- **ingredient:** Comparative protocol that evaluates extraction objectives across data regimes and deployment generalization settings
- **why_structurally_necessary:** The claim is a comparative empirical finding about bottlenecks and scaling behavior. It requires an analysis protocol that systematically varies data properties and observes how extraction choices affect final performance and generalization.
- **why_not_lower_level_substeps:** This should remain one protocol-level ingredient rather than being split into separate dataset-size sweeps, quality sweeps, and metric definitions.
- **why_not_adjacent_implementation_details:** Individual plots, metrics, or visualization choices are not separate ingredients; they are local parts of the broader comparative protocol.
- **necessary:** true
- **from_prior_work:** false
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** none
- **role:** EVALUATION_PROTOCOL
- **contribution:** Provides the systematic empirical protocol that isolates policy extraction and generalization bottlenecks across environments and data regimes.
- **rationale:** A finding about what the main bottleneck is cannot be established from one aggregate return table alone. The paper needs a structured comparative protocol that varies data size, quality, and coverage while comparing extraction methods under shared critics. That protocol is built in the target paper itself, so `NONE` is the correct grounding.
- **evidence_span:** “We use data size, quality, and coverage as levers for systematically controlling their impacts”

---

## 5. Excluded Tempting Non-Ingredients

### Claim C1 exclusions
- **Offline RL setting with fixed logged data and no online interaction**
  - Excluded because it is too broad and field-level. The more precise structural ingredient is the **decoupled offline RL setup** that lets the paper isolate value learning from extraction.

- **Advantage-weighted regression as its own standalone ingredient**
  - Excluded as a separate final ingredient because it is better represented as an **additional grounding** under the broader comparative extraction-objective ingredient, rather than being promoted to its own parallel ingredient.

- **Standard offline RL benchmark datasets as a separate ingredient**
  - Excluded because the finding depends more centrally on the paper’s broader comparative protocol than on one benchmark substrate alone. The datasets are part of that empirical protocol rather than a standalone higher-level ingredient.

- **Test-time generalization as a separate conceptual ingredient**
  - Excluded because it is already captured within the rewritten claim and within the comparative evaluation protocol ingredient. Keeping it separate would over-decompose the finding.

---

## 6. Candidate Grounding Decisions

## Claim C1: Offline RL bottleneck finding

### Ingredient C1.I1
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The finding relies on a family of decoupled offline RL methods rather than one single representative study.
- **Candidate:** Offline RL Without Off-Policy Evaluation
- **decision:** accepted_additional
- **why:** Representative prior decoupled offline RL method explicitly listed by the paper.
- **Candidate:** Offline Reinforcement Learning with Implicit Q-Learning
- **decision:** accepted_additional
- **why:** Another representative decoupled method used to motivate the analysis setup.
- **Candidate:** Contrastive Learning as Goal-Conditioned Reinforcement Learning
- **decision:** accepted_additional
- **why:** Included as a representative decoupled method family member rather than the single canonical grounding.

### Ingredient C1.I2
- **Candidate:** A Minimalist Approach to Offline Reinforcement Learning
- **decision:** accepted_canonical
- **why:** Cleanest representative study for behavior-constrained policy-gradient extraction, which the paper identifies as the stronger extraction family.
- **Candidate:** Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning
- **decision:** accepted_additional
- **why:** Necessary comparison grounding for the weaker value-weighted extraction family contrasted in the finding.
- **Candidate:** Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling
- **decision:** accepted_additional
- **why:** Relevant additional comparison study for alternative policy extraction objectives.

### Ingredient C1.I3
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The comparative data-scaling and deployment-generalization protocol is constructed in the target paper rather than inherited from one prior study.

---

## 7. Final Grounding Decisions

## Claim C1
- **C1.I1** → NONE → CONCEPTUAL_FRAMEWORK  
  - additional: Offline RL Without Off-Policy Evaluation; Offline Reinforcement Learning with Implicit Q-Learning; Contrastive Learning as Goal-Conditioned Reinforcement Learning
- **C1.I2** → A Minimalist Approach to Offline Reinforcement Learning → CORE_METHOD  
  - additional: Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning; Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling
- **C1.I3** → NONE → EVALUATION_PROTOCOL

---

## 8. Final Rationales And Evidence

## Claim C1: Offline RL bottleneck finding

### C1.I1
- **ingredient:** Offline RL methods with decoupled value learning and policy extraction phases
- **canonical study:** NONE
- **additional studies:** Offline RL Without Off-Policy Evaluation; Offline Reinforcement Learning with Implicit Q-Learning; Contrastive Learning as Goal-Conditioned Reinforcement Learning
- **role:** CONCEPTUAL_FRAMEWORK
- **contribution:** Provides the offline RL setup in which value learning is trained independently of policy extraction, enabling the paper’s bottleneck analysis.
- **rationale:** To show that policy extraction rather than value learning is often the limiting factor, the paper must analyze settings where the critic can be held fixed while different extraction procedures are compared. That dependence is structurally on a family of decoupled offline RL methods rather than on one single canonical paper, so `NONE` is the faithful canonical choice and the relevant prior methods are better represented as additional studies.
- **evidence_span:** “we focus on offline RL methods with decoupled value and policy training phases”

### C1.I2
- **ingredient:** Behavior-constrained policy-gradient extraction objectives for policies learned from fixed critics
- **canonical study:** A Minimalist Approach to Offline Reinforcement Learning
- **additional studies:** Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning; Offline Reinforcement Learning via High-Fidelity Generative Behavior Modeling
- **role:** CORE_METHOD
- **contribution:** Provides the representative behavior-regularized policy-gradient extraction objective that the paper identifies as stronger and more scalable than common alternatives.
- **rationale:** The paper’s empirical conclusion depends on contrasting extraction families and finding that DDPG+BC-style extraction often outperforms widely used value-weighted or sampling-based alternatives. DDPG+BC is the clearest representative study for that stronger extraction family, while AWR and SfBC-style methods serve as additional comparison groundings rather than the canonical representative.
- **evidence_span:** “switching to behavior-constrained policy gradient objectives (e.g., DDPG+BC) often leads to substantial improvements”

### C1.I3
- **ingredient:** Comparative protocol that evaluates extraction objectives across data regimes and deployment generalization settings
- **canonical study:** NONE
- **role:** EVALUATION_PROTOCOL
- **contribution:** Provides the systematic empirical protocol that isolates policy extraction and generalization bottlenecks across environments and data regimes.
- **rationale:** A finding about what the main bottleneck is cannot be established from one aggregate return table alone. The paper needs a structured comparative protocol that varies data size, quality, and coverage while comparing extraction methods under shared critics. That protocol is built in the target paper itself, so `NONE` is the correct grounding.
- **evidence_span:** “We use data size, quality, and coverage as levers for systematically controlling their impacts”
"""


EXAMPLE_3 = """
# EXAMPLE: IndicXTREME / IndicBERT

## 1. Cluster Evidence

Based on downstream cluster contributions, the paper introduces two distinct downstream-used contributions:

- a **benchmark** for multilingual zero-shot evaluation on Indic languages
- a **pretrained multilingual encoder** for Indic NLU

Downstream evidence shows these are reused differently and should therefore be split into two atomic claims.

---

## 2. Claim Split Decision

### Claim C1
- **claim_id:** C1
- **artifact_type:** Benchmark
- **decision:** YES_SUFFICIENT
- **evidence_cluster_id:** C5

### Claim C2
- **claim_id:** C2
- **artifact_type:** Resource
- **decision:** YES_SUFFICIENT
- **evidence_cluster_id:** C2

### Why this split is correct

The paper contributes two separate artifacts with distinct downstream roles:

- **IndicXTREME** is reused as a multilingual benchmark for zero-shot evaluation.
- **IndicBERT v2** is reused as a pretrained model for transfer and downstream fine-tuning.

These have different structural ingredients, so they should not be annotated as one merged claim.

---

## 3. Rewritten Claims

### Claim C1
- **claim_id:** C1
- **artifact_type:** Benchmark
- **rewritten_claim:** **Benchmark: A multilingual Indic-language evaluation suite, enabling standardized zero-shot assessment of pretrained multilingual models across diverse NLU tasks.**
- **why_this_is_atomic:** This claim isolates the benchmark artifact and its evaluation role, without bundling the pretrained model.
- **optional cluster_id:** C5
- **decision:** YES_SUFFICIENT

### Claim C2
- **claim_id:** C2
- **artifact_type:** Resource
- **rewritten_claim:** **Resource: A pretrained multilingual encoder for Indic languages, enabling zero-shot transfer and downstream adaptation on Indic NLU tasks.**
- **why_this_is_atomic:** This claim isolates the pretrained encoder as the reused artifact, separate from the benchmark used to evaluate such models.
- **optional cluster_id:** C2
- **decision:** YES_SUFFICIENT

---

## 4. Minimal Ingredient Set Per Claim

## Claim C1: Indic benchmark

### Ingredient C1.I1
- **ingredient_id:** C1.I1
- **ingredient:** Multilingual, multi-task, zero-shot evaluation framework for cross-lingual NLU benchmarking
- **why_structurally_necessary:** The benchmark is explicitly framed as a multilingual zero-shot evaluation suite across tasks and languages. Without this benchmark framing, it would not exist in its claimed form as a standardized cross-lingual evaluation resource.
- **why_not_lower_level_substeps:** This should remain one benchmark-paradigm ingredient rather than being split into separate benchmark papers or separate task families.
- **why_not_adjacent_implementation_details:** The structural dependency is the zero-shot multilingual benchmark framing itself, not local benchmark formatting or reporting choices.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization
- **additional groundings if any:** XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation; XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation
- **role:** CONCEPTUAL_FRAMEWORK
- **contribution:** Provides the multilingual zero-shot benchmark framing that the paper adapts to Indic languages.
- **rationale:** To construct a benchmark whose point is standardized zero-shot evaluation across many languages and tasks, the paper depends on an existing multilingual benchmark paradigm. XTREME is the cleanest canonical grounding because it most directly established that framing; XTREME-R and XGLUE are relevant supporting studies but are less direct representatives.
- **evidence_span:** “aims to test the multilingual zero-shot capabilities of pretrained language models”

### Ingredient C1.I2
- **ingredient_id:** C1.I2
- **ingredient:** NLU task datasets and source resources used to assemble the benchmark
- **why_structurally_necessary:** A multi-task benchmark requires the actual upstream task substrates from which its evaluation sets are translated, adapted, corrected, or incorporated.
- **why_not_lower_level_substeps:** This should remain one composite data-source ingredient rather than being split into separate ingredients for each task dataset.
- **why_not_adjacent_implementation_details:** The key dependency is the heterogeneous task substrate itself, not each local adaptation step used for one task.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning; XNLI: Evaluating Cross-lingual Sentence Representations; IndicXNLI: Evaluating Multilingual Inference for Indian Languages; Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages; The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation; MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
- **role:** DATA_SOURCE
- **contribution:** Supplies the task definitions and raw/source evaluation material that the benchmark translates, subsets, verifies, or incorporates.
- **rationale:** Constructing a multi-task benchmark requires the underlying task substrates that define its evaluation sets. Because these substrates are inherently composite across multiple tasks and datasets, they do not map cleanly to a single prior study, making `NONE` the appropriate canonical choice, with the task-specific sources listed as additional studies.
- **evidence_span:** “We manually translate the COPA test set into 18 Indic languages to create IndicCOPA.”

### Ingredient C1.I3
- **ingredient_id:** C1.I3
- **ingredient:** Human-supervised multilingual curation, translation, and verification protocol
- **why_structurally_necessary:** Human supervision is one of the benchmark’s defining claimed properties. Without it, the benchmark would not exist in the same form as a human-validated evaluation resource.
- **why_not_lower_level_substeps:** This should remain one protocol-level ingredient rather than being split into manual translation, verification, correction, and annotation as separate ingredients.
- **why_not_adjacent_implementation_details:** The structural dependency is the benchmark-wide human-supervision workflow, not each local annotation step.
- **necessary:** true
- **from_prior_work:** false
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** none
- **role:** EVALUATION_PROTOCOL
- **contribution:** Provides the manual benchmark-construction process ensuring that all included evaluation sets are created, translated, edited, or verified by humans.
- **rationale:** This ingredient is necessary because the benchmark’s defining property is that all evaluation sets are created or verified under human supervision. Removing it would fundamentally change the nature of the benchmark, reducing it to a weaker, non-human-validated collection of datasets. Because this workflow is benchmark-specific and realized in the target paper, `NONE` is the correct grounding.
- **evidence_span:** “ALL the evaluation sets included in IndicXTREME were created with human supervision”

---

## Claim C2: Indic multilingual encoder

### Ingredient C2.I1
- **ingredient_id:** C2.I1
- **ingredient:** Transformer-based masked language modeling architecture
- **why_structurally_necessary:** The resource is explicitly a BERT-style pretrained encoder. Without the transformer MLM architecture, it would not exist in its claimed form.
- **why_not_lower_level_substeps:** This should remain one architecture-level ingredient rather than being split into encoder layers, attention heads, or hyperparameter choices.
- **why_not_adjacent_implementation_details:** Model size, vocabulary size, and optimizer settings are implementation details; the structural dependency is the BERT-style MLM architecture itself.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **additional groundings if any:** none
- **role:** CORE_METHOD
- **contribution:** Provides the transformer encoder architecture and masked language modeling objective used to pretrain the model.
- **rationale:** To build a multilingual BERT-style encoder, the paper requires a transformer architecture trained with masked language modeling. BERT is the clearest and most faithful canonical grounding for that dependency.
- **evidence_span:** “We use the default hyperparameters of BERT-Base”

### Ingredient C2.I2
- **ingredient_id:** C2.I2
- **ingredient:** Translation Language Modeling objective for cross-lingual alignment
- **why_structurally_necessary:** The paper explicitly uses TLM as a second core objective to improve cross-lingual transfer by aligning languages through parallel data.
- **why_not_lower_level_substeps:** This should remain one objective-level ingredient rather than being split into masking details, parallel-pair formatting, or loss implementation details.
- **why_not_adjacent_implementation_details:** The structural dependency is the TLM objective itself, not specific data-preparation or training-engineering choices around it.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** Cross-lingual Language Model Pretraining
- **additional groundings if any:** none
- **role:** CORE_METHOD
- **contribution:** Provides the cross-lingual pretraining objective used to align representations across languages.
- **rationale:** The model is not only trained with MLM; it explicitly incorporates TLM as a core objective for cross-lingual alignment. That objective maps cleanly to XLM as the canonical prior study.
- **evidence_span:** “Translation Language Modeling (Conneau and Lample, 2019, TLM)”

### Ingredient C2.I3
- **ingredient_id:** C2.I3
- **ingredient:** Large-scale Indic-language pretraining text
- **why_structurally_necessary:** A multilingual encoder for Indic languages cannot be pretrained in its claimed form without a large-scale Indic-language text substrate.
- **why_not_lower_level_substeps:** This should remain one training-data ingredient rather than being split into IndicCorp, Wikipedia, OSCAR, Samanantar-derived text, or synthetic translations separately.
- **why_not_adjacent_implementation_details:** Specific corpus components and augmentation choices are subordinate pieces of the broader pretraining-text ingredient and should not be promoted to standalone structural ingredients.
- **necessary:** true
- **from_prior_work:** partially
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages; Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages
- **role:** TRAINING_DATA
- **contribution:** Provides the large-scale Indic-language text substrate used to pretrain the multilingual encoder.
- **rationale:** Recreating the model requires the actual large-scale Indic-language training text. But the operative pretraining text is structurally composite: it is dominated by corpus material created or assembled in the target paper, while also incorporating prior resources such as IndicCorp lineage and Samanantar-derived text. Because no single prior study cleanly represents that full training substrate, `NONE` is the correct canonical choice, with prior contributing resources listed as additional studies.
- **evidence_span:** “we merge data from IndicCorp v2 with Indic language data from Wikipedia and OSCAR”

---

## 5. Excluded Tempting Non-Ingredients

### Claim C1 exclusions
- **Machine translation model for auto-translating some source datasets**
  - Excluded because it is not structural to the benchmark as a whole and is only used to help construct a small part of one task family rather than defining the benchmark itself.

- **English training datasets used for zero-shot evaluation**
  - Excluded because these are used to apply the benchmark in evaluation setups, not to construct the benchmark artifact itself.

- **Individual task datasets as separate top-level ingredients**
  - Excluded because the benchmark’s task substrate is better represented as one composite DATA_SOURCE ingredient rather than over-decomposed into one ingredient per task.

### Claim C2 exclusions
- **IndicTrans as a standalone ingredient**
  - Excluded because it is a helper tool for generating some synthetic parallel data, not a core structural ingredient of the model in its claimed form.

- **Temperature-based upsampling / balancing tricks**
  - Excluded because these are recipe-level training choices rather than artifact-defining ingredients.

- **Language ID tokens / tokenizer / WordPiece design**
  - Excluded because these are implementation-level configuration details, not part of the minimal structurally sufficient ingredient set.

- **Samanantar as the canonical grounding for the full training-text ingredient**
  - Excluded because it represents only one smaller component of the broader pretraining substrate; assigning it as canonical would distort the true structure of the ingredient.

---

## 6. Candidate Grounding Decisions

## Claim C1: Indic benchmark

### Ingredient C1.I1
- **Candidate:** XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization
- **decision:** accepted_canonical
- **why:** Cleanest representative study for the multilingual multi-task zero-shot benchmark framing adapted here.
- **Candidate:** XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation
- **decision:** accepted_additional
- **why:** Important related multilingual benchmark extension, but not the clearest canonical origin.
- **Candidate:** XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation
- **decision:** accepted_additional
- **why:** Relevant supporting multilingual benchmark study, but less direct as the primary framing.

### Ingredient C1.I2
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The task substrate is structurally composite across multiple datasets and source resources.
- **Candidate:** Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning
- **decision:** accepted_additional
- **why:** Source task for IndicCOPA.
- **Candidate:** XNLI: Evaluating Cross-lingual Sentence Representations
- **decision:** accepted_additional
- **why:** Important upstream NLI source task.
- **Candidate:** IndicXNLI: Evaluating Multilingual Inference for Indian Languages
- **decision:** accepted_additional
- **why:** Important Indic-specific benchmark source.
- **Candidate:** Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages
- **decision:** accepted_additional
- **why:** Upstream NER task resource.
- **Candidate:** The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation
- **decision:** accepted_additional
- **why:** Upstream retrieval/evaluation source resource.
- **Candidate:** MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
- **decision:** accepted_additional
- **why:** Upstream intent/slot resource.

### Ingredient C1.I3
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The human-supervised curation and verification workflow is benchmark-specific and realized in the target paper.

## Claim C2: Indic multilingual encoder

### Ingredient C2.I1
- **Candidate:** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **decision:** accepted_canonical
- **why:** Cleanest representative study for the transformer MLM architecture used by the model.

### Ingredient C2.I2
- **Candidate:** Cross-lingual Language Model Pretraining
- **decision:** accepted_canonical
- **why:** Cleanest representative study for the TLM objective explicitly used by the paper.

### Ingredient C2.I3
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The pretraining text is dominated by same-paper corpus construction and is structurally composite across several sources.
- **Candidate:** IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages
- **decision:** accepted_additional
- **why:** Important prior Indic corpus lineage that the paper extends.
- **Candidate:** Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages
- **decision:** accepted_additional
- **why:** Important contributing prior text source, but not representative enough to be canonical for the full pretraining-text ingredient.

---

## 7. Final Grounding Decisions

## Claim C1
- **C1.I1** → XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization → CONCEPTUAL_FRAMEWORK  
  - additional: XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation; XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation
- **C1.I2** → NONE → DATA_SOURCE  
  - additional: Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning; XNLI: Evaluating Cross-lingual Sentence Representations; IndicXNLI: Evaluating Multilingual Inference for Indian Languages; Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages; The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation; MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
- **C1.I3** → NONE → EVALUATION_PROTOCOL

## Claim C2
- **C2.I1** → BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding → CORE_METHOD
- **C2.I2** → Cross-lingual Language Model Pretraining → CORE_METHOD
- **C2.I3** → NONE → TRAINING_DATA  
  - additional: IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages; Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages

---

## 8. Final Rationales And Evidence

## Claim C1: Indic benchmark

### C1.I1
- **ingredient:** Multilingual, multi-task, zero-shot evaluation framework for cross-lingual NLU benchmarking
- **canonical study:** XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalization
- **additional studies:** XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation; XGLUE: A New Benchmark Dataset for Cross-lingual Pre-training, Understanding and Generation
- **role:** CONCEPTUAL_FRAMEWORK
- **contribution:** Provides the multilingual zero-shot benchmark framing that the paper adapts to Indic languages.
- **rationale:** To construct a benchmark whose point is standardized zero-shot evaluation across many languages and tasks, the paper depends on an existing multilingual benchmark paradigm. XTREME is the cleanest canonical grounding because it most directly established that framing; XTREME-R and XGLUE are relevant supporting studies but are less direct representatives.
- **evidence_span:** “aims to test the multilingual zero-shot capabilities of pretrained language models”

### C1.I2
- **ingredient:** NLU task datasets and source resources used to assemble the benchmark
- **canonical study:** NONE
- **additional studies:** Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning; XNLI: Evaluating Cross-lingual Sentence Representations; IndicXNLI: Evaluating Multilingual Inference for Indian Languages; Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages; The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation; MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages
- **role:** DATA_SOURCE
- **contribution:** Supplies the task definitions and raw/source evaluation material that the benchmark translates, subsets, verifies, or incorporates.
- **rationale:** Constructing a multi-task benchmark requires the underlying task substrates that define its evaluation sets. Because these substrates are inherently composite across multiple tasks and datasets, they do not map cleanly to a single prior study, making `NONE` the appropriate canonical choice, with the task-specific sources listed as additional studies.
- **evidence_span:** “We manually translate the COPA test set into 18 Indic languages to create IndicCOPA.”

### C1.I3
- **ingredient:** Human-supervised multilingual curation, translation, and verification protocol
- **canonical study:** NONE
- **role:** EVALUATION_PROTOCOL
- **contribution:** Provides the manual benchmark-construction process ensuring that all included evaluation sets are created, translated, edited, or verified by humans.
- **rationale:** This ingredient is necessary because the benchmark’s defining property is that all evaluation sets are created or verified under human supervision. Removing it would fundamentally change the nature of the benchmark, reducing it to a weaker, non-human-validated collection of datasets. Because this workflow is benchmark-specific and realized in the target paper, `NONE` is the correct grounding.
- **evidence_span:** “ALL the evaluation sets included in IndicXTREME were created with human supervision”

## Claim C2: Indic multilingual encoder

### C2.I1
- **ingredient:** Transformer-based masked language modeling architecture
- **canonical study:** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **role:** CORE_METHOD
- **contribution:** Provides the transformer encoder architecture and masked language modeling objective used to pretrain the model.
- **rationale:** To build a multilingual BERT-style encoder, the paper requires a transformer architecture trained with masked language modeling. BERT is the clearest and most faithful canonical grounding for that dependency.
- **evidence_span:** “We use the default hyperparameters of BERT-Base”

### C2.I2
- **ingredient:** Translation Language Modeling objective for cross-lingual alignment
- **canonical study:** Cross-lingual Language Model Pretraining
- **role:** CORE_METHOD
- **contribution:** Provides the cross-lingual pretraining objective used to align representations across languages.
- **rationale:** The model is not only trained with MLM; it explicitly incorporates TLM as a core objective for cross-lingual alignment. That objective maps cleanly to XLM as the canonical prior study.
- **evidence_span:** “Translation Language Modeling (Conneau and Lample, 2019, TLM)”

### C2.I3
- **ingredient:** Large-scale Indic-language pretraining text
- **canonical study:** NONE
- **additional studies:** IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages; Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages
- **role:** TRAINING_DATA
- **contribution:** Provides the large-scale Indic-language text substrate used to pretrain the multilingual encoder.
- **rationale:** Recreating the model requires the actual large-scale Indic-language training text. But the operative pretraining text is structurally composite: it is dominated by corpus material created or assembled in the target paper, while also incorporating prior resources such as IndicCorp lineage and Samanantar-derived text. Because no single prior study cleanly represents that full training substrate, `NONE` is the correct canonical choice, with prior contributing resources listed as additional studies.
- **evidence_span:** “we merge data from IndicCorp v2 with Indic language data from Wikipedia and OSCAR”
"""


EXAMPLE_4 = """
# EXAMPLE: MassiveDS datastore scaling resource

## 1. Cluster Evidence

**The MassiveDS datastore resource** is used in the literature

The paper also presents a pipeline reordering trick that makes datastore scaling computationally feasible. However, the downstream evidence does **not** show that this pipeline is reused as a separate artifact on its own. Therefore, the correct annotation is **one resource claim**, with the pipeline-reordering idea represented as an important ingredient rather than a separate claim.

---

## 2. Claim Split Decision

### Claim C1
- **claim_id:** C1
- **artifact_type:** Resource
- **rewritten_claim:** **Dataset: A large-scale multi-domain datastore and experimental pipeline, enabling systematic study and evaluation of retrieval-based language model scaling.**
- **decision:** YES_SUFFICIENT
- **evidence_cluster_id:** C3

### Why this split is correct

The downstream reuse is centered on **one released resource**: the MassiveDS datastore.

- Later work uses the **datastore itself** as the reusable artifact.
- The paper also introduces a pipeline-reordering idea, but the evidence does **not** show that this pipeline is being reused downstream as a separate artifact.
- Because claim splitting is driven by **distinct downstream-used contributions**, the correct annotation is to keep **one claim**.

The pipeline reordering is still important, but its role is different:

- it is a **structural ingredient** that makes the datastore resource computationally feasible in its claimed form
- it is **not** a separate discovery claim unless downstream evidence shows reuse of the pipeline itself

---

## 3. Rewritten Claims

### Claim C1
- **claim_id:** C1
- **artifact_type:** Resource
- **rewritten_claim:** **Resource: A massive open multi-domain datastore that combines general web and domain-specific text, enabling large-scale retrieval-in-context language modeling and datastore-scaling studies.**
- **why_this_is_atomic:** This claim isolates the released datastore resource that downstream work reuses. It does not split off the pipeline-reordering idea as a separate claim because that idea is not independently substantiated as a downstream-used artifact.
- **optional cluster_id:** C3
- **decision:** YES_SUFFICIENT

---

## 4. Minimal Ingredient Set Per Claim

## Claim C1: MassiveDS datastore resource

### Ingredient C1.I1
- **ingredient_id:** C1.I1
- **ingredient:** Multi-source open text corpora spanning general web and domain-specific sources
- **why_structurally_necessary:** The datastore is defined by both its scale and its domain diversity. Without large open corpora spanning general web and specialized domains, the released resource would not exist in its claimed form.
- **why_not_lower_level_substeps:** This should remain one composite data-source ingredient rather than being split into separate final ingredients for web data, books, scientific papers, math, biomedical text, code, and other source families.
- **why_not_adjacent_implementation_details:** The structural dependency is the multi-source raw text substrate itself, not every individual source-selection or curation decision.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** RedPajama: An Open Source Recipe to Reproduce LLaMA Training Dataset; PubMed Baseline Repository; S2ORC: The Semantic Scholar Open Research Corpus; peS2o (Pretraining Efficiently on S2ORC) Dataset; NaturalProofs: Mathematical Theorem Proving in Natural Language; Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- **role:** DATA_SOURCE
- **contribution:** Supplies the large-scale open text sources used to assemble the datastore across both general web and specialized domains.
- **rationale:** Recreating MassiveDS requires the raw text substrate that gives it both scale and domain coverage. That substrate is inherently composite: the datastore mixes general web data with multiple distinct domain-specific resources. No single prior study faithfully represents that entire source mixture, so `NONE` is the correct canonical grounding, with the main contributing source resources listed as additional studies.
- **evidence_span:** “comprising 1.4 trillion tokens of both general web data and domain specific data”

### Ingredient C1.I2
- **ingredient_id:** C1.I2
- **ingredient:** Pretrained dense retriever for document embedding
- **why_structurally_necessary:** The released datastore is not just raw text; it is an operational retrieval datastore with dense-vector retrieval over passages. Without a pretrained retriever to embed documents, it would not function in its claimed form.
- **why_not_lower_level_substeps:** This should remain one retriever ingredient rather than being split into encoder architecture, embedding dimensionality, or indexing internals.
- **why_not_adjacent_implementation_details:** Retriever hyperparameters and ablation choices are implementation details. The structural dependency is the pretrained dense retriever used to embed the datastore.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** Unsupervised Dense Information Retrieval with Contrastive Learning
- **additional groundings if any:** none
- **role:** IMPLEMENTATION_TOOLING
- **contribution:** Provides the pretrained retriever used to represent datastore documents as dense vectors for retrieval.
- **rationale:** To reproduce the datastore as a usable retrieval resource, the corpus must be embedded into a searchable dense space. The paper explicitly uses Contriever-MSMARCO for this purpose, making Contriever the cleanest canonical grounding.
- **evidence_span:** “we use CONTRIEVER-MSMARCO (Izacard et al., 2022), which represents every document in the datastore as a dense vector”

### Ingredient C1.I3
- **ingredient_id:** C1.I3
- **ingredient:** Retrieval-in-context language modeling framework
- **why_structurally_necessary:** MassiveDS is introduced specifically to support retrieve-in-context language models rather than as a generic corpus release. Without this framing, the datastore would lose the main capability it is meant to enable.
- **why_not_lower_level_substeps:** This should remain one framework-level ingredient rather than being split into retrieval prompting, concatenation order, or model-specific usage details.
- **why_not_adjacent_implementation_details:** The structural dependency is the inference-time retrieval framework itself, not one particular LM or evaluation setup.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** In-Context Retrieval-Augmented Language Models
- **additional groundings if any:** none
- **role:** CONCEPTUAL_FRAMEWORK
- **contribution:** Defines the retrieval-in-context framing in which documents are retrieved from an external datastore and prepended to LM inputs at inference time.
- **rationale:** The datastore is meaningful here because it supports retrieval-based language models that use retrieved documents in context. Ram et al. is the cleanest canonical grounding because it directly represents the retrieve-in-context setup that the paper focuses on.
- **evidence_span:** “We focus on retrieve-in-context language models (RIC-LMs)”

### Ingredient C1.I4
- **ingredient_id:** C1.I4
- **ingredient:** Data filtering workflow for deduplication, decontamination, and quality control
- **why_structurally_necessary:** The released resource is not merely a raw trillion-token dump; it is a constructed datastore whose quality depends on filtering, deduplication, and decontamination steps. Without this workflow, the datastore would be substantially weaker as a reusable retrieval resource.
- **why_not_lower_level_substeps:** This should remain one filtering-workflow ingredient rather than being split into deduplication, decontamination, and quality filtering as separate ingredients.
- **why_not_adjacent_implementation_details:** Individual thresholds, heuristics, or local filtering choices are not the structural dependency; the key dependency is the broader filtering workflow that makes the datastore usable.
- **necessary:** true
- **from_prior_work:** true
- **maps_cleanly_to_one_study:** true
- **canonical grounding decision:** Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
- **additional groundings if any:** none
- **role:** IMPLEMENTATION_TOOLING
- **contribution:** Provides the filtering workflow used to clean the datastore through deduplication, decontamination, and quality controls.
- **rationale:** Building a reusable datastore at this scale requires systematic filtering so the resulting resource is not just large, but also usable and reasonably clean. The paper explicitly ties this step to Dolma, making it the best canonical grounding.
- **evidence_span:** “data filtering, including deduplication, decontamination, and quality filters”

### Ingredient C1.I5
- **ingredient_id:** C1.I5
- **ingredient:** Pipeline reordering to amortize expensive indexing and retrieval across datastore variants
- **why_structurally_necessary:** The datastore is released together with an experimental pipeline that makes large-scale datastore scaling feasible on modest compute. Without this reordering idea, the resource would lose an important defining property: practical support for scalable datastore studies.
- **why_not_lower_level_substeps:** This should remain one pipeline-level ingredient rather than being split into retrieval-overfetching, post-hoc subsampling, and late-stage filtering as separate ingredients.
- **why_not_adjacent_implementation_details:** The structural dependency is the reordering principle itself—running expensive steps once and sharing them—not the local code or systems details used to implement it.
- **necessary:** true
- **from_prior_work:** false
- **maps_cleanly_to_one_study:** false
- **canonical grounding decision:** NONE
- **additional groundings if any:** none
- **role:** CONCEPTUAL_FRAMEWORK
- **contribution:** Provides the core pipeline idea that makes datastore scaling computationally feasible by sharing expensive indexing and retrieval across many variants.
- **rationale:** A key part of the released contribution is not only the datastore’s scale, but the fact that the paper makes large-scale scaling studies feasible by reordering operations so indexing and retrieval are amortized. This idea is introduced in the target paper rather than inherited from a single prior study, so `NONE` is the correct grounding.
- **evidence_span:** “the most expensive ones—indexing and retrieval—are run only once at the start”

---

## 5. Excluded Tempting Non-Ingredients

### Claim C1 exclusions
- **Pipeline reordering as a separate second claim**
  - Excluded because the downstream reuse evidence centers on the datastore resource, not on the pipeline as an independently reused artifact. The pipeline is important, but it is best represented as a structural ingredient of the single datastore claim.

- **Passage chunking and indexing into fixed-length retrieval units**
  - Excluded because this is a lower-level construction step rather than one of the smallest higher-level structural ingredients. It is subordinate to the broader retriever-and-datastore construction pipeline.

- **FAISS / exact top-K retrieval implementation**
  - Excluded because this is too implementation-specific. It supports the pipeline but is not needed at the abstraction level of the minimal structurally sufficient ingredient set.

- **Separate final ingredients for general web data and domain-specific data**
  - Excluded because the higher-level composite ingredient is the multi-source open text substrate spanning both. Splitting them would over-decompose the datastore’s source basis.

- **Contriever as MODEL_INITIALIZATION instead of IMPLEMENTATION_TOOLING**
  - Excluded because its role here is not initializing the target discovery as a model checkpoint, but operationally embedding the datastore documents for retrieval.

---

## 6. Candidate Grounding Decisions

## Claim C1: MassiveDS datastore resource

### Ingredient C1.I1
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The datastore’s source substrate is structurally composite across general web and many domain-specific resources.
- **Candidate:** RedPajama: An Open Source Recipe to Reproduce LLaMA Training Dataset
- **decision:** accepted_additional
- **why:** Representative open source for books and other large-scale text components, but not sufficient alone to represent the whole source mixture.
- **Candidate:** PubMed Baseline Repository
- **decision:** accepted_additional
- **why:** Important biomedical source contributing to the domain-specific side of the datastore.
- **Candidate:** S2ORC: The Semantic Scholar Open Research Corpus
- **decision:** accepted_additional
- **why:** Important scientific-paper source for the domain-specific mixture.
- **Candidate:** peS2o (Pretraining Efficiently on S2ORC) Dataset
- **decision:** accepted_additional
- **why:** Additional scientific corpus contribution.
- **Candidate:** NaturalProofs: Mathematical Theorem Proving in Natural Language
- **decision:** accepted_additional
- **why:** Representative mathematical-language source for the domain-specific mixture.
- **Candidate:** Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- **decision:** accepted_additional
- **why:** Relevant additional grounding because C4 is explicitly cited as one of the general web data sources.

### Ingredient C1.I2
- **Candidate:** Unsupervised Dense Information Retrieval with Contrastive Learning
- **decision:** accepted_canonical
- **why:** Cleanest representative prior study for Contriever, the pretrained dense retriever actually used to embed datastore passages.

### Ingredient C1.I3
- **Candidate:** In-Context Retrieval-Augmented Language Models
- **decision:** accepted_canonical
- **why:** Cleanest representative framing for the retrieval-in-context LM setup the datastore is designed to support.

### Ingredient C1.I4
- **Candidate:** Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
- **decision:** accepted_canonical
- **why:** Explicitly cited by the paper for the filtering stage and is the cleanest representative for the deduplication / decontamination / quality-control workflow.

### Ingredient C1.I5
- **Candidate:** NONE
- **decision:** accepted_none
- **why:** The pipeline-reordering idea is introduced in the target paper itself and is not cleanly attributable to one prior study.

---

## 7. Final Grounding Decisions

## Claim C1
- **C1.I1** → NONE → DATA_SOURCE  
  - additional: RedPajama: An Open Source Recipe to Reproduce LLaMA Training Dataset; PubMed Baseline Repository; S2ORC: The Semantic Scholar Open Research Corpus; peS2o (Pretraining Efficiently on S2ORC) Dataset; NaturalProofs: Mathematical Theorem Proving in Natural Language; Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- **C1.I2** → Unsupervised Dense Information Retrieval with Contrastive Learning → IMPLEMENTATION_TOOLING
- **C1.I3** → In-Context Retrieval-Augmented Language Models → CONCEPTUAL_FRAMEWORK
- **C1.I4** → Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research → IMPLEMENTATION_TOOLING
- **C1.I5** → NONE → CONCEPTUAL_FRAMEWORK

---

## 8. Final Rationales And Evidence

## Claim C1: MassiveDS datastore resource

### C1.I1
- **ingredient:** Multi-source open text corpora spanning general web and domain-specific sources
- **canonical study:** NONE
- **additional studies:** RedPajama: An Open Source Recipe to Reproduce LLaMA Training Dataset; PubMed Baseline Repository; S2ORC: The Semantic Scholar Open Research Corpus; peS2o (Pretraining Efficiently on S2ORC) Dataset; NaturalProofs: Mathematical Theorem Proving in Natural Language; Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
- **role:** DATA_SOURCE
- **contribution:** Supplies the large-scale open text sources used to assemble the datastore across both general web and specialized domains.
- **rationale:** Recreating MassiveDS requires the raw text substrate that gives it both scale and domain coverage. That substrate is inherently composite: the datastore mixes general web data with multiple distinct domain-specific resources. No single prior study faithfully represents that entire source mixture, so `NONE` is the correct canonical grounding, with the main contributing source resources listed as additional studies.
- **evidence_span:** “comprising 1.4 trillion tokens of both general web data and domain specific data”

### C1.I2
- **ingredient:** Pretrained dense retriever for document embedding
- **canonical study:** Unsupervised Dense Information Retrieval with Contrastive Learning
- **role:** IMPLEMENTATION_TOOLING
- **contribution:** Provides the pretrained retriever used to represent datastore documents as dense vectors for retrieval.
- **rationale:** To reproduce the datastore as a usable retrieval resource, the corpus must be embedded into a searchable dense space. The paper explicitly uses Contriever-MSMARCO for this purpose, making Contriever the cleanest canonical grounding.
- **evidence_span:** “we use CONTRIEVER-MSMARCO (Izacard et al., 2022), which represents every document in the datastore as a dense vector”

### C1.I3
- **ingredient:** Retrieval-in-context language modeling framework
- **canonical study:** In-Context Retrieval-Augmented Language Models
- **role:** CONCEPTUAL_FRAMEWORK
- **contribution:** Defines the retrieval-in-context framing in which documents are retrieved from an external datastore and prepended to LM inputs at inference time.
- **rationale:** The datastore is meaningful here because it supports retrieval-based language models that use retrieved documents in context. Ram et al. is the cleanest canonical grounding because it directly represents the retrieve-in-context setup that the paper focuses on.
- **evidence_span:** “We focus on retrieve-in-context language models (RIC-LMs)”

### C1.I4
- **ingredient:** Data filtering workflow for deduplication, decontamination, and quality control
- **canonical study:** Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
- **role:** IMPLEMENTATION_TOOLING
- **contribution:** Provides the filtering workflow used to clean the datastore through deduplication, decontamination, and quality controls.
- **rationale:** Building a reusable datastore at this scale requires systematic filtering so the resulting resource is not just large, but also usable and reasonably clean. The paper explicitly ties this step to Dolma, making it the best canonical grounding.
- **evidence_span:** “data filtering, including deduplication, decontamination, and quality filters”

### C1.I5
- **ingredient:** Pipeline reordering to amortize expensive indexing and retrieval across datastore variants
- **canonical study:** NONE
- **role:** CONCEPTUAL_FRAMEWORK
- **contribution:** Provides the core pipeline idea that makes datastore scaling computationally feasible by sharing expensive indexing and retrieval across many variants.
- **rationale:** A key part of the released contribution is not only the datastore’s scale, but the fact that the paper makes large-scale scaling studies feasible by reordering operations so indexing and retrieval are amortized. This idea is introduced in the target paper rather than inherited from a single prior study, so `NONE` is the correct grounding.
- **evidence_span:** “the most expensive ones—indexing and retrieval—are run only once at the start”
"""
