from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


ALLOWED_ARTIFACT_TYPES = ["Resource", "Finding", "Method", "Benchmark", "Dataset", "Tool", "Other"]
ALLOWED_ROLES = [
    "CONCEPTUAL_FRAMEWORK",
    "CORE_METHOD",
    "DATA_SOURCE",
    "MODEL_INITIALIZATION",
    "EVALUATION_PROTOCOL",
]


class ReasoningCandidate(BaseModel):
    study: str
    decision: Literal["accepted_canonical", "accepted_additional", "accepted_none", "rejected_candidate"]
    why: str


class ReasoningIngredient(BaseModel):
    ingredient_id: str
    ingredient: str
    necessary: bool
    from_prior_work: bool
    maps_cleanly_to_one_study: bool
    notes: str
    canonical_grounding_decision: Dict[str, Any]
    additional_groundings: List[Dict[str, Any]]
    candidate_studies_considered: List[ReasoningCandidate]
    role: Optional[Literal["CONCEPTUAL_FRAMEWORK", "CORE_METHOD", "DATA_SOURCE", "MODEL_INITIALIZATION", "EVALUATION_PROTOCOL", "IMPLEMENTATION_TOOLING", "TRAINING_DATA"]] = None
    contribution: str
    rationale: str
    evidence_span: str


class ReasoningClaim(BaseModel):
    claim_id: str
    artifact_type: Literal["Resource", "Finding", "Method", "Benchmark", "Dataset", "Tool", "Other"]
    rewritten_claim: str
    cluster_id: str = ""
    decision: Literal["YES_SUFFICIENT", "NO_NOT_DISCOVERY"] = "YES_SUFFICIENT"
    notes: str = ""
    why_this_is_atomic: str
    ingredients: List[ReasoningIngredient]


class ReasoningOutput(BaseModel):
    original_discovery_claim: str
    claim_split_decision: Dict[str, Any]
    rewritten_claims: List[ReasoningClaim]
    paper_level_notes: str = ""


class GroundingRecord(BaseModel):
    ref_id: Optional[str] = None
    bib_key: Optional[str] = None
    paper_id: Optional[str] = None
    external_ids: Optional[Dict[str, Any]] = None
    ref_title: Optional[str] = None
    ref_year: Optional[str] = None
    ref_authors: Optional[str] = None


class CanonicalAnnotation(BaseModel):
    role: Optional[Literal["CONCEPTUAL_FRAMEWORK", "CORE_METHOD", "DATA_SOURCE", "MODEL_INITIALIZATION", "EVALUATION_PROTOCOL", "IMPLEMENTATION_TOOLING", "TRAINING_DATA"]] = None
    roles: List[Literal["CONCEPTUAL_FRAMEWORK", "CORE_METHOD", "DATA_SOURCE", "MODEL_INITIALIZATION", "EVALUATION_PROTOCOL", "IMPLEMENTATION_TOOLING", "TRAINING_DATA"]]
    contribution: str
    rationale: str
    evidence_span: str


class IngredientPayload(BaseModel):
    ingredient_id: str
    ingredient: str
    canonical_ref_id: str
    canonical_grounding: Optional[GroundingRecord] = None
    additional_ref_ids: List[str]
    additional_groundings: List[GroundingRecord]
    canonical_annotation: CanonicalAnnotation


class EnablingDiscoveryPayload(GroundingRecord):
    ingredient_id: str
    ingredient: str
    role: Optional[Literal["CONCEPTUAL_FRAMEWORK", "CORE_METHOD", "DATA_SOURCE", "MODEL_INITIALIZATION", "EVALUATION_PROTOCOL", "IMPLEMENTATION_TOOLING", "TRAINING_DATA"]] = None
    roles: List[Literal["CONCEPTUAL_FRAMEWORK", "CORE_METHOD", "DATA_SOURCE", "MODEL_INITIALIZATION", "EVALUATION_PROTOCOL", "IMPLEMENTATION_TOOLING", "TRAINING_DATA"]]
    contribution: str
    rationale: str
    evidence_span: str


class ClaimPayload(BaseModel):
    claim_id: str
    text: str
    rewritten_claim: str
    cluster_id: str = ""
    decision: Literal["YES_SUFFICIENT", "NO_NOT_DISCOVERY", "UNCERTAIN"] = "YES_SUFFICIENT"
    notes: str = ""
    ingredients: List[IngredientPayload]
    enabling_discoveries: List[EnablingDiscoveryPayload]


class JudgeCandidateScore(BaseModel):
    candidate_id: str
    candidate_index: int
    score: int
    assessment: str


class JudgeResult(BaseModel):
    selected_candidate_index: int
    selected_candidate_id: str
    selected_reason: str
    candidate_scores: List[JudgeCandidateScore]


class UIPayload(BaseModel):
    target_paper_id: str
    target_title: Optional[str] = None
    target_year: Optional[int] = None
    annotator_id: str
    active_claim_id: Optional[str] = None
    claims: List[ClaimPayload]
