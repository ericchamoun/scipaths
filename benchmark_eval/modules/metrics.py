import re

# ---------------------------------------------------------------------------
# Type helpers (shared between llm_as_a_judge.py and task_a.py)
# ---------------------------------------------------------------------------

TYPE_SYNONYMS = {
    frozenset({"Benchmark", "Metric"}): "EvaluationArtifact",
}

TYPE_TO_ARTIFACT_SYNONYMS: dict[str, set[str]] = {
    "benchmark": {"benchmark", "metric", "evaluation"},
    "metric":    {"benchmark", "metric", "evaluation"},
    "dataset":   {"dataset", "corpus", "data", "lexicon"},
    "corpus":    {"dataset", "corpus", "data"},
    "lexicon":   {"dataset", "corpus", "data", "lexicon"},
    "model":     {"model", "tool", "system"},
    "method":    {"method", "model", "algorithm"},
    "framework": {"framework", "system", "tool", "model"},
    "algorithm": {"algorithm", "method"},
    "system":    {"system", "model", "tool", "framework"},
}


def extract_llm_type(claim_text: str) -> str:
    m = re.match(r"^([A-Za-z]+)\s*:", claim_text)
    return m.group(1).strip().title() if m else "Unknown"


def extract_manual_artifact_types(clusters: list[dict]) -> set[str]:
    types: set[str] = set()
    for c in clusters:
        parts = c.get("cluster_key", "").lower().split("|")
        if len(parts) >= 2:
            types.add(parts[1])
    return types


def type_match_score(llm_type: str, manual_type: str) -> float:
    a, b = llm_type.strip().title(), manual_type.strip().title()
    if a == b:
        return 1.0
    for group in TYPE_SYNONYMS:
        if a in group and b in group:
            return 0.5
    return 0.0


def c4_type_coverage(llm_type: str, clusters: list[dict]) -> float:
    manual_artifacts = extract_manual_artifact_types(clusters)
    if not manual_artifacts:
        return 0.0
    normalized = llm_type.lower()
    if normalized in manual_artifacts:
        return 1.0
    expanded = TYPE_TO_ARTIFACT_SYNONYMS.get(normalized, {normalized})
    return 0.5 if expanded & manual_artifacts else 0.0


# ---------------------------------------------------------------------------
# Enabling contribution generation metrics
# ---------------------------------------------------------------------------

def maximum_full_matching(matches: list[dict], n_references: int, n_predicted: int) -> list[dict]:
    """Return one-to-one maximum matching over full reference/predicted edges.

    Indices in matches are 1-based to match the JSON output shown to users.
    """
    edges: dict[int, list[dict]] = {i: [] for i in range(1, n_references + 1)}
    seen: set[tuple[int, int]] = set()
    for match in matches:
        if match.get("match") != "full":
            continue
        ref_idx = match.get("reference_idx")
        pred_idx = match.get("predicted_idx")
        if not isinstance(ref_idx, int) or not isinstance(pred_idx, int):
            continue
        if not (1 <= ref_idx <= n_references and 1 <= pred_idx <= n_predicted):
            continue
        key = (ref_idx, pred_idx)
        if key in seen:
            continue
        seen.add(key)
        edges[ref_idx].append(match)

    for ref_edges in edges.values():
        ref_edges.sort(key=lambda m: m["predicted_idx"])

    pred_to_match: dict[int, dict] = {}

    def _try_match(ref_idx: int, visited: set[int]) -> bool:
        for edge in edges.get(ref_idx, []):
            pred_idx = edge["predicted_idx"]
            if pred_idx in visited:
                continue
            visited.add(pred_idx)
            previous = pred_to_match.get(pred_idx)
            if previous is None or _try_match(previous["reference_idx"], visited):
                pred_to_match[pred_idx] = edge
                return True
        return False

    for ref_idx in range(1, n_references + 1):
        _try_match(ref_idx, set())

    return sorted(pred_to_match.values(), key=lambda m: (m["reference_idx"], m["predicted_idx"]))


def compute_matching_recall(n_matched: int, n_references: int) -> float:
    if n_references == 0:
        return 0.0
    return n_matched / n_references


def compute_matching_precision(n_matched: int, n_predicted: int) -> float | None:
    if n_predicted == 0:
        return None
    return n_matched / n_predicted

def compute_paper_recall(judgments: list[dict]) -> float:
    if not judgments:
        return 0.0
    return sum(1 for j in judgments if j.get("covered", False)) / len(judgments)


def compute_paper_precision(judgments: list[dict], n_predicted: int) -> float | None:
    """Fraction of predicted ingredients that matched at least one silver.

    Derived for free from recall judgments via unique best_match_idx values —
    no extra LLM calls needed.
    """
    if n_predicted == 0:
        return None
    matched = {j["best_match_idx"] for j in judgments if j.get("covered") and j.get("best_match_idx") is not None}
    return len(matched) / n_predicted


def compute_paper_f1(recall: float | None, precision: float | None) -> float | None:
    if recall is None or precision is None:
        return None
    if recall + precision == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 3)


def aggregate_task_a_results(results: list[dict]) -> dict:
    recalls     = [r["recall"]    for r in results if r.get("recall")    is not None]
    precisions  = [r["precision"] for r in results if r.get("precision") is not None]
    f1s         = [r["f1"]        for r in results if r.get("f1")        is not None]
    n = len(results)
    n_gen_errors   = sum(1 for r in results if r.get("error"))
    n_judge_errors = sum(
        1 for r in results
        for j in r.get("recall_judgments", [])
        if "error:" in j.get("reasoning", "")
    )

    def _mean(vals): return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "n_papers": n,
        "n_gen_errors":   n_gen_errors,
        "n_judge_errors": n_judge_errors,
        "mean_recall":    _mean(recalls),
        "mean_precision": _mean(precisions),
        "mean_f1":        _mean(f1s),
        "full_recall_rate":    round(sum(1 for r in recalls if r == 1.0) / len(recalls), 3) if recalls else None,
        "partial_recall_rate": round(sum(1 for r in recalls if 0 < r < 1.0) / len(recalls), 3) if recalls else None,
        "zero_recall_rate":    round(sum(1 for r in recalls if r == 0.0) / len(recalls), 3) if recalls else None,
        "mean_n_reference": round(sum(r.get("n_reference", r.get("n_silver", 0)) for r in results) / n, 2) if n else None,
        "mean_n_silver":    round(sum(r.get("n_reference", r.get("n_silver", 0)) for r in results) / n, 2) if n else None,
        "mean_n_predicted": round(sum(r.get("n_predicted", 0) for r in results) / n, 2) if n else None,
    }


# ---------------------------------------------------------------------------
# Comparison metrics (llm_as_a_judge.py)
# ---------------------------------------------------------------------------

def aggregate_comparison_results(results: list[dict]) -> dict:
    c1_scores = [r["c1_score"] for r in results if r.get("c1_score") is not None]
    c2_scores = [r["c2_type_match"] for r in results]
    c4_scores = [r["c4_type_coverage"] for r in results]
    n = len(results)
    return {
        "n_papers": n,
        "c1_mean_score": round(sum(c1_scores) / len(c1_scores), 3) if c1_scores else None,
        "c1_agreement_rate": round(sum(1 for s in c1_scores if s >= 2) / len(c1_scores), 3) if c1_scores else None,
        "c2_exact_match_rate": round(sum(1 for s in c2_scores if s == 1.0) / n, 3) if n else None,
        "c2_partial_plus_rate": round(sum(1 for s in c2_scores if s >= 0.5) / n, 3) if n else None,
        "c4_mean_coverage": round(sum(c4_scores) / n, 3) if n else None,
        "c4_exact_match_rate": round(sum(1 for s in c4_scores if s == 1.0) / n, 3) if n else None,
        "c4_partial_plus_rate": round(sum(1 for s in c4_scores if s >= 0.5) / n, 3) if n else None,
    }
