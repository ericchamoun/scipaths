from pathlib import Path
import json
import re

DATA_DIR = Path("data")
MANUAL_DIR = DATA_DIR / "currently_manual_annotation"
GEMINI_DIR = DATA_DIR / "gemini_llm_annotations"
DEFAULT_GOLD_FILE = Path("../data/dev.json")

_UPDATED_CLAIMS: dict[str, list[str]] | None = None
_GOLD_EXPORTS: dict[Path, list[dict]] = {}


def _load_updated_claims() -> dict[str, list[str]]:
    global _UPDATED_CLAIMS
    if _UPDATED_CLAIMS is None:
        path = DATA_DIR / "updated_claims.json"
        _UPDATED_CLAIMS = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _UPDATED_CLAIMS


def get_updated_claims(paper_id: str) -> list[str]:
    """Return the updated discovery claims for a paper (may be multiple)."""
    return _load_updated_claims().get(paper_id, [])


def get_paper_ids(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")}


def get_overlapping_ids() -> list[str]:
    return sorted(get_paper_ids(MANUAL_DIR) & get_paper_ids(GEMINI_DIR))


def load_gold_export(path: str | Path = DEFAULT_GOLD_FILE) -> list[dict]:
    """Load the standalone gold/manual annotation export."""
    p = Path(path)
    if p not in _GOLD_EXPORTS:
        _GOLD_EXPORTS[p] = json.loads(p.read_text(encoding="utf-8"))
    return _GOLD_EXPORTS[p]


def _is_claim_level_export(rows: list[dict]) -> bool:
    return bool(rows) and "target_contribution" in rows[0] and "claims" not in rows[0]


def _claim_level_paper_id(row: dict) -> str:
    if row.get("target_paper_id"):
        return str(row["target_paper_id"])
    if row.get("idx") is not None:
        return str(row["idx"])
    return ""


def _claim_level_claim_idx(row: dict) -> int:
    return int(row.get("claim_idx", row.get("idx", 0)) or 0)


def get_gold_claim_tasks(
    path: str | Path = DEFAULT_GOLD_FILE,
    claim_field: str = "target_contribution",
    conference_year: int | None = None,
) -> list[tuple[str, int, str]]:
    """Return one evaluation task per annotated claim.

    Each tuple is (paper_id, claim_idx, claim_text). claim_field selects which
    claim text to use, with fallback to older export fields when needed.
    """
    fallback_fields = [field for field in ["target_contribution", "rewritten_capability", "rewritten_claim"] if field != claim_field]
    tasks: list[tuple[str, int, str]] = []
    rows = load_gold_export(path)
    if _is_claim_level_export(rows):
        for row in rows:
            if conference_year is not None and row.get("conference_year") != conference_year:
                continue
            paper_id = _claim_level_paper_id(row)
            claim_idx = _claim_level_claim_idx(row)
            claim_text = row.get(claim_field) or next((row.get(field) for field in fallback_fields if row.get(field)), "")
            if paper_id and claim_text:
                tasks.append((paper_id, claim_idx, claim_text))
        return tasks

    for paper in rows:
        if conference_year is not None and paper.get("conference_year") != conference_year:
            continue
        paper_id = paper.get("target_paper_id", "")
        for claim_idx, claim in enumerate(paper.get("claims") or []):
            claim_text = claim.get(claim_field) or next((claim.get(field) for field in fallback_fields if claim.get(field)), "")
            if paper_id and claim_text:
                tasks.append((paper_id, claim_idx, claim_text))
    return tasks


def get_gold_paper_title(paper_id: str, path: str | Path = DEFAULT_GOLD_FILE) -> str:
    rows = load_gold_export(path)
    if _is_claim_level_export(rows):
        for row in rows:
            row_id = _claim_level_paper_id(row)
            if row_id == str(paper_id):
                return row.get("target_title") or str(row.get("idx") or paper_id)
        return str(paper_id)

    for paper in rows:
        if paper.get("target_paper_id") == paper_id:
            return paper.get("target_title") or paper_id
    return paper_id


def get_gold_ingredients_for_claim(
    paper_id: str,
    claim_idx: int,
    path: str | Path = DEFAULT_GOLD_FILE,
) -> list[dict]:
    rows = load_gold_export(path)
    if _is_claim_level_export(rows):
        for row in rows:
            row_id = _claim_level_paper_id(row)
            row_claim_idx = _claim_level_claim_idx(row)
            if row_id == str(paper_id) and row_claim_idx == claim_idx:
                return row.get("enabling_contributions") or []
        return []

    for paper in rows:
        if paper.get("target_paper_id") != paper_id:
            continue
        claims = paper.get("claims") or []
        if claim_idx < len(claims):
            return claims[claim_idx].get("ingredients") or []
    return []


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_manual(paper_id: str) -> dict | None:
    return _load_json(MANUAL_DIR / paper_id / "usage_discovery_from_contributions.json")


def load_llm_annotation(paper_id: str) -> dict | None:
    folder = GEMINI_DIR / paper_id
    if not folder.exists():
        return None
    files = list(folder.glob("llm_annotation__*.json"))
    return _load_json(files[0]) if files else None


def load_paper_metadata(paper_id: str) -> dict:
    data = _load_json(MANUAL_DIR / paper_id / "paper_metadata.json")
    if isinstance(data, list) and data:
        return data[0]
    return data or {}


def get_semantic_scholar_paper_id(paper_id: str) -> str | None:
    """Return the target paper's Semantic Scholar paperId when local metadata has it."""
    metadata = load_paper_metadata(paper_id)
    s2_id = metadata.get("paperId")
    return str(s2_id) if s2_id else None


def get_active_claim(paper_id: str) -> dict | None:
    annotation = load_llm_annotation(paper_id)
    if not annotation:
        return None
    active_id = annotation.get("active_claim_id", "C1")
    for claim in annotation.get("claims", []):
        if claim["claim_id"] == active_id:
            return claim
    return None


def get_rewritten_claim(paper_id: str) -> str | None:
    claim = get_active_claim(paper_id)
    if not claim:
        return None
    return claim.get("rewritten_claim") or claim.get("text")


def get_silver_ingredients(paper_id: str) -> list[dict]:
    claim = get_active_claim(paper_id)
    return claim.get("ingredients", []) if claim else []


def get_silver_ingredients_for_claim(paper_id: str, claim_idx: int) -> list[dict]:
    """Return silver ingredients for the claim at position claim_idx (0-based).

    updated_claims.json lists claims in the same order as the Gemini annotation's
    claims[], so claim_idx 0 → claims[0], claim_idx 1 → claims[1], etc.
    Falls back to the active claim if claim_idx is out of range.
    """
    annotation = load_llm_annotation(paper_id)
    if not annotation:
        return []
    claims = annotation.get("claims", [])
    if claim_idx < len(claims):
        return claims[claim_idx].get("ingredients", [])
    # fallback: active claim
    return get_silver_ingredients(paper_id)


def get_paper_title(paper_id: str) -> str:
    annotation = load_llm_annotation(paper_id)
    if annotation and annotation.get("target_title"):
        return annotation["target_title"]
    return load_paper_metadata(paper_id).get("title", paper_id)


def get_references(paper_id: str) -> list[dict]:
    """Return cited papers as [{title, contexts, is_influential}]."""
    raw = _load_json(MANUAL_DIR / paper_id / "references_metadata.json")
    if not isinstance(raw, list):
        return []
    refs = []
    for entry in raw:
        cited = entry.get("citedPaper", {})
        title = cited.get("title", "").strip()
        if not title:
            continue
        contexts = [
            c.get("context", "").strip()
            for c in entry.get("contextsWithIntent", [])
            if c.get("context")
        ]
        refs.append({
            "title": title,
            "contexts": contexts,
            "is_influential": entry.get("isInfluential", False),
        })
    return refs


def get_related_work(paper_id: str) -> str:
    """Return the target paper's Related Work section text when available."""
    paper_dir = MANUAL_DIR / paper_id
    sections_dir = paper_dir / "sections"
    if sections_dir.exists():
        candidates = sorted(
            p for p in sections_dir.iterdir()
            if p.is_file() and "related" in p.stem.lower()
        )
        if candidates:
            return candidates[0].read_text(encoding="utf-8", errors="ignore").strip()

    processed = paper_dir / "processed_main.tex"
    if not processed.exists():
        return ""

    text = processed.read_text(encoding="utf-8", errors="ignore")
    section_re = re.compile(r"\\(?:section|subsection)\*?\{([^{}]+)\}")
    matches = list(section_re.finditer(text))
    for i, match in enumerate(matches):
        title = match.group(1).lower()
        if "related" not in title:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        return text[match.start():end].strip()
    return ""
