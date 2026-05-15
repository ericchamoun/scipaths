from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel


SECTION_FILES = [
    "abstract.txt",
    "introduction.tex",
    "related_work.tex",
    "tldr.txt",
]


class PaperPackage(BaseModel):
    paper_dir: Path
    paper_metadata: Dict[str, Any]
    extracted_discovery_claim: str
    downstream_cluster_evidence: List[Dict[str, Any]]
    paper_text: Dict[str, str]
    full_processed_text: str
    bibliography: List[Dict[str, Any]]
    citation_contexts: List[Dict[str, Any]]

    def to_prompt_payload(self) -> Dict[str, Any]:
        return {
            "paper_metadata": self.paper_metadata,
            "extracted_discovery_claim": self.extracted_discovery_claim,
            "downstream_cluster_evidence": self.downstream_cluster_evidence,
            "paper_text": self.paper_text,
            "full_processed_text": self.full_processed_text,
            "bibliography": self.bibliography,
            "citation_contexts": self.citation_contexts,
        }


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _read_text(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def _normalize_dict_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                return item
    return {}


def _collect_sections(paper_dir: Path) -> Dict[str, str]:
    sections_dir = paper_dir / "sections"
    out: Dict[str, str] = {}
    for name in SECTION_FILES:
        text = _read_text(sections_dir / name).strip()
        if text:
            out[name] = text[:12000]
    if not out:
        processed = _read_text(paper_dir / "processed_main.tex").strip()
        if processed:
            out["processed_main.tex"] = processed[:24000]
    return out


def _collect_full_processed_text(paper_dir: Path) -> str:
    processed = _read_text(paper_dir / "processed_main.tex").strip()
    if processed:
        return processed

    sections_dir = paper_dir / "sections"
    parts: List[str] = []
    if sections_dir.exists():
        for path in sorted(sections_dir.iterdir()):
            if not path.is_file():
                continue
            text = _read_text(path).strip()
            if text:
                parts.append(f"[{path.name}]\n{text}")
    return "\n\n".join(parts)


def _extract_year(value: Any) -> Any:
    if value:
        return value
    return None


def _normalise_reference_record(ref: Dict[str, Any]) -> Dict[str, Any]:
    cited = ref.get("citedPaper")
    source = cited if isinstance(cited, dict) else ref
    external_ids = source.get("external_ids") or source.get("externalIds") or {}
    return {
        "ref_id": (
            ref.get("ref_id")
            or ref.get("bib_key")
            or source.get("ref_id")
            or source.get("bib_key")
            or source.get("paperId")
            or source.get("paper_id")
            or external_ids.get("ACL")
            or external_ids.get("ArXiv")
            or external_ids.get("DOI")
        ),
        "title": source.get("title") or source.get("ref_title"),
        "authors": source.get("authors") or source.get("ref_authors"),
        "year": _extract_year(source.get("year") or source.get("ref_year")),
        "external_ids": external_ids,
    }


def _parse_bibtex_entries(text: str, limit: int) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for match in re.finditer(r"@\w+\s*\{\s*([^,]+),(.*?)(?=\n@\w+\s*\{|\Z)", text, re.S):
        key = match.group(1).strip()
        body = match.group(2)
        fields: Dict[str, str] = {}
        for field in ("title", "author", "year", "doi", "url", "eprint"):
            field_match = re.search(
                rf"\b{field}\s*=\s*(\{{(?:[^{{}}]|\{{[^{{}}]*\}})*\}}|\"[^\"]*\"|[^,\n]+)",
                body,
                re.I | re.S,
            )
            if field_match:
                value = field_match.group(1).strip().strip(",")
                if (value.startswith("{") and value.endswith("}")) or (
                    value.startswith('"') and value.endswith('"')
                ):
                    value = value[1:-1]
                fields[field] = re.sub(r"\s+", " ", value).strip()
        if fields:
            external_ids: Dict[str, Any] = {}
            if fields.get("doi"):
                external_ids["DOI"] = fields["doi"]
            if fields.get("eprint"):
                external_ids["ArXiv"] = fields["eprint"]
            entries.append(
                {
                    "ref_id": key,
                    "title": fields.get("title"),
                    "authors": fields.get("author"),
                    "year": fields.get("year"),
                    "external_ids": external_ids,
                }
            )
        if len(entries) >= limit:
            break
    return entries


def _collect_bibtex_citation_contexts(paper_dir: Path, limit: int = 60) -> List[Dict[str, Any]]:
    bibtex = _read_text(paper_dir / "references.bib")
    processed = _read_text(paper_dir / "processed_main.tex")
    if not bibtex or not processed:
        return []

    refs = _parse_bibtex_entries(bibtex, limit=500)
    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for ref in refs:
        ref_id = ref.get("ref_id")
        if not ref_id:
            continue
        for match in re.finditer(rf"\\cite\w*\s*(?:\[[^\]]*\]\s*)*\{{[^}}]*\b{re.escape(str(ref_id))}\b[^}}]*\}}", processed):
            key = (str(ref_id), match.start())
            if key in seen:
                continue
            seen.add(key)
            start = max(0, match.start() - 350)
            end = min(len(processed), match.end() + 350)
            snippet = re.sub(r"\s+", " ", processed[start:end]).strip()
            out.append(
                {
                    "ref_id": ref_id,
                    "citation_marker": ref.get("title") or ref_id,
                    "text": snippet,
                    "section": None,
                    "intents": [],
                }
            )
            if len(out) >= limit:
                return out
    return out


def _collect_bibliography(paper_dir: Path, limit: int = 80) -> List[Dict[str, Any]]:
    refs = _load_json(paper_dir / "references_metadata.json", [])
    if isinstance(refs, list) and refs:
        return [_normalise_reference_record(ref) for ref in refs[:limit] if isinstance(ref, dict)]

    bibtex = _read_text(paper_dir / "references.bib")
    if bibtex:
        return _parse_bibtex_entries(bibtex, limit)
    return []


def _collect_citation_contexts(paper_dir: Path, limit: int = 60) -> List[Dict[str, Any]]:
    refs = _load_json(paper_dir / "references_metadata.json", [])
    out = []
    if isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            ref_record = _normalise_reference_record(ref)
            for context in ref.get("contextsWithIntent") or []:
                if not isinstance(context, dict):
                    continue
                text = context.get("context") or context.get("text") or ""
                if not text:
                    continue
                out.append(
                    {
                        "ref_id": ref_record.get("ref_id"),
                        "citation_marker": ref_record.get("title"),
                        "text": text,
                        "section": context.get("section"),
                        "intents": context.get("intents", []),
                    }
                )
                if len(out) >= limit:
                    return out
    contexts = _load_json(paper_dir / "usage_contexts.json", [])
    if isinstance(contexts, list):
        for item in contexts:
            entry = {
                "ref_id": item.get("ref_id") or item.get("bib_key"),
                "citation_marker": item.get("citation_marker"),
                "text": item.get("text") or item.get("text_raw") or "",
                "section": item.get("section"),
            }
            if entry["text"]:
                out.append(entry)
            if len(out) >= limit:
                break
    if not out:
        out = _collect_bibtex_citation_contexts(paper_dir, limit=limit)
    return out


def _collect_downstream_cluster_evidence(paper_dir: Path) -> List[Dict[str, Any]]:
    discovery = _normalize_dict_payload(_load_json(paper_dir / "usage_discovery_from_contributions.json", {}))
    clusters = discovery.get("clusters", [])
    out = []
    for cluster in clusters:
        out.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "representative_claim": cluster.get("representative_claim") or cluster.get("cluster_title"),
                "cluster_title": cluster.get("cluster_title"),
                "count": cluster.get("count"),
                "merge_rationale": cluster.get("merge_rationale"),
            }
        )
    return out


def load_paper_package(paper_dir: str | Path, extracted_claim_override: str | None = None) -> PaperPackage:
    paper_dir = Path(paper_dir)
    discovery = _normalize_dict_payload(_load_json(paper_dir / "usage_discovery_from_contributions.json", {}))
    paper_metadata = _normalize_dict_payload(_load_json(paper_dir / "paper_metadata.json", {}))
    claim = extracted_claim_override or (
        discovery.get("most_impactful_contribution_self_contained")
        or discovery.get("most_impactful_contribution")
        or ""
    )
    return PaperPackage(
        paper_dir=paper_dir,
        paper_metadata=paper_metadata,
        extracted_discovery_claim=claim,
        downstream_cluster_evidence=_collect_downstream_cluster_evidence(paper_dir),
        paper_text=_collect_sections(paper_dir),
        full_processed_text=_collect_full_processed_text(paper_dir),
        bibliography=_collect_bibliography(paper_dir),
        citation_contexts=_collect_citation_contexts(paper_dir),
    )
