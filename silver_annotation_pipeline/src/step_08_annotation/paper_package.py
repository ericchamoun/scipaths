from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from common.paper_package import (
    PaperPackage,
    _collect_bibliography,
    _collect_citation_contexts,
    _collect_full_processed_text,
    _collect_sections,
    _load_json,
    _normalize_dict_payload,
)


def _collect_all_cluster_evidence(paper_dir: Path) -> List[Dict[str, Any]]:
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
                "cluster_key": cluster.get("cluster_key"),
                "claim_indices": cluster.get("claim_indices", []),
                "source_cluster_ids": cluster.get("source_cluster_ids", []),
                "merge_rationale": cluster.get("merge_rationale"),
            }
        )
    return out


def load_paper_package(paper_dir: str | Path, extracted_claim_override: str | None = None) -> PaperPackage:
    paper_dir = Path(paper_dir)
    paper_metadata = _normalize_dict_payload(_load_json(paper_dir / "paper_metadata.json", {}))
    cluster_evidence = _collect_all_cluster_evidence(paper_dir)
    seed = extracted_claim_override or ""
    return PaperPackage(
        paper_dir=paper_dir,
        paper_metadata=paper_metadata,
        extracted_discovery_claim=seed,
        downstream_cluster_evidence=cluster_evidence,
        paper_text=_collect_sections(paper_dir),
        full_processed_text=_collect_full_processed_text(paper_dir),
        bibliography=_collect_bibliography(paper_dir),
        citation_contexts=_collect_citation_contexts(paper_dir),
    )
