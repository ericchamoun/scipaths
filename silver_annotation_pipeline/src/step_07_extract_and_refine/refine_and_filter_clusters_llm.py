import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.llm_client import LLMClient

PAPER_META_FILE = "paper_metadata.json"
CONTRIB_FILE = "usage_contributions.json"
DISCOVERY_FILE = "usage_discovery_from_contributions.json"
OUT_FILE = "usage_discovery_from_contributions_refined.json"

REFINE_SCHEMA = {
    "type": "object",
    "properties": {
        "kept_groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_ids": {"type": "array", "items": {"type": "string"}},
                    "merged_title": {"type": "string"},
                    "merged_key": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["cluster_ids", "merged_title", "merged_key", "rationale"],
            },
        },
        "dropped_clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "cluster_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["cluster_id", "reason"],
            },
        },
    },
    "required": ["kept_groups", "dropped_clusters"],
}


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_paper_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / PAPER_META_FILE).exists()], key=lambda p: p.name)


def _to_int_indices(raw_indices: List[Any]) -> List[int]:
    out: List[int] = []
    for i in raw_indices or []:
        try:
            out.append(int(i))
        except Exception:
            continue
    return out


def _parse_key(key: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in str(key or "").split("|")]
    if len(parts) >= 3:
        return parts[0].upper(), parts[1], parts[2]
    return "", "", ""


def _dominant_key(member_clusters: List[Dict[str, Any]]) -> str:
    rel_count: Dict[str, int] = {}
    art_count: Dict[str, int] = {}
    pur_count: Dict[str, int] = {}
    for c in member_clusters:
        rel, art, pur = _parse_key(c.get("cluster_key", ""))
        if rel:
            rel_count[rel] = rel_count.get(rel, 0) + 1
        if art:
            art_count[art] = art_count.get(art, 0) + 1
        if pur:
            pur_count[pur] = pur_count.get(pur, 0) + 1
    rel = max(rel_count, key=rel_count.get) if rel_count else "USES"
    art = max(art_count, key=art_count.get) if art_count else "contribution"
    pur = max(pur_count, key=pur_count.get) if pur_count else "unspecified"
    return f"{rel}|{art}|{pur}"


def _extract_title(meta: Any) -> str:
    if isinstance(meta, list) and meta:
        meta = meta[0]
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("title", ""))


def _title_from_cluster_key(cluster_key: str) -> str:
    parts = [p.strip() for p in str(cluster_key or "").split("|")]
    if len(parts) >= 3:
        relation, artifact, purpose = parts[0], parts[1], parts[2]
        relation_txt = "Uses" if relation.upper() == "USES" else "Extends"
        artifact_txt = artifact.replace("_", " ")
        purpose_txt = purpose.replace("_", " ")
        return f"{relation_txt} {artifact_txt} for {purpose_txt}".strip()
    return cluster_key or ""


def _cluster_by_exact_keys(keys: List[str]) -> List[List[int]]:
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for i, key in enumerate(keys):
        k = (key or "").strip()
        if not k:
            k = f"__EMPTY__::{i}"
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(i)
    return [groups[k] for k in order]


def _build_initial_clusters_from_contributions(contrib: Dict[str, Any]) -> List[Dict[str, Any]]:
    contributions = [
        c for c in contrib.get("contributions", []) or []
        if c.get("label") in {"USES", "EXTENDS"} and (c.get("paper_claim") or c.get("claim"))
    ]
    if not contributions:
        return []
    cluster_keys_all: List[str] = []
    for c in contributions:
        key = (c.get("cluster_key") or "").strip()
        if not key:
            label = str(c.get("label", "USES")).upper()
            if label not in {"USES", "EXTENDS"}:
                label = "USES"
            key = f"{label}|contribution|unspecified"
        cluster_keys_all.append(key)
    clusters = _cluster_by_exact_keys(cluster_keys_all)
    out: List[Dict[str, Any]] = []
    for idx, cluster in enumerate(clusters, start=1):
        first = contributions[cluster[0]]
        key = cluster_keys_all[cluster[0]]
        title = (first.get("cluster_title") or "").strip() or _title_from_cluster_key(key)
        out.append({
            "cluster_id": f"C{idx}",
            "count": str(len(cluster)),
            "representative_claim": title,
            "cluster_key": key,
            "cluster_title": title,
            "claim_indices": [str(i) for i in cluster],
        })
    return out


def _cluster_support_summary(cluster: Dict[str, Any], contributions: List[Dict[str, Any]]) -> Dict[str, Any]:
    indices = _to_int_indices(cluster.get("claim_indices") or [])
    items: List[Dict[str, Any]] = []
    for i in indices:
        if 0 <= i < len(contributions):
            items.append(contributions[i])
    labels = [str(item.get("label", "")).upper() for item in items if item.get("label")]
    examples: List[str] = []
    for item in items:
        text = str(item.get("paper_claim") or item.get("claim") or "").strip()
        if text:
            examples.append(text)
        if len(examples) >= 3:
            break
    rationales = [str(item.get("rationale", "")).strip() for item in items if item.get("rationale")][:2]
    use_count = sum(1 for x in labels if x == "USES")
    ext_count = sum(1 for x in labels if x == "EXTENDS")
    return {
        "examples": examples,
        "rationales": rationales,
        "uses_count": use_count,
        "extends_count": ext_count,
        "member_count": len(items),
    }


def build_prompt(paper_title: str, centroids: List[Dict[str, Any]]) -> str:
    lines: List[str] = [
        "You are refining downstream citation contribution clusters for one target paper.",
        "Input clusters are already built. Your job is to (a) conservatively merge near-duplicate downstream-usage clusters and (b) drop clusters that do not actually show substantive downstream usage of the target contribution.",
        "",
        f"Target paper: {paper_title}",
        "",
        "Rules:",
        "- Operate only at cluster level. Do not invent new instances.",
        "- Prefer conservative merges. If unsure, keep clusters separate.",
        "- You may drop clusters only when they fail to show real downstream use or extension of the target contribution.",
        "- Drop clusters that are clearly mere mention, loose comparison, background citation, noisy extraction, or off-target usage.",
        "- Never merge USES and EXTENDS clusters together.",
        "- Every input cluster_id must either appear in exactly one kept group or in dropped_clusters.",
        "- kept merged_key must be in format RELATION|artifact|purpose.",
        "- merged_title must be a short natural-language summary (5-12 words).",
        "",
        "Input clusters:",
    ]
    for c in centroids:
        lines.append(
            f"- {c['cluster_id']}: key={c.get('cluster_key','')}; title={c.get('cluster_title','')}; count={c.get('count', 0)}; uses={c.get('uses_count',0)}; extends={c.get('extends_count',0)}; examples={' | '.join(c.get('examples',[])[:2])}; rationales={' | '.join(c.get('rationales',[])[:1])}"
        )
    lines += [
        "",
        "Return JSON only with this shape:",
        "{",
        '  "kept_groups": [',
        "    {",
        '      "cluster_ids": ["C1","C3"],',
        '      "merged_title": "Uses target dataset for evaluation",',
        '      "merged_key": "USES|dataset|evaluation",',
        '      "rationale": "Both clusters describe the same downstream dataset use."',
        "    }",
        "  ],",
        '  "dropped_clusters": [',
        '    {"cluster_id": "C7", "reason": "Only background mention; no substantive downstream use."}',
        "  ]",
        "}",
    ]
    return "\n".join(lines)


def _normalize_decision(data: Dict[str, Any], original_clusters: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid_ids = [c.get("cluster_id", "") for c in original_clusters if c.get("cluster_id")]
    valid_set = set(valid_ids)
    assigned = set()
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for item in data.get("dropped_clusters") or []:
        cid = item.get("cluster_id")
        if cid in valid_set and cid not in assigned:
            assigned.add(cid)
            dropped.append({"cluster_id": cid, "reason": str(item.get("reason", "")).strip() or "Dropped by LLM filter."})

    for g in data.get("kept_groups") or []:
        ids = [cid for cid in (g.get("cluster_ids") or []) if cid in valid_set and cid not in assigned]
        if not ids:
            continue
        for cid in ids:
            assigned.add(cid)
        kept.append({
            "cluster_ids": ids,
            "merged_title": str(g.get("merged_title", "")).strip(),
            "merged_key": str(g.get("merged_key", "")).strip(),
            "rationale": str(g.get("rationale", "")).strip(),
        })

    for cid in valid_ids:
        if cid not in assigned:
            kept.append({
                "cluster_ids": [cid],
                "merged_title": "",
                "merged_key": "",
                "rationale": "Auto-singleton fallback.",
            })

    order = {cid: i for i, cid in enumerate(valid_ids)}
    kept.sort(key=lambda g: min(order[cid] for cid in g["cluster_ids"]))
    dropped.sort(key=lambda x: order.get(x["cluster_id"], 10**9))
    return kept, dropped


def refine_paper(paper_dir: Path, overwrite: bool, inplace: bool) -> str:
    disc_path = paper_dir / DISCOVERY_FILE
    contrib_path = paper_dir / CONTRIB_FILE
    meta_path = paper_dir / PAPER_META_FILE

    disc = load_json(disc_path)
    contrib = load_json(contrib_path)
    meta = load_json(meta_path)
    if not isinstance(contrib, dict):
        return "missing_inputs"

    if not isinstance(disc, dict):
        disc = {"paper_id": contrib.get("paper_id"), "decision": "", "justification": "", "clusters": []}

    clusters = disc.get("clusters") or []
    if not clusters:
        clusters = _build_initial_clusters_from_contributions(contrib)
    if not clusters:
        payload = dict(disc)
        payload["clusters"] = []
        payload["dropped_clusters"] = []
        payload["cluster_refine_method"] = "llm_centroid_merge_filter"
        payload["cluster_refine_source"] = CONTRIB_FILE
        out_path = disc_path if inplace else (paper_dir / OUT_FILE)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return "empty_clusters"

    out_path = disc_path if inplace else (paper_dir / OUT_FILE)
    if out_path.exists() and not overwrite:
        return "skipped"

    contributions = contrib.get("contributions") or []
    centroids: List[Dict[str, Any]] = []
    auto_dropped: List[Dict[str, Any]] = []
    active_clusters: List[Dict[str, Any]] = []

    for c in clusters:
        cid = c.get("cluster_id", "")
        summary = _cluster_support_summary(c, contributions)
        rel, _, _ = _parse_key(c.get("cluster_key", ""))
        if summary["uses_count"] + summary["extends_count"] == 0 or rel not in {"USES", "EXTENDS"}:
            auto_dropped.append({"cluster_id": cid, "reason": "No verified USES/EXTENDS support in member contributions."})
            continue
        row = {
            "cluster_id": cid,
            "cluster_key": c.get("cluster_key", ""),
            "cluster_title": c.get("cluster_title") or c.get("representative_claim") or "",
            "count": int(c.get("count", summary["member_count"]) or summary["member_count"]),
            **summary,
        }
        centroids.append(row)
        active_clusters.append(c)

    if not active_clusters:
        payload = dict(disc)
        payload["clusters"] = []
        payload["dropped_clusters"] = auto_dropped
        payload["cluster_refine_method"] = "llm_centroid_merge_filter"
        payload["cluster_refine_source"] = CONTRIB_FILE if not load_json(disc_path) else DISCOVERY_FILE
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return "refined"

    prompt = build_prompt(_extract_title(meta), centroids)
    client = LLMClient()
    raw = client.call(prompt, schema=REFINE_SCHEMA)
    data = json.loads(raw)
    kept_groups, llm_dropped = _normalize_decision(data, active_clusters)

    id_to_cluster = {c.get("cluster_id"): c for c in active_clusters if c.get("cluster_id")}
    merged_clusters: List[Dict[str, Any]] = []
    for idx, g in enumerate(kept_groups, start=1):
        member_ids = g["cluster_ids"]
        members = [id_to_cluster[mid] for mid in member_ids if mid in id_to_cluster]
        merged_indices: List[int] = []
        for m in members:
            for i in _to_int_indices(m.get("claim_indices") or []):
                if i not in merged_indices:
                    merged_indices.append(i)
        merged_indices.sort()
        merged_key = g.get("merged_key") or _dominant_key(members)
        rel, _, _ = _parse_key(merged_key)
        if rel not in {"USES", "EXTENDS"}:
            merged_key = _dominant_key(members)
        merged_title = g.get("merged_title") or (members[0].get("cluster_title") if members else "")
        if not merged_title:
            merged_title = members[0].get("representative_claim", "") if members else ""
        merged_clusters.append({
            "cluster_id": f"C{idx}",
            "count": str(len(merged_indices)),
            "representative_claim": merged_title,
            "cluster_key": merged_key,
            "cluster_title": merged_title,
            "claim_indices": [str(i) for i in merged_indices],
            "source_cluster_ids": member_ids,
            "merge_rationale": g.get("rationale", ""),
        })

    payload = dict(disc)
    payload["clusters"] = merged_clusters
    payload["dropped_clusters"] = auto_dropped + llm_dropped
    payload["cluster_refine_method"] = "llm_centroid_merge_filter"
    payload["cluster_refine_source"] = CONTRIB_FILE if not load_json(disc_path) else DISCOVERY_FILE
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return "refined"


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM centroid-level merge/filter pass for downstream contribution clusters.")
    parser.add_argument("--root", type=str, default="runs/processed_papers", help="Root directory containing processed paper directories.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    parser.add_argument("--inplace", action="store_true", help="Write back to usage_discovery_from_contributions.json.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    paper_dirs = iter_paper_dirs(root)
    print(f"[INFO] Found {len(paper_dirs)} paper dirs under {root}")
    counts = {"refined": 0, "skipped": 0, "missing_inputs": 0, "empty_clusters": 0}
    for paper_dir in paper_dirs:
        status = refine_paper(paper_dir, overwrite=args.overwrite, inplace=args.inplace)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{status.upper()}] {paper_dir.name}")
    print("[SUMMARY] refined={refined}, skipped={skipped}, missing_inputs={missing_inputs}, empty_clusters={empty_clusters}".format(**counts))


if __name__ == "__main__":
    main()
