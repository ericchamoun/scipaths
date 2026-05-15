import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.llm_client import LLMClient

from prompts import build_contribution_prompt
from schemas import CONTRIBUTION_JSON_SCHEMA


PAPER_META_FILE = "paper_metadata.json"
USAGE_CONTEXTS_FILE = "usage_contexts.json"
ARXIV_PARAGRAPHS_FILE = "usage_citing_paragraphs.json"
VERIFIED_FILE = "usage_uses_extends_verified.json"
OUT_FILE = "usage_contributions.json"


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_paper_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for child in root.iterdir():
        if child.is_dir() and (child / PAPER_META_FILE).exists():
            out.append(child)
    return out


def _normalize_author_last(name: str) -> str:
    parts = [p for p in (name or "").split() if p.strip()]
    return parts[-1] if parts else ""


def extract_target_info(meta: Any) -> Dict[str, str]:
    if isinstance(meta, list) and meta:
        meta = meta[0]
    if not isinstance(meta, dict):
        return {
            "title": "",
            "first_author_last": "",
            "year": "",
            "tldr": "",
            "abstract": "",
        }
    authors = meta.get("authors") or []
    first_author = authors[0]["name"] if authors else ""
    tldr = ""
    tldr_obj = meta.get("tldr")
    if isinstance(tldr_obj, dict):
        tldr = tldr_obj.get("text", "")
    return {
        "title": meta.get("title", ""),
        "first_author_last": _normalize_author_last(first_author),
        "year": str(meta.get("year", "")),
        "tldr": tldr,
        "abstract": meta.get("abstract", ""),
    }


def build_citing_contexts_map_from_paragraphs(
    arxiv_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    citing_map: Dict[str, Dict[str, Any]] = {}
    for entry in arxiv_data.get("citing_papers", []) or []:
        if not isinstance(entry, dict):
            continue
        citing_id = entry.get("citing_paper_id") or ""
        contexts = []
        seen = set()
        for paragraph in entry.get("target_citing_paragraphs", []) or []:
            paragraph = (paragraph or "").strip()
            if not paragraph:
                continue
            combined = f"Target-citing paragraph: {paragraph}"
            norm = " ".join(combined.split()).lower()
            if norm in seen:
                continue
            seen.add(norm)
            contexts.append(combined)
        citing_map[citing_id] = {
            "title": entry.get("citing_title", ""),
            "paper_id": citing_id,
            "contexts": contexts,
            "source": "arxiv_paragraphs",
        }
    return citing_map


def build_citing_contexts_map_from_usage(
    usage: Dict[str, Any],
    confirmed_texts_by_citing: Dict[str, set] | None,
) -> Dict[str, Dict[str, Any]]:
    citing_map: Dict[str, Dict[str, Any]] = {}
    for entry in usage.get("citing_papers", []) or []:
        if not isinstance(entry, dict):
            continue
        citing_id = entry.get("citing_paper_id") or ""
        allowed_texts = confirmed_texts_by_citing.get(citing_id) if confirmed_texts_by_citing else None
        contexts = []
        seen = set()
        for c in entry.get("contexts", []) or []:
            if not isinstance(c, dict):
                continue
            text = (c.get("context_with_marker") or c.get("text") or "").strip()
            if not text:
                continue
            if allowed_texts is not None and text not in allowed_texts:
                continue
            norm = " ".join(text.split()).lower()
            if norm in seen:
                continue
            seen.add(norm)
            contexts.append(f"Target sentence: {text}")
        citing_map[citing_id] = {
            "title": entry.get("title", ""),
            "paper_id": citing_id,
            "contexts": contexts,
            "source": "usage_contexts_fallback",
        }
    return citing_map


def extract_contribution(
    client: LLMClient,
    target_info: Dict[str, str],
    citing_info: Dict[str, Any],
) -> Dict[str, Any]:
    contexts = citing_info.get("contexts", [])
    prompt = build_contribution_prompt(target_info, citing_info, contexts)
    raw = client.call(prompt, schema=CONTRIBUTION_JSON_SCHEMA)
    data = _parse_llm_json(raw)
    if not isinstance(data, dict):
        return {
            "citing_paper_id": citing_info.get("paper_id", ""),
            "citing_title": citing_info.get("title", ""),
            "label": "NOT_CONFIRMED",
            "paper_claim": "",
            "claim": "",
            "cluster_title": "",
            "cluster_key": "",
            "evidence_span": "",
            "rationale": "",
            "contexts": contexts,
            "source": citing_info.get("source", "unknown"),
        }
    label = data.get("label", "NOT_CONFIRMED")
    paper_claim = data.get("paper_claim", "") or data.get("claim", "")
    cluster_title = data.get("cluster_title", "") or data.get("cluster_claim", "")
    cluster_key = data.get("cluster_key", "")
    evidence_span = data.get("evidence_span", "")
    if not evidence_span:
        label = "NOT_CONFIRMED"
        paper_claim = ""
        cluster_title = ""
        cluster_key = ""
    if label in {"USES", "EXTENDS"} and not cluster_title:
        cluster_title = paper_claim
    if label in {"USES", "EXTENDS"} and not cluster_key:
        cluster_key = f"{label}|contribution|unspecified"
    return {
        "citing_paper_id": citing_info.get("paper_id", ""),
        "citing_title": citing_info.get("title", ""),
        "label": label,
        "paper_claim": paper_claim,
        "claim": paper_claim,
        "cluster_title": cluster_title,
        "cluster_key": cluster_key,
        "evidence_span": evidence_span,
        "rationale": data.get("rationale", ""),
        "contexts": contexts,
        "source": citing_info.get("source", "unknown"),
    }


def _parse_llm_json(raw: str) -> Any | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def process_paper(
    paper_dir: Path,
    client: LLMClient,
    overwrite: bool,
    resume: bool,
) -> str:
    verified = load_json(paper_dir / VERIFIED_FILE)
    if not isinstance(verified, dict):
        return "missing_verified"
    out_path = paper_dir / OUT_FILE
    if out_path.exists() and (resume or not overwrite):
        return "skipped"

    if verified.get("final_label") == "NOT_CONFIRMED":
        payload = {
            "paper_id": verified.get("paper_id"),
            "final_label": "NOT_CONFIRMED",
            "contributions": [],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return "no_confirmed"

    arxiv_data = load_json(paper_dir / ARXIV_PARAGRAPHS_FILE)
    if not isinstance(arxiv_data, dict):
        return "missing_arxiv_paragraphs"

    target_info = extract_target_info(load_json(paper_dir / PAPER_META_FILE))
    citing_map = build_citing_contexts_map_from_paragraphs(arxiv_data)
    usage = load_json(paper_dir / USAGE_CONTEXTS_FILE)
    confirmed_texts_by_citing: Dict[str, set] = {}
    for item in verified.get("confirmed", []) or []:
        citing_id = item.get("citing_paper_id") or ""
        text = item.get("text") or ""
        if not citing_id or not text:
            continue
        confirmed_texts_by_citing.setdefault(citing_id, set()).add(text)
    usage_map = (
        build_citing_contexts_map_from_usage(usage, confirmed_texts_by_citing)
        if isinstance(usage, dict)
        else {}
    )

    confirmed = verified.get("confirmed", [])
    confirmed_ids = {item.get("citing_paper_id") for item in confirmed if item.get("citing_paper_id")}
    contributions: List[Dict[str, Any]] = []
    fallback_citing_ids: List[str] = []
    for citing_id in confirmed_ids:
        citing_info = citing_map.get(citing_id)
        if citing_info and not citing_info.get("contexts"):
            citing_info = None
        if not citing_info:
            fallback = usage_map.get(citing_id)
            if fallback and fallback.get("contexts"):
                citing_info = fallback
                fallback_citing_ids.append(citing_id)
            else:
                continue
        contributions.append(extract_contribution(client, target_info, citing_info))

    payload = {
        "paper_id": verified.get("paper_id"),
        "final_label": verified.get("final_label"),
        "contributions": contributions,
        "source": "arxiv_paragraphs",
        "fallback_citing_ids": fallback_citing_ids,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return "labeled"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-citing-paper contribution claims from verified USES/EXTENDS."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="runs/processed_papers",
        help="Root directory containing processed paper directories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing usage_contributions.json files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip papers with existing output files (even if --overwrite is set).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    client = LLMClient()
    paper_dirs = sorted(iter_paper_dirs(root), key=lambda p: p.name)
    print(f"[INFO] Found {len(paper_dirs)} paper dirs under {root}")

    counts = {
        "labeled": 0,
        "skipped": 0,
        "missing_verified": 0,
        "missing_arxiv_paragraphs": 0,
        "no_confirmed": 0,
    }
    for paper_dir in paper_dirs:
        status = process_paper(paper_dir, client, args.overwrite, args.resume)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{status.upper()}] {paper_dir.name}")

    print(
        "[SUMMARY] labeled={labeled}, skipped={skipped}, missing_verified={missing_verified}, "
        "missing_arxiv_paragraphs={missing_arxiv_paragraphs}, no_confirmed={no_confirmed}".format(**counts)
    )


if __name__ == "__main__":
    main()
