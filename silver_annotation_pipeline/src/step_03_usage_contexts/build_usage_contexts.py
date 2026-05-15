import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


PAPER_META_FILE = "paper_metadata.json"
CITATIONS_FILE = "citations_metadata.json"
DEFAULT_OUT_NAME = "usage_contexts.json"


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] could not parse JSON at {path}: {e}")
        return None


def iter_paper_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for child in root.iterdir():
        if child.is_dir() and (child / PAPER_META_FILE).exists():
            out.append(child)
    return out


def _extract_contexts(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []

    raw = item.get("contextsWithIntent") or []
    if isinstance(raw, list) and raw:
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            text_raw = (entry.get("context") or "").strip()
            text = (entry.get("context_with_marker") or text_raw).strip()
            intents = entry.get("intents") or []
            contexts.append(
                {
                    "text": text,
                    "text_raw": text_raw,
                    "intents": intents,
                }
            )

    # Fallback for older schema that only stores raw context strings.
    if not contexts:
        raw_alt = item.get("contexts") or []
        if isinstance(raw_alt, list):
            for text in raw_alt:
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if text:
                    contexts.append(
                        {
                            "text": text,
                            "intents": [],
                        }
                    )

    return contexts


def build_usage_contexts_for_paper(paper_dir: Path) -> Optional[Dict[str, Any]]:
    citations_path = paper_dir / CITATIONS_FILE
    data = load_json(citations_path)
    if data is None:
        return None

    if not isinstance(data, list):
        print(f"[WARN] {paper_dir.name}: {CITATIONS_FILE} is not a list")
        return None

    citing_entries: List[Dict[str, Any]] = []
    total_contexts = 0
    citing_with_context = 0
    influential_citations = 0
    influential_with_context = 0
    influential_contexts: List[Dict[str, Any]] = []

    for item in data:
        if not isinstance(item, dict):
            continue
        citing = item.get("citingPaper") or {}

        contexts = _extract_contexts(item)
        is_influential = bool(item.get("isInfluential", False))
        if is_influential:
            influential_citations += 1
        if contexts:
            citing_with_context += 1
            total_contexts += len(contexts)
            if is_influential:
                influential_with_context += 1

        citing_entries.append(
            {
                "citing_paper_id": citing.get("paperId"),
                "title": citing.get("title"),
                "external_ids": citing.get("externalIds") or {},
                "is_influential": is_influential,
                "contexts": contexts,
            }
        )
        if is_influential and contexts:
            influential_contexts.append(
                {
                    "citing_paper_id": citing.get("paperId"),
                    "title": citing.get("title"),
                    "external_ids": citing.get("externalIds") or {},
                    "contexts": contexts,
                }
            )

    payload = {
        "paper_id": paper_dir.name,
        "total_citations": len(data),
        "num_contexts": total_contexts,
        "num_citing_with_context": citing_with_context,
        "num_citing_without_context": len(data) - citing_with_context,
        "num_influential_citations": influential_citations,
        "num_influential_with_context": influential_with_context,
        "influential_contexts": influential_contexts,
        "citing_papers": citing_entries,
    }
    return payload


def run(root: Path, out_name: str, overwrite: bool) -> None:
    root = root.resolve()
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    paper_dirs = sorted(iter_paper_dirs(root), key=lambda p: p.name)
    print(f"[INFO] Found {len(paper_dirs)} paper dirs under {root}")

    for paper_dir in paper_dirs:
        out_path = paper_dir / out_name
        if out_path.exists() and not overwrite:
            print(f"[SKIP] {paper_dir.name}: {out_name} already exists")
            continue
        payload = build_usage_contexts_for_paper(paper_dir)
        if payload is None:
            continue

        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[OK] {paper_dir.name}: wrote {out_name} "
            f"({payload['num_contexts']} contexts from {payload['total_citations']} citations)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build usage_contexts.json from citations_metadata.json files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="processed_papers/acl_2024",
        help="Root directory containing processed_papers/acl_2024/<paper_id> dirs.",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default=DEFAULT_OUT_NAME,
        help="Output filename to write inside each paper dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing usage_contexts.json files.",
    )
    args = parser.parse_args()

    run(Path(args.root), out_name=args.out_name, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
