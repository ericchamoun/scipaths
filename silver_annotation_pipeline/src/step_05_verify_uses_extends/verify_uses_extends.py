import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.llm_client import LLMClient

from prompts import build_uses_extends_verification_prompt
from schemas import USES_EXTENDS_VERIFICATION_JSON_SCHEMA


PAPER_META_FILE = "paper_metadata.json"
USAGE_LABELS_FILE = "usage_context_labels.json"
OUT_FILE = "usage_uses_extends_verified.json"

USE_LABELS = {"Uses", "Extends"}


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
        return {"title": "", "first_author_last": "", "year": ""}
    authors = meta.get("authors") or []
    first_author = authors[0]["name"] if authors else ""
    return {
        "title": meta.get("title", ""),
        "first_author_last": _normalize_author_last(first_author),
        "year": str(meta.get("year", "")),
    }


def verify_candidates(
    client: LLMClient,
    target_info: Dict[str, str],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    prompt = build_uses_extends_verification_prompt(target_info, candidates)
    try:
        raw = client.call(prompt, schema=USES_EXTENDS_VERIFICATION_JSON_SCHEMA)
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}. Marking all candidates NOT_CONFIRMED.")
        return [
            {
                "id": item.get("id"),
                "label": "NOT_CONFIRMED",
                "cue_span": "",
                "rationale": "",
                "text": item.get("text", ""),
                "citing_paper_id": item.get("citing_paper_id", ""),
                "citing_title": item.get("citing_title", ""),
                "original_label": item.get("original_label", ""),
            }
            for item in candidates
        ]
    data = _parse_llm_json(raw)
    if not isinstance(data, dict):
        print("[WARN] Failed to parse LLM JSON response; marking all candidates NOT_CONFIRMED.")
        return [
            {
                "id": item.get("id"),
                "label": "NOT_CONFIRMED",
                "cue_span": "",
                "rationale": "",
                "text": item.get("text", ""),
                "citing_paper_id": item.get("citing_paper_id", ""),
                "citing_title": item.get("citing_title", ""),
                "original_label": item.get("original_label", ""),
            }
            for item in candidates
        ]
    labels = data.get("labels", [])
    by_id = {item.get("id"): item for item in labels if isinstance(item, dict)}

    verified: List[Dict[str, Any]] = []
    for candidate in candidates:
        item_id = candidate["id"]
        model = by_id.get(item_id, {})
        label = model.get("label", "NOT_CONFIRMED")
        cue_span = model.get("cue_span", "")
        if not cue_span:
            label = "NOT_CONFIRMED"
        verified.append(
            {
                "id": item_id,
                "label": label,
                "cue_span": cue_span,
                "rationale": model.get("rationale", ""),
                "text": candidate.get("text", ""),
                "citing_paper_id": candidate.get("citing_paper_id", ""),
                "citing_title": candidate.get("citing_title", ""),
                "original_label": candidate.get("original_label", ""),
            }
        )
    return verified


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
    k: int,
    batch_size: int,
    overwrite: bool,
    resume: bool,
) -> str:
    labels_path = paper_dir / USAGE_LABELS_FILE
    payload = load_json(labels_path)
    if not isinstance(payload, dict):
        return "missing_labels"

    out_path = paper_dir / OUT_FILE
    if out_path.exists() and (resume or not overwrite):
        return "skipped"

    labels = payload.get("labels", [])
    candidates_all = []
    for item in labels:
        if item.get("label") in USE_LABELS:
            candidates_all.append(
                {
                    "id": item.get("id"),
                    "text": item.get("text", ""),
                    "citing_paper_id": item.get("citing_paper_id", ""),
                    "citing_title": item.get("citing_title", ""),
                    "original_label": item.get("label"),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                }
            )

    if not candidates_all:
        result = {
            "paper_id": payload.get("paper_id"),
            "target": {},
            "candidates_total": 0,
            "candidates_considered": 0,
            "verified": [],
            "confirmed": [],
        }
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return "no_candidates"

    # Keep top-k highest-confidence USES/EXTENDS contexts for LLM verification.
    # If k <= 0, verify all candidates.
    candidates_all = sorted(
        candidates_all,
        key=lambda x: x.get("confidence", 0.0),
        reverse=True,
    )
    candidates = candidates_all if k <= 0 else candidates_all[:k]

    target_info = extract_target_info(load_json(paper_dir / PAPER_META_FILE))
    verified: List[Dict[str, Any]] = []
    if batch_size <= 0:
        batch_size = 25
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        verified.extend(verify_candidates(client, target_info, batch))
    confirmed = [v for v in verified if v["label"] in {"USES", "EXTENDS"}]
    if any(item["label"] == "EXTENDS" for item in confirmed):
        final_label = "EXTENDS"
    elif confirmed:
        final_label = "USES"
    else:
        final_label = "NOT_CONFIRMED"

    result = {
        "paper_id": payload.get("paper_id"),
        "target": target_info,
        "candidates_total": len(candidates_all),
        "candidates_considered": len(candidates),
        "verification_batch_size": int(batch_size),
        "verification_num_batches": (len(candidates) + batch_size - 1) // batch_size if candidates else 0,
        "candidates_selected": len(confirmed),
        "verified": verified,
        "confirmed": confirmed,
        "confirmed_extends": sum(1 for x in confirmed if x.get("label") == "EXTENDS"),
        "confirmed_uses": sum(1 for x in confirmed if x.get("label") == "USES"),
        "final_label": final_label,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return "verified"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify USES/EXTENDS candidates via LLM and select top-K."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="runs/processed_papers",
        help="Root directory containing processed paper directories.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Verify top-k USES/EXTENDS candidates ranked by classifier confidence (<=0 means all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of candidates per LLM verification batch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing usage_uses_extends_verified.json files.",
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

    counts = {"verified": 0, "skipped": 0, "missing_labels": 0, "no_candidates": 0}
    for paper_dir in paper_dirs:
        status = process_paper(
            paper_dir,
            client,
            args.k,
            args.batch_size,
            args.overwrite,
            args.resume,
        )
        counts[status] = counts.get(status, 0) + 1
        print(f"[{status.upper()}] {paper_dir.name}")

    print(
        "[SUMMARY] verified={verified}, skipped={skipped}, missing_labels={missing_labels}, "
        "no_candidates={no_candidates}".format(**counts)
    )


if __name__ == "__main__":
    main()
