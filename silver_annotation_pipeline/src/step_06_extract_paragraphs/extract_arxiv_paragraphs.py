import argparse
import json
import random
import re
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


PAPER_META_FILE = "paper_metadata.json"
USAGE_CONTEXTS_FILE = "usage_contexts.json"
VERIFIED_FILE = "usage_uses_extends_verified.json"
OUT_FILE = "usage_citing_paragraphs.json"


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


def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    for member in tar.getmembers():
        member_path = path / member.name
        if not str(member_path.resolve()).startswith(str(path.resolve())):
            raise RuntimeError(f"Blocked path traversal in tar: {member.name}")
    tar.extractall(path)


_ARXIV_LAST_TS = 0.0


def _arxiv_min_interval_sleep() -> None:
    """Global throttle to avoid arXiv API rate limits."""
    global _ARXIV_LAST_TS
    min_interval = float(os.getenv("ARXIV_MIN_INTERVAL", "1.0"))
    now = time.monotonic()
    elapsed = now - _ARXIV_LAST_TS
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _ARXIV_LAST_TS = time.monotonic()


def download_arxiv_source(arxiv_id: str, tmpdir: Path) -> Optional[Path]:
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    archive_path = tmpdir / f"{arxiv_id.replace('/', '_')}.tar"
    max_retries = int(os.getenv("ARXIV_MAX_RETRIES", "6"))
    base_sleep = float(os.getenv("ARXIV_BASE_SLEEP", "2.0"))
    max_sleep = float(os.getenv("ARXIV_MAX_BACKOFF", "60"))

    for attempt in range(max_retries):
        try:
            _arxiv_min_interval_sleep()
            urllib.request.urlretrieve(url, archive_path)  # noqa: S310
            try:
                with tarfile.open(archive_path) as tar:
                    safe_extract(tar, tmpdir)
                return tmpdir
            except tarfile.ReadError as exc:
                print(f"[WARN] Invalid arXiv archive for {arxiv_id}: {exc}")
                return None
        except Exception as exc:
            # arXiv sometimes returns 429; treat any network error as retryable.
            sleep = min(base_sleep * (2 ** attempt), max_sleep) + random.uniform(0.0, 0.5)
            print(f"[WARN] Failed to download arXiv source for {arxiv_id}: {exc}")
            print(f"[WARN] arXiv download retrying in {sleep:.2f}s")
            time.sleep(sleep)
            continue

    print(f"[ERROR] Giving up after {max_retries} attempts for arXiv {arxiv_id}")
    return None


def find_main_tex(root: Path) -> Optional[Path]:
    tex_files = list(root.rglob("*.tex"))
    if not tex_files:
        return None

    candidates: List[Tuple[int, Path]] = []
    for path in tex_files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        score = 0
        if "\\begin{document}" in text:
            score += 3
        if "\\documentclass" in text:
            score += 2
        score += len(text) // 1000
        candidates.append((score, path))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1] if candidates else None


def read_bib_files(root: Path) -> Dict[str, str]:
    bibs: Dict[str, str] = {}
    for path in root.rglob("*.bib"):
        try:
            bibs[str(path.relative_to(root))] = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
    return bibs


def normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if t]


def paragraphize(text: str) -> List[str]:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n\s*\n", "\n\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def strip_latex_comments(text: str) -> str:
    # Remove explicit comment environments first.
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", text, flags=re.DOTALL)

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        out_chars: List[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "%":
                # Keep escaped percent (\%) and continue parsing.
                if i > 0 and line[i - 1] == "\\":
                    out_chars.append(ch)
                    i += 1
                    continue
                # Unescaped percent starts a LaTeX comment; ignore rest of the line.
                break
            out_chars.append(ch)
            i += 1
        cleaned_lines.append("".join(out_chars))
    return "\n".join(cleaned_lines)


def parse_bib_entries(bib_text: str) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    matches = list(re.finditer(r"@[\w]+\s*\{\s*([^,]+),", bib_text))
    for i, match in enumerate(matches):
        key = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(bib_text)
        body = bib_text[start:end]
        fields = {}
        for f_match in re.finditer(r"(\w+)\s*=\s*[{|\"](.+?)[}|\"]\s*,", body, re.DOTALL):
            fields[f_match.group(1).lower()] = f_match.group(2).strip()
        entries.append({"key": key, **fields})
    return entries


def find_target_bib_keys(
    bib_texts: Dict[str, str],
    target_info: Dict[str, str],
) -> List[str]:
    target_title = normalize_text(target_info.get("title", ""))
    target_author = normalize_text(target_info.get("first_author_last", ""))
    target_year = target_info.get("year", "")
    if not target_title and not target_author:
        return []

    keys: List[str] = []
    for bib_text in bib_texts.values():
        for entry in parse_bib_entries(bib_text):
            title = normalize_text(entry.get("title", ""))
            author = normalize_text(entry.get("author", ""))
            year = str(entry.get("year", ""))
            has_title = bool(title)
            title_match = target_title and (target_title in title or title in target_title)
            author_match = target_author and target_author in author
            year_match = target_year and target_year in year

            if title_match and author_match:
                keys.append(entry["key"])
            elif not has_title and author_match and year_match:
                keys.append(entry["key"])
            elif author_match and year_match:
                keys.append(entry["key"])
    return keys


def replace_target_citations(text: str, target_keys: List[str], target_info: Dict[str, str]) -> str:
    key_set = set(target_keys or [])
    author = target_info.get("first_author_last", "").lower()
    year = target_info.get("year", "")
    alt_years = {year}
    if year.isdigit():
        alt_years.add(str(int(year) - 1))
        alt_years.add(str(int(year) + 1))

    def repl(match: re.Match) -> str:
        keys = [k.strip() for k in match.group(1).split(",")]
        for key in keys:
            if key in key_set:
                return "<CITED HERE>"
            key_lc = key.lower()
            if author and author in key_lc and any(y in key_lc for y in alt_years if y):
                return "<CITED HERE>"
        return match.group(0)

    return re.sub(r"\\cite[a-zA-Z]*\s*\{([^}]+)\}", repl, text)


def match_paragraphs(
    paragraphs: List[str],
    contexts: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    para_tokens = [set(tokenize(p)) for p in paragraphs]

    for idx, ctx in enumerate(contexts, start=1):
        ctx_text = ctx.get("text", "")
        ctx_tokens = set(tokenize(ctx_text))
        if not ctx_tokens:
            continue
        best = None
        best_score = 0.0
        for p_idx, tokens in enumerate(para_tokens):
            if not tokens:
                continue
            overlap = len(ctx_tokens & tokens) / max(1, len(ctx_tokens))
            if overlap > best_score:
                best = p_idx
                best_score = overlap
        if best is not None and best_score >= 0.5:
            paragraph = paragraphs[best]
            results.append(
                {
                    "context_id": idx,
                    "context": ctx_text,
                    "context_with_marker": ctx.get("text_with_marker", ctx_text),
                    "paragraph": paragraph,
                    "overlap": round(best_score, 3),
                }
            )
    return results


def _normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _normalize_for_match(text: str) -> str:
    text = text.replace("<CITED HERE>", "")
    text = re.sub(r"\[[^\]]+\]", "", text)
    return _normalize_text(text)


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


def build_citing_contexts_map(
    usage: Dict[str, Any],
    confirmed_texts_by_citing: Dict[str, set] | None,
) -> Dict[str, Dict[str, Any]]:
    citing_map: Dict[str, Dict[str, Any]] = {}
    for entry in usage.get("citing_papers", []) or []:
        if not isinstance(entry, dict):
            continue
        citing_id = entry.get("citing_paper_id") or ""
        allowed_texts = confirmed_texts_by_citing.get(citing_id) if confirmed_texts_by_citing else None
        allowed_norms = (
            {_normalize_for_match(text) for text in allowed_texts} if allowed_texts else None
        )
        contexts = []
        seen = set()
        for c in entry.get("contexts", []) or []:
            if not isinstance(c, dict):
                continue
            text_raw = (c.get("text") or "").strip()
            text_with_marker = (c.get("context_with_marker") or text_raw).strip()
            if not text_raw:
                continue
            norm = _normalize_for_match(text_raw)
            if allowed_norms is not None and norm not in allowed_norms:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            contexts.append({"text": text_raw, "text_with_marker": text_with_marker})
        if allowed_texts is not None and not contexts:
            for text in allowed_texts:
                norm = _normalize_for_match(text)
                if norm in seen:
                    continue
                seen.add(norm)
                contexts.append({"text": text, "text_with_marker": text})
        citing_map[citing_id] = {
            "title": entry.get("title", ""),
            "paper_id": citing_id,
            "arxiv_id": (entry.get("external_ids") or {}).get("ArXiv", ""),
            "contexts": contexts,
        }
    return citing_map


def process_citing_paper(citing: Dict[str, Any]) -> Dict[str, Any]:
    target_info = citing.get("target_info", {})
    arxiv_id = citing.get("arxiv_id", "")
    if not arxiv_id:
        return {"error": "missing_arxiv_id", **citing}

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        if not download_arxiv_source(arxiv_id, tmpdir):
            return {"error": "bad_arxiv_archive", **citing}
        main_tex = find_main_tex(tmpdir)
        if not main_tex:
            return {"error": "missing_main_tex", **citing}

        tex_text = main_tex.read_text(encoding="utf-8", errors="ignore")
        tex_text = strip_latex_comments(tex_text)
        bibs = read_bib_files(tmpdir)
        target_keys = find_target_bib_keys(bibs, target_info)
        tex_text = replace_target_citations(tex_text, target_keys, target_info)
        paragraphs = paragraphize(tex_text)
        target_citing_paragraphs = [p for p in paragraphs if "<CITED HERE>" in p]
        matched = match_paragraphs(paragraphs, citing.get("contexts", []))

        return {
            "citing_paper_id": citing.get("paper_id", ""),
            "citing_title": citing.get("title", ""),
            "arxiv_id": arxiv_id,
            "main_tex_file": str(main_tex.relative_to(tmpdir)),
            "bib_files": list(bibs.keys()),
            "bib_texts": bibs,
            "target_bib_keys": target_keys,
            "contexts": citing.get("contexts", []),
            "target_citing_paragraphs": target_citing_paragraphs,
            "matched_paragraphs": matched,
        }


def process_paper(root: Path, overwrite: bool, include_all: bool, resume: bool) -> str:
    usage = load_json(root / USAGE_CONTEXTS_FILE)
    if not isinstance(usage, dict):
        return "missing_usage"

    out_path = root / OUT_FILE
    if out_path.exists() and (resume or not overwrite):
        return "skipped"

    verified = None
    confirmed_texts_by_citing: Dict[str, set] = {}
    if not include_all:
        verified = load_json(root / VERIFIED_FILE)
        if not isinstance(verified, dict):
            return "missing_verified"
        for item in verified.get("confirmed", []) or []:
            citing_id = item.get("citing_paper_id") or ""
            text = item.get("text") or ""
            if not citing_id or not text:
                continue
            confirmed_texts_by_citing.setdefault(citing_id, set()).add(text)

    target_info = extract_target_info(load_json(root / PAPER_META_FILE))
    citing_map = build_citing_contexts_map(
        usage,
        confirmed_texts_by_citing if confirmed_texts_by_citing else None,
    )
    if not citing_map:
        out_path.write_text(
            json.dumps({"paper_id": usage.get("paper_id"), "citing_papers": []}, indent=2),
            encoding="utf-8",
        )
        return "empty_citing"

    confirmed_ids: Optional[set] = None
    if not include_all and isinstance(verified, dict):
        confirmed = verified.get("confirmed", [])
        confirmed_ids = {
            item.get("citing_paper_id")
            for item in confirmed
            if item.get("citing_paper_id")
        }

    citing_papers = []
    for citing_id, citing in citing_map.items():
        if confirmed_ids is not None and citing_id not in confirmed_ids:
            continue
        citing["target_info"] = target_info
        citing_papers.append(process_citing_paper(citing))

    payload = {"paper_id": usage.get("paper_id"), "citing_papers": citing_papers}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return "processed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download arXiv sources and extract citation-local paragraphs."
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
        help="Overwrite existing usage_citing_paragraphs.json files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all citing papers (not just confirmed USES/EXTENDS).",
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

    paper_dirs = sorted(iter_paper_dirs(root), key=lambda p: p.name)
    print(f"[INFO] Found {len(paper_dirs)} paper dirs under {root}")

    counts = {
        "processed": 0,
        "skipped": 0,
        "missing_usage": 0,
        "missing_verified": 0,
        "empty_citing": 0,
    }
    for paper_dir in paper_dirs:
        status = process_paper(paper_dir, args.overwrite, args.all, args.resume)
        counts[status] = counts.get(status, 0) + 1
        print(f"[{status.upper()}] {paper_dir.name}")

    print(
        "[SUMMARY] processed={processed}, skipped={skipped}, missing_usage={missing_usage}, "
        "missing_verified={missing_verified}, empty_citing={empty_citing}".format(**counts)
    )


if __name__ == "__main__":
    main()
