from pathlib import Path
import argparse
import json
import os
import random
import re
import tarfile
import time

import arxiv
import requests

from config import ACL_IDS_PATH
from process_tex_source import preprocess_tex, extract_introduction_and_related
from semanticscholar_client import get_paper, get_paper_links, search_by_title


def load_ids(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


_ARXIV_LAST_TS = 0.0


def _cleanup_partial_source_dir(source_dir: Path) -> None:
    for pattern in ("*.tar.gz", "*.tgz", "*.tar"):
        for path in source_dir.glob(pattern):
            try:
                path.unlink()
            except Exception:
                pass


def _download_arxiv_source_with_retries(paper, source_dir: Path, arxiv_id: str) -> Path | None:
    max_retries = int(os.getenv("ARXIV_SOURCE_MAX_RETRIES", "4"))
    base_sleep = float(os.getenv("ARXIV_SOURCE_BASE_SLEEP", "2.0"))
    max_sleep = float(os.getenv("ARXIV_MAX_BACKOFF", "60"))
    last_exc = None

    for attempt in range(max_retries):
        _cleanup_partial_source_dir(source_dir)
        try:
            _arxiv_min_interval_sleep()
            tar_path = Path(paper.download_source(dirpath=str(source_dir)))
            if not tar_path.exists():
                raise FileNotFoundError(f"download_source returned {tar_path}, but the file does not exist")
            if tar_path.stat().st_size < 1024:
                raise IOError(f"downloaded source archive is unexpectedly small ({tar_path.stat().st_size} bytes)")
            return tar_path
        except Exception as exc:
            last_exc = exc
            sleep = min(base_sleep * (2**attempt), max_sleep) + random.uniform(0.0, 0.5)
            print(f"[WARN] Failed to download source for {arxiv_id} on attempt {attempt + 1}/{max_retries}: {exc}")
            if attempt + 1 < max_retries:
                print(f"[INFO] Retrying source download in {sleep:.2f}s")
                time.sleep(sleep)

    print(f"[WARN] Source download failed for {arxiv_id} after {max_retries} attempts: {last_exc}")
    return None


def _arxiv_min_interval_sleep() -> None:
    """Global throttle to avoid arXiv API rate limits."""
    global _ARXIV_LAST_TS
    min_interval = float(os.getenv("ARXIV_MIN_INTERVAL", "1.0"))
    now = time.monotonic()
    elapsed = now - _ARXIV_LAST_TS
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _ARXIV_LAST_TS = time.monotonic()


def download_arxiv_tex(arxiv_id: str, base_dir: Path) -> Path | None:
    """
    Download LaTeX source from arXiv and return the path to a merged TeX file.

    - arxiv_id: e.g. "2410.22815"
    - base_dir: paper directory where source should be unpacked
    """
    source_dir = base_dir / f"tex_{arxiv_id}"
    source_dir.mkdir(parents=True, exist_ok=True)
    search = arxiv.Search(id_list=[arxiv_id])
    max_retries = int(os.getenv("ARXIV_MAX_RETRIES", "6"))
    base_sleep = float(os.getenv("ARXIV_BASE_SLEEP", "2.0"))
    max_sleep = float(os.getenv("ARXIV_MAX_BACKOFF", "60"))
    paper = None

    for attempt in range(max_retries):
        try:
            _arxiv_min_interval_sleep()
            paper = next(search.results())
            break
        except StopIteration:
            print(f"[WARN] No arXiv paper found for ID {arxiv_id}")
            return None
        except arxiv.HTTPError as exc:
            if getattr(exc, "status", None) == 429 or "429" in str(exc):
                sleep = min(base_sleep * (2**attempt), max_sleep) + random.uniform(0.0, 0.5)
                print(f"[WARN] arXiv 429 → retrying in {sleep:.2f}s")
                time.sleep(sleep)
                continue
            print(f"[WARN] arXiv HTTP error for {arxiv_id}: {exc}")
            return None
        except Exception as exc:
            sleep = min(base_sleep * (2**attempt), max_sleep) + random.uniform(0.0, 0.5)
            print(f"[WARN] arXiv error {exc} → retrying in {sleep:.2f}s")
            time.sleep(sleep)
            continue

    if paper is None:
        print(f"[ERROR] Giving up after {max_retries} attempts for arXiv ID {arxiv_id}")
        return None

    tar_path = _download_arxiv_source_with_retries(paper, source_dir, arxiv_id)
    if tar_path is None:
        return None

    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=source_dir)
        os.remove(tar_path)
    except Exception as exc:
        print(f"[WARN] Failed to extract source for {arxiv_id}: {exc}")
        return None

    processed_tex = preprocess_tex(source_dir)
    if processed_tex:
        extract_introduction_and_related(processed_tex)

    if not processed_tex or not processed_tex.exists():
        print(f"[WARN] Could not produce merged TeX for {arxiv_id}")
        return None

    print(f"[INFO] Processed LaTeX for {arxiv_id} at {processed_tex}")
    return processed_tex


def _extract_arxiv_id_from_text(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b", text)
    if match:
        return match.group(1)
    match = re.search(r"arxiv[:\s/]*(\d{4}\.\d{4,5}(?:v\d+)?)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _safe_write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _query_openreview_for_paper(openreview_id: str) -> dict | None:
    """Query OpenReview using a real OpenReview note/forum id."""
    if not openreview_id:
        return None

    try_urls = [
        f"https://api.openreview.net/notes?forum={openreview_id}",
        f"https://api2.openreview.net/notes?forum={openreview_id}",
        f"https://api.openreview.net/notes?id={openreview_id}",
        f"https://api2.openreview.net/notes?id={openreview_id}",
    ]

    for url in try_urls:
        try:
            response = requests.get(url, timeout=20)
            if response.status_code != 200:
                continue
            payload = response.json()
        except Exception:
            continue

        notes = None
        if isinstance(payload, dict) and isinstance(payload.get("notes"), list):
            notes = payload["notes"]
        elif isinstance(payload, dict) and payload.get("content"):
            notes = [payload]
        elif isinstance(payload, list):
            notes = payload

        if not notes:
            continue

        note = notes[0]
        content = note.get("content") if isinstance(note, dict) else None
        title = None
        arxiv_id = None
        pdf_url = None

        if isinstance(content, dict):
            raw_title = content.get("title") or content.get("paperTitle")
            title = raw_title.get("value") if isinstance(raw_title, dict) else raw_title

            raw_pdf = content.get("pdf")
            pdf_url = raw_pdf.get("value") if isinstance(raw_pdf, dict) else raw_pdf

            for value in content.values():
                if isinstance(value, dict):
                    value = value.get("value")
                if isinstance(value, list):
                    value = " ".join(str(item) for item in value)
                if isinstance(value, str):
                    arxiv_id = _extract_arxiv_id_from_text(value)
                    if arxiv_id:
                        break

        if not title and isinstance(note, dict):
            title = note.get("title") or note.get("forumTitle")

        if not arxiv_id and isinstance(note, dict):
            for value in note.values():
                if isinstance(value, str):
                    arxiv_id = _extract_arxiv_id_from_text(value)
                    if arxiv_id:
                        break

        return {
            "title": title,
            "arxiv_id": arxiv_id,
            "pdf_url": pdf_url,
            "openreview_id": openreview_id,
            "source_url": url,
        }

    return None


def _treat_as_openreview(paper: dict) -> bool:
    acl_id = str(paper.get("id", "")).lower()
    id_type = str(paper.get("id_type", "")).lower()
    return (
        id_type == "openreview"
        or bool(paper.get("openreview_id"))
        or acl_id.startswith("neurips-")
        or acl_id.startswith("icml-")
    )


def _fetch_s2_by_title(title: str, acl_id: str) -> tuple[int, dict | None]:
    if not title:
        print(f"[WARN] no title available for {acl_id} → skipping.")
        return 0, None
    hit = search_by_title(title)
    if not hit:
        print(f"[WARN] no S2 match for {acl_id} ({title}) → skipping.")
        return 0, None
    s2_id = hit["paperId"]
    print(f"[DEBUG] title search matched semantic scholar paperId={s2_id}")
    return get_paper(s2_id, id_type="SemanticScholar")


def _best_arxiv_id(*values: str) -> str | None:
    for value in values:
        arxiv_id = _extract_arxiv_id_from_text(value or "")
        if arxiv_id:
            return arxiv_id
    return None


def _write_openreview_snapshot(paper_dir: Path, payload: dict) -> None:
    if payload:
        _safe_write_json(paper_dir / "openreview_metadata.json", payload)


def _write_metadata_outputs(paper_dir: Path, acl_id: str, data: dict) -> None:
    meta_path = paper_dir / "paper_metadata.json"
    _safe_write_json(meta_path, [data])
    print(f"[DEBUG] wrote metadata to {meta_path}")

    external_ids = data.get("externalIds", {}) or {}
    arxiv_id = external_ids.get("ArXiv")
    if arxiv_id:
        download_arxiv_tex(arxiv_id=arxiv_id, base_dir=paper_dir)

    sections_dir = paper_dir / "sections"
    sections_dir.mkdir(exist_ok=True)

    abstract = data.get("abstract")
    if abstract:
        _safe_write_text(sections_dir / "abstract.txt", abstract)

    tldr_obj = data.get("tldr")
    if isinstance(tldr_obj, dict) and tldr_obj.get("text"):
        _safe_write_text(sections_dir / "tldr.txt", tldr_obj["text"])

    semantic_id = data.get("paperId")
    if not semantic_id:
        print(f"[WARN] no semantic_id for {acl_id} → skip refs/cites.")
        return

    citation_count = data.get("citationCount", 0)
    reference_count = data.get("referenceCount", 0)

    ref_status, refs = get_paper_links(semantic_id, "references", reference_count)
    if ref_status == 200:
        _safe_write_json(paper_dir / "references_metadata.json", refs)

    cit_status, cits = get_paper_links(semantic_id, "citations", citation_count)
    if cit_status == 200:
        _safe_write_json(paper_dir / "citations_metadata.json", cits)

    if "ArXiv" not in external_ids:
        _safe_write_text(paper_dir / "no_arxiv.txt", "no arxiv for this paper")


def fetch_one_acl_id(paper: dict, base_dir: Path):
    acl_id = paper["id"]
    title = (paper.get("title") or "").strip()
    id_type = paper.get("id_type", "ACL")
    openreview_id = paper.get("openreview_id", "")
    input_pdf_url = paper.get("pdf_url", "")
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    print(
        f"[DEBUG] fetch_one_acl_id: id={acl_id} id_type={id_type} "
        f"title_len={len(title)} s2_key_present={'yes' if bool(s2_key) else 'no'} "
        f"s2_key_len={len(s2_key)}"
    )

    paper_dir = base_dir / acl_id
    ensure_dir(paper_dir)
    meta_path = paper_dir / "paper_metadata.json"

    if meta_path.exists():
        return

    status, data = 0, None
    fetch_label = f"{id_type}:{acl_id}"
    is_openreview = _treat_as_openreview(paper)
    openreview_meta = None
    attempted_title_search = False

    if is_openreview:
        try:
            openreview_meta = _query_openreview_for_paper(openreview_id or acl_id)
        except Exception as exc:
            print(f"[WARN] OpenReview lookup failed for {acl_id}: {exc}")
            openreview_meta = None

        if openreview_meta:
            _write_openreview_snapshot(paper_dir, openreview_meta)
            or_title = (openreview_meta.get("title") or title or "").strip()
            arxiv_id = (
                _best_arxiv_id(
                    openreview_meta.get("arxiv_id", ""),
                    openreview_meta.get("pdf_url", ""),
                    input_pdf_url,
                )
                or ""
            )
            if arxiv_id:
                print(f"[DEBUG] OpenReview -> found ArXiv {arxiv_id} for {acl_id}")
                status, data = get_paper(arxiv_id, id_type="ArXiv")
                fetch_label = f"ArXiv:{arxiv_id}"
                title = or_title or title
            elif or_title:
                print(f"[DEBUG] OpenReview -> no arXiv for {acl_id}, title-searching")
                status, data = _fetch_s2_by_title(or_title, acl_id)
                fetch_label = f"title:{or_title[:80]}"
                title = or_title
                attempted_title_search = True
            else:
                print(f"[WARN] OpenReview metadata for {acl_id} had neither title nor arXiv")
        else:
            print(f"[WARN] no OpenReview metadata for {acl_id} (openreview_id={openreview_id or acl_id})")

        if data is None and title and not attempted_title_search:
            print(f"[DEBUG] OpenReview fallback -> title-searching extracted title for {acl_id}")
            status, data = _fetch_s2_by_title(title, acl_id)
            fetch_label = f"title:{title[:80]}"
            attempted_title_search = True

    if data is None and not is_openreview:
        status, data = get_paper(acl_id, id_type=id_type)
        fetch_label = f"{id_type}:{acl_id}"

    if data is None and not attempted_title_search:
        print(
            f"[WARN] direct fetch failed for {fetch_label} "
            f"(status={status}) → trying title search with title_len={len(title)}"
        )
        status, data = _fetch_s2_by_title(title, acl_id)

    if status != 200 or data is None:
        print(f"[WARN] still no data for {acl_id} → skipping.")
        return

    _write_metadata_outputs(paper_dir, acl_id, data)
    print("[SUCCESS]")


def fetch_all_metadata(acl_ids_path: Path, out_dir: Path, start_from: str | None = None, resume: bool = False):
    raw = json.loads(acl_ids_path.read_text(encoding="utf-8"))
    papers = raw if isinstance(raw[0], dict) else [{"id": x, "title": ""} for x in raw]

    start_seen = start_from is None
    for paper in papers:
        pid = str(paper.get("id", ""))
        if not start_seen:
            if pid == start_from:
                start_seen = True
            else:
                continue
        if resume:
            paper_dir = out_dir / pid
            if (paper_dir / "paper_metadata.json").exists():
                continue
        fetch_one_acl_id(paper, out_dir)
    return "Meta Data Completed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=str, required=True, help="Path to JSON file with paper IDs.")
    parser.add_argument("--outdir", type=str, default="papers", help="Output directory for metadata.")
    parser.add_argument("--start-from", type=str, default=None, help="Start from this paper ID.")
    parser.add_argument("--resume", action="store_true", help="Skip papers that already have paper_metadata.json.")
    args = parser.parse_args()

    ACL_IDS_PATH = Path(args.ids).expanduser().resolve()
    OUTDIR = Path(args.outdir).expanduser().resolve()

    if not ACL_IDS_PATH.exists():
        raise FileNotFoundError(f"Could not find {ACL_IDS_PATH}")

    print(f"[INFO] Using ID list from {ACL_IDS_PATH}")
    print(f"[INFO] Output will be saved to {OUTDIR}")

    start = time.time()
    fetch_all_metadata(acl_ids_path=ACL_IDS_PATH, out_dir=OUTDIR, start_from=args.start_from, resume=args.resume)
    print("done in", time.time() - start, "s")
