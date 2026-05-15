import time
import random
import requests
from typing import Optional, Tuple, Any
from config import SEMANTIC_SCHOLAR_API_KEY
import os

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"

_LAST_REQUEST_TS = 0.0


def _min_interval_sleep() -> None:
    """Global throttle to avoid hammering Semantic Scholar."""
    global _LAST_REQUEST_TS
    min_interval = float(os.getenv("S2_MIN_INTERVAL", "1.0"))
    now = time.monotonic()
    elapsed = now - _LAST_REQUEST_TS
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _LAST_REQUEST_TS = time.monotonic()


def robust_request(url, params=None, headers=None, max_retries=8, base_sleep=2.0):
    """
    Make a GET request with exponential backoff.
    Retries on:
        - connection errors
        - 429 (Too Many Requests)
        - 500–599 server errors
        - invalid JSON
    Returns (status_code, json_or_None).
    """

    for attempt in range(max_retries):
        try:
            _min_interval_sleep()
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            status = resp.status_code

            if status == 200:
                try:
                    return 200, resp.json()
                except Exception:
                    print(f"[WARN] JSON decode failed on attempt {attempt+1}/{max_retries}")

            if status == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep = float(retry_after)
                    except Exception:
                        sleep = base_sleep * (2 ** attempt)
                else:
                    sleep = base_sleep * (2 ** attempt)
                max_sleep = float(os.getenv("S2_MAX_BACKOFF", "60"))
                sleep = min(sleep, max_sleep)
                sleep += random.uniform(0.0, 0.5)
                print(f"[WARN] 429 Too Many Requests → retrying in {sleep:.2f}s")
                time.sleep(sleep)
                continue

            if 500 <= status < 600:
                sleep = base_sleep * (2 ** attempt)
                max_sleep = float(os.getenv("S2_MAX_BACKOFF", "60"))
                sleep = min(sleep, max_sleep)
                sleep += random.uniform(0.0, 0.5)
                print(f"[WARN] Server error {status} → retrying in {sleep:.2f}s")
                time.sleep(sleep)
                continue

            return status, None

        except requests.exceptions.RequestException as e:
            sleep = base_sleep * (2 ** attempt)
            max_sleep = float(os.getenv("S2_MAX_BACKOFF", "60"))
            sleep = min(sleep, max_sleep)
            sleep += random.uniform(0.0, 0.5)
            print(f"[WARN] Network error {e} → retrying in {sleep:.2f}s")
            time.sleep(sleep)
            continue

    print(f"[ERROR] Giving up after {max_retries} attempts for URL: {url}")
    return None, None
def get_paper(paper_id: str, id_type: str = "ACL") -> Tuple[int, Optional[dict]]:
    """
    id_type can be "ACL" or "SemanticScholar" or "ArXiv" etc.
    """
    if id_type == "SemanticScholar":
        full_id = paper_id
    else:
        full_id = f"{id_type}:{paper_id}"

    url = f"{BASE_URL}/{full_id}"
    params = {
        "fields": (
            "title,year,publicationDate,authors,url,venue,externalIds,"
            "tldr,abstract,citationCount,referenceCount,openAccessPdf"
        )
    }

    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}

    status, data = robust_request(url, params=params, headers=headers, max_retries=5, base_sleep=1.0)
    if status == 200 and data is not None:
        return status, data
    else:
        print(f"[WARN] {status} on {full_id}")
        return status or 0, None


def get_paper_links(semantic_id: str, target_type: str, total: int, limit: int = 1000):
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
    loops = total // limit + 1 if total else 0
    collected = []

    for i in range(loops):
        offset = i * limit
        url = f"{BASE_URL}/{semantic_id}/{target_type}"
        params = {
            "offset": offset,
            "limit": limit,
            "fields": "paperId,title,isInfluential,externalIds,contextsWithIntent,openAccessPdf",
        }

        status, data = robust_request(url, params=params, headers=headers, max_retries=5, base_sleep=1.0)

        if status != 200 or data is None:
            print(f"[WARN] {target_type} fetch failed for {semantic_id} (status {status})")
            return status or 0, []

        items = data.get("data")
        if not isinstance(items, list):
            print(f"[WARN] malformed {target_type} response for {semantic_id}")
            return status, []

        collected.extend(items)

    return 200, collected


def search_by_title(title: str, limit: int = 1):
    """Search Semantic Scholar by paper title."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": title,
        "limit": limit,
        "fields": "paperId,title,year,venue,externalIds",
    }
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}

    status, data = robust_request(url, params=params, headers=headers, max_retries=5, base_sleep=1.0)
    if status == 200 and data is not None:
        items = data.get("data", [])
        return items[0] if items else None
    else:
        print(f"[WARN] title search failed for '{title[:60]}...' (status {status})")
        return None
