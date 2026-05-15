from __future__ import annotations

import asyncio
import json
import os

from dataclasses import dataclass, field
from tenacity import RetryError

try:
    from agents import RunContextWrapper, function_tool
except ModuleNotFoundError:
    RunContextWrapper = object

    def function_tool(func):
        return func


@dataclass
class SearchContext:
    """Tracks per-search state for Recall@k computation and verbose logging."""
    sem: asyncio.Semaphore
    searches: list[list[str]] = field(default_factory=list)   # ordered paper IDs per call
    search_results: list[list[dict]] = field(default_factory=list)  # full result dicts per call
    queries: list[str] = field(default_factory=list)           # one entry per call
    targets: set[str] = field(default_factory=set)             # ground-truth IDs (verbose hit-checking)
    source_paper_id: str = ""                                  # excluded from results (anti-leakage)
    source_paper_ids: set[str] = field(default_factory=set)     # all known IDs for source paper
    source_filtered: list[bool] = field(default_factory=list)  # True if source was removed each call
    max_year: int | None = None                                # exclude known papers after this year
    year_filtered: list[int] = field(default_factory=list)      # number of known future-year papers removed
    max_searches: int | None = None                            # hard cap on tool calls for this task


def _make_client() -> AsyncSemanticScholar:
    from semanticscholar import AsyncSemanticScholar

    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None
    return AsyncSemanticScholar(api_key=api_key)


def _paper_tldr(paper) -> str:
    tldr = getattr(paper, "tldr", None)
    if isinstance(tldr, dict):
        return tldr.get("text") or ""
    if tldr is not None:
        return getattr(tldr, "text", "") or ""
    return ""


async def s2_search(
    query: str,
    sem: asyncio.Semaphore,
    limit: int = 20,
    max_year: int | None = None,
    include_snippets: bool = False,
) -> list[dict]:
    """Search Semantic Scholar. Built-in retry handles 429 automatically."""
    sch = _make_client()
    request_limit = min(100, max(limit, limit * 5)) if max_year is not None else limit
    async with sem:
        fields = ["paperId", "title", "year", "authors"]
        if include_snippets:
            fields.extend(["abstract", "tldr"])
        results = await sch.search_paper(query, limit=request_limit, fields=fields)
    # PaginatedResults auto-fetches all pages when iterated; slice to enforce limit.
    papers = []
    for paper in results[:request_limit]:
        if max_year is not None and paper.year is not None and paper.year > max_year:
            continue
        authors = [a.name for a in (paper.authors or [])[:3]]
        item = {
            "paperId": paper.paperId,
            "title": paper.title,
            "year": paper.year,
            "authors": authors,
        }
        if include_snippets:
            item["abstract"] = getattr(paper, "abstract", None) or ""
            item["tldr"] = _paper_tldr(paper)
        papers.append(item)
        if len(papers) >= limit:
            break
    return papers


@function_tool
async def search_semantic_scholar(ctx: RunContextWrapper[SearchContext], query: str) -> str:
    """Search Semantic Scholar for academic papers matching the query.
    Returns up to 20 results ranked by relevance, each with paperId, title, year, authors.
    Use focused queries — paper title keywords or distinctive technical terms work best.
    """
    if ctx.context.max_searches is not None and len(ctx.context.queries) >= ctx.context.max_searches:
        return json.dumps({
            "error": (
                f"Search budget exhausted after {ctx.context.max_searches} searches. "
                "Do not call search again; return the final JSON using the papers already retrieved."
            )
        })
    ctx.context.queries.append(query)  # record before API call so we know it was attempted
    try:
        papers = await s2_search(query, ctx.context.sem, max_year=ctx.context.max_year)
        ctx.context.year_filtered.append(0)
    except PermissionError:
        return json.dumps({"error": "Semantic Scholar API key rejected (403). Check SEMANTIC_SCHOLAR_API_KEY in .env."})
    except RetryError:
        has_key = bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
        hint = "." if has_key else ". Set SEMANTIC_SCHOLAR_API_KEY in .env to raise rate limits."
        return json.dumps({"error": f"Semantic Scholar rate limit exceeded after retries{hint}"})
    except Exception as e:
        return json.dumps({"error": str(e)})
    source_ids = set(ctx.context.source_paper_ids)
    if ctx.context.source_paper_id:
        source_ids.add(ctx.context.source_paper_id)
    filtered = bool(source_ids) and any(p.get("paperId") in source_ids for p in papers)
    if filtered:
        papers = [p for p in papers if p.get("paperId") not in source_ids]
    ctx.context.source_filtered.append(bool(filtered))

    ids = [p["paperId"] for p in papers if p.get("paperId")]
    ctx.context.searches.append(ids)
    ctx.context.search_results.append(papers)

    return json.dumps(
        [{"rank": i + 1, **{k: p.get(k) for k in ("paperId", "title", "year", "authors")}}
         for i, p in enumerate(papers)],
        ensure_ascii=False,
    )
