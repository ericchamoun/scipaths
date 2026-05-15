import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


PAPER_META_FILE = "paper_metadata.json"
USAGE_CLAIMS_FILE = "usage_claims.json"
USAGE_CONTEXTS_FILE = "usage_contexts.json"
CITATIONS_FILE = "citations_metadata.json"
PROCESSED_MAIN_FILE = "processed_main.tex"
REFERENCES_META_FILE = "references_metadata.json"


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def iter_paper_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for child in root.iterdir():
        if child.is_dir() and (child / PAPER_META_FILE).exists():
            out.append(child)
    return out


def load_paper_metadata(paper_dir: Path) -> Dict[str, Any]:
    meta = load_json(paper_dir / PAPER_META_FILE)
    if isinstance(meta, list) and meta:
        return meta[0]
    if isinstance(meta, dict):
        return meta
    return {}


def _is_structurally_complete(paper_dir: Path) -> bool:
    return (
        (paper_dir / PAPER_META_FILE).exists()
        and (paper_dir / PROCESSED_MAIN_FILE).exists()
        and (paper_dir / REFERENCES_META_FILE).exists()
    )


def _author_last_names(authors: List[Any]) -> List[str]:
    last_names: List[str] = []
    for author in authors:
        if isinstance(author, dict):
            name = author.get("name")
        else:
            name = author
        if not isinstance(name, str):
            continue
        parts = [p for p in re.split(r"\s+", name.strip()) if p]
        if not parts:
            continue
        last_names.append(parts[-1])
    return list(dict.fromkeys(last_names))


def _title_aliases(title: str) -> List[str]:
    aliases = [title]
    if ":" in title:
        aliases.append(title.split(":", 1)[0])
    acronym = "".join([c for c in title if c.isupper()])
    if 3 <= len(acronym) <= 10:
        aliases.append(acronym)
    return list(dict.fromkeys([a for a in aliases if a]))


def _artifact_aliases(paper_dir: Path) -> List[str]:
    aliases: List[str] = []
    usage_claims = load_json(paper_dir / USAGE_CLAIMS_FILE)
    if isinstance(usage_claims, dict):
        caps = usage_claims.get("capabilities") or []
        if isinstance(caps, list):
            for cap in caps:
                if not isinstance(cap, dict):
                    continue
                name = cap.get("artifact_name")
                if isinstance(name, str) and name.strip():
                    aliases.append(name.strip())
    return list(dict.fromkeys(aliases))


def _loose_alias_pattern(alias: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", alias)
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return r"\b" + r"[-\s]*".join(map(re.escape, parts)) + r"\b"


def build_patterns(
    meta: Dict[str, Any],
    paper_dir: Path,
) -> Tuple[List[re.Pattern], List[re.Pattern], str | None]:
    year = meta.get("year")
    year_str = str(year) if isinstance(year, int) else None
    authors = meta.get("authors") if isinstance(meta.get("authors"), list) else []
    last_names = _author_last_names(authors)
    title = meta.get("title") if isinstance(meta.get("title"), str) else ""
    aliases = _title_aliases(title) + _artifact_aliases(paper_dir)
    aliases = [a for a in aliases if a]

    author_patterns: List[re.Pattern] = []
    alias_patterns: List[re.Pattern] = []

    if year_str and last_names:
        year_pat = rf"{re.escape(year_str)}[a-z]?"
        first_last = re.escape(last_names[0])
        author_patterns.append(
            re.compile(
                rf"\b{first_last}\s+et\s+al\.?\s*(?:,\s*|\s*){year_pat}",
                re.IGNORECASE,
            )
        )

    for alias in aliases:
        pat = _loose_alias_pattern(alias)
        if pat:
            alias_patterns.append(re.compile(pat, re.IGNORECASE))

    first_last = last_names[0] if last_names else None
    return author_patterns, alias_patterns, first_last


def _replace_author_span(text: str, first_last: str) -> Tuple[str, bool]:
    occurrences = list(re.finditer(rf"\b{re.escape(first_last)}\b", text, re.IGNORECASE))
    if len(occurrences) != 1:
        return text, False
    author_pat = re.compile(
        rf"\(?\b{re.escape(first_last)}\b"
        rf"\s+(?:et\s+al\.?|and|&)\s*"
        rf"(?:,?\s*\(?\d{{4}}[a-z]?\)?)?"
        rf"\)?",
        re.IGNORECASE,
    )
    new_text, count = author_pat.subn("<CITED HERE>", text, count=1)
    return new_text, count > 0


_BRACKET_NUM_RE = re.compile(r"\[[0-9,;\s]+\]")
_BRACKET_GROUP_RE = re.compile(r"\[([0-9,;\s]+)\]")


def _extract_bracket_numbers(text: str) -> List[str]:
    numbers: List[str] = []
    for match in _BRACKET_GROUP_RE.finditer(text):
        parts = re.split(r"[,\s;]+", match.group(1).strip())
        for part in parts:
            if part.isdigit():
                numbers.append(part)
    return numbers


def _dominant_bracket(contexts: List[Dict[str, Any]]) -> str | None:
    counts: Dict[str, int] = {}
    for ctx in contexts:
        if not isinstance(ctx, dict):
            continue
        text = ctx.get("context") or ctx.get("text")
        if not isinstance(text, str):
            continue
        for num in _extract_bracket_numbers(text):
            counts[num] = counts.get(num, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    winners = [num for num, count in counts.items() if count == best]
    if len(winners) == 1:
        return winners[0]
    return None


def _single_bracket_candidate(contexts: List[Dict[str, Any]]) -> str | None:
    counts: Dict[str, int] = {}
    for ctx in contexts:
        if not isinstance(ctx, dict):
            continue
        text = ctx.get("context") or ctx.get("text")
        if not isinstance(text, str):
            continue
        matches = list(_BRACKET_GROUP_RE.finditer(text))
        if len(matches) == 1:
            nums = _extract_bracket_numbers(text)
            if len(nums) != 1:
                continue
            num = nums[0]
            counts[num] = counts.get(num, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    winners = [num for num, count in counts.items() if count == best]
    if len(winners) == 1:
        return winners[0]
    return None


def _replace_single_bracket(text: str, dominant: str | None) -> Tuple[str, bool]:
    matches = list(_BRACKET_GROUP_RE.finditer(text))
    if len(matches) != 1:
        return text, False
    nums = _extract_bracket_numbers(text)
    if len(nums) != 1:
        return text, False
    num = nums[0]
    if dominant is not None and num != dominant:
        return text, False
    start, end = matches[0].span()
    return text[:start] + "<CITED HERE>" + text[end:], True


def replace_with_marker(
    text: str,
    author_patterns: List[re.Pattern],
    alias_patterns: List[re.Pattern],
    dominant_bracket: str | None = None,
    first_author_last: str | None = None,
) -> Tuple[str, bool]:
    def _collapse_markers(value: str) -> str:
        value = re.sub(r"(?:<CITED HERE>[\s()\[\],;:]*){2,}", "<CITED HERE> ", value)
        value = re.sub(r"<CITED HERE>(?:\s+<CITED HERE>)+", "<CITED HERE>", value)
        return value.strip()

    updated = text
    changed = False

    author_changed = False
    if first_author_last:
        new, author_changed = _replace_author_span(updated, first_author_last)
        if author_changed:
            changed = True
            updated = _collapse_markers(new)

    if dominant_bracket:
        def _replace_if_contains(match: re.Match) -> str:
            nums = re.split(r"[,\s;]+", match.group(1).strip())
            if any(n == dominant_bracket for n in nums if n.isdigit()):
                return "<CITED HERE>"
            return match.group(0)

        new = _BRACKET_GROUP_RE.sub(_replace_if_contains, updated)
        if new != updated:
            changed = True
            updated = _collapse_markers(new)

    for pat in author_patterns:
        new = pat.sub("<CITED HERE>", updated)
        if new != updated:
            changed = True
            updated = _collapse_markers(new)

    if not author_changed:
        for pat in alias_patterns:
            new = pat.sub("<CITED HERE>", updated)
            if new != updated:
                changed = True
                updated = _collapse_markers(new)

    new, bracket_changed = _replace_single_bracket(updated, dominant_bracket)
    if bracket_changed:
        changed = True
        updated = _collapse_markers(new)

    updated = _collapse_markers(updated)
    return updated, changed


def _process_contexts(
    contexts: List[Dict[str, Any]],
    author_patterns: List[re.Pattern],
    alias_patterns: List[re.Pattern],
    dominant_bracket: str | None,
    first_author_last: str | None,
) -> Tuple[int, int]:
    updated_count = 0
    total = 0
    for ctx in contexts:
        if not isinstance(ctx, dict):
            continue
        text = ctx.get("context") or ctx.get("text")
        if not isinstance(text, str):
            continue
        total += 1
        new_text, changed = replace_with_marker(
            text,
            author_patterns=author_patterns,
            alias_patterns=alias_patterns,
            dominant_bracket=dominant_bracket,
            first_author_last=first_author_last,
        )
        if changed:
            updated_count += 1
        ctx["context_with_marker"] = new_text
    return updated_count, total


def update_citations_file(
    paper_dir: Path,
    author_patterns: List[re.Pattern],
    alias_patterns: List[re.Pattern],
    first_author_last: str | None,
) -> Tuple[int, int]:
    path = paper_dir / CITATIONS_FILE
    data = load_json(path)
    if not isinstance(data, list):
        return 0, 0
    updated = 0
    total = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        ctxs = entry.get("contextsWithIntent") or []
        if isinstance(ctxs, list):
            dominant = _dominant_bracket(ctxs)
            if dominant is None:
                dominant = _single_bracket_candidate(ctxs)
            upd, tot = _process_contexts(
                ctxs,
                author_patterns,
                alias_patterns,
                dominant,
                first_author_last,
            )
            updated += upd
            total += tot
    save_json(path, data)
    return updated, total


def update_usage_contexts_file(
    paper_dir: Path,
    author_patterns: List[re.Pattern],
    alias_patterns: List[re.Pattern],
    first_author_last: str | None,
) -> Tuple[int, int]:
    path = paper_dir / USAGE_CONTEXTS_FILE
    data = load_json(path)
    if not isinstance(data, dict):
        return 0, 0
    updated = 0
    total = 0
    for entry in data.get("citing_papers", []) or []:
        if not isinstance(entry, dict):
            continue
        ctxs = entry.get("contexts") or []
        if isinstance(ctxs, list):
            dominant = _dominant_bracket(ctxs)
            if dominant is None:
                dominant = _single_bracket_candidate(ctxs)
            upd, tot = _process_contexts(
                ctxs,
                author_patterns,
                alias_patterns,
                dominant,
                first_author_last,
            )
            updated += upd
            total += tot
    save_json(path, data)
    return updated, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace citation mentions with <CITED HERE> in context fields."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="runs/processed_papers",
        help="Root directory containing processed paper directories.",
    )
    parser.add_argument(
        "--usage-contexts",
        action="store_true",
        help="Also update usage_contexts.json.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root directory does not exist: {root}")

    paper_dirs = sorted(iter_paper_dirs(root), key=lambda p: p.name)
    print(f"[INFO] Found {len(paper_dirs)} paper dirs under {root}")
    total_updated = 0
    total_contexts = 0
    skipped_incomplete = 0

    for paper_dir in paper_dirs:
        if not _is_structurally_complete(paper_dir):
            skipped_incomplete += 1
            continue
        meta = load_paper_metadata(paper_dir)
        if not meta:
            continue
        author_patterns, alias_patterns, first_author_last = build_patterns(meta, paper_dir)
        if not (author_patterns or alias_patterns):
            continue
        updated, total = update_citations_file(
            paper_dir,
            author_patterns,
            alias_patterns,
            first_author_last,
        )
        total_updated += updated
        total_contexts += total
        if args.usage_contexts:
            upd_usage, tot_usage = update_usage_contexts_file(
                paper_dir,
                author_patterns,
                alias_patterns,
                first_author_last,
            )
            updated += upd_usage
            total += tot_usage
            total_updated += upd_usage
            total_contexts += tot_usage
        if total:
            print(f"[OK] {paper_dir.name}: updated {updated} contexts over {total}")

    print(
        f"[SUMMARY] total_updated={total_updated} over {total_contexts}; "
        f"skipped_incomplete={skipped_incomplete}"
    )


if __name__ == "__main__":
    main()
