import os
import re
from pathlib import Path
import shutil

def read_tex(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def resolve_inputs(tex: str, base_dir: Path, seen=None) -> str:
    """
    Recursively replace \\input{...} and \\include{...} with file contents.
    """
    if seen is None:
        seen = set()

    pattern = r'\\(?:input|include)\{([^}]+)\}'

    def repl(match):
        name = match.group(1)
        if not name.endswith(".tex"):
            name += ".tex"

        full = base_dir / name

        if full in seen:
            return f"% WARNING: skipped circular input {full}\n"

        if not full.exists():
            return f"% WARNING: missing file {full}\n"

        seen.add(full)
        content = read_tex(full)
        return resolve_inputs(content, full.parent, seen)

    return re.sub(pattern, repl, tex)


def find_main_tex(source_dir: Path) -> Path | None:
    """
    Heuristic to find the main .tex file:
    1. match .bbl → .tex
    2. else top-level .tex that contains \\begin{document}
    3. else first .tex in directory
    """
    bbls = list(source_dir.glob("*.bbl"))
    if bbls:
        main_candidate = source_dir / (bbls[0].stem + ".tex")
        if main_candidate.exists():
            return main_candidate

    for tex in source_dir.glob("*.tex"):
        if "\\begin{document}" in read_tex(tex):
            return tex

    tex_files = list(source_dir.glob("*.tex"))
    return tex_files[0] if tex_files else None


def preprocess_tex(source_dir: Path) -> Path | None:
    """
    Given an extracted arXiv source directory, produce:
      - a merged TeX file named 'processed_main.tex'
      - a concatenated BibTeX file named 'references.bib'
    Both are written in the parent directory of source_dir (the paper dir).

    Then delete the extracted source_dir.
    """
    main_tex = find_main_tex(source_dir)
    if not main_tex:
        print(f"[WARN] No main .tex found in {source_dir}")
        shutil.rmtree(source_dir, ignore_errors=True)
        return None

    raw = read_tex(main_tex)
    merged = resolve_inputs(raw, main_tex.parent)

    paper_dir = source_dir.parent
    out_tex_path = paper_dir / "processed_main.tex"
    out_tex_path.write_text(merged, encoding="utf-8")

    bib_files = list(source_dir.rglob("*.bib"))
    if bib_files:
        bib_texts = []
        for bib in bib_files:
            try:
                bib_texts.append(bib.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                print(f"[WARN] Could not read bib file {bib}")
        if bib_texts:
            bib_out = paper_dir / "references.bib"
            bib_out.write_text("\n\n".join(bib_texts), encoding="utf-8")
            print(f"[INFO] Wrote combined BibTeX to {bib_out}")

    shutil.rmtree(source_dir, ignore_errors=True)

    return out_tex_path

def _load_tex(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


SECTION_PATTERN = re.compile(
    r'\\section\*?\{([^}]*)\}',
    flags=re.IGNORECASE
)


def _split_into_sections(tex: str):
    """
    Returns a list of (section_title, content) in order.
    Title is the raw LaTeX title text (without braces).
    Content is the text from this \\section line up to (but not including)
    the next \\section or end of document.
    """
    sections = []
    matches = list(SECTION_PATTERN.finditer(tex))

    if not matches:
        return sections

    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(tex)
        content = tex[start:end]
        sections.append((title, content))

    return sections


def _normalize_title(title: str) -> str:
    """Lowercase and strip punctuation-ish stuff for robust matching."""
    t = title.lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def _find_best_section(sections, candidates):
    """
    sections: list of (raw_title, content)
    candidates: list of strings to match against normalized title
    Returns the content of the best-matching section or None.
    """
    norm_candidates = [c.lower() for c in candidates]

    for raw_title, content in sections:
        nt = _normalize_title(raw_title)
        for cand in norm_candidates:
            if nt == cand or cand in nt:
                return content
    return None


def extract_introduction_and_related(
    processed_tex_path: Path,
    out_dir: Path | None = None,
) -> dict:
    """
    Given path to processed_main.tex, extract Introduction and Related Work sections
    into separate .tex files.

    Returns a dict with keys:
        {
          "introduction": Path | None,
          "related_work": Path | None
        }
    """
    if out_dir is None:
        out_dir = processed_tex_path.parent / "sections"

    out_dir.mkdir(parents=True, exist_ok=True)

    tex = _load_tex(processed_tex_path)
    sections = _split_into_sections(tex)

    intro_candidates = ["introduction"]
    related_candidates = ["related work"]

    intro_content = _find_best_section(sections, intro_candidates)
    related_content = _find_best_section(sections, related_candidates)

    results = {"introduction": None, "related_work": None}

    if intro_content:
        intro_path = out_dir / "introduction.tex"
        intro_path.write_text(intro_content, encoding="utf-8")
        results["introduction"] = intro_path
    else:
        print(f"[WARN] No Introduction section found in {processed_tex_path}")

    if related_content:
        rw_path = out_dir / "related_work.tex"
        rw_path.write_text(related_content, encoding="utf-8")
        results["related_work"] = rw_path
    else:
        print(f"[WARN] No Related Work section found in {processed_tex_path}")

    return results
