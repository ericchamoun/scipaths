from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PIPELINE_ROOT / "src"


def load_dotenv() -> None:
    env_path = PIPELINE_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


STEP_LABELS = {
    1: "Fetch metadata + LaTeX for input paper",
    2: "Add citation markers",
    3: "Build usage contexts",
    4: "Label citation functions",
    5: "Verify USES/EXTENDS",
    6: "Extract arXiv paragraphs",
    7: "Extract target contributions and refine clusters",
    8: "Annotate target contributions and enabling contributions",
}


@dataclass(frozen=True)
class PipelineSettings:
    provider: str = "gemini"
    model: str = "gemini-3.1-pro-preview"
    verification_model: str = "gemini-3-flash-preview"
    annotation_model: str = "gemini/gemini-3.1-pro-preview"
    formatter_model: str = "gemini/gemini-3.1-pro-preview"
    judge_model: str = "gemini/gemini-3.1-pro-preview"
    candidate_count: int = 3
    device: str = "cpu"


def parse_arxiv_id(paper: str) -> str:
    value = (paper or "").strip()
    if not value:
        raise ValueError("Paper input is required.")
    if "arxiv.org" in value:
        match = re.search(r"arxiv\.org/(abs|pdf)/([^/?#]+)", value)
        if not match:
            raise ValueError(f"Could not parse arXiv ID from URL: {value}")
        value = match.group(2)
    value = value.replace(".pdf", "")
    value = re.sub(r"v\d+$", "", value)
    if not re.match(r"^[0-9]{4}\.[0-9]{4,5}$", value):
        raise ValueError(f"Invalid arXiv ID: {value}")
    return value


def create_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        run_dir = output_root / run_id
        try:
            run_dir.mkdir(parents=False, exist_ok=False)
            return run_dir
        except FileExistsError:
            time.sleep(0.1)
    raise RuntimeError(f"Could not create a unique run directory under {output_root}")


def write_ids_file(run_dir: Path, arxiv_ids: Iterable[str]) -> Path:
    payload = [{"id": paper_id, "title": "", "id_type": "ArXiv"} for paper_id in arxiv_ids]
    ids_path = run_dir / "input_ids.json"
    ids_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ids_path


def write_run_config(run_dir: Path, payload: dict) -> None:
    (run_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def iter_paper_dirs(processed_root: Path) -> list[Path]:
    if not processed_root.exists():
        return []
    return sorted(
        [path for path in processed_root.iterdir() if path.is_dir() and (path / "paper_metadata.json").exists()],
        key=lambda path: path.name,
    )


def log_tail(path: Path, max_lines: int = 60) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def env_for_step(settings: PipelineSettings, step: int) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    env["LLM_PROVIDER"] = settings.provider
    env["LLM_MODEL"] = settings.verification_model if step == 5 else settings.model
    return env


def step_commands(step: int, run_dir: Path, ids_path: Path | None, settings: PipelineSettings) -> list[list[str]]:
    processed_root = run_dir / "processed_papers"
    py = sys.executable
    if step == 1:
        if ids_path is None:
            raise ValueError("--ids is required for step 1.")
        return [[
            py,
            "src/step_01_fetch/fetch_metadata.py",
            "--ids",
            str(ids_path),
            "--outdir",
            str(processed_root),
            "--resume",
        ]]
    if step == 2:
        return [[py, "src/step_02_mark_citations/replace_citation_markers.py", "--root", str(processed_root)]]
    if step == 3:
        return [[py, "src/step_03_usage_contexts/build_usage_contexts.py", "--root", str(processed_root), "--out-name", "usage_contexts.json"]]
    if step == 4:
        return [[
            py,
            "src/step_04_label_citations/label_citation_functions.py",
            "--root",
            str(processed_root),
            "--model-path",
            "Deep-Citation/Workspace/acl_scicite_wksp_trl/best_model.pt",
            "--model-data-dir",
            "Deep-Citation/Data",
            "--model-class-def",
            "Deep-Citation/Data/class_def.json",
            "--model-lm",
            "allenai/scibert_scivocab_uncased",
            "--device",
            settings.device,
        ]]
    if step == 5:
        return [[
            py,
            "src/step_05_verify_uses_extends/verify_uses_extends.py",
            "--root",
            str(processed_root),
            "--k",
            "0",
            "--batch-size",
            "25",
            "--resume",
        ]]
    if step == 6:
        return [[py, "src/step_06_extract_paragraphs/extract_arxiv_paragraphs.py", "--root", str(processed_root)]]
    if step == 7:
        return [
            [py, "src/step_07_extract_and_refine/extract_contributions_from_citations.py", "--root", str(processed_root)],
            [py, "src/step_07_extract_and_refine/refine_and_filter_clusters_llm.py", "--root", str(processed_root), "--inplace", "--overwrite"],
        ]
    raise ValueError(f"Unknown step: {step}")


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(PIPELINE_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            print(line, end="")
        code = proc.wait()
        log.write(f"\n[exit_code] {code}\n")
        return code


def run_pipeline_step(step: int, run_dir: Path, ids_path: Path | None, settings: PipelineSettings) -> None:
    log_path = run_dir / "logs" / f"step_{step:02d}.log"
    if log_path.exists():
        log_path.unlink()
    for cmd in step_commands(step, run_dir, ids_path, settings):
        code = run_command(cmd, log_path, env_for_step(settings, step))
        if code != 0:
            tail = log_tail(log_path)
            raise RuntimeError(f"Step {step} failed with exit code {code}.\n\n{tail}")


def count_verified_uses_extends(payload: dict) -> int:
    records = payload.get("confirmed") or payload.get("verified_contexts") or payload.get("contexts") or payload.get("items") or []
    if not isinstance(records, list):
        return 0
    return sum(1 for item in records if isinstance(item, dict) and item.get("label") in {"USES", "EXTENDS", "Uses", "Extends"})


def stop_reason_after_step(step: int, paper_dir: Path) -> str | None:
    if step == 1:
        if not paper_dir.exists():
            return "metadata could not be fetched for this paper"
        if not (paper_dir / "processed_main.tex").exists():
            return "arXiv source could not be retrieved or converted for this paper"
        citations = load_json(paper_dir / "citations_metadata.json", [])
        if not isinstance(citations, list) or not citations:
            return "Semantic Scholar returned no citing papers for this target paper"
    if step == 3:
        usage = load_json(paper_dir / "usage_contexts.json", {})
        if not isinstance(usage, dict) or int(usage.get("num_contexts") or 0) == 0:
            return "no citation usage contexts were found"
    if step == 4:
        labels = load_json(paper_dir / "usage_context_labels.json", {})
        contexts = labels.get("labels") if isinstance(labels, dict) else None
        if not isinstance(contexts, list) or not contexts:
            return "citation-function labeling produced no labeled contexts"
    if step == 5:
        verified = load_json(paper_dir / "usage_uses_extends_verified.json", {})
        if not isinstance(verified, dict):
            return "USES/EXTENDS verification did not produce an output file"
        if count_verified_uses_extends(verified) == 0:
            return "no downstream citations were verified as USES or EXTENDS"
    if step == 6:
        paragraphs = load_json(paper_dir / "usage_citing_paragraphs.json", {})
        citing = paragraphs.get("citing_papers") if isinstance(paragraphs, dict) else None
        if not isinstance(citing, list) or not citing:
            return "no citing-paper paragraphs could be extracted from arXiv"
        usable = [
            item for item in citing
            if isinstance(item, dict)
            and not item.get("error")
            and (item.get("matched_paragraphs") or item.get("target_citing_paragraphs"))
        ]
        if not usable:
            return "arXiv paragraph extraction returned no usable citing-paper text"
    if step == 7:
        contributions = load_json(paper_dir / "usage_contributions.json", {})
        items = contributions.get("contributions") if isinstance(contributions, dict) else None
        if not isinstance(items, list) or not items:
            return "no downstream target-contribution evidence could be extracted"
        refined = load_json(paper_dir / "usage_discovery_from_contributions.json", {})
        clusters = refined.get("clusters") if isinstance(refined, dict) else None
        if not isinstance(clusters, list) or not clusters:
            return "no valid downstream usage clusters survived refinement"
    return None


def paper_has_refined_clusters(paper_dir: Path) -> bool:
    refined = load_json(paper_dir / "usage_discovery_from_contributions.json", {})
    clusters = refined.get("clusters") if isinstance(refined, dict) else None
    return isinstance(clusters, list) and bool(clusters)


def run_annotation_for_paper(paper_dir: Path, run_dir: Path, settings: PipelineSettings) -> Path:
    output_root = run_dir / "two_pass_outputs"
    log_path = run_dir / "logs" / f"step_08_{paper_dir.name}.log"
    cmd = [
        sys.executable,
        "-m",
        "step_08_annotation.cli",
        "run",
        "--paper-dir",
        str(paper_dir),
        "--provider",
        settings.provider,
        "--model",
        settings.annotation_model,
        "--formatter-model",
        settings.formatter_model,
        "--judge-model",
        settings.judge_model,
        "--candidate-count",
        str(settings.candidate_count),
        "--output-root",
        str(output_root),
        "--run-label",
        paper_dir.name,
    ]
    code = run_command(cmd, log_path, env_for_step(settings, 8))
    if code != 0:
        raise RuntimeError(f"Step 8 failed for {paper_dir.name}.\n\n{log_tail(log_path)}")
    return output_root
