from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_utils import (
    PipelineSettings,
    STEP_LABELS,
    iter_paper_dirs,
    paper_has_refined_clusters,
    run_annotation_for_paper,
    run_pipeline_step,
    write_run_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one silver-annotation pipeline step over a local batch run directory."
    )
    parser.add_argument("--step", type=int, required=True, choices=range(1, 9), help="Pipeline step to run, 1 through 8.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Batch run directory containing processed_papers/.")
    parser.add_argument("--ids", type=Path, help="JSON paper ID file. Required for step 1.")
    parser.add_argument("--provider", default="gemini", help="LLM provider for generation steps.")
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Model for target-contribution extraction/refinement.")
    parser.add_argument("--verification-model", default="gemini-3-flash-preview", help="Model for USES/EXTENDS verification.")
    parser.add_argument("--annotation-model", default="gemini/gemini-3.1-pro-preview", help="Model for final annotation.")
    parser.add_argument("--formatter-model", default="gemini/gemini-3.1-pro-preview", help="Formatter model for final annotation.")
    parser.add_argument("--judge-model", default="gemini/gemini-3.1-pro-preview", help="Judge model for final annotation.")
    parser.add_argument("--candidate-count", type=int, default=3, help="Number of annotation candidates before judging.")
    parser.add_argument("--device", default="cpu", help="Device for the citation-function classifier.")
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    settings = PipelineSettings(
        provider=args.provider,
        model=args.model,
        verification_model=args.verification_model,
        annotation_model=args.annotation_model,
        formatter_model=args.formatter_model,
        judge_model=args.judge_model,
        candidate_count=args.candidate_count,
        device=args.device,
    )
    write_run_config(
        args.run_dir,
        {
            "mode": "batch_step",
            "last_step_requested": args.step,
            "step_label": STEP_LABELS[args.step],
            "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )

    if args.step == 1 and args.ids is None:
        raise SystemExit("--ids is required when --step 1 is used.")
    if args.step < 8:
        print(f"[step {args.step}/8] {STEP_LABELS[args.step]}")
        run_pipeline_step(args.step, args.run_dir, args.ids if args.step == 1 else None, settings)
        print(f"Step {args.step} complete. Outputs saved in {args.run_dir}")
        return

    processed_root = args.run_dir / "processed_papers"
    paper_dirs = [paper_dir for paper_dir in iter_paper_dirs(processed_root) if paper_has_refined_clusters(paper_dir)]
    if not paper_dirs:
        raise SystemExit(f"No papers with refined clusters found under {processed_root}")

    completed: list[str] = []
    for paper_dir in paper_dirs:
        print(f"[step 8/8] Annotating {paper_dir.name}")
        run_annotation_for_paper(paper_dir, args.run_dir, settings)
        completed.append(paper_dir.name)

    (args.run_dir / "step_08_summary.json").write_text(
        json.dumps({"annotated_papers": completed, "count": len(completed)}, indent=2),
        encoding="utf-8",
    )
    print(f"Step 8 complete for {len(completed)} paper(s). Outputs saved in {args.run_dir}")


if __name__ == "__main__":
    main()
