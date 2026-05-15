from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_utils import (
    PipelineSettings,
    STEP_LABELS,
    create_run_dir,
    parse_arxiv_id,
    run_annotation_for_paper,
    run_pipeline_step,
    stop_reason_after_step,
    write_ids_file,
    write_run_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full silver-annotation pipeline for one arXiv paper and save outputs locally."
    )
    parser.add_argument("--paper", required=True, help="arXiv URL or ID, for example https://arxiv.org/abs/2505.17978")
    parser.add_argument("--output-root", type=Path, default=Path("runs"), help="Directory where run folders are created.")
    parser.add_argument("--provider", default="gemini", help="LLM provider for generation steps.")
    parser.add_argument("--model", default="gemini-3.1-pro-preview", help="Model for target-contribution extraction/refinement.")
    parser.add_argument("--verification-model", default="gemini-3-flash-preview", help="Model for USES/EXTENDS verification.")
    parser.add_argument("--annotation-model", default="gemini/gemini-3.1-pro-preview", help="Model for final annotation.")
    parser.add_argument("--formatter-model", default="gemini/gemini-3.1-pro-preview", help="Formatter model for final annotation.")
    parser.add_argument("--judge-model", default="gemini/gemini-3.1-pro-preview", help="Judge model for final annotation.")
    parser.add_argument("--candidate-count", type=int, default=3, help="Number of annotation candidates before judging.")
    parser.add_argument("--device", default="cpu", help="Device for the citation-function classifier.")
    args = parser.parse_args()

    arxiv_id = parse_arxiv_id(args.paper)
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

    run_dir = create_run_dir(args.output_root)
    ids_path = write_ids_file(run_dir, [arxiv_id])
    write_run_config(
        run_dir,
        {
            "paper": args.paper,
            "arxiv_id": arxiv_id,
            "steps": list(STEP_LABELS.values()),
            "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )

    summary_lines = [f"Run directory: {run_dir}", f"Input paper: {arxiv_id}"]
    print(f"[run] {run_dir}")

    paper_dir = run_dir / "processed_papers" / arxiv_id
    for step in range(1, 8):
        print(f"\n[step {step}/8] {STEP_LABELS[step]}")
        run_pipeline_step(step, run_dir, ids_path if step == 1 else None, settings)
        summary_lines.append(f"Step {step} complete: {STEP_LABELS[step]}")
        reason = stop_reason_after_step(step, paper_dir)
        if reason:
            message = f"Pipeline stopped cleanly after step {step}: {reason}."
            summary_lines.append(message)
            (run_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
            print(f"\n{message}")
            print(f"Outputs saved in {run_dir}")
            return

    print(f"\n[step 8/8] {STEP_LABELS[8]}")
    run_annotation_for_paper(paper_dir, run_dir, settings)
    summary_lines.append(f"Step 8 complete: {STEP_LABELS[8]}")
    summary_lines.append("Pipeline completed successfully.")
    (run_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    (run_dir / "run_status.json").write_text(
        json.dumps({"status": "completed", "run_dir": str(run_dir), "paper_id": arxiv_id}, indent=2),
        encoding="utf-8",
    )
    print(f"\nPipeline completed successfully.")
    print(f"Outputs saved in {run_dir}")


if __name__ == "__main__":
    main()
