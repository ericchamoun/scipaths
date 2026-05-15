from __future__ import annotations

import json
from pathlib import Path

import typer

from .paper_package import load_paper_package

from .pipeline import TwoPassAnnotationPipeline


app = typer.Typer(help="Run step 8: derive target contributions, enabling contributions, and groundings.")


def _default_output_root() -> Path:
    return Path("runs/two_pass_outputs")


@app.command()
def run(
    paper_dir: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    provider: str = typer.Option("openai", help="Provider family: openai or gemini."),
    model: str = typer.Option("openai/gpt-5", help="Reasoning model used for target-contribution derivation and annotation."),
    formatter_model: str | None = typer.Option(
        None,
        help="Optional model override for pass 2 formatting, e.g. openai/gpt-5-mini or openai/gpt-5.4-pro.",
    ),
    judge_model: str | None = typer.Option(
        None,
        help="Optional model override for pass 1 candidate ranking. Ignored when --candidate-count=1.",
    ),
    candidate_count: int = typer.Option(
        1,
        help="Number of reasoning candidates to generate. If set to 1, no judge call is made.",
    ),
    formatter_max_attempts: int = typer.Option(
        3,
        help="Formatter-only retry attempts after pass 1 has succeeded.",
    ),
    include_reference_examples: bool = typer.Option(
        True,
        "--include-reference-examples/--no-include-reference-examples",
        help="Include the built-in reference examples in the pass-1 reasoning prompt.",
    ),
    prompt_profile: str = typer.Option(
        "full",
        help="Reasoning prompt profile: full or generic.",
    ),
    output_root: Path = typer.Option(
        _default_output_root(),
        help="Directory to store run outputs.",
    ),
    run_label: str | None = typer.Option(None, help="Optional label to include in the saved run directory name."),
    annotator_id: str = typer.Option("llm", help="Annotator id to embed in the final UI payload."),
    extracted_claim: str | None = typer.Option(None, help="Optional override for the extracted target contribution."),
) -> None:
    paper = load_paper_package(paper_dir, extracted_claim_override=extracted_claim)
    pipeline = TwoPassAnnotationPipeline(
        provider=provider,
        model=model,
        formatter_model=formatter_model,
        judge_model=judge_model,
        output_root=output_root,
        run_label=run_label,
        annotator_id=annotator_id,
        candidate_count=candidate_count,
        formatter_max_attempts=formatter_max_attempts,
        include_reference_examples=include_reference_examples,
        prompt_profile=prompt_profile,
        progress_callback=typer.echo,
    )
    result = pipeline.run(paper)
    typer.echo(str(result.run_dir / "run_output.json"))


@app.command()
def summarize(run_output: Path = typer.Option(..., exists=True, dir_okay=False, file_okay=True)) -> None:
    data = json.loads(run_output.read_text())
    payload = data.get("ui_payload") or {}
    claims = payload.get("claims") or []
    summary = {
        "paper_id": data.get("paper_id"),
        "target_contribution_count": len(claims),
        "target_contributions": [
            {
                "claim_id": claim.get("claim_id"),
                "rewritten_claim": claim.get("rewritten_claim"),
                "decision": claim.get("decision"),
                "enabling_contribution_count": len(claim.get("ingredients") or []),
            }
            for claim in claims
        ],
    }
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
