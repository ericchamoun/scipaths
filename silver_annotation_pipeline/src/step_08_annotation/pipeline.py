from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict

from common.model_client import ModelConfig, MultiProviderLLMClient
from common.paper_package import PaperPackage

from .final_prompts import (
    SYSTEM_TWO_PASS_FORMATTER,
    SYSTEM_TWO_PASS_JUDGE,
    SYSTEM_TWO_PASS_REASONING,
    formatter_prompt,
    judge_prompt,
    reasoning_prompt,
)
from .schemas import JudgeResult, UIPayload


@dataclass
class TwoPassPipelineResult:
    run_dir: Path
    result: Dict[str, Any]


class FormatterStageError(RuntimeError):
    def __init__(self, message: str, run_dir: Path):
        super().__init__(message)
        self.run_dir = run_dir


class TwoPassAnnotationPipeline:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        formatter_model: str | None,
        judge_model: str | None,
        output_root: Path,
        run_label: str | None = None,
        annotator_id: str = "llm",
        temperature: float = 0.2,
        max_tokens: int = 16000,
        candidate_count: int = 1,
        formatter_max_attempts: int = 3,
        include_reference_examples: bool = True,
        prompt_profile: str = "full",
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.output_root = output_root
        self.annotator_id = annotator_id
        self.progress_callback = progress_callback
        self.run_label = run_label
        self.candidate_count = max(1, candidate_count)
        self.formatter_max_attempts = max(1, formatter_max_attempts)
        self.include_reference_examples = include_reference_examples
        self.prompt_profile = prompt_profile
        self.use_judge = self.candidate_count > 1
        stage_models = {}
        if formatter_model:
            stage_models["two_pass_formatter"] = formatter_model
        if judge_model and self.use_judge:
            stage_models["two_pass_judge"] = judge_model
        self.client = MultiProviderLLMClient(
            default_config=ModelConfig(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            stage_models=stage_models,
        )

    def run(self, paper: PaperPackage) -> TwoPassPipelineResult:
        run_dir = self._make_run_dir(paper)
        payload = {
            **paper.to_prompt_payload(),
            "paper_dir": paper.paper_dir,
            "full_processed_text": self._load_full_processed_text(paper),
        }
        formatter_config = self.client.config_for_stage("two_pass_formatter")
        self._log(
            f"[run] paper={paper.paper_dir.name} provider={self.client.default_config.provider} model={self.client.default_config.model_name}"
        )
        self._log(f"[run] formatter_model={formatter_config.model_name}")
        self._log(f"[run] include_reference_examples={self.include_reference_examples}")
        self._log(f"[run] prompt_profile={self.prompt_profile}")
        if self.use_judge:
            judge_config = self.client.config_for_stage("two_pass_judge")
            self._log(f"[run] judge_model={judge_config.model_name}")
        else:
            self._log("[run] judge_model=disabled (candidate_count=1)")
        self._log(f"[run] output={run_dir}")

        reasoning_user_prompt = reasoning_prompt(payload, include_reference_examples=self.include_reference_examples, prompt_profile=self.prompt_profile)
        self._write_text(run_dir / "pass_1_reasoning.prompt.txt", reasoning_user_prompt)
        self._log(
            f"[pass 1] free-form reasoning ({self.candidate_count} candidate{'s' if self.candidate_count != 1 else ''})"
        )

        candidate_texts: list[str] = []
        candidate_paths: list[str] = []
        for index in range(self.candidate_count):
            reasoning_text = self.client.generate_text(
                stage_name="two_pass_reasoning",
                system_prompt=SYSTEM_TWO_PASS_REASONING,
                user_prompt=reasoning_user_prompt,
            )
            candidate_id = f"candidate_{index + 1}"
            candidate_path = run_dir / f"pass_1_reasoning.output.{candidate_id}.md"
            self._write_text(candidate_path, reasoning_text)
            candidate_texts.append(reasoning_text)
            candidate_paths.append(str(candidate_path))

        selected_candidate_index = 0
        selected_candidate_id = "candidate_1"
        selected_reasoning_text = candidate_texts[0]
        judge_output_path: Path | None = None

        if self.use_judge:
            judge_user_prompt = judge_prompt(payload, candidate_texts)
            self._write_text(run_dir / "pass_1_reasoning.judge.prompt.txt", judge_user_prompt)
            self._log("[pass 1] candidate judging")
            judge_result = self.client.generate_structured(
                stage_name="two_pass_judge",
                system_prompt=SYSTEM_TWO_PASS_JUDGE,
                user_prompt=judge_user_prompt,
                response_model=JudgeResult,
            )
            judge_output_path = run_dir / "pass_1_reasoning.judge.output.json"
            self._write_json(judge_output_path, judge_result.model_dump())
            selected_candidate_index = judge_result.selected_candidate_index
            selected_candidate_id = judge_result.selected_candidate_id
            selected_reasoning_text = candidate_texts[selected_candidate_index]

        selected_reasoning_path = run_dir / "pass_1_reasoning.selected.md"
        self._write_text(selected_reasoning_path, selected_reasoning_text)

        formatter_user_prompt = formatter_prompt(payload, selected_reasoning_text, self.annotator_id)
        self._write_text(run_dir / "pass_2_formatter.prompt.txt", formatter_user_prompt)
        final_payload: UIPayload | None = None
        formatter_attempts: list[dict[str, Any]] = []
        for attempt in range(1, self.formatter_max_attempts + 1):
            self._log(
                f"[pass 2] strict ui json formatting (attempt {attempt}/{self.formatter_max_attempts})"
            )
            try:
                final_payload = self.client.generate_structured(
                    stage_name="two_pass_formatter",
                    system_prompt=SYSTEM_TWO_PASS_FORMATTER,
                    user_prompt=formatter_user_prompt,
                    response_model=UIPayload,
                )
                formatter_attempts.append({"attempt": attempt, "status": "success"})
                break
            except Exception as exc:
                error_text = "".join(traceback.format_exception(exc)).strip()
                error_path = run_dir / f"pass_2_formatter.attempt_{attempt}.error.txt"
                self._write_text(error_path, error_text)
                formatter_attempts.append(
                    {
                        "attempt": attempt,
                        "status": "failed",
                        "error": str(exc),
                        "error_path": str(error_path),
                    }
                )
                if attempt < self.formatter_max_attempts:
                    self._log("[pass 2] formatter failed; retrying formatter only")

        if final_payload is None:
            self._write_json(run_dir / "formatter_attempts.json", {"attempts": formatter_attempts})
            raise FormatterStageError(
                f"Formatter failed after {self.formatter_max_attempts} attempts; pass 1 outputs kept in {run_dir}",
                run_dir,
            )

        self._write_json(run_dir / "formatter_attempts.json", {"attempts": formatter_attempts})
        self._write_json(run_dir / "pass_2_ui_payload.json", final_payload.model_dump())

        result = {
            "paper_id": paper.paper_dir.name,
            "paper_dir": str(paper.paper_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reasoner_model": self.client.default_config.model_name,
            "formatter_model": formatter_config.model_name,
            "judge_model": judge_config.model_name if self.use_judge else None,
            "candidate_count": self.candidate_count,
            "include_reference_examples": self.include_reference_examples,
            "prompt_profile": self.prompt_profile,
            "reasoning_candidate_paths": [str(path) for path in candidate_paths],
            "selected_reasoning_candidate": selected_candidate_id,
            "selected_candidate_index": selected_candidate_index,
            "selected_reasoning_path": str(selected_reasoning_path),
            "judge_output_path": str(judge_output_path) if judge_output_path is not None else None,
            "formatter_attempts": formatter_attempts,
            "ui_payload_path": str(run_dir / "pass_2_ui_payload.json"),
            "ui_payload": final_payload.model_dump(),
        }
        self._write_json(run_dir / "run_output.json", result)
        self._log("[run] complete")
        return TwoPassPipelineResult(run_dir=run_dir, result=result)

    def _make_run_dir(self, paper: PaperPackage) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = stamp
        if self.run_label:
            run_name = f"{self._slugify(self.run_label)}__{stamp}"
        run_dir = self.output_root / paper.paper_dir.name / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    @staticmethod
    def _slugify(value: str) -> str:
        slug = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value.strip())
        slug = "-".join(part for part in slug.split("-") if part)
        return slug[:160] or "run"

    def _load_full_processed_text(self, paper: PaperPackage) -> str:
        processed_path = paper.paper_dir / "processed_main.tex"
        if processed_path.exists():
            try:
                return processed_path.read_text()
            except Exception:
                pass

        sections_dir = paper.paper_dir / "sections"
        parts: list[str] = []
        if sections_dir.exists():
            for path in sorted(sections_dir.iterdir()):
                if not path.is_file():
                    continue
                try:
                    text = path.read_text().strip()
                except Exception:
                    continue
                if text:
                    parts.append(f"[{path.name}]\n{text}")
        return "\n\n".join(parts)
