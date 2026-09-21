"""Filesystem-backed run storage.

Layout:
    runs/<run_id>/
        manifest.json          rewritten after every change (crash-safe)
        spec.json
        llm_calls.jsonl        one LLMCallRecord per line, append-only
        attempts/00/
            prompt.json
            keyframe.png
            eval.json
        attempts/01/...
        final.mp4
        video_eval.json
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import (
    Attempt, Evaluation, LLMCallRecord, RunManifest, RunStatus, ShotSpec, VideoPrompt,
)


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


class RunHandle:
    """Mutable view of one run. Every mutating call persists immediately."""

    def __init__(self, run_dir: Path, manifest: RunManifest):
        self.dir = run_dir
        self.manifest = manifest
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "attempts").mkdir(exist_ok=True)
        self.save()

    # ---- paths ----
    @property
    def run_id(self) -> str:
        return self.manifest.run_id

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def llm_log_path(self) -> Path:
        return self.dir / "llm_calls.jsonl"

    def attempt_dir(self, index: int) -> Path:
        d = self.dir / "attempts" / f"{index:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- persistence ----
    def save(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(self.manifest.model_dump_json(indent=2))
        tmp.replace(self.manifest_path)  # atomic on POSIX

    def set_spec(self, spec: ShotSpec) -> None:
        self.manifest.spec = spec
        (self.dir / "spec.json").write_text(spec.model_dump_json(indent=2))
        self.save()

    def record_attempt(self, attempt: Attempt) -> Attempt:
        """Copies the keyframe into the attempt dir, writes prompt/eval, updates manifest.
        Returns the attempt with image_path pointing at the stored copy."""
        d = self.attempt_dir(attempt.index)
        dest = d / f"keyframe{attempt.image_path.suffix or '.png'}"
        if attempt.image_path.resolve() != dest.resolve():
            shutil.copy2(attempt.image_path, dest)
        stored = attempt.model_copy(update={"image_path": dest})
        (d / "prompt.json").write_text(stored.prompt.model_dump_json(indent=2))
        (d / "eval.json").write_text(stored.evaluation.model_dump_json(indent=2))
        # replace if re-recording same index, else append
        self.manifest.attempts = [a for a in self.manifest.attempts if a.index != stored.index]
        self.manifest.attempts.append(stored)
        self.manifest.attempts.sort(key=lambda a: a.index)
        self.save()
        return stored

    def record_llm_call(self, rec: LLMCallRecord) -> None:
        with self.llm_log_path.open("a") as f:
            f.write(rec.model_dump_json() + "\n")

    def set_video(self, prompt: VideoPrompt, video_path: Path,
                  evaluation: Evaluation | None = None) -> Path:
        dest = self.dir / f"final{video_path.suffix or '.mp4'}"
        if video_path.resolve() != dest.resolve():
            shutil.copy2(video_path, dest)
        self.manifest.video_prompt = prompt
        self.manifest.video_path = dest
        if evaluation is not None:
            self.manifest.video_evaluation = evaluation
            (self.dir / "video_eval.json").write_text(evaluation.model_dump_json(indent=2))
        self.save()
        return dest

    def finalize(self, status: RunStatus, chosen_attempt: int | None = None,
                 error: str | None = None) -> None:
        self.manifest.status = status
        if chosen_attempt is not None:
            self.manifest.chosen_attempt = chosen_attempt
        if error is not None:
            self.manifest.error = error
        self.save()

    def llm_calls(self) -> list[LLMCallRecord]:
        if not self.llm_log_path.exists():
            return []
        return [LLMCallRecord.model_validate_json(line)
                for line in self.llm_log_path.read_text().splitlines() if line.strip()]


class RunStore:
    def __init__(self, root: Path | str = "runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_run(self, description: str, config: dict[str, Any] | None = None,
                run_id: str | None = None) -> RunHandle:
        run_id = run_id or _new_run_id()
        manifest = RunManifest(run_id=run_id, description=description, config=config or {})
        return RunHandle(self.root / run_id, manifest)

    def open(self, run_id: str) -> RunHandle:
        run_dir = self.root / run_id
        manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
        return RunHandle(run_dir, manifest)

    def list_runs(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir()
                      if (p / "manifest.json").exists())

    def load_manifest(self, run_id: str) -> RunManifest:
        return RunManifest.model_validate_json(
            (self.root / run_id / "manifest.json").read_text()
        )