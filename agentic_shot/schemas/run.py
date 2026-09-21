"""Records of what the pipeline did. RunManifest is what the Stage 2 harness reads."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .evaluation import Evaluation
from .prompt import ImagePrompt, VideoPrompt
from .spec import ShotSpec


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_SOFT_FAILURES = "completed_with_soft_failures"
    COMPLETED_WITH_CRITICAL_FAILURES = "completed_with_critical_failures"  # only via on_exhausted="always"
    FAILED_KEYFRAME = "failed_keyframe"   # exhausted attempts, critical field still failing
    FAILED_VIDEO = "failed_video"         # animation step errored
    ERROR = "error"                       # unexpected exception


class LLMCallRecord(BaseModel):
    id: str
    timestamp: datetime = Field(default_factory=_now)
    agent: str = Field(description="planner | prompter | critic | video_critic")
    model: str
    messages: list[dict[str, Any]]
    response: Any
    latency_s: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    attempt_index: int | None = None


class Attempt(BaseModel):
    index: int
    prompt: ImagePrompt
    seed: int
    image_path: Path
    evaluation: Evaluation
    cache_hit: bool = False
    gen_latency_s: float | None = None
    timestamp: datetime = Field(default_factory=_now)


class RunManifest(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=_now)
    description: str
    spec: ShotSpec | None = None
    attempts: list[Attempt] = Field(default_factory=list)
    chosen_attempt: int | None = None
    video_prompt: VideoPrompt | None = None
    video_path: Path | None = None
    video_evaluation: Evaluation | None = None
    status: RunStatus = RunStatus.RUNNING
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Snapshot of backends, profile, seed policy, thresholds used.",
    )
    error: str | None = None

    def best_attempt(self) -> Attempt | None:
        if not self.attempts:
            return None
        return max(self.attempts, key=lambda a: a.evaluation.rank_key())
