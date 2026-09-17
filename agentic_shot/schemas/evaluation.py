"""Per-field verdicts and the aggregate pass rule."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from .spec import EvaluableIn, ShotSpec


class VerdictStatus(str, Enum):
    EVALUATED = "evaluated"
    SKIPPED = "skipped"  # e.g. video-only field during keyframe evaluation


class FieldVerdict(BaseModel):
    field: str
    status: VerdictStatus = VerdictStatus.EVALUATED
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    passed: bool | None = None
    description: str | None = Field(
        default=None, description="Critic's describe-step output, before comparison."
    )
    critique: str | None = None
    suggested_fix: str | None = None
    samples: list[float] = Field(
        default_factory=list, description="Raw per-sample scores when multi-sampled."
    )

    @model_validator(mode="after")
    def _consistent(self) -> "FieldVerdict":
        if self.field not in ShotSpec.FIELD_META:
            raise ValueError(f"Unknown field: {self.field}")
        if self.status is VerdictStatus.EVALUATED and (self.score is None or self.passed is None):
            raise ValueError("Evaluated verdict requires score and passed.")
        return self

    @classmethod
    def skipped(cls, field: str) -> "FieldVerdict":
        return cls(field=field, status=VerdictStatus.SKIPPED)

    @property
    def sample_spread(self) -> float | None:
        """Max - min of samples; a cheap disagreement signal. None if <2 samples."""
        if len(self.samples) < 2:
            return None
        return max(self.samples) - min(self.samples)


class Evaluation(BaseModel):
    verdicts: list[FieldVerdict]
    stage: EvaluableIn = Field(
        description="Which artifact was judged: image (keyframe) or video."
    )
    critic_model: str | None = None
    human_verdicts: list[FieldVerdict] | None = Field(
        default=None, description="Filled in later by a human. Ground truth for Stage 2."
    )

    def by_field(self) -> dict[str, FieldVerdict]:
        return {v.field: v for v in self.verdicts}

    def evaluated(self) -> list[FieldVerdict]:
        return [v for v in self.verdicts if v.status is VerdictStatus.EVALUATED]

    def critical_pass(self) -> bool:
        return all(
            v.passed for v in self.evaluated() if ShotSpec.meta(v.field).critical
        )

    def weighted_score(self) -> float:
        ev = self.evaluated()
        total = sum(ShotSpec.meta(v.field).weight for v in ev)
        if total == 0:
            return 0.0
        return sum(ShotSpec.meta(v.field).weight * v.score for v in ev) / total

    def overall_pass(self, threshold: float) -> bool:
        """All critical fields pass AND weighted mean >= threshold."""
        return self.critical_pass() and self.weighted_score() >= threshold

    def fields_to_revise(self) -> list[str]:
        """Fields whose score is below their own revise_below threshold."""
        return [
            v.field for v in self.evaluated()
            if v.score < ShotSpec.meta(v.field).revise_below
        ]

    def rank_key(self) -> tuple[bool, float]:
        """Sort key for best-attempt selection: critical pass first, then score."""
        return (self.critical_pass(), self.weighted_score())