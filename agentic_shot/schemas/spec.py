"""ShotSpec: the structured creative intent, plus per-field metadata that
drives evaluation and retry policy."""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field


class EvaluableIn(str, Enum):
    IMAGE = "image"  # judgeable on a single keyframe
    VIDEO = "video"  # only judgeable on the animated clip


class FieldMeta(BaseModel):
    """Evaluation policy for one ShotSpec field."""

    weight: float = Field(ge=0.0, description="Weight in the soft weighted mean.")
    critical: bool = Field(
        description="If True, a failing score vetoes the attempt regardless of other fields."
    )
    revise_below: float = Field(
        ge=0.0, le=1.0,
        description="Score below which this field triggers a prompt revision. "
                    "Lower = critic is less trusted on this field.",
    )
    evaluable_in: EvaluableIn


class ShotSpec(BaseModel):
    subject: str = Field(description="Who or what the shot is about.")
    setting: str = Field(description="Location, environment, time of day, weather.")
    action: str = Field(description="What the subject is doing.")
    camera_angle: str = Field(description="Framing and position: e.g. 'behind the subject, eye level, medium-wide'.")
    camera_movement: str = Field(description="Motion: e.g. 'slow dolly forward', 'static', 'handheld'.")
    lighting: str = Field(description="Light sources, quality, direction, color.")
    mood: str = Field(description="Emotional tone.")
    style: str = Field(description="Visual/cinematic reference: film stock, director, genre look.")

    # Canonical field order. Used by ImagePrompt.compile() and by the critic
    # when reporting verdicts. Do not reorder casually — it changes prompt strings
    # and therefore cache keys.
    FIELD_ORDER: ClassVar[tuple[str, ...]] = (
        "subject", "setting", "action", "camera_angle",
        "camera_movement", "lighting", "mood", "style",
    )

    # Default per-field policy. Tune from human-verdict data in Stage 2.
    FIELD_META: ClassVar[dict[str, FieldMeta]] = {
        "subject":         FieldMeta(weight=3.0, critical=True,  revise_below=0.6, evaluable_in=EvaluableIn.IMAGE),
        "setting":         FieldMeta(weight=2.0, critical=False, revise_below=0.6, evaluable_in=EvaluableIn.IMAGE),
        "action":          FieldMeta(weight=2.0, critical=True,  revise_below=0.5, evaluable_in=EvaluableIn.VIDEO),
        "camera_angle":    FieldMeta(weight=1.5, critical=False, revise_below=0.3, evaluable_in=EvaluableIn.IMAGE),
        "camera_movement": FieldMeta(weight=1.5, critical=False, revise_below=0.3, evaluable_in=EvaluableIn.VIDEO),
        "lighting":        FieldMeta(weight=1.5, critical=False, revise_below=0.3, evaluable_in=EvaluableIn.IMAGE),
        "mood":            FieldMeta(weight=1.0, critical=False, revise_below=0.3, evaluable_in=EvaluableIn.IMAGE),
        "style":           FieldMeta(weight=1.0, critical=False, revise_below=0.3, evaluable_in=EvaluableIn.IMAGE),
    }

    @classmethod
    def fields_for(cls, where: EvaluableIn) -> tuple[str, ...]:
        return tuple(f for f in cls.FIELD_ORDER if cls.FIELD_META[f].evaluable_in is where)

    @classmethod
    def image_fields(cls) -> tuple[str, ...]:
        return cls.fields_for(EvaluableIn.IMAGE)

    @classmethod
    def video_fields(cls) -> tuple[str, ...]:
        return cls.fields_for(EvaluableIn.VIDEO)

    @classmethod
    def meta(cls, field: str) -> FieldMeta:
        return cls.FIELD_META[field]

    def as_dict(self) -> dict[str, str]:
        """Field -> value, in canonical order."""
        return {f: getattr(self, f) for f in self.FIELD_ORDER}