"""Prompts are structured by ShotSpec field so a revision can touch one
segment and leave the rest byte-identical."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .spec import ShotSpec


class ImagePrompt(BaseModel):
    segments: dict[str, str] = Field(
        description="ShotSpec field name -> prompt text for that aspect."
    )
    negative: str = ""
    version: int = Field(default=1, description="Incremented on every revise().")

    @field_validator("segments")
    @classmethod
    def _known_fields_only(cls, v: dict[str, str]) -> dict[str, str]:
        unknown = set(v) - set(ShotSpec.FIELD_ORDER)
        if unknown:
            raise ValueError(f"Unknown segment keys: {sorted(unknown)}")
        return v

    def compile(self) -> str:
        """Join segments in canonical field order. Empty segments are skipped.
        This string (plus negative, seed, profile) is the image cache key."""
        parts = [self.segments[f].strip() for f in ShotSpec.FIELD_ORDER
                 if self.segments.get(f, "").strip()]
        return ", ".join(parts)

    def with_segments(self, updates: dict[str, str]) -> "ImagePrompt":
        """Return a new prompt with only the given segments replaced."""
        return ImagePrompt(
            segments={**self.segments, **updates},
            negative=self.negative,
            version=self.version + 1,
        )

    def diff(self, other: "ImagePrompt") -> list[str]:
        """Field names whose segment text differs between self and other."""
        keys = set(self.segments) | set(other.segments)
        return [f for f in ShotSpec.FIELD_ORDER
                if f in keys and self.segments.get(f) != other.segments.get(f)]


class VideoPrompt(BaseModel):
    text: str
    duration_s: float = 5.0

    @classmethod
    def from_spec(cls, spec: ShotSpec, duration_s: float = 5.0) -> "VideoPrompt":
        return cls(
            text=f"{spec.camera_movement}. {spec.subject} {spec.action}. {spec.mood}.",
            duration_s=duration_s,
        )