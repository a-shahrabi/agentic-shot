"""Prompts are structured by ShotSpec field so a revision can touch one
segment and leave the rest byte-identical."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from .spec import ShotSpec

# Order segments are compiled in. Separate from ShotSpec.FIELD_ORDER on purpose:
# FLUX weights early tokens most, so framing goes first; mood (least literal) goes last.
# camera_movement is listed for completeness but never has a segment (invisible in a still).
COMPILE_ORDER: tuple[str, ...] = (
    "camera_angle", "subject", "action", "setting",
    "lighting", "style", "mood", "camera_movement",
)
assert set(COMPILE_ORDER) == set(ShotSpec.FIELD_ORDER), "COMPILE_ORDER must cover every field"

_TRAILING = " \t\n.,;:"


def _clean(segment: str) -> str:
    s = segment.strip().rstrip(_TRAILING).strip()
    return s[:1].upper() + s[1:] if s else s


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
        """Join segments in COMPILE_ORDER as sentences. Empty segments are skipped.
        This string (plus negative, seed, profile) is the image cache key."""
        parts = [_clean(self.segments.get(f, "")) for f in COMPILE_ORDER]
        parts = [p for p in parts if p]
        return ". ".join(parts) + ("." if parts else "")

    def word_count(self) -> int:
        return len(self.compile().split())

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
