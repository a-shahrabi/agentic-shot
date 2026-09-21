from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SeedPolicy(BaseModel):
    initial_seed: int | None = Field(default=None, description="None = random.")
    hold_for: int = Field(
        default=2, ge=1,
        description="Keep the same seed for this many attempts before a reseed is allowed.",
    )
    reseed_on_repeat_failure: bool = Field(
        default=True,
        description="If a field fails twice in a row and hold_for is exhausted, change seed.",
    )
    reroll_when_nothing_to_revise: bool = Field(
        default=True,
        description="If the attempt fails overall but no field is below its revise_below, "
                    "retry the same prompt with a new seed instead of stopping.",
    )


OnExhausted = Literal["animate_if_critical_pass", "always", "never"]


class PipelineConfig(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    pass_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    seed: SeedPolicy = Field(default_factory=SeedPolicy)
    on_exhausted: OnExhausted = "animate_if_critical_pass"
    image_profile: str = Field(default="dev", description="Recorded in manifest; consumed by the image backend.")
    video_duration_s: float = 5.0