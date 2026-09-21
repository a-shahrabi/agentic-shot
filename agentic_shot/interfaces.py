"""Protocols for every swappable component. Real and fake implementations
both satisfy these; the pipeline depends on nothing else."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from .schemas import Evaluation, ImagePrompt, ShotSpec, VideoPrompt


@dataclass
class GenResult:
    path: Path
    cache_hit: bool = False
    latency_s: float | None = None


@runtime_checkable
class ImageGenerator(Protocol):
    name: str

    def generate(self, prompt: ImagePrompt, *, seed: int,
                 refs: Sequence[Path] = ()) -> GenResult: ...


@runtime_checkable
class VideoGenerator(Protocol):
    name: str

    def animate(self, keyframe: Path, prompt: VideoPrompt) -> Path: ...


@runtime_checkable
class Planner(Protocol):
    def plan(self, description: str) -> ShotSpec: ...


@runtime_checkable
class Prompter(Protocol):
    def write(self, spec: ShotSpec) -> ImagePrompt: ...

    def revise(self, prompt: ImagePrompt, evaluation: Evaluation, spec: ShotSpec,
               *, escalate: bool = False) -> ImagePrompt:
        """Return a new prompt changing only the segments flagged by
        `evaluation.fields_to_revise()`. `escalate=True` means the same field
        has failed before: rewrite rather than nudge."""
        ...


@runtime_checkable
class Critic(Protocol):
    def evaluate(self, image: Path, spec: ShotSpec) -> Evaluation: ...


@runtime_checkable
class VideoCritic(Protocol):
    def evaluate(self, video: Path, spec: ShotSpec) -> Evaluation: ...