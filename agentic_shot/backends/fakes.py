"""Fake implementations of every component. Permanent test fixtures —
this is how loop logic gets tested without models or API keys."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..interfaces import GenResult
from ..schemas import (
    EvaluableIn, Evaluation, FieldVerdict, ImagePrompt, ShotSpec, VideoPrompt,
)


def _write_png(path: Path, text: str) -> None:
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (512, 288), (40, 40, 60))
        ImageDraw.Draw(img).multiline_text((10, 10), text[:600], fill=(230, 230, 230))
        img.save(path)
    except ImportError:
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + text.encode()[:600])


@dataclass
class FakeImageGenerator:
    out_dir: Path
    name: str = "fake-image"
    calls: list[tuple[str, int]] = field(default_factory=list)   # (compiled_prompt, seed)

    def generate(self, prompt: ImagePrompt, *, seed: int, refs: Sequence[Path] = ()) -> GenResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        compiled = prompt.compile()
        self.calls.append((compiled, seed))
        path = self.out_dir / f"gen_{len(self.calls):03d}_seed{seed}.png"
        _write_png(path, f"seed={seed}\n{compiled}")
        return GenResult(path=path, cache_hit=False, latency_s=0.0)


@dataclass
class FakeVideoGenerator:
    out_dir: Path
    name: str = "fake-video"
    calls: list[tuple[Path, VideoPrompt]] = field(default_factory=list)
    fail: bool = False

    def animate(self, keyframe: Path, prompt: VideoPrompt) -> Path:
        if self.fail:
            raise RuntimeError("fake video failure")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.calls.append((keyframe, prompt))
        dest = self.out_dir / f"video_{len(self.calls):03d}.mp4"
        shutil.copy2(keyframe, dest)
        return dest


@dataclass
class FakePlanner:
    spec: ShotSpec

    def plan(self, description: str) -> ShotSpec:
        return self.spec


@dataclass
class FakePrompter:
    """write(): one segment per image field, straight from the spec.
    revise(): rewrites each flagged segment deterministically."""
    revise_calls: list[dict] = field(default_factory=list)
    no_op: bool = False   # if True, revise() returns the prompt unchanged (tests forced reseed)

    def write(self, spec: ShotSpec) -> ImagePrompt:
        return ImagePrompt(segments={f: getattr(spec, f) for f in ShotSpec.image_fields()})

    def revise(self, prompt: ImagePrompt, evaluation: Evaluation, spec: ShotSpec,
               *, escalate: bool = False) -> ImagePrompt:
        flagged = evaluation.fields_to_revise()
        self.revise_calls.append({
            "base_version": prompt.version, "fields": flagged, "escalate": escalate,
        })
        if self.no_op:
            return prompt
        tag = "REWRITE" if escalate else "nudge"
        return prompt.with_segments(
            {f: f"{prompt.segments.get(f, '')} [{tag} v{prompt.version + 1}]" for f in flagged}
        )


@dataclass
class ScriptedCritic:
    """Returns verdicts from a script: one dict of {field: score} per call.
    Fields not in the dict score 0.9. Last entry repeats when exhausted."""
    script: list[dict[str, float]]
    pass_at: float = 0.5
    seen: list[Path] = field(default_factory=list)

    def evaluate(self, image: Path, spec: ShotSpec) -> Evaluation:
        self.seen.append(image)
        idx = min(len(self.seen) - 1, len(self.script) - 1)
        scores = self.script[idx]
        verdicts = []
        for f in ShotSpec.FIELD_ORDER:
            if f in ShotSpec.video_fields():
                verdicts.append(FieldVerdict.skipped(f))
            else:
                s = scores.get(f, 0.9)
                verdicts.append(FieldVerdict(field=f, score=s, passed=s >= self.pass_at,
                                             critique=None if s >= self.pass_at else f"{f} wrong"))
        return Evaluation(verdicts=verdicts, stage=EvaluableIn.IMAGE, critic_model="scripted")


class NoopVideoCritic:
    """Stage 1 placeholder: marks every video field as skipped, honestly."""

    def evaluate(self, video: Path, spec: ShotSpec) -> Evaluation:
        return Evaluation(
            verdicts=[FieldVerdict.skipped(f) for f in ShotSpec.video_fields()],
            stage=EvaluableIn.VIDEO, critic_model=None,
        )