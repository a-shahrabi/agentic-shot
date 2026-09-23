"""The critic: judges a keyframe against a ShotSpec, field by field.

Each pass is two calls:
  1. describe  - the model sees the image but NOT the spec, and describes each field.
                 Keeping it blind stops the model from "seeing" what it was told to expect.
  2. compare   - the model sees the image, the blind description and the spec, and scores
                 each field 0-1 with a critique and a suggested prompt fix.

Strong fields (subject, setting) get one pass. Weak fields (camera, lighting, mood,
style) get `samples` passes; their score is the mean and the spread is a confidence signal.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from ..backends.llm import LLM
from ..schemas import EvaluableIn, Evaluation, FieldVerdict, ShotSpec
from .prompts import (
    CRITIC_COMPARE_SYSTEM, CRITIC_COMPARE_VERSION,
    CRITIC_DESCRIBE_SYSTEM, CRITIC_DESCRIBE_VERSION, CRITIC_FIELD_QUESTIONS,
)

WEAK_BELOW = 0.5   # a field whose revise_below is under this is "weak" -> multi-sampled


# ---- LLM output schemas -----------------------------------------------------

class FieldObservation(BaseModel):
    field: str
    observation: str


class Observations(BaseModel):
    observations: list[FieldObservation]


class FieldJudgment(BaseModel):
    field: str
    score: float
    critique: str
    suggested_fix: str


class Judgments(BaseModel):
    judgments: list[FieldJudgment]


@dataclass
class _Sample:
    score: float
    observation: str
    critique: str
    suggested_fix: str


# ---- helpers ----------------------------------------------------------------

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_image(path: Path, max_side: int = 1568) -> Path:
    """Downscale large images (and convert odd formats) before sending to the VLM.
    Returns the original path if Pillow is missing or the image is already fine."""
    path = Path(path)
    try:
        from PIL import Image
        with Image.open(path) as im:
            ok_format = path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
            if ok_format and max(im.size) <= max_side:
                return path
            out_dir = Path(tempfile.gettempdir()) / "agentic_shot_critic"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{file_sha256(path)[:16]}_{max_side}.jpg"
            if not out.exists():
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side))
                im.save(out, quality=90)
            return out
    except Exception:
        return path


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# ---- critic -----------------------------------------------------------------

class VLMCritic:
    version = f"{CRITIC_DESCRIBE_VERSION}+{CRITIC_COMPARE_VERSION}"

    def __init__(self, llm: LLM, *, samples: int = 3, pass_at: float = 0.5,
                 temperature: float = 0.7, max_image_side: int = 1568,
                 weak_fields: Sequence[str] | None = None):
        if samples < 1:
            raise ValueError("samples must be >= 1")
        self.llm = llm
        self.samples = samples
        self.pass_at = pass_at
        self.temperature = temperature
        self.max_image_side = max_image_side
        image_fields = ShotSpec.image_fields()
        self.weak_fields = tuple(
            weak_fields if weak_fields is not None
            else [f for f in image_fields if ShotSpec.meta(f).revise_below < WEAK_BELOW]
        )

    @property
    def model(self) -> str:
        return self.llm.model

    def identity(self) -> dict:
        """Everything that changes the critic's output. Used as part of the cache key."""
        return {
            "model": self.model, "version": self.version, "samples": self.samples,
            "pass_at": self.pass_at, "temperature": self.temperature,
            "max_image_side": self.max_image_side, "weak_fields": list(self.weak_fields),
        }

    # ------------------------------------------------------------------ public

    def evaluate(self, image: Path, spec: ShotSpec) -> Evaluation:
        img = prepare_image(Path(image), self.max_image_side)
        fields = list(ShotSpec.image_fields())
        weak = [f for f in fields if f in self.weak_fields]

        passes = [self._one_pass(img, spec, fields)]
        for _ in range(self.samples - 1):
            if not weak:
                break
            passes.append(self._one_pass(img, spec, weak))

        verdicts: list[FieldVerdict] = []
        for f in ShotSpec.FIELD_ORDER:
            if f not in fields:
                verdicts.append(FieldVerdict.skipped(f))
                continue
            results = [p[f] for p in passes if f in p]
            scores = [r.score for r in results]
            mean = round(sum(scores) / len(scores), 3)
            passed = mean >= self.pass_at
            # When failing, surface the most negative sample's critique: it's the most
            # specific about what's wrong, which is what the prompter needs.
            rep = results[0] if passed else min(results, key=lambda r: r.score)
            verdicts.append(FieldVerdict(
                field=f, score=mean, passed=passed,
                description=results[0].observation,
                critique=rep.critique.strip() or None,
                suggested_fix=rep.suggested_fix.strip() or None,
                samples=scores,
            ))
        return Evaluation(verdicts=verdicts, stage=EvaluableIn.IMAGE, critic_model=self.model)

    # ----------------------------------------------------------------- passes

    def _one_pass(self, img: Path, spec: ShotSpec, fields: list[str]) -> dict[str, _Sample]:
        observations = self._describe(img, fields)
        judgments = self._compare(img, spec, fields, observations)
        return {
            f: _Sample(score=_clamp(j.score), observation=observations.get(f, ""),
                       critique=j.critique, suggested_fix=j.suggested_fix)
            for f, j in judgments.items()
        }

    def _describe(self, img: Path, fields: list[str]) -> dict[str, str]:
        questions = {f: CRITIC_FIELD_QUESTIONS[f] for f in fields}
        messages = [
            {"role": "system", "content": CRITIC_DESCRIBE_SYSTEM},
            {"role": "user", "content": "Fields to describe:\n" + json.dumps(questions, indent=2)},
        ]
        out: Observations = self.llm.complete(
            messages, schema=Observations, images=[img], temperature=self.temperature
        )
        got = {o.field: o.observation.strip() for o in out.observations if o.field in questions}
        # A missing observation is tolerable: the compare step still sees the image.
        return {f: got.get(f, "(no observation)") for f in fields}

    def _compare(self, img: Path, spec: ShotSpec, fields: list[str],
                 observations: dict[str, str]) -> dict[str, FieldJudgment]:
        payload = [
            {"field": f, "spec": getattr(spec, f), "blind_description": observations[f]}
            for f in fields
        ]
        messages = [
            {"role": "system", "content": CRITIC_COMPARE_SYSTEM},
            {"role": "user", "content": "Fields to check:\n" + json.dumps(payload, indent=2)},
        ]
        # One retry if the model skips a field; a verdict must exist for every field.
        for attempt in range(2):
            out: Judgments = self.llm.complete(
                messages, schema=Judgments, images=[img], temperature=self.temperature
            )
            got = {j.field: j for j in out.judgments if j.field in fields}
            missing = [f for f in fields if f not in got]
            if not missing:
                return got
        raise ValueError(f"Critic returned no judgment for {missing} after retry")


# ---- cache ------------------------------------------------------------------

class CachedCritic:
    """Wraps a VLMCritic. Key = image bytes + spec + critic identity (model, prompt
    versions, sampling). Re-running an eval after changing loop logic costs nothing;
    editing a critic prompt (and bumping its version) invalidates exactly the right entries."""

    def __init__(self, inner: VLMCritic, cache_dir: Path | str = "cache/critic"):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self.inner.model

    def key(self, image: Path, spec: ShotSpec) -> str:
        blob = json.dumps({
            "image": file_sha256(Path(image)),
            "spec": spec.model_dump(),
            "critic": self.inner.identity(),
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def evaluate(self, image: Path, spec: ShotSpec) -> Evaluation:
        path = self.cache_dir / f"{self.key(image, spec)}.json"
        if path.exists():
            self.hits += 1
            return Evaluation.model_validate_json(path.read_text())
        ev = self.inner.evaluate(image, spec)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(ev.model_dump_json(indent=2))
        tmp.replace(path)
        self.misses += 1
        return ev