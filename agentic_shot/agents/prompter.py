from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ..backends.llm import LLM
from ..schemas import Evaluation, ImagePrompt, ShotSpec
from .prompts import (
    PROMPTER_REVISE_SYSTEM, PROMPTER_REVISE_VERSION,
    PROMPTER_WRITE_SYSTEM, PROMPTER_WRITE_VERSION,
)

# camera_movement is excluded: it is invisible in a still frame.
PROMPT_FIELDS: tuple[str, ...] = tuple(
    f for f in ShotSpec.FIELD_ORDER if f != "camera_movement"
)


class SegmentsOut(BaseModel):
    subject: str
    setting: str
    action: str
    camera_angle: str
    lighting: str
    mood: str
    style: str


class SegmentRevision(BaseModel):
    field: str = Field(description="One of the failing field names, exactly as given.")
    text: str = Field(description="The full replacement segment.")
    rationale: str = Field(description="One sentence: what caused the failure and what changed.")


class RevisionOut(BaseModel):
    revisions: list[SegmentRevision]


class LLMPrompter:
    write_version = PROMPTER_WRITE_VERSION
    revise_version = PROMPTER_REVISE_VERSION

    def __init__(self, llm: LLM, write_temperature: float = 0.4,
                 revise_temperature: float = 0.7):
        self.llm = llm
        self.write_temperature = write_temperature
        self.revise_temperature = revise_temperature
        self.last_rationales: dict[str, str] = {}

    # ------------------------------------------------------------------ write

    def write(self, spec: ShotSpec) -> ImagePrompt:
        messages = [
            {"role": "system", "content": PROMPTER_WRITE_SYSTEM},
            {"role": "user", "content": "Shot spec:\n" + json.dumps(spec.as_dict(), indent=2)},
        ]
        out: SegmentsOut = self.llm.complete(
            messages, schema=SegmentsOut, temperature=self.write_temperature
        )
        return ImagePrompt(segments={f: getattr(out, f).strip() for f in PROMPT_FIELDS})

    # ----------------------------------------------------------------- revise

    def revise(self, prompt: ImagePrompt, evaluation: Evaluation, spec: ShotSpec,
               *, escalate: bool = False) -> ImagePrompt:
        flagged = [f for f in evaluation.fields_to_revise() if f in PROMPT_FIELDS]
        self.last_rationales = {}
        if not flagged:
            return prompt

        verdicts = evaluation.by_field()
        failing = [
            {
                "field": f,
                "spec_intent": getattr(spec, f),
                "current_segment": prompt.segments.get(f, ""),
                "critic_saw": verdicts[f].description,
                "critique": verdicts[f].critique,
                "critic_suggestion": verdicts[f].suggested_fix,
                "score": verdicts[f].score,
            }
            for f in flagged
        ]
        frozen = {f: t for f, t in prompt.segments.items() if f not in flagged}
        user = (
            f"Mode: {'REWRITE' if escalate else 'NUDGE'}\n\n"
            f"Failing fields (rewrite these only):\n{json.dumps(failing, indent=2)}\n\n"
            f"Frozen segments (context only, do not return):\n{json.dumps(frozen, indent=2)}"
        )
        messages = [
            {"role": "system", "content": PROMPTER_REVISE_SYSTEM},
            {"role": "user", "content": user},
        ]
        out: RevisionOut = self.llm.complete(
            messages, schema=RevisionOut, temperature=self.revise_temperature
        )

        # Enforce the contract in code, not just in the prompt:
        # only flagged fields, only non-empty text.
        updates = {r.field: r.text.strip() for r in out.revisions
                   if r.field in flagged and r.text.strip()}
        self.last_rationales = {r.field: r.rationale for r in out.revisions if r.field in updates}
        if not updates:
            return prompt  # pipeline detects the unchanged prompt and reseeds
        return prompt.with_segments(updates)
