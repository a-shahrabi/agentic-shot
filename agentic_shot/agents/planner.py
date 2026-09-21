from __future__ import annotations

from ..backends.llm import LLM
from ..schemas import ShotSpec
from .prompts import PLANNER_SYSTEM, PLANNER_VERSION


class LLMPlanner:
    version = PLANNER_VERSION

    def __init__(self, llm: LLM, temperature: float = 0.7):
        self.llm = llm
        self.temperature = temperature

    def plan(self, description: str) -> ShotSpec:
        description = description.strip()
        if not description:
            raise ValueError("Empty scene description.")
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Scene description:\n{description}"},
        ]
        spec = self.llm.complete(messages, schema=ShotSpec, temperature=self.temperature)
        empty = [f for f in ShotSpec.FIELD_ORDER if not getattr(spec, f).strip()]
        if empty:
            raise ValueError(f"Planner returned empty fields: {empty}")
        return spec
