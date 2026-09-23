import json

import pytest

from agentic_shot.agents import LLMPlanner, LLMPrompter
from agentic_shot.agents.prompter import PROMPT_FIELDS
from agentic_shot.backends.llm import parse_structured
from agentic_shot.schemas import EvaluableIn, Evaluation, FieldVerdict, ImagePrompt, ShotSpec
from tests.test_schemas import make_prompt, make_spec


class ScriptedLLM:
    """Returns queued JSON replies in order; records every call."""
    model = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, *, schema=None, images=(), temperature=0.0):
        self.calls.append({"messages": messages, "schema": schema, "temperature": temperature})
        reply = self.replies.pop(0)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return parse_structured(text, schema) if schema else text


def eval_failing(**scores) -> Evaluation:
    verdicts = []
    for f in ShotSpec.FIELD_ORDER:
        if f in ShotSpec.video_fields():
            verdicts.append(FieldVerdict.skipped(f))
        else:
            s = scores.get(f, 0.9)
            verdicts.append(FieldVerdict(field=f, score=s, passed=s >= 0.5,
                                         description=f"saw {f}", critique=f"{f} wrong"))
    return Evaluation(verdicts=verdicts, stage=EvaluableIn.IMAGE)


# ---- planner ----------------------------------------------------------------

def test_planner_returns_spec_and_sends_description():
    llm = ScriptedLLM(make_spec().model_dump())
    spec = LLMPlanner(llm).plan("  a samurai in rain  ")
    assert spec == make_spec()
    call = llm.calls[0]
    assert call["schema"] is ShotSpec
    assert call["messages"][0]["role"] == "system"
    assert "a samurai in rain" in call["messages"][1]["content"]


def test_planner_rejects_empty_description():
    with pytest.raises(ValueError):
        LLMPlanner(ScriptedLLM()).plan("   ")


def test_planner_rejects_empty_fields():
    bad = make_spec().model_dump() | {"lighting": "  "}
    with pytest.raises(ValueError, match="lighting"):
        LLMPlanner(ScriptedLLM(bad)).plan("x")


# ---- prompter.write ---------------------------------------------------------

def test_write_builds_segments_without_camera_movement():
    segs = {f: f"{f} text" for f in PROMPT_FIELDS}
    llm = ScriptedLLM(segs)
    p = LLMPrompter(llm).write(make_spec())
    assert set(p.segments) == set(PROMPT_FIELDS)
    assert "camera_movement" not in p.segments
    assert "slow dolly forward" in llm.calls[0]["messages"][1]["content"]   # spec was sent


# ---- prompter.revise --------------------------------------------------------

def test_revise_nothing_flagged_makes_no_call():
    llm = ScriptedLLM()
    p = make_prompt()
    assert LLMPrompter(llm).revise(p, eval_failing(), make_spec()) is p
    assert llm.calls == []


def test_revise_changes_only_flagged_even_if_llm_oversteps():
    llm = ScriptedLLM({"revisions": [
        {"field": "camera_angle", "text": "seen from directly behind, face not visible", "rationale": "r"},
        {"field": "lighting", "text": "SHOULD BE IGNORED", "rationale": "r"},   # not flagged
    ]})
    prompter = LLMPrompter(llm)
    p = make_prompt()
    new = prompter.revise(p, eval_failing(camera_angle=0.1), make_spec())
    assert p.diff(new) == ["camera_angle"]
    assert new.segments["lighting"] == p.segments["lighting"]
    assert prompter.last_rationales == {"camera_angle": "r"}


def test_revise_sends_mode_critic_context_and_frozen_segments():
    llm = ScriptedLLM({"revisions": [{"field": "camera_angle", "text": "x", "rationale": "r"}]})
    LLMPrompter(llm).revise(make_prompt(), eval_failing(camera_angle=0.1), make_spec(), escalate=True)
    user = llm.calls[0]["messages"][1]["content"]
    assert "Mode: REWRITE" in user
    assert "saw camera_angle" in user            # critic description passed through
    assert "from behind, eye level" in user      # spec intent passed through
    frozen_part = user.split("Frozen segments")[1]
    assert "camera_angle" not in frozen_part


def test_revise_nudge_mode():
    llm = ScriptedLLM({"revisions": [{"field": "camera_angle", "text": "x", "rationale": "r"}]})
    LLMPrompter(llm).revise(make_prompt(), eval_failing(camera_angle=0.1), make_spec())
    assert "Mode: NUDGE" in llm.calls[0]["messages"][1]["content"]


def test_revise_empty_llm_output_returns_prompt_unchanged():
    llm = ScriptedLLM({"revisions": [{"field": "camera_angle", "text": "  ", "rationale": "r"}]})
    p = make_prompt()
    assert LLMPrompter(llm).revise(p, eval_failing(camera_angle=0.1), make_spec()) is p


def test_revise_multiple_fields():
    llm = ScriptedLLM({"revisions": [
        {"field": "camera_angle", "text": "a", "rationale": "r"},
        {"field": "subject", "text": "b", "rationale": "r"},
    ]})
    new = LLMPrompter(llm).revise(make_prompt(), eval_failing(camera_angle=0.1, subject=0.2), make_spec())
    assert set(make_prompt().diff(new)) == {"camera_angle", "subject"}


# ---- real agents inside the real loop ---------------------------------------

def test_real_agents_drive_the_pipeline(tmp_path):
    from agentic_shot.backends.fakes import (
        FakeImageGenerator, FakeVideoGenerator, NoopVideoCritic, ScriptedCritic,
    )
    from agentic_shot.config import PipelineConfig, SeedPolicy
    from agentic_shot.pipeline import Pipeline
    from agentic_shot.schemas import RunStatus
    from agentic_shot.storage import RunStore

    llm = ScriptedLLM(
        make_spec().model_dump(),                                      # planner
        {f: f"{f} text" for f in PROMPT_FIELDS},                       # write
        {"revisions": [{"field": "camera_angle",                       # revise
                        "text": "seen from directly behind, face not visible",
                        "rationale": "r"}]},
    )
    p = Pipeline(
        planner=LLMPlanner(llm), prompter=LLMPrompter(llm),
        image_gen=FakeImageGenerator(tmp_path / "gen"),
        critic=ScriptedCritic([{"camera_angle": 0.1}, {}]),
        video_gen=FakeVideoGenerator(tmp_path / "vid"),
        video_critic=NoopVideoCritic(),
        store=RunStore(tmp_path / "runs"),
        config=PipelineConfig(seed=SeedPolicy(initial_seed=7)),
    )
    m = p.run("samurai")
    assert m.status is RunStatus.COMPLETED
    a0, a1 = m.attempts
    assert a0.prompt.diff(a1.prompt) == ["camera_angle"]
    assert a1.prompt.segments["camera_angle"] == "seen from directly behind, face not visible"
    assert a0.seed == a1.seed == 7
    assert llm.replies == []   # exactly 3 LLM calls


# ---- word budget ------------------------------------------------------------

def _long(n=40):
    return {f: " ".join(["word"] * n) for f in PROMPT_FIELDS}


def test_short_prompt_makes_no_compress_call():
    llm = ScriptedLLM({f: f"{f} text" for f in PROMPT_FIELDS})
    pr = LLMPrompter(llm)
    pr.write(make_spec())
    assert len(llm.calls) == 1 and pr.last_warnings == []


def test_over_budget_triggers_one_compress_call():
    short = {f: "a few words here" for f in PROMPT_FIELDS}
    llm = ScriptedLLM(_long(), short)
    pr = LLMPrompter(llm)
    p = pr.write(make_spec())
    assert len(llm.calls) == 2
    assert "shorten" in llm.calls[1]["messages"][0]["content"]
    assert p.word_count() <= pr.word_budget
    assert pr.last_warnings == []


def test_compress_that_fails_warns_and_stops():
    llm = ScriptedLLM(_long(), _long())      # compression returns the same length
    pr = LLMPrompter(llm)
    p = pr.write(make_spec())
    assert len(llm.calls) == 2               # exactly one retry, no loop
    assert p.word_count() > pr.word_budget
    assert pr.last_warnings and "truncated" in pr.last_warnings[0]


def test_revise_over_budget_warns_but_keeps_frozen_segments():
    base = ImagePrompt(segments=_long(20))   # 140 words, under budget
    llm = ScriptedLLM({"revisions": [{"field": "camera_angle",
                                      "text": " ".join(["new"] * 60), "rationale": "r"}]})
    pr = LLMPrompter(llm)
    new = pr.revise(base, eval_failing(camera_angle=0.1), make_spec(), escalate=True)
    assert base.diff(new) == ["camera_angle"]   # frozen segments untouched
    assert pr.last_warnings and "budget" in pr.last_warnings[0]
