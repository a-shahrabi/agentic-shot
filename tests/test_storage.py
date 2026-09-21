import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from agentic_shot.backends.llm import LLM, LoggingLLM, attach_images, parse_structured
from agentic_shot.schemas import (
    Attempt, EvaluableIn, Evaluation, FieldVerdict, ImagePrompt, RunStatus, VideoPrompt,
)
from agentic_shot.storage import RunStore
from tests.test_schemas import make_eval, make_prompt, make_spec


# ---- fixtures ---------------------------------------------------------------

@pytest.fixture
def png(tmp_path: Path) -> Path:
    p = tmp_path / "src.png"
    p.write_bytes(b"\x89PNG fake")
    return p


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


# ---- RunStore ---------------------------------------------------------------

def test_new_run_creates_layout_and_manifest(store: RunStore):
    h = store.new_run("samurai", config={"profile": "dev"})
    assert (h.dir / "manifest.json").exists()
    assert (h.dir / "attempts").is_dir()
    assert store.load_manifest(h.run_id).status is RunStatus.RUNNING
    assert store.load_manifest(h.run_id).config == {"profile": "dev"}


def test_record_attempt_copies_image_and_persists(store: RunStore, png: Path):
    h = store.new_run("samurai")
    h.set_spec(make_spec())
    a = Attempt(index=0, prompt=make_prompt(), seed=7, image_path=png, evaluation=make_eval())
    stored = h.record_attempt(a)

    d = h.dir / "attempts" / "00"
    assert stored.image_path == d / "keyframe.png"
    assert (d / "keyframe.png").read_bytes() == png.read_bytes()
    assert (d / "prompt.json").exists() and (d / "eval.json").exists()

    m = store.load_manifest(h.run_id)
    assert len(m.attempts) == 1
    assert m.attempts[0].seed == 7
    assert m.spec == make_spec()


def test_manifest_rewritten_after_every_attempt(store: RunStore, png: Path):
    h = store.new_run("samurai")
    for i in range(3):
        h.record_attempt(Attempt(index=i, prompt=make_prompt(), seed=i,
                                 image_path=png, evaluation=make_eval()))
        assert len(store.load_manifest(h.run_id).attempts) == i + 1


def test_rerecording_same_index_replaces(store: RunStore, png: Path):
    h = store.new_run("samurai")
    for seed in (1, 2):
        h.record_attempt(Attempt(index=0, prompt=make_prompt(), seed=seed,
                                 image_path=png, evaluation=make_eval()))
    m = store.load_manifest(h.run_id)
    assert len(m.attempts) == 1 and m.attempts[0].seed == 2


def test_set_video_and_finalize(store: RunStore, tmp_path: Path):
    h = store.new_run("samurai")
    h.set_spec(make_spec())
    vid = tmp_path / "out.mp4"
    vid.write_bytes(b"mp4")
    ev = Evaluation(stage=EvaluableIn.VIDEO, verdicts=[FieldVerdict.skipped("camera_movement")])
    dest = h.set_video(VideoPrompt.from_spec(make_spec()), vid, ev)
    h.finalize(RunStatus.COMPLETED, chosen_attempt=0)

    m = store.load_manifest(h.run_id)
    assert m.video_path == dest and dest.exists()
    assert m.video_evaluation.stage is EvaluableIn.VIDEO
    assert m.status is RunStatus.COMPLETED and m.chosen_attempt == 0


def test_open_and_list(store: RunStore):
    ids = [store.new_run(f"d{i}").run_id for i in range(2)]
    assert store.list_runs() == sorted(ids)
    h = store.open(ids[0])
    assert h.manifest.description == "d0"


# ---- LLM helpers ------------------------------------------------------------

class Out(BaseModel):
    x: int


def test_parse_structured_strips_fences():
    assert parse_structured('```json\n{"x": 3}\n```', Out).x == 3
    assert parse_structured('{"x": 4}', Out).x == 4


def test_attach_images_goes_on_last_user_message(png: Path):
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "look"}]
    out = attach_images(msgs, [png])
    parts = out[1]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert msgs[1]["content"] == "look"  # original untouched


# ---- LoggingLLM -------------------------------------------------------------

class FakeLLM:
    model = "fake-1"
    last_usage = {"tokens_in": 10, "tokens_out": 5}

    def __init__(self, reply="hello", fail=False):
        self.reply, self.fail = reply, fail

    def complete(self, messages, *, schema=None, images=(), temperature=0.0):
        if self.fail:
            raise RuntimeError("boom")
        return parse_structured(self.reply, schema) if schema else self.reply


def test_fake_satisfies_protocol():
    assert isinstance(FakeLLM(), LLM)


def test_logging_llm_records_text_call(store: RunStore):
    h = store.new_run("d")
    llm = LoggingLLM(FakeLLM("hi"), h.record_llm_call, agent="planner")
    llm.attempt_index = 1
    assert llm.complete([{"role": "user", "content": "yo"}]) == "hi"

    calls = h.llm_calls()
    assert len(calls) == 1
    c = calls[0]
    assert c.agent == "planner" and c.model == "fake-1" and c.attempt_index == 1
    assert c.response == "hi" and c.tokens_in == 10 and c.latency_s >= 0


def test_logging_llm_records_structured_call_and_images(store: RunStore, png: Path):
    h = store.new_run("d")
    llm = LoggingLLM(FakeLLM('{"x": 9}'), h.record_llm_call, agent="critic")
    out = llm.complete([{"role": "user", "content": "judge"}], schema=Out, images=[png])
    assert out.x == 9
    c = h.llm_calls()[0]
    assert c.response == {"x": 9}
    assert c.messages[-1] == {"role": "_images", "content": [str(png)]}


def test_logging_llm_records_errors_then_reraises(store: RunStore):
    h = store.new_run("d")
    llm = LoggingLLM(FakeLLM(fail=True), h.record_llm_call, agent="prompter")
    with pytest.raises(RuntimeError):
        llm.complete([{"role": "user", "content": "x"}])
    assert h.llm_calls()[0].response == {"error": "RuntimeError: boom"}


def test_for_agent_shares_sink(store: RunStore):
    h = store.new_run("d")
    base = LoggingLLM(FakeLLM(), h.record_llm_call, agent="planner")
    base.attempt_index = 2
    critic = base.for_agent("critic")
    critic.complete([{"role": "user", "content": "x"}])
    assert h.llm_calls()[0].agent == "critic"
    assert h.llm_calls()[0].attempt_index == 2
