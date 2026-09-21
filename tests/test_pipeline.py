from pathlib import Path

import pytest

from agentic_shot.backends.fakes import (
    FakeImageGenerator, FakePlanner, FakePrompter, FakeVideoGenerator,
    NoopVideoCritic, ScriptedCritic,
)
from agentic_shot.config import PipelineConfig, SeedPolicy
from agentic_shot.pipeline import Pipeline
from agentic_shot.schemas import RunStatus, VerdictStatus
from agentic_shot.storage import RunStore
from tests.test_schemas import make_spec

GOOD = {}                      # everything defaults to 0.9 -> pass
BAD_CAMERA = {"camera_angle": 0.1}
BAD_SUBJECT = {"subject": 0.1}
SOFT_FAIL = {"lighting": 0.35, "mood": 0.35, "style": 0.35}  # above revise_below (0.3), below pass


def build(tmp_path: Path, script, *, config=None, prompter=None, video_fail=False):
    store = RunStore(tmp_path / "runs")
    p = Pipeline(
        planner=FakePlanner(make_spec()),
        prompter=prompter or FakePrompter(),
        image_gen=FakeImageGenerator(tmp_path / "gen"),
        critic=ScriptedCritic(script),
        video_gen=FakeVideoGenerator(tmp_path / "vid", fail=video_fail),
        video_critic=NoopVideoCritic(),
        store=store,
        config=config or PipelineConfig(seed=SeedPolicy(initial_seed=1000)),
    )
    return p, store


# ---- happy path -------------------------------------------------------------

def test_passes_first_attempt_and_stops(tmp_path):
    p, store = build(tmp_path, [GOOD])
    m = p.run("samurai")
    assert m.status is RunStatus.COMPLETED
    assert len(m.attempts) == 1 and m.chosen_attempt == 0
    assert m.video_path.exists()
    assert len(p.image_gen.calls) == 1


def test_recovers_on_second_attempt(tmp_path):
    p, store = build(tmp_path, [BAD_CAMERA, GOOD])
    m = p.run("samurai")
    assert m.status is RunStatus.COMPLETED
    assert len(m.attempts) == 2 and m.chosen_attempt == 1
    assert p.prompter.revise_calls[0]["fields"] == ["camera_angle"]
    assert p.prompter.revise_calls[0]["escalate"] is False


# ---- revision targeting -----------------------------------------------------

def test_revision_touches_only_flagged_segment(tmp_path):
    p, _ = build(tmp_path, [BAD_CAMERA, GOOD])
    m = p.run("samurai")
    a0, a1 = m.attempts
    assert a0.prompt.diff(a1.prompt) == ["camera_angle"]
    assert a1.prompt.version == a0.prompt.version + 1


def test_soft_failure_above_revise_below_does_not_trigger_revision(tmp_path):
    # fails overall (weighted mean below threshold) but no field is below revise_below
    p, _ = build(tmp_path, [SOFT_FAIL, GOOD])
    m = p.run("samurai")
    assert p.prompter.revise_calls == []          # nothing was revised
    assert p.image_gen.calls[0][1] != p.image_gen.calls[1][1]   # rerolled the seed instead


def test_reroll_disabled_stops_early(tmp_path):
    cfg = PipelineConfig(seed=SeedPolicy(initial_seed=1000, reroll_when_nothing_to_revise=False))
    p, _ = build(tmp_path, [SOFT_FAIL], config=cfg)
    m = p.run("samurai")
    assert len(m.attempts) == 1
    assert m.status is RunStatus.COMPLETED_WITH_SOFT_FAILURES


# ---- seed policy ------------------------------------------------------------

def test_seed_held_for_first_two_attempts(tmp_path):
    p, _ = build(tmp_path, [BAD_CAMERA, GOOD])
    p.run("samurai")
    seeds = [s for _, s in p.image_gen.calls]
    assert seeds[0] == seeds[1] == 1000


def test_repeat_failure_triggers_reseed_on_third(tmp_path):
    p, _ = build(tmp_path, [BAD_CAMERA, BAD_CAMERA, GOOD])
    p.run("samurai")
    seeds = [s for _, s in p.image_gen.calls]
    assert seeds[0] == seeds[1] == 1000
    assert seeds[2] != 1000
    assert p.prompter.revise_calls[1]["escalate"] is True   # same field twice -> rewrite


def test_unchanged_prompt_forces_reseed(tmp_path):
    p, _ = build(tmp_path, [BAD_CAMERA, GOOD], prompter=FakePrompter(no_op=True))
    p.run("samurai")
    seeds = [s for _, s in p.image_gen.calls]
    assert seeds[0] != seeds[1]


def test_reseed_disabled_keeps_seed(tmp_path):
    cfg = PipelineConfig(seed=SeedPolicy(initial_seed=1000, reseed_on_repeat_failure=False))
    p, _ = build(tmp_path, [BAD_CAMERA, BAD_CAMERA, GOOD], config=cfg)
    p.run("samurai")
    assert {s for _, s in p.image_gen.calls} == {1000}


# ---- regression guard -------------------------------------------------------

def test_revises_from_best_not_latest(tmp_path):
    # attempt 0: camera bad. attempt 1: regresses subject (critical). attempt 2 must
    # revise from attempt 0, so it should be flagged on camera_angle, not subject.
    p, _ = build(tmp_path, [BAD_CAMERA, BAD_SUBJECT, GOOD])
    m = p.run("samurai")
    assert p.prompter.revise_calls[1]["fields"] == ["camera_angle"]
    assert p.prompter.revise_calls[1]["base_version"] == m.attempts[0].prompt.version


# ---- best-attempt selection -------------------------------------------------

def test_best_attempt_prefers_critical_pass(tmp_path):
    # attempt 2 has a high soft score but fails subject; attempt 1 is the honest best
    p, _ = build(tmp_path, [BAD_SUBJECT, BAD_CAMERA, BAD_SUBJECT])
    m = p.run("samurai")
    assert m.chosen_attempt == 1


def test_ties_prefer_earlier_attempt(tmp_path):
    p, _ = build(tmp_path, [BAD_CAMERA, BAD_CAMERA, BAD_CAMERA])
    m = p.run("samurai")
    assert m.chosen_attempt == 0


# ---- on_exhausted -----------------------------------------------------------

def test_soft_failure_animates_and_flags(tmp_path):
    p, _ = build(tmp_path, [BAD_CAMERA, BAD_CAMERA, BAD_CAMERA])
    m = p.run("samurai")
    assert m.status is RunStatus.COMPLETED_WITH_SOFT_FAILURES
    assert m.video_path is not None and m.video_path.exists()


def test_critical_failure_bails_without_animating(tmp_path):
    p, _ = build(tmp_path, [BAD_SUBJECT, BAD_SUBJECT, BAD_SUBJECT])
    m = p.run("samurai")
    assert m.status is RunStatus.FAILED_KEYFRAME
    assert m.video_path is None
    assert p.video_gen.calls == []


def test_on_exhausted_always_animates_critical_failure(tmp_path):
    cfg = PipelineConfig(seed=SeedPolicy(initial_seed=1000), on_exhausted="always")
    p, _ = build(tmp_path, [BAD_SUBJECT] * 3, config=cfg)
    m = p.run("samurai")
    assert m.status is RunStatus.COMPLETED_WITH_CRITICAL_FAILURES
    assert m.video_path.exists()


def test_on_exhausted_never_skips_video(tmp_path):
    cfg = PipelineConfig(seed=SeedPolicy(initial_seed=1000), on_exhausted="never")
    p, _ = build(tmp_path, [BAD_CAMERA] * 3, config=cfg)
    m = p.run("samurai")
    assert m.status is RunStatus.FAILED_KEYFRAME
    assert p.video_gen.calls == []


def test_max_attempts_respected(tmp_path):
    cfg = PipelineConfig(max_attempts=2, seed=SeedPolicy(initial_seed=1000))
    p, _ = build(tmp_path, [BAD_CAMERA] * 5, config=cfg)
    m = p.run("samurai")
    assert len(m.attempts) == 2


# ---- video failure ----------------------------------------------------------

def test_video_failure_recorded(tmp_path):
    p, _ = build(tmp_path, [GOOD], video_fail=True)
    m = p.run("samurai")
    assert m.status is RunStatus.FAILED_VIDEO
    assert "fake video failure" in m.error


# ---- persistence ------------------------------------------------------------

def test_manifest_on_disk_matches_and_is_honest_about_video_fields(tmp_path):
    p, store = build(tmp_path, [BAD_CAMERA, GOOD])
    m = p.run("samurai")
    on_disk = store.load_manifest(m.run_id)
    assert on_disk.status is m.status
    assert len(on_disk.attempts) == 2
    assert on_disk.spec == make_spec()
    # video fields never silently reported as passing
    vid = {v.field: v for v in on_disk.video_evaluation.verdicts}
    assert all(v.status is VerdictStatus.SKIPPED for v in vid.values())
    key = {v.field: v for v in on_disk.attempts[0].evaluation.verdicts}
    assert key["camera_movement"].status is VerdictStatus.SKIPPED


def test_config_snapshot_recorded(tmp_path):
    p, store = build(tmp_path, [GOOD])
    m = p.run("samurai")
    assert m.config["image_gen"] == "fake-image"
    assert m.config["max_attempts"] == 3
    assert m.config["on_exhausted"] == "animate_if_critical_pass"
