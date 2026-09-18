from pathlib import Path

from agentic_shot.schemas import (
    Attempt, EvaluableIn, Evaluation, FieldVerdict, ImagePrompt,
    RunManifest, RunStatus, ShotSpec, VerdictStatus, VideoPrompt,
)


def make_spec() -> ShotSpec:
    return ShotSpec(
        subject="a lone samurai in a black kimono",
        setting="rain-soaked neon Tokyo street at night",
        action="walks slowly away from camera",
        camera_angle="from behind, eye level, medium-wide",
        camera_movement="slow dolly forward",
        lighting="neon signs, wet reflections, cool blue key with magenta fill",
        mood="solitary, tense",
        style="Blade Runner meets Kurosawa, anamorphic, 35mm grain",
    )


def make_prompt() -> ImagePrompt:
    return ImagePrompt(
        segments={
            "subject": "a lone samurai in a black kimono",
            "setting": "rain-soaked neon Tokyo street at night",
            "camera_angle": "shot from behind, eye level, medium-wide",
            "lighting": "neon reflections on wet asphalt, cool blue key, magenta fill",
            "mood": "solitary, tense",
            "style": "anamorphic 35mm film grain, cinematic",
        },
        negative="blurry, cartoon, text",
    )


def verdict(field: str, score: float, passed: bool) -> FieldVerdict:
    return FieldVerdict(field=field, score=score, passed=passed, critique="x")


def make_eval(subject_ok: bool = True, camera: float = 0.9) -> Evaluation:
    return Evaluation(
        stage=EvaluableIn.IMAGE,
        critic_model="test",
        verdicts=[
            verdict("subject", 0.95 if subject_ok else 0.1, subject_ok),
            verdict("setting", 0.9, True),
            verdict("camera_angle", camera, camera >= 0.5),
            verdict("lighting", 0.8, True),
            verdict("mood", 0.8, True),
            verdict("style", 0.8, True),
            FieldVerdict.skipped("action"),
            FieldVerdict.skipped("camera_movement"),
        ],
    )


# ---- round trips -----------------------------------------------------------

def test_spec_roundtrip():
    s = make_spec()
    assert ShotSpec.model_validate_json(s.model_dump_json()) == s


def test_prompt_roundtrip_and_compile():
    p = make_prompt()
    assert ImagePrompt.model_validate_json(p.model_dump_json()) == p
    compiled = p.compile()
    assert compiled.startswith("a lone samurai")
    assert compiled.index("shot from behind") < compiled.index("neon reflections")


def test_manifest_roundtrip(tmp_path: Path):
    m = RunManifest(run_id="r1", description="samurai", spec=make_spec())
    m.attempts.append(Attempt(
        index=0, prompt=make_prompt(), seed=42,
        image_path=tmp_path / "keyframe.png", evaluation=make_eval(),
    ))
    m.video_prompt = VideoPrompt.from_spec(m.spec)
    m.status = RunStatus.COMPLETED
    back = RunManifest.model_validate_json(m.model_dump_json())
    assert back == m


# ---- field metadata --------------------------------------------------------

def test_field_classification():
    assert "camera_movement" in ShotSpec.video_fields()
    assert "camera_angle" in ShotSpec.image_fields()
    assert set(ShotSpec.image_fields()) | set(ShotSpec.video_fields()) == set(ShotSpec.FIELD_ORDER)


def test_unknown_segment_rejected():
    import pytest
    with pytest.raises(ValueError):
        ImagePrompt(segments={"lens": "50mm"})


# ---- prompt revision -------------------------------------------------------

def test_with_segments_touches_only_target():
    p = make_prompt()
    p2 = p.with_segments({"camera_angle": "rear view, following the subject"})
    assert p.diff(p2) == ["camera_angle"]
    assert p2.version == p.version + 1
    assert p2.segments["lighting"] == p.segments["lighting"]


# ---- pass rule -------------------------------------------------------------

def test_critical_failure_vetoes_despite_high_soft_scores():
    e = make_eval(subject_ok=False)
    assert e.weighted_score() > 0.5
    assert not e.overall_pass(threshold=0.5)


def test_all_critical_pass_and_mean_above_threshold():
    e = make_eval()
    assert e.overall_pass(threshold=0.7)
    assert not e.overall_pass(threshold=0.95)


def test_fields_to_revise_respects_per_field_threshold():
    # camera_angle at 0.4 is below pass but above its revise_below (0.3): logged, no retry
    assert make_eval(camera=0.4).fields_to_revise() == []
    # at 0.2 it is below revise_below: triggers revision
    assert make_eval(camera=0.2).fields_to_revise() == ["camera_angle"]


def test_best_attempt_prefers_critical_pass_over_score():
    m = RunManifest(run_id="r", description="d", spec=make_spec())
    good_soft_bad_critical = make_eval(subject_ok=False, camera=0.95)
    ok = make_eval(subject_ok=True, camera=0.5)
    for i, ev in enumerate([good_soft_bad_critical, ok]):
        m.attempts.append(Attempt(index=i, prompt=make_prompt(), seed=1,
                                  image_path=Path(f"{i}.png"), evaluation=ev))
    assert m.best_attempt().index == 1


def test_skipped_verdicts_ignored_in_score():
    e = make_eval()
    assert all(v.status is VerdictStatus.EVALUATED for v in e.evaluated())
    assert len(e.evaluated()) == 6