"""Run descriptions through the real planner + prompter and print what they produce.

    python -m tools.try_agents                      # all built-in descriptions
    python -m tools.try_agents -n 3                 # first 3
    python -m tools.try_agents -d "your own scene"  # one custom description
    python -m tools.try_agents --model <model>      # or set AGENTIC_SHOT_LLM_MODEL

Output + every LLM call are saved under scratch/agents_<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from datetime import datetime
from pathlib import Path

from agentic_shot.agents import LLMPlanner, LLMPrompter
from agentic_shot.backends.llm import LoggingLLM, OpenAICompatibleLLM
from agentic_shot.schemas import EvaluableIn, Evaluation, FieldVerdict, ShotSpec

DEFAULT_MODEL = os.environ.get("AGENTIC_SHOT_LLM_MODEL")

DESCRIPTIONS = [
    "A lone samurai walks through rain-soaked neon Tokyo streets at night, dolly shot from behind.",
    "A gunslinger's eyes in extreme close-up as a church bell rings over a dusty square at noon.",
    "Two sisters argue in a cramped Seoul apartment kitchen, handheld, fluorescent light.",
    "An astronaut floats alone inside a derelict space station, slow push in.",
    "A priest sits motionless in an empty diner at 3am, static wide shot.",
    "A woman in a red coat crosses a snowy train platform as a steam train arrives.",
    "Low angle of a boxer resting on the ropes between rounds, sweat and smoke.",
    "A child runs through a Mexico City street market at golden hour, long tracking shot.",
    "A detective lights a cigarette under a bridge, fog, 1970s film look.",
    "A delivery robot stops at an abandoned crosswalk in a flooded future city at dawn.",
]

WRAP = 100


def show(title: str, mapping: dict[str, str]) -> None:
    print(f"\n  {title}")
    for k, v in mapping.items():
        lines = textwrap.wrap(v, WRAP - 20) or [""]
        print(f"    {k:<16}{lines[0]}")
        for line in lines[1:]:
            print(f"    {'':<16}{line}")


def fake_camera_failure(spec: ShotSpec) -> Evaluation:
    """A plausible critic verdict: everything fine except the camera angle."""
    verdicts = []
    for f in ShotSpec.FIELD_ORDER:
        if f in ShotSpec.video_fields():
            verdicts.append(FieldVerdict.skipped(f))
        elif f == "camera_angle":
            verdicts.append(FieldVerdict(
                field=f, score=0.1, passed=False,
                description="Medium shot from the front; the subject walks toward the camera and the face is fully visible.",
                critique="Spec calls for a view from behind; the image shows the subject from the front.",
                suggested_fix="Make the rear viewpoint explicit and state that the face is not visible.",
            ))
        else:
            verdicts.append(FieldVerdict(field=f, score=0.9, passed=True))
    return Evaluation(verdicts=verdicts, stage=EvaluableIn.IMAGE, critic_model="demo")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None, help="e.g. http://localhost:11434/v1 for Ollama")
    ap.add_argument("-d", "--description", action="append", help="Custom description (repeatable).")
    ap.add_argument("-n", type=int, default=None, help="Use only the first N built-in descriptions.")
    ap.add_argument("--no-revise-demo", action="store_true")
    args = ap.parse_args()
    if not args.model:
        ap.error("pass --model <name> or set AGENTIC_SHOT_LLM_MODEL "
                 "(any OpenAI chat model that supports structured output)")

    descriptions = args.description or DESCRIPTIONS[: args.n]
    out_dir = Path("scratch") / f"agents_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "llm_calls.jsonl"

    def sink(rec):
        with log_path.open("a") as f:
            f.write(rec.model_dump_json() + "\n")

    base = LoggingLLM(OpenAICompatibleLLM(args.model, base_url=args.base_url), sink)
    planner = LLMPlanner(base.for_agent("planner"))
    prompter = LLMPrompter(base.for_agent("prompter"))

    print(f"model: {args.model}   output: {out_dir}")
    results = []
    for i, desc in enumerate(descriptions, 1):
        print("\n" + "=" * WRAP)
        print(f"[{i}/{len(descriptions)}] {desc}")
        try:
            spec = planner.plan(desc)
            prompt = prompter.write(spec)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results.append({"description": desc, "error": str(e)})
            continue
        show("SPEC", spec.as_dict())
        show("PROMPT SEGMENTS", prompt.segments)
        print(f"\n  COMPILED ({len(prompt.compile().split())} words)")
        print(textwrap.indent(textwrap.fill(prompt.compile(), WRAP - 4), "    "))
        results.append({"description": desc, "spec": spec.model_dump(),
                        "prompt": prompt.model_dump(), "compiled": prompt.compile()})

    first = next((r for r in results if "spec" in r), None)
    if first and not args.no_revise_demo:
        print("\n" + "=" * WRAP)
        print("REVISE DEMO: pretend the critic said the camera angle is wrong")
        spec = ShotSpec.model_validate(first["spec"])
        from agentic_shot.schemas import ImagePrompt
        prompt = ImagePrompt.model_validate(first["prompt"])
        ev = fake_camera_failure(spec)
        for escalate in (False, True):
            revised = prompter.revise(prompt, ev, spec, escalate=escalate)
            mode = "REWRITE" if escalate else "NUDGE"
            changed = prompt.diff(revised)
            print(f"\n  {mode}: changed fields = {changed}")
            for f in changed:
                show(f"{f}  (before)", {f: prompt.segments.get(f, "")})
                show(f"{f}  (after)", {f: revised.segments[f]})
                print(f"    rationale: {prompter.last_rationales.get(f, '-')}")
            results.append({"revise_demo": mode, "changed": changed,
                            "revised": revised.model_dump()})

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_dir}/results.json and {log_path.name}")


if __name__ == "__main__":
    main()
