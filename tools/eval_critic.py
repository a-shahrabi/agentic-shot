"""Measure the critic against your labels.

    python -m tools.eval_critic --model gpt-5.4-mini
    python -m tools.eval_critic --model gpt-5.4-mini --model gpt-5.5     # compare critics
    python -m tools.eval_critic --model gpt-5.4-mini --samples 1          # cheaper, no sampling

Results are cached in cache/critic/, so re-running is free unless the model,
critic prompts, sampling settings, image or spec changed.
Reports go to scratch/critic_eval_<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from agentic_shot.agents.critic import CachedCritic, VLMCritic
from agentic_shot.backends.llm import LoggingLLM, OpenAICompatibleLLM
from agentic_shot.critic_eval import (
    DEFAULT_DIR, FieldStats, compare, image_path, load_labels, overall_agreement,
)
from agentic_shot.schemas import Evaluation, ShotSpec


def pct(x: float | None) -> str:
    return "  -  " if x is None else f"{x * 100:4.0f}%"


def num(x: float | None) -> str:
    return "  -  " if x is None else f"{x:5.2f}"


def print_table(stats: dict[str, FieldStats]) -> None:
    print(f"\n  {'field':<14}{'n':>3} {'agree':>6} {'false':>6} {'false':>6}"
          f" {'score|':>7} {'score|':>7} {'best':>6} {'@best':>6} {'rev':>5} {'spread':>7}")
    print(f"  {'':<14}{'':>3} {'':>6} {'fail':>6} {'pass':>6}"
          f" {'hum OK':>7} {'hum X':>7} {'thr':>6} {'':>6} {'below':>5} {'':>7}")
    for f, s in stats.items():
        thr, acc = s.best_threshold()
        print(f"  {f:<14}{s.n:>3} {pct(s.agreement):>6} {s.false_fail:>6} {s.false_pass:>6}"
              f" {num(s.mean_when_human_pass):>7} {num(s.mean_when_human_fail):>7}"
              f" {num(thr):>6} {pct(acc):>6} {ShotSpec.meta(f).revise_below:>5.2f}"
              f" {num(s.mean_spread):>7}")
    print(f"\n  overall agreement: {pct(overall_agreement(stats)).strip()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append",
                    help="Critic model (repeatable to compare). Needs vision support.")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--show", type=int, default=15, help="Max disagreements to print per model.")
    args = ap.parse_args()
    models = args.model or ([os.environ["AGENTIC_SHOT_CRITIC_MODEL"]]
                            if os.environ.get("AGENTIC_SHOT_CRITIC_MODEL") else None)
    if not models:
        ap.error("pass --model (repeatable) or set AGENTIC_SHOT_CRITIC_MODEL")

    items = load_labels(args.dir)
    if not items:
        ap.error(f"no labels in {args.dir}/labels. Run: python -m tools.label --model <model>")

    out_dir = Path("scratch") / f"critic_eval_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "llm_calls.jsonl"

    def sink(rec):
        with log_path.open("a") as f:
            f.write(rec.model_dump_json() + "\n")

    report: dict = {"labels": len(items), "models": {}}
    print(f"{len(items)} labeled images   output: {out_dir}")

    for model in models:
        llm = LoggingLLM(OpenAICompatibleLLM(model), sink, agent="critic")
        inner = VLMCritic(llm, samples=args.samples)
        critic = inner if args.no_cache else CachedCritic(inner)

        print("\n" + "=" * 96 + f"\ncritic: {model}   samples: {args.samples}")
        evaluations: dict[str, Evaluation] = {}
        errors: dict[str, str] = {}
        for i, item in enumerate(items, 1):
            print(f"  [{i}/{len(items)}] {item.filename}", end="", flush=True)
            try:
                evaluations[item.filename] = critic.evaluate(image_path(args.dir, item.filename), item.spec)
                print()
            except Exception as e:
                errors[item.filename] = f"{type(e).__name__}: {e}"
                print(f"  ERROR {errors[item.filename]}")
        if isinstance(critic, CachedCritic):
            print(f"  cache: {critic.hits} hits, {critic.misses} new")

        stats, disagreements = compare(items, evaluations)
        print_table(stats)

        # false passes first: those are the ones that let bad shots through
        disagreements.sort(key=lambda d: (d.human_pass, d.field))
        if disagreements:
            print(f"\n  DISAGREEMENTS ({len(disagreements)}, showing {min(len(disagreements), args.show)})")
        for d in disagreements[: args.show]:
            kind = "FALSE PASS" if not d.human_pass else "false fail"
            print(f"\n  {kind}  {d.filename}  [{d.field}]  critic score {d.critic_score:.2f}")
            for label, text in (("spec", d.spec_value), ("critic saw", d.critic_saw),
                                ("critique", d.critique), ("your note", d.human_note)):
                if text:
                    print(textwrap.indent(textwrap.fill(f"{label}: {text}", 88), "      "))

        report["models"][model] = {
            "samples": args.samples,
            "overall_agreement": overall_agreement(stats),
            "fields": {f: {**asdict(s), "agreement": s.agreement,
                           "best_threshold": s.best_threshold()[0],
                           "accuracy_at_best": s.best_threshold()[1],
                           "revise_below": ShotSpec.meta(f).revise_below}
                       for f, s in stats.items()},
            "disagreements": [asdict(d) for d in disagreements],
            "errors": errors,
            "evaluations": {k: json.loads(v.model_dump_json()) for k, v in evaluations.items()},
        }

    if len(models) > 1:
        print("\n" + "=" * 96 + "\nSUMMARY (agreement per field)")
        fields = list(ShotSpec.image_fields())
        print(f"  {'model':<22}" + "".join(f"{f[:10]:>11}" for f in fields) + f"{'overall':>9}")
        for m, r in report["models"].items():
            row = "".join(f"{pct(r['fields'][f]['agreement']):>11}" for f in fields)
            print(f"  {m:<22}{row}{pct(r['overall_agreement']):>9}")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved {out_dir}/report.json")


if __name__ == "__main__":
    main()