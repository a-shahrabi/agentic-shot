"""Hand-label images against planned specs. Your labels are the ground truth
the critic is measured against.

    python -m tools.label --model gpt-5.4-mini
    python -m tools.label --model gpt-5.4-mini --only samurai_back.jpg --relabel

For each image: it opens the image, plans a spec from the description, then asks
you field by field whether the IMAGE matches the SPEC:
    y  matches        n  doesn't match (optionally add a note)
    e  edit the spec text first (use this when the planner invented a detail
       the description never asked for, e.g. a wardrobe color)
    s  skip this field (excluded from scoring)
    q  quit (the current image is not saved)

Already-labeled images are skipped unless --relabel.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from agentic_shot.agents import LLMPlanner
from agentic_shot.backends.llm import OpenAICompatibleLLM
from agentic_shot.critic_eval import (
    DEFAULT_DIR, LabeledItem, human_verdict, image_path, label_path,
    read_descriptions, save_label,
)
from agentic_shot.schemas import FieldVerdict, ShotSpec


class Quit(Exception):
    pass


def open_image(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        print(f"  (open manually: {path})")


def ask(prompt: str, allowed: str) -> str:
    while True:
        a = input(prompt).strip().lower()
        if a[:1] in allowed and a:
            return a[0]
        print(f"    please type one of: {' / '.join(allowed)}")


def label_field(spec: ShotSpec, f: str) -> tuple[ShotSpec, FieldVerdict]:
    while True:
        value = getattr(spec, f)
        print(f"\n  {f.upper()}")
        print(textwrap.indent(textwrap.fill(value, 90), "    "))
        a = ask("    image matches?  [y]es [n]o [e]dit spec [s]kip [q]uit > ", "ynesq")
        if a == "q":
            raise Quit
        if a == "s":
            return spec, FieldVerdict.skipped(f)
        if a == "e":
            new = input("    new spec text > ").strip()
            if new:
                spec = spec.model_copy(update={f: new})
            continue
        note = None
        if a == "n":
            note = input("    what's wrong? (optional, Enter to skip) > ").strip() or None
        return spec, human_verdict(f, matches=(a == "y"), note=note)


def label_one(planner: LLMPlanner, root: Path, filename: str, description: str,
              model: str) -> LabeledItem:
    print("\n" + "=" * 96)
    print(f"{filename}\n  {description}")
    open_image(image_path(root, filename))
    print("  planning spec...")
    spec = planner.plan(description)

    human: list[FieldVerdict] = []
    for f in ShotSpec.FIELD_ORDER:
        if f in ShotSpec.video_fields():
            human.append(FieldVerdict.skipped(f))
            continue
        spec, verdict = label_field(spec, f)
        human.append(verdict)
    return LabeledItem(filename=filename, description=description, spec=spec,
                       human=human, planner_model=model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("AGENTIC_SHOT_LLM_MODEL"),
                    help="Planner model (same one the pipeline uses).")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--only", action="append", help="Label only this filename (repeatable).")
    ap.add_argument("--relabel", action="store_true")
    args = ap.parse_args()
    if not args.model:
        ap.error("pass --model or set AGENTIC_SHOT_LLM_MODEL")

    rows, missing = read_descriptions(args.dir)
    for m in missing:
        print(f"WARNING: {m} is in descriptions.csv but not in images/")
    if args.only:
        rows = [r for r in rows if r[0] in set(args.only)]
    todo = [r for r in rows if args.relabel or not label_path(args.dir, r[0]).exists()]
    print(f"{len(rows)} images, {len(rows) - len(todo)} already labeled, {len(todo)} to go")

    planner = LLMPlanner(OpenAICompatibleLLM(args.model))
    done = 0
    try:
        for filename, description in todo:
            item = label_one(planner, args.dir, filename, description, args.model)
            path = save_label(args.dir, item)
            done += 1
            print(f"\n  saved {path}")
    except (Quit, KeyboardInterrupt):
        print("\n\nstopped; the current image was not saved.")
    print(f"labeled {done} this session")


if __name__ == "__main__":
    main()