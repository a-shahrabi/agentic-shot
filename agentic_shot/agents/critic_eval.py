"""Human-labeled critic dataset + agreement metrics.

Dataset layout:
    data/critic_set/
        images/             the images
        descriptions.csv    filename,description
        labels/             one <filename>.json per labeled image (written by tools/label.py)

This module is the seed of the Stage 2 harness: it only knows about specs,
human verdicts and Evaluations, not about any particular critic or model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .schemas import Evaluation, FieldVerdict, ShotSpec, VerdictStatus

DEFAULT_DIR = Path("data/critic_set")


# ---- dataset ----------------------------------------------------------------

class LabeledItem(BaseModel):
    filename: str
    description: str
    spec: ShotSpec
    human: list[FieldVerdict] = Field(description="Human pass/fail per field (score 1.0 or 0.0).")
    labeled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    planner_model: str | None = None

    def human_by_field(self) -> dict[str, FieldVerdict]:
        return {v.field: v for v in self.human if v.status is VerdictStatus.EVALUATED}


def read_descriptions(root: Path = DEFAULT_DIR) -> tuple[list[tuple[str, str]], list[str]]:
    """Returns ([(filename, description)], [missing filenames])."""
    csv_path = Path(root) / "descriptions.csv"
    rows, missing = [], []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower() for c in (reader.fieldnames or [])}
        if not {"filename", "description"} <= cols:
            raise ValueError(f"{csv_path} needs a header row: filename,description (got {reader.fieldnames})")
        for row in reader:
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            name, desc = row.get("filename", ""), row.get("description", "")
            if not name or not desc:
                continue
            if not (Path(root) / "images" / name).exists():
                missing.append(name)
                continue
            rows.append((name, desc))
    return rows, missing


def image_path(root: Path, filename: str) -> Path:
    return Path(root) / "images" / filename


def label_path(root: Path, filename: str) -> Path:
    return Path(root) / "labels" / f"{filename}.json"


def save_label(root: Path, item: LabeledItem) -> Path:
    p = label_path(root, item.filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(item.model_dump_json(indent=2))
    return p


def load_labels(root: Path = DEFAULT_DIR) -> list[LabeledItem]:
    d = Path(root) / "labels"
    if not d.exists():
        return []
    return [LabeledItem.model_validate_json(p.read_text()) for p in sorted(d.glob("*.json"))]


def human_verdict(field_name: str, matches: bool, note: str | None = None) -> FieldVerdict:
    return FieldVerdict(field=field_name, score=1.0 if matches else 0.0, passed=matches,
                        critique=note or None)


# ---- metrics ----------------------------------------------------------------

def best_threshold(pass_scores: list[float], fail_scores: list[float]) -> tuple[float | None, float | None]:
    """Score threshold that best separates human-pass from human-fail
    (critic says pass iff score >= t). Returns (threshold, accuracy).
    Ties prefer the threshold closest to 0.5."""
    n = len(pass_scores) + len(fail_scores)
    if n == 0 or not pass_scores or not fail_scores:
        return None, None
    points = sorted(set(pass_scores) | set(fail_scores))
    candidates = {0.0, 1.01} | {round((a + b) / 2, 3) for a, b in zip(points, points[1:])}
    def acc(t: float) -> float:
        return (sum(s >= t for s in pass_scores) + sum(s < t for s in fail_scores)) / n
    best = max(candidates, key=lambda t: (acc(t), -abs(t - 0.5)))
    return best, acc(best)


@dataclass
class FieldStats:
    field: str
    n: int = 0
    agree: int = 0
    false_fail: int = 0            # critic fail, human pass
    false_pass: int = 0            # critic pass, human fail  <- the dangerous one
    pass_scores: list[float] = field(default_factory=list)   # critic scores where human passed
    fail_scores: list[float] = field(default_factory=list)   # critic scores where human failed
    spreads: list[float] = field(default_factory=list)

    @property
    def agreement(self) -> float | None:
        return self.agree / self.n if self.n else None

    @property
    def mean_when_human_pass(self) -> float | None:
        return sum(self.pass_scores) / len(self.pass_scores) if self.pass_scores else None

    @property
    def mean_when_human_fail(self) -> float | None:
        return sum(self.fail_scores) / len(self.fail_scores) if self.fail_scores else None

    @property
    def mean_spread(self) -> float | None:
        return sum(self.spreads) / len(self.spreads) if self.spreads else None

    def best_threshold(self) -> tuple[float | None, float | None]:
        return best_threshold(self.pass_scores, self.fail_scores)


@dataclass
class Disagreement:
    filename: str
    field: str
    human_pass: bool
    critic_score: float
    spec_value: str
    critic_saw: str | None
    critique: str | None
    human_note: str | None


def compare(items: list[LabeledItem], evaluations: dict[str, Evaluation]
            ) -> tuple[dict[str, FieldStats], list[Disagreement]]:
    """evaluations: filename -> critic Evaluation. Items without one are skipped."""
    stats = {f: FieldStats(f) for f in ShotSpec.image_fields()}
    disagreements: list[Disagreement] = []
    for item in items:
        ev = evaluations.get(item.filename)
        if ev is None:
            continue
        critic = {v.field: v for v in ev.evaluated()}
        for f, h in item.human_by_field().items():
            c = critic.get(f)
            if c is None or f not in stats:
                continue
            s = stats[f]
            s.n += 1
            (s.pass_scores if h.passed else s.fail_scores).append(c.score)
            if c.sample_spread is not None:
                s.spreads.append(c.sample_spread)
            if c.passed == h.passed:
                s.agree += 1
                continue
            if h.passed:
                s.false_fail += 1
            else:
                s.false_pass += 1
            disagreements.append(Disagreement(
                filename=item.filename, field=f, human_pass=bool(h.passed),
                critic_score=c.score, spec_value=getattr(item.spec, f),
                critic_saw=c.description, critique=c.critique, human_note=h.critique,
            ))
    return stats, disagreements


def overall_agreement(stats: dict[str, FieldStats]) -> float | None:
    n = sum(s.n for s in stats.values())
    return sum(s.agree for s in stats.values()) / n if n else None