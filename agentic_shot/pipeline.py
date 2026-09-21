"""The agent loop. This is the only module that knows the order of operations."""

from __future__ import annotations

import random
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .backends.llm import LoggingLLM
from .config import PipelineConfig
from .interfaces import (
    Critic, ImageGenerator, Planner, Prompter, VideoCritic, VideoGenerator,
)
from .schemas import Attempt, RunManifest, RunStatus, ShotSpec, VideoPrompt
from .storage import RunHandle, RunStore


@dataclass
class Pipeline:
    planner: Planner
    prompter: Prompter
    image_gen: ImageGenerator
    critic: Critic
    video_gen: VideoGenerator
    video_critic: VideoCritic
    store: RunStore
    config: PipelineConfig = field(default_factory=PipelineConfig)
    loggers: Sequence[LoggingLLM] = ()   # tagged with attempt_index as the loop advances
    rng: random.Random = field(default_factory=random.Random)

    # ------------------------------------------------------------------ public

    def run(self, description: str) -> RunManifest:
        run = self.store.new_run(description, config=self._config_snapshot())
        try:
            spec = self.planner.plan(description)
            run.set_spec(spec)
            best = self._keyframe_loop(run, spec)
            self._finish(run, spec, best)
        except Exception as e:
            run.finalize(RunStatus.ERROR, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            raise
        return run.manifest

    # ------------------------------------------------------------ keyframe loop

    def _keyframe_loop(self, run: RunHandle, spec: ShotSpec) -> Attempt:
        cfg = self.config
        prompt = self.prompter.write(spec)
        seed = cfg.seed.initial_seed if cfg.seed.initial_seed is not None else self._new_seed()
        attempts: list[Attempt] = []
        prev_flagged: set[str] = set()

        for i in range(cfg.max_attempts):
            self._tag_loggers(i)
            gen = self.image_gen.generate(prompt, seed=seed)
            ev = self.critic.evaluate(gen.path, spec)
            attempt = run.record_attempt(Attempt(
                index=i, prompt=prompt, seed=seed, image_path=gen.path,
                evaluation=ev, cache_hit=gen.cache_hit, gen_latency_s=gen.latency_s,
            ))
            attempts.append(attempt)

            if ev.overall_pass(cfg.pass_threshold):
                break
            if i == cfg.max_attempts - 1:
                break

            # Regression guard: always revise from the best attempt so far,
            # not from the latest one (which may have regressed a critical field).
            base = self._best(attempts)
            flagged = set(base.evaluation.fields_to_revise())

            if not flagged:
                if not cfg.seed.reroll_when_nothing_to_revise:
                    break
                prompt, seed = base.prompt, self._new_seed()
                prev_flagged = set()
                continue

            repeat = bool(flagged & prev_flagged)
            new_prompt = self.prompter.revise(base.prompt, base.evaluation, spec, escalate=repeat)

            reseed = (
                cfg.seed.reseed_on_repeat_failure and repeat and (i + 1) >= cfg.seed.hold_for
            ) or new_prompt.compile() == base.prompt.compile()  # unchanged prompt + same seed = same image

            prompt = new_prompt
            seed = self._new_seed() if reseed else base.seed
            prev_flagged = flagged

        return self._best(attempts)

    # ------------------------------------------------------------------ finish

    def _finish(self, run: RunHandle, spec: ShotSpec, best: Attempt) -> None:
        cfg = self.config
        passed = best.evaluation.overall_pass(cfg.pass_threshold)
        critical_ok = best.evaluation.critical_pass()

        if passed:
            status = RunStatus.COMPLETED
        elif cfg.on_exhausted == "never":
            run.finalize(RunStatus.FAILED_KEYFRAME, chosen_attempt=best.index)
            return
        elif cfg.on_exhausted == "animate_if_critical_pass":
            if not critical_ok:
                run.finalize(RunStatus.FAILED_KEYFRAME, chosen_attempt=best.index)
                return
            status = RunStatus.COMPLETED_WITH_SOFT_FAILURES
        else:  # "always"
            status = (RunStatus.COMPLETED_WITH_SOFT_FAILURES if critical_ok
                      else RunStatus.COMPLETED_WITH_CRITICAL_FAILURES)

        video_prompt = VideoPrompt.from_spec(spec, duration_s=cfg.video_duration_s)
        try:
            self._tag_loggers(None)
            video_path = self.video_gen.animate(best.image_path, video_prompt)
            video_ev = self.video_critic.evaluate(video_path, spec)
            run.set_video(video_prompt, video_path, video_ev)
        except Exception as e:
            run.finalize(RunStatus.FAILED_VIDEO, chosen_attempt=best.index,
                         error=f"{type(e).__name__}: {e}")
            return

        run.finalize(status, chosen_attempt=best.index)

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _best(attempts: list[Attempt]) -> Attempt:
        # max() keeps the first on ties, so earlier attempts win equal scores.
        return max(attempts, key=lambda a: a.evaluation.rank_key())

    def _new_seed(self) -> int:
        return self.rng.randint(0, 2**31 - 1)

    def _tag_loggers(self, attempt_index: int | None) -> None:
        for lg in self.loggers:
            lg.attempt_index = attempt_index

    def _config_snapshot(self) -> dict:
        return {
            **self.config.model_dump(),
            "image_gen": getattr(self.image_gen, "name", type(self.image_gen).__name__),
            "video_gen": getattr(self.video_gen, "name", type(self.video_gen).__name__),
            "critic": type(self.critic).__name__,
            "planner": type(self.planner).__name__,
            "prompter": type(self.prompter).__name__,
        }
