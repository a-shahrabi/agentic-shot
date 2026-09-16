# agentic-shot

An agentic pipeline that turns a one-line scene description into a video shot.
An LLM writes a structured shot spec, generates a keyframe, critiques its own
output field by field, revises the prompt where it missed, and animates the
best attempt.

**Status:** in development (Stage 1 — single shot)

## How it works

description → shot spec → image prompt → keyframe → critique → revise → animate

The critique loop is the point. The critic scores each spec field separately
(subject, camera, lighting, mood, style) and the prompter revises only the
segments that failed.

## Stack

Python · OpenAI API (planner/prompter/critic) · mflux for local Flux on Apple
Silicon · Runway for video · Gradio UI

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

## Roadmap

- **Stage 1** — single shot, evaluate-and-revise loop
- **Stage 2** — evaluation harness, critic scored against human labels
- **Stage 3** — multi-model routing
- **Stage 4** — multi-shot sequences with style consistency
