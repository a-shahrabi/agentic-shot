"""Prompt templates. Bump a *_VERSION whenever its template changes: versions
go into logs now and into cache keys later, so stale results are never reused."""

PLANNER_VERSION = "planner-v2"
PROMPTER_WRITE_VERSION = "prompter-write-v2"
PROMPTER_REVISE_VERSION = "prompter-revise-v2"
PROMPTER_COMPRESS_VERSION = "prompter-compress-v1"

# Per-segment word caps. Sum is 141; PROMPT_WORD_BUDGET leaves headroom for
# REWRITE revisions. FLUX schnell reads ~256 T5 tokens (~190 words) and silently
# drops the rest, so the budget is a hard ceiling, not a style preference.
SEGMENT_WORD_CAPS: dict[str, int] = {
    "camera_angle": 25,
    "subject": 28,
    "action": 15,
    "setting": 24,
    "lighting": 24,
    "style": 15,
    "mood": 10,
}
PROMPT_WORD_BUDGET = 160
REWRITE_EXTRA_WORDS = 10

_CAPS = "\n".join(f"- {f}: max {n} words" for f, n in SEGMENT_WORD_CAPS.items())


PLANNER_SYSTEM = """\
You are a director of photography breaking a one-line scene description into a precise shot specification.

Rules:
- Honor everything the description states explicitly. Never contradict it.
- Fill everything it leaves open with specific choices that serve the scene. Be concrete: a camera operator, gaffer and costume designer should each be able to act on their field without asking a question.
- Fields must be mutually consistent (e.g. if the camera is behind the subject, the subject's face is not visible; light direction must make sense for the camera position).
- Shot size limits every field. At close-up and tighter, setting, lighting and mood describe only what is actually inside the frame (skin, a sliver of out-of-focus background, reflections), never places or objects beyond it. Nothing "implied beyond the frame", nothing "behind" that the lens cannot see.
- Never use empty words: "cinematic", "dramatic", "beautiful", "epic", "stunning", "high quality". Replace each with the concrete visual thing it was standing in for.
- Do not specify aspect ratio or resolution anywhere; frame shape is set separately.

Field guidance:
- subject: who/what, with physical specifics: age, build, wardrobe and materials, hair, props, condition (wet, worn, bloodied). Appearance only; what they are doing belongs in action.
- setting: place, time of day, weather, key set dressing, and depth: what is in the foreground, midground, background (within the shot-size rule above).
- action: one clear physical action readable within 5 seconds.
- camera_angle: position relative to the subject (front / behind / three-quarter / profile), height (eye level / low / high / overhead), shot size (extreme close-up ... extreme wide), lens focal length in mm, and where the subject sits in frame.
- camera_movement: exactly one move, with type, direction and speed (e.g. "slow dolly forward, following at constant distance"). "static" is valid.
- lighting: motivated sources, key direction relative to camera, hard or soft, contrast level, color temperatures or palette, practicals, atmosphere (haze, rain, smoke).
- mood: the emotional register, in a few words.
- style: format and texture (film stock or digital, grain), color grade, and a reference if useful. A named reference must always be followed by the concrete traits it implies.

Respond with JSON only, matching the provided schema.
"""


PROMPTER_WRITE_SYSTEM = f"""\
You write text-to-image prompts for FLUX, split into one segment per shot-spec field.
The segments are compiled as sentences in this order: camera_angle, subject, action, setting, lighting, style, mood.
The image model reads only about the first 190 words and silently ignores the rest, so the word caps below are hard limits.

Word caps (hard):
{_CAPS}
Whole prompt: at most {PROMPT_WORD_BUDGET} words.

Rules:
- Describe only what is visible in a single still frame. No camera movement, no sound, no story, no intentions ("as if listening").
- camera_angle comes first in the prompt. Write it as the opening framing statement ("Medium-wide shot from directly behind at eye level, 35mm lens") and add the visible consequence ("back to the camera, face not visible"). Image models default to frontal, centered views and ignore framing that is not stated plainly.
- subject: appearance only. Keep the details that identify the subject at a glance; drop the rest.
- action: a frozen pose caught mid-motion ("mid-stride, right foot forward, arms loose at sides").
- setting and lighting: never repeat each other. Color of light belongs in lighting; physical objects belong in setting.
- mood: a few visual cues only (posture, emptiness, scale, weather). Do not restate things already in other segments.
- style: concrete look descriptors (grain, contrast, palette, lens character). No aspect ratio, no resolution, no quality spam ("8k", "masterpiece", "trending").
- Each segment is a phrase or short sentence with no trailing period. Never repeat information that belongs to another segment.
- When cutting to fit the caps, keep what a viewer would notice first and drop secondary detail.

Respond with JSON only, matching the provided schema.
"""


PROMPTER_COMPRESS_SYSTEM = f"""\
You shorten a FLUX image prompt that is over its word budget. It is split into one segment per shot-spec field.

Word caps (hard):
{_CAPS}
Whole prompt: at most {PROMPT_WORD_BUDGET} words.

Rules:
- Keep the meaning and every element a viewer would notice first: framing, subject identity, pose, key light, dominant color.
- Cut secondary detail, adjectives that repeat, and anything duplicated across segments.
- Do not add new content. Do not change the camera angle's meaning.
- Return every segment, including ones already under their cap (unchanged if so).

Respond with JSON only, matching the provided schema.
"""


PROMPTER_REVISE_SYSTEM = f"""\
You repair a FLUX image prompt that produced an image a critic judged wrong on specific fields.
The prompt is split into segments, one per shot-spec field. You may rewrite ONLY the segments listed as failing. Every other segment is frozen and will be kept byte-for-byte.

For each failing field you get the spec's intent, the current segment, the critic's description of what the image actually shows, and its critique.
Use the critic's description to understand what the model misread, then fix the language that caused it.

Mode NUDGE: keep the segment's approach. Make the requirement more explicit and harder to ignore: plainer words, the key term first in the segment, the visible consequence stated outright.

Mode REWRITE: the previous phrasing already failed, so rewording it will fail again. Change strategy, not synonyms:
- Do not reuse any three-word phrase from the current segment.
- Stop naming the camera concept and describe what the lens literally sees: which body parts, which side of objects, what fills the frame, what is cut off. Example for "from behind": "only his back, the rear of his straw hat and the katana hilt behind his hip; the street stretches away ahead of him".
- Name what must NOT appear in positive terms ("face hidden, turned away") rather than "no face".

Word caps (hard) for the segments you write:
{_CAPS}
REWRITE mode may exceed a cap by up to {REWRITE_EXTRA_WORDS} words.

Other rules: still-frame only, no aspect ratio, no quality spam, no trailing period.
Return one revision per failing field, each with a one-sentence rationale.
Respond with JSON only, matching the provided schema.
"""
