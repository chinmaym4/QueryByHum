# earworm

Query-by-humming over a personal music library (up to ~1,000 songs). Full design is in SPEC.md — read that first in any new session. This file is standing project memory: conventions, commands, and how to approach the hard parts. It doesn't change across milestones; SPEC.md does.

## Commands

- Install: `uv sync`
- Tests: `uv run pytest`
- CLI: `uv run earworm <command>`
- ffmpeg must be on PATH — check with `ffmpeg -version` before debugging any audio decode issue

## Conventions

- Numba-jitted functions live in `matchers/*.py` next to the `Matcher` class that uses them, not in a shared utils file.
- Tunable parameters go in `config.toml`, not hardcoded. If you catch yourself writing a bare number into matching logic, stop and put it in config instead.
- `data/` is gitignored and never committed (stems, features, index.db).
- Audio is always normalized to 16kHz mono float32 before pitch extraction — see `audio.py`. Don't re-implement decoding elsewhere.

## Current milestone

M1: matching engine + eval harness, CLI only. No web app, no OAuth, no LLM, no lyrics matcher yet. SPEC.md's "Future hooks" section describes interfaces to leave room for, not things to build now.

## How to approach the matching work

This is the actual hard part of the project, and it's easy to fool yourself into thinking it works when it doesn't. Hold to this process:

1. **Form a hypothesis before changing code.** If a matcher is underperforming, write down in your response what you think is wrong and why, before editing. "Contour matcher does worse on short hums" is a hypothesis. "Let me try tweaking the weights" is not — that's poking at it.
2. **Use the synthetic test suite to localize the problem.** It exists precisely so you can isolate whether a failure is in pitch extraction, note segmentation, or the alignment itself, without the confound of messy real audio. If a synthetic test fails, find which stage is wrong before touching the eval harness or real hums.
3. **A failing test means the implementation is wrong, not the threshold.** Never loosen a synthetic test's accuracy threshold, change its tolerance, or mark it as expected-to-fail in order to make it pass. If you believe a threshold in SPEC.md is genuinely unreasonable, say so explicitly, explain why, and let the user decide — don't quietly adjust it.
4. **Same rule for the real-hum eval.** If top-5 accuracy is bad, the response is to debug (look at `earworm explain` plots, check whether the failure correlates with `part` or `in_preview` in the eval breakdown) — not to redefine success, not to cherry-pick which hums count.
5. **When a bug resists a fix, change your model of the problem, not just the parameters.** If you've adjusted a threshold or weight three times without improvement, that's a signal the bug is somewhere else — in pitch extraction, in note segmentation, in the offset bookkeeping — not in the number you keep nudging. Go look at intermediate output (plot the pitch curve, print the token string) instead of continuing to guess at the same layer.
6. **Report what you actually found.** When something doesn't work, say so plainly, along with what you ruled out and what you think the real cause is — rather than describing a partial fix as though it closes the issue.

The goal is a matcher that is actually accurate on real hums, not one that passes tests by construction. A thorough, skeptical pass that finds and fixes a real bug is worth more than three quick parameter tweaks.

## Gotchas

- torchcrepe model: `tiny` on CPU, `full` on GPU — check device in `pitch.py`, don't hardcode.
- Deezer preview API may have changed since the spec was written — verify it actually returns previews before building the rest of ingestion around it, and say clearly if it doesn't.
