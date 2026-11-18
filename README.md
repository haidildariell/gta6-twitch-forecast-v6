# GTA VI Twitch CCU Forecast — gta6-twitch-forecast-v6

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/haidildariell/gta6-twitch-forecast-v6/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/haidildariell/gta6-twitch-forecast-v6/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Competition: https://make.fun/bounty/gta6-twitch-ccu-forecast-01

## What this entry does

`main.py` forecasts a game's launch-day Twitch concurrent viewers hour by
hour. Given a case file describing the target game (its known hour-1 viewer
count, its pre-launch history, and a library of comparable past launches
with full hourly curves), it emits quantile forecasts — the 5%, 25%, 50%,
75% and 95% levels — for hours 2 through 24 of release day.

Everything is modelled in log space: each launch becomes a curve of
log-ratios to its own hour 1, plus a handful of summary features (hour-1
size, the pre-launch surge into release, the ratio to the pre-launch peak,
a console-only flag, and an unlock regime parsed from the release notes).

## Method: routed experts by unlock regime

A single global model blurs together launches that behave very
differently, so the forecaster routes each target to a specialist:

- **Single-time (STEP) unlocks** — an aligned template: comparable STEP
  curves are shifted so their unlock hours line up, a per-offset median
  template is read off at the target's unlock hour, and a ridge-estimated
  amplitude shift adjusts the post-unlock level for the target's size and
  surge. If the target's unlock hour is unknown, the forecast is a
  mixture over the unlock hours seen in the pool.
- **Console launches** — per-hour ridge regressors over the console pool.
  When the target is far smaller than every console analogue, the entry
  switches to a median-anchored functional fallback: a per-hour median
  curve plus three smooth basis shapes (a day trend and two humps) whose
  coefficients are regressed on size and surge, so an out-of-range target
  still gets a sensibly shaped day.
- **Mixed PC/console rolling launches** with a distinct PC unlock — a
  per-hour ridge using the full feature set, including the pre-launch
  peak ratio and the console flag.
- **All other launches** — an archetype mixture: the rolling analogues
  are clustered with k-medoids over standardised curves, each cluster
  contributes its median template, and clusters are weighted by how close
  the target's features sit to the cluster centroid.

In every branch the predictive spread is calibrated with leave-one-out
residuals computed inside that same expert, and spreads widen when the
target is larger than any analogue it was trained on. If anything about a
case cannot be modelled, the program falls back to the house baseline
rather than emitting nothing.

The program is pure Python standard library, uses no network, and is
fully deterministic: the same input always produces byte-identical
output.

## How to run

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

- Input: every `*.in` file in `MAKEFUN_INPUT_DIR` (default `/input`), in
  filename order, each a JSON case of schema
  `makefun/twitch-ccu-case/v1`.
- Output: one `<stem>.out` file per case in `MAKEFUN_OUTPUT_DIR`
  (default `/output`, created if missing), containing
  `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for
  hours 2..24.
