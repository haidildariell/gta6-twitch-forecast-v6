#!/usr/bin/env python3
"""Forecast a game's launch-day Twitch viewers, hour by hour.

Entry for make.fun competition gta6-twitch-ccu-forecast-01.

The program reads every ``*.in`` case file from ``MAKEFUN_INPUT_DIR``
(default ``/input``), in filename order, and writes one ``*.out`` file per
case to ``MAKEFUN_OUTPUT_DIR`` (default ``/output``).  Each output holds
quantile forecasts (levels 0.05/0.25/0.5/0.75/0.95) for hours 2..24 of the
release day, expressed as multiples of the known hour-1 viewer count and
converted back to absolute viewers.

Method (all logarithms natural; curves are log-ratios to hour 1):

* Every comparable launch is summarised by its log-ratio curve, its
  hour-1 size, its pre-launch surge and peak ratios, a console flag and
  an unlock regime (STEP = one simultaneous unlock time, ROLL = rolling
  midnight-local unlock, OTHER).
* Routing picks an expert by the target's regime:
    - STEP targets use an aligned template around the unlock hour plus
      a ridge-estimated amplitude shift; an unknown unlock hour is a
      mixture over the pool's unlock hours.
    - Console targets use per-hour ridge regression, or, when the
      target is far smaller than every console analogue, a functional
      model: a per-hour median curve plus three smooth basis shapes
      whose coefficients are ridge-regressed on size and surge.
    - Mixed PC/console rolling launches with a distinct PC unlock use a
      per-hour ridge with the full feature set.
    - Everything else uses an archetype mixture: k-medoids clusters of
      standardised analogue curves, weighted by how close the target's
      size/surge/peak features sit to each cluster centroid.
* Spreads come from leave-one-out residuals inside the chosen expert,
  widened when the target is larger than every analogue.

Standard library only, no network, fully deterministic.
"""

import json
import math
import os
import re
from datetime import datetime, timedelta

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
HOURS = list(range(2, 25))  # forecast hours


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def mean(vals):
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def pstdev(vals):
    vals = list(vals)
    if not vals:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def rms(vals):
    vals = list(vals)
    if not vals:
        return 0.0
    return math.sqrt(sum(v * v for v in vals) / len(vals))


# --------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------

def pt(s):
    """Parse an ISO timestamp into an aware datetime."""
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def plat_text(platforms):
    """Platforms may arrive as a string or a list; normalise to text."""
    if platforms is None:
        return ""
    if isinstance(platforms, (list, tuple)):
        return ", ".join(str(p) for p in platforms)
    return str(platforms)


def classify_regime(unlock_text, platforms):
    """Classify an unlock description as STEP / ROLL / OTHER."""
    t = (unlock_text or "").lower()
    console_only = "PC" not in plat_text(platforms)
    has_midnight = ("midnight local" in t) or ("rolling" in t)
    single = any(s in t for s in (
        "single global", "single simultaneous", "single time",
        "simultaneous across", "one global"))
    if single and not has_midnight:
        return "STEP"
    if has_midnight:
        return "ROLL"
    if ("global unlock" in t) or ("simultaneous" in t):
        return "STEP"
    if console_only:
        return "ROLL"
    return "OTHER"


_UNLOCK_TS = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})")


def parse_unlock_hour(unlock_text, day_start):
    """Hour (1-based) of the first explicit timestamp in the unlock text."""
    if not unlock_text:
        return None
    m = _UNLOCK_TS.search(str(unlock_text))
    if not m:
        return None
    ds = day_start if isinstance(day_start, datetime) else pt(day_start)
    y, mo, d, hh, mm = (int(g) for g in m.groups())
    ts = datetime(y, mo, d, hh, mm, tzinfo=ds.tzinfo)
    return int((ts - ds).total_seconds() // 3600) + 1


def target_from_notes(notes):
    """Recover the target's platforms and unlock text from the notes."""
    text = notes or ""
    platforms = ""
    unlock = ""
    m = re.search(r"released [^;]*? on (.*?); real unlock:", text, re.DOTALL)
    if m:
        platforms = m.group(1)
    m = re.search(r"real unlock: (.*?)(?:\. day_start is|Preload live\.| Source:|$)",
                  text, re.DOTALL)
    if m:
        unlock = m.group(1)
    if not unlock:
        m = re.search(r"real unlock: (.*)", text, re.DOTALL)
        if m:
            unlock = m.group(1)[:400]
    return platforms, unlock


def series_before(points, ds):
    """(viewers 24h before ds, peak viewers in the 14 days before ds)."""
    target_pre = ds - timedelta(hours=24)
    window_start = ds - timedelta(days=14)
    pre1 = None
    latest_t = None
    latest_v = None
    peak = None
    for p in points or []:
        if not isinstance(p, dict):
            continue
        t_raw = p.get("at") or p.get("from")
        v = p.get("viewers")
        if t_raw is None or v is None:
            continue
        try:
            t = pt(t_raw)
            v = float(v)
        except (TypeError, ValueError):
            continue
        if t == target_pre:
            pre1 = v
        if t < ds and (latest_t is None or t > latest_t):
            latest_t, latest_v = t, v
        if window_start <= t < ds and (peak is None or v > peak):
            peak = v
    if pre1 is None:
        pre1 = latest_v
    return pre1, peak


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

class Rec(object):
    """One launch (analogue or target) reduced to the model's features."""
    __slots__ = ("name", "x", "lp1", "surge", "peakratio", "console",
                 "regime", "u", "unlock_text")

    def __init__(self, name, x, lp1, surge, peakratio, console, regime, u,
                 unlock_text=""):
        self.name = name
        self.x = x                  # dict hour -> ln(viewers_h / viewers_1)
        self.lp1 = lp1              # ln(viewers in hour 1)
        self.surge = surge          # ln(hour1 / viewers 24h before)
        self.peakratio = peakratio  # ln(hour1 / pre-launch peak)
        self.console = console
        self.regime = regime
        self.u = u                  # unlock hour for STEP launches
        self.unlock_text = unlock_text


def build_analog(c):
    """Build a Rec from one comparable, or None if it is unusable."""
    name = str(c.get("name") or "")
    if name.startswith("Grand Theft Auto VI category"):
        return None
    ds = pt(c["day_start"])
    hourly = {}
    for point in c.get("hourly") or []:
        try:
            h = int((pt(point["from"]) - ds).total_seconds() // 3600) + 1
            v = point.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
        if v and 1 <= h <= 24:
            hourly[h] = float(v)
    if any(h not in hourly for h in range(1, 25)):
        return None
    x = {h: math.log(hourly[h] / hourly[1]) for h in range(1, 25)}
    lp1 = math.log(hourly[1])

    # largest single-hour jump in the log curve (first maximum wins)
    jump_v = None
    jump_h = None
    for h in range(2, 25):
        d = x[h] - x[h - 1]
        if jump_v is None or d > jump_v:
            jump_v, jump_h = d, h

    unlock = c.get("unlock") or ""
    platforms = c.get("platforms") or ""
    regime = classify_regime(unlock, platforms)
    u = parse_unlock_hour(unlock, c.get("day_start"))
    if regime == "STEP" and u is None:
        u = jump_h
    if regime != "STEP" and jump_v is not None and jump_v >= 1.0 \
            and x[jump_h - 1] < 0.6:
        regime = "STEP"
        u = jump_h

    pre1, peak = series_before(c.get("hourly") or [], ds)
    surge = math.log(hourly[1] / pre1) if pre1 else 0.0
    peakratio = math.log(hourly[1] / peak) if peak else 0.0
    console = "PC" not in plat_text(platforms)
    return Rec(name, x, lp1, surge, peakratio, console, regime, u, unlock)


def build_target(case):
    """Build the target Rec (plus hour-1 viewers) from a case."""
    plat, unlock = target_from_notes(case.get("notes"))
    if not plat:
        game = case.get("game") or {}
        if isinstance(game, dict):
            plat = game.get("name") or ""
        else:
            plat = str(game)
    regime = classify_regime(unlock, plat)
    u = parse_unlock_hour(unlock, case.get("day_start")) if regime == "STEP" else None
    h1 = float(case["known"][0]["viewers"])
    hh = case.get("history_hourly")
    points = hh if hh else (case.get("history") or [])
    pre1, peak = series_before(points, pt(case["day_start"]))
    lp1 = math.log(max(1.0, h1))
    surge = math.log(h1 / pre1) if pre1 else 0.0
    peakratio = math.log(h1 / peak) if peak else 0.0
    console = "PC" not in plat_text(plat)
    rec = Rec(str(plat), {1: 0.0}, lp1, surge, peakratio, console, regime, u,
              unlock)
    return rec, h1


# --------------------------------------------------------------------------
# weighted ridge regression (Gaussian elimination, partial pivoting)
# --------------------------------------------------------------------------

def solve_linear(A, b):
    """Solve A beta = b; tiny pivots leave the coefficient at zero."""
    n = len(b)
    M = [list(A[i]) for i in range(n)]
    y = list(b)
    for i in range(n):
        piv_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[piv_row][i]) < 1e-12:
            continue
        if piv_row != i:
            M[i], M[piv_row] = M[piv_row], M[i]
            y[i], y[piv_row] = y[piv_row], y[i]
        piv = M[i][i]
        for r in range(i + 1, n):
            factor = M[r][i] / piv if abs(piv) > 1e-12 else 0.0
            if factor == 0.0:
                continue
            for cc in range(i, n):
                M[r][cc] -= factor * M[i][cc]
            y[r] -= factor * y[i]
    beta = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) <= 1e-12:
            beta[i] = 0.0
            continue
        s = y[i] - sum(M[i][j] * beta[j] for j in range(i + 1, n))
        beta[i] = s / M[i][i]
    return beta


def fit_ridge(X, y, weights, lam):
    """Ridge with an unpenalised intercept: (X'WX + lam D) beta = X'Wy."""
    if not X:
        return []
    p = len(X[0])
    if weights is None:
        weights = [1.0] * len(X)
    A = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for row, yi, w in zip(X, y, weights):
        if w == 0.0:
            continue
        for i in range(p):
            b[i] += w * row[i] * yi
            ri = row[i]
            for j in range(p):
                A[i][j] += w * ri * row[j]
    for i in range(1, p):
        A[i][i] += lam
    return solve_linear(A, b)


def dot(beta, feat):
    return sum(b * f for b, f in zip(beta, feat))


# --------------------------------------------------------------------------
# feature functions (centre = mean lp1 of the pool in use)
# --------------------------------------------------------------------------

def f_small(rec, center):
    return [1.0, rec.lp1 - center, rec.surge]


def f_a(rec, center):
    return [1.0, rec.lp1 - center, rec.surge, rec.peakratio,
            1.0 if rec.console else 0.0]


FEATS = {"small": f_small, "a": f_a}


def size_scale(target, pool):
    """Widen spreads when the target outgrows every analogue."""
    if not pool:
        return 1.0
    return 1.0 + 0.30 * max(0.0, target.lp1 - max(a.lp1 for a in pool))


# --------------------------------------------------------------------------
# BRANCH R: per-hour ridge over a pool
# --------------------------------------------------------------------------

def branch_r(pool, target, feat_name, lam, weights=None):
    if not pool:
        raise ValueError("empty pool")
    feat = FEATS[feat_name]
    center = mean(a.lp1 for a in pool)
    tf = feat(target, center)
    mu = {}
    resid = {h: [] for h in HOURS}
    for h in HOURS:
        beta = fit_ridge([feat(a, center) for a in pool],
                         [a.x[h] for a in pool], weights, lam)
        mu[h] = dot(beta, tf)
    # leave-one-out residuals for the spread
    for idx, ai in enumerate(pool):
        sub = [a for j, a in enumerate(pool) if j != idx]
        if not sub:
            continue
        sub_w = None
        if weights is not None:
            sub_w = [w for j, w in enumerate(weights) if j != idx]
        c_sub = mean(a.lp1 for a in sub)
        fi = feat(ai, c_sub)
        for h in HOURS:
            beta = fit_ridge([feat(a, c_sub) for a in sub],
                             [a.x[h] for a in sub], sub_w, lam)
            resid[h].append(ai.x[h] - dot(beta, fi))
    scale = size_scale(target, pool)
    sd = {}
    for h in HOURS:
        base = rms(resid[h]) if resid[h] else 0.6
        sd[h] = clamp(base, 0.22, 1.30) * scale
    return mu, sd


# --------------------------------------------------------------------------
# BRANCH S: STEP template aligned on the unlock hour + amplitude shift
# --------------------------------------------------------------------------

def step_template(pool):
    T = {}
    for k in range(-14, 15):
        vals = [a.x[a.u + k] for a in pool
                if a.u is not None and 1 <= a.u + k <= 24]
        if len(vals) >= 2:
            T[k] = median(vals)
    return T


def step_shift(pool, target):
    """Ridge-estimated post-unlock level shift for the target."""
    rows = []
    levels = []
    for a in pool:
        if a.u is None:
            continue
        vals = [a.x[a.u + k] for k in range(0, 9) if 1 <= a.u + k <= 24]
        if len(vals) >= 4:
            rows.append(a)
            levels.append(mean(vals))
    if len(rows) < 3:
        return 0.0
    center = mean(a.lp1 for a in rows)
    beta = fit_ridge([f_small(a, center) for a in rows], levels, None, 1.0)
    return dot(beta, f_small(target, center)) - median(levels)


def step_predict(T, shift, uT):
    if not T:
        raise ValueError("empty template")
    keys = sorted(T)
    mu = {}
    for h in HOURS:
        k = h - uT
        if k in T:
            base = T[k]
        else:
            nearest = min(keys, key=lambda kk: (abs(kk - k), kk))
            base = T[nearest]
        taper = 1.0 if k <= 8 else max(0.7, 1.0 - (k - 8) * 0.075)
        mu[h] = base + (shift * taper if k >= 0 else 0.0)
    return mu


def step_known_u(pool, target, uT):
    T = step_template(pool)
    shift = step_shift(pool, target)
    mu = step_predict(T, shift, uT)
    resid = {h: [] for h in HOURS}
    for idx, a in enumerate(pool):
        if a.u is None:
            continue
        sub = [x for j, x in enumerate(pool) if j != idx]
        T_loo = step_template(sub)
        if not T_loo:
            continue
        shift_loo = step_shift(sub, a)
        pred = step_predict(T_loo, shift_loo, a.u)
        for h in HOURS:
            if (h - a.u) in T_loo:
                resid[h].append(a.x[h] - pred[h])
    scale = size_scale(target, pool)
    sd = {}
    for h in HOURS:
        base = rms(resid[h]) if resid[h] else 0.7
        sd[h] = clamp(base, 0.22, 1.30) * scale
        if abs(h - uT) <= 1:
            sd[h] = min(1.30, sd[h] * 1.30)
    return mu, sd


def branch_s(pool, target, uT):
    if not pool:
        raise ValueError("empty step pool")
    if uT is not None:
        return step_known_u(pool, target, uT)
    # mixture over the pool's unlock hours
    counts = {}
    for a in pool:
        if a.u is not None:
            counts[a.u] = counts.get(a.u, 0) + 1
    if not counts:
        raise ValueError("no unlock hours in step pool")
    total = sum(c + 0.5 for c in counts.values())
    mu = {h: 0.0 for h in HOURS}
    var = {h: 0.0 for h in HOURS}
    for u in sorted(counts):
        w = (counts[u] + 0.5) / total
        mu_u, sd_u = step_known_u(pool, target, u)
        for h in HOURS:
            mu[h] += w * mu_u[h]
            var[h] += w * (sd_u[h] ** 2 + mu_u[h] ** 2)
    sd = {h: math.sqrt(max(0.0, var[h] - mu[h] ** 2)) for h in HOURS}
    return mu, sd


# --------------------------------------------------------------------------
# BRANCH C: console targets
# --------------------------------------------------------------------------

def _basis():
    b1 = {h: (h - 13) / 11.0 for h in HOURS}
    b2 = {h: math.exp(-((h - 10) ** 2) / 50.0) for h in HOURS}
    b3 = {h: math.exp(-((h - 20) ** 2) / 50.0) for h in HOURS}
    return [b1, b2, b3]


def _functional_predict(pool, target, basis):
    med = {h: median([a.x[h] for a in pool]) for h in HOURS}
    center = mean(a.lp1 for a in pool)
    pred_k = []
    for b in basis:
        bnorm = sum(b[h] ** 2 for h in HOURS)
        coeffs = [sum((a.x[h] - med[h]) * b[h] for h in HOURS) / bnorm
                  for a in pool]
        beta = fit_ridge([[1.0, a.lp1 - center, a.surge] for a in pool],
                         coeffs, None, 0.25)
        pred_k.append(dot(beta, [1.0, target.lp1 - center, target.surge]))
    mu = {}
    for h in HOURS:
        mu[h] = med[h] + sum(pred_k[k] * basis[k][h] for k in range(3))
    return mu


def branch_c(cons, multi, target):
    if len(cons) >= 5 and target.lp1 < min(a.lp1 for a in cons) - 1.0:
        # FUNCTIONAL sub-branch: target far below every console analogue
        basis = _basis()
        mu = _functional_predict(cons, target, basis)
        resid = {h: [] for h in HOURS}
        for idx, ai in enumerate(cons):
            sub = [a for j, a in enumerate(cons) if j != idx]
            if not sub:
                continue
            mu_i = _functional_predict(sub, ai, basis)
            for h in HOURS:
                resid[h].append(ai.x[h] - mu_i[h])
        scale = size_scale(target, cons)
        sd = {}
        for h in HOURS:
            base = rms(resid[h]) if resid[h] else 0.5
            sd[h] = clamp(base, 0.22, 1.30) * scale * 0.9
        return mu, sd
    # RIDGE sub-branch
    if len(cons) >= 5:
        pool = list(cons)
        weights = None
    else:
        pool = list(cons) + list(multi)
        weights = [1.0] * len(cons) + [0.25] * len(multi)
    if not pool:
        raise ValueError("empty console pool")
    mu, sd = branch_r(pool, target, "small", 0.25, weights)
    if cons:
        med = {h: median([a.x[h] for a in cons]) for h in HOURS}
        for h in HOURS:
            mu[h] = 0.8 * mu[h] + 0.2 * med[h]
    for h in HOURS:
        sd[h] *= 0.9
    return mu, sd


# --------------------------------------------------------------------------
# BRANCH A: archetype mixture over the rolling pool
# --------------------------------------------------------------------------

def _peak_hour(a):
    best_h = 2
    best_v = None
    for h in HOURS:
        if best_v is None or a.x[h] > best_v:
            best_v, best_h = a.x[h], h
    return best_h


def _cluster(pool):
    """K-medoids (K=5) on standardised curves; returns list of groups."""
    n = len(pool)
    if n <= 5:
        return [[a] for a in pool]
    mean_h = {h: mean(a.x[h] for a in pool) for h in HOURS}
    sd_h = {}
    for h in HOURS:
        s = pstdev([a.x[h] for a in pool])
        sd_h[h] = s if s != 0.0 else 1.0
    vecs = {id(a): [(a.x[h] - mean_h[h]) / sd_h[h] for h in HOURS]
            for a in pool}

    def dist(a, b):
        va, vb = vecs[id(a)], vecs[id(b)]
        return math.sqrt(sum((p - q) ** 2 for p, q in zip(va, vb)))

    order = sorted(pool, key=lambda a: (_peak_hour(a), a.name))
    medoids = [order[int(i * n / 5)] for i in range(5)]
    for _ in range(30):
        def nearest(a):
            return min(medoids, key=lambda m: (dist(a, m), m.name))
        groups = {id(m): [] for m in medoids}
        for a in pool:
            groups[id(nearest(a))].append(a)
        new_medoids = []
        for m in medoids:
            group = groups[id(m)]
            if not group:
                new_medoids.append(m)
                continue
            best = min(group,
                       key=lambda cand: (sum(dist(cand, o) for o in group),
                                         cand.name))
            new_medoids.append(best)
        if [m.name for m in new_medoids] == [m.name for m in medoids]:
            medoids = new_medoids
            break
        medoids = new_medoids
    final = {id(m): [] for m in medoids}
    for a in pool:
        m = min(medoids, key=lambda mm: (dist(a, mm), mm.name))
        final[id(m)].append(a)
    return [g for g in final.values() if g]


def _feat_stats(pool):
    stats = []
    for attr in ("lp1", "surge", "peakratio"):
        vals = [getattr(a, attr) for a in pool]
        s = pstdev(vals)
        stats.append((mean(vals), s if s != 0.0 else 1.0))
    return stats


def _fvec(rec, stats):
    out = []
    for i, attr in enumerate(("lp1", "surge", "peakratio")):
        m, s = stats[i]
        out.append((getattr(rec, attr) - m) / s)
    out.append(1.0 if rec.console else 0.0)
    return out


def _mixture_mu(pool, target, stats):
    groups = _cluster(pool)
    tf = _fvec(target, stats)
    templates = []
    disps = []
    weights = []
    for g in groups:
        T = {h: median([a.x[h] for a in g]) for h in HOURS}
        D = {h: 1.4826 * median([abs(a.x[h] - T[h]) for a in g])
             for h in HOURS}
        cen = [mean([_fvec(a, stats)[i] for a in g]) for i in range(4)]
        d2 = sum((tf[i] - cen[i]) ** 2 for i in range(3)) \
            + 2.0 * (tf[3] - cen[3]) ** 2
        w = math.exp(-d2 / (2.0 * 1.15 ** 2))
        if not any(a.console == target.console for a in g):
            w *= 0.15
        templates.append(T)
        disps.append(D)
        weights.append(w)
    total = sum(weights)
    if total <= 0.0:
        weights = [1.0 / len(groups)] * len(groups)
    else:
        weights = [w / total for w in weights]
    mu = {}
    var = {}
    for h in HOURS:
        m = sum(w * T[h] for w, T in zip(weights, templates))
        mu[h] = m
        var[h] = sum(w * (D[h] ** 2 + (T[h] - m) ** 2)
                     for w, D, T in zip(weights, disps, templates))
    return mu, var


def branch_a(pool, target):
    if not pool:
        raise ValueError("empty rolling pool")
    stats = _feat_stats(pool)
    mu, var = _mixture_mu(pool, target, stats)
    resid = {h: [] for h in HOURS}
    for idx, ai in enumerate(pool):
        sub = [a for j, a in enumerate(pool) if j != idx]
        if not sub:
            continue
        mu_i, _ = _mixture_mu(sub, ai, stats)
        for h in HOURS:
            resid[h].append(ai.x[h] - mu_i[h])
    scale = size_scale(target, pool)
    sd = {}
    for h in HOURS:
        loo = rms(resid[h]) if resid[h] else 0.0
        base = max(math.sqrt(max(0.0, var[h])), 0.8 * loo)
        sd[h] = clamp(base, 0.22, 1.30) * scale
    return mu, sd


# --------------------------------------------------------------------------
# routing + output assembly
# --------------------------------------------------------------------------

def route(analogs, target):
    step = [a for a in analogs if a.regime == "STEP"]
    roll = [a for a in analogs if a.regime != "STEP"]
    cons = [a for a in roll if a.console]
    multi = [a for a in roll if not a.console]
    if target.regime == "STEP" and step:
        return branch_s(step, target, target.u)
    if target.regime == "OTHER" and not target.console \
            and target.lp1 < 9.0 and step:
        return branch_s(step, target, None)
    if target.console:
        return branch_c(cons, multi, target)
    t = (target.unlock_text or "").lower()
    mixed = ("steam" in t and "unlock" in t) or ("pc single global" in t) \
        or ("pc unlock later" in t) or ("pc" in t and "global unlock" in t)
    if target.regime == "ROLL" and not target.console and mixed:
        return branch_r(roll, target, "a", 1.0)
    return branch_a(roll, target)


def smooth(mu):
    out = {}
    for h in HOURS:
        num = 0.5 * mu[h]
        den = 0.5
        if h - 1 in mu:
            num += 0.25 * mu[h - 1]
            den += 0.25
        if h + 1 in mu:
            num += 0.25 * mu[h + 1]
            den += 0.25
        out[h] = num / den
    return out


def assemble(mu, sd, h1):
    # Sanitise before exponentiating: the output contract requires finite
    # quantiles, and a single non-finite mu/sd from an edge-case branch
    # must not poison the whole case (json would emit invalid Infinity).
    mu = {h: (mu[h] if math.isfinite(mu[h]) else 0.0) for h in HOURS}
    sd = {h: (sd[h] if math.isfinite(sd[h]) and sd[h] >= 0.0 else 0.6)
          for h in HOURS}
    mu2 = smooth(mu)
    p1 = max(1.0, h1)
    rows = []
    for h in HOURS:
        qs = []
        for z in Z:
            arg = mu2[h] + z * sd[h]
            val = p1 * math.exp(min(arg, 50.0))
            qs.append(min(1e9, max(1e-6, val)))
        qs.sort()
        rows.append({"hour": h, "q": qs})
    return rows


def baseline_rows(p1):
    """House baseline used when a case cannot be forecast at all."""
    p1 = max(1.0, p1)
    rows = []
    for h in HOURS:
        qs = [min(1e9, max(1e-6, p1 * math.exp(z * 0.15 * math.sqrt(h - 1))))
              for z in Z]
        qs.sort()
        rows.append({"hour": h, "q": qs})
    return rows


def forecast_case(case):
    analogs = []
    for c in case.get("comparables") or []:
        try:
            rec = build_analog(c)
        except Exception:
            rec = None
        if rec is not None:
            analogs.append(rec)
    target, h1 = build_target(case)
    try:
        mu, sd = route(analogs, target)
    except Exception:
        mu = {h: 0.0 for h in HOURS}
        sd = {h: 0.6 for h in HOURS}
    return assemble(mu, sd, h1)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(in_dir):
        return
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        stem = name[:-3]
        case = None
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            rows = forecast_case(case)
        except Exception:
            p1 = 1.0
            try:
                p1 = float(case["known"][0]["viewers"])
            except Exception:
                pass
            rows = baseline_rows(p1)
        payload = json.dumps({"hours": rows}, separators=(",", ":"))
        with open(os.path.join(out_dir, stem + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(payload)


if __name__ == "__main__":
    main()
