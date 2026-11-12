#!/usr/bin/env python3
"""Pump.fun top-5 market-cap forecaster (make.fun pumpfun-top5-forecast-01).

Reads the house's 10-minute snapshots from /input (or $MAKEFUN_INPUT_DIR) and
writes one forecast, /output/forecast.json (or $MAKEFUN_OUTPUT_DIR): for each
of the 5 places, a probability for each coin named being at that place, the
top 5 in order, and for each place a probability for each market-cap bin.

Model: every coin's market cap (log10, in SOL) is projected forward to each
answer read with a Monte Carlo over log-normal paths. Each coin gets an
age-aware drift (recent momentum, decaying with horizon toward a negative
base drift: pump.fun coins pump fast and fade), a per-coin volatility
shrunk toward a global prior, and curve coins near the graduation threshold
get a graduation jump (observed jumps are mixed, mean about -0.3 log10).
Place probabilities and size-bin probabilities are the Monte Carlo
frequencies averaged over the seven reads. Sizes mix the simulated
order-statistic distribution with a "fresh leader" component, because the
top of the chart is a renewing population: coins born after the cut, which
no forecast can name, regularly take top-5 places, so the cap level of a
place is steadier than any single coin's path.

Probability allocation is greedy per place: a coin is named only while its
simulated probability beats the unnamed share it would otherwise take, and
a reserve (larger for longer horizons) is always left unnamed so that
newcomers and unmodeled movers still score better than the floor.

Deterministic: one fixed seed, sorted iteration everywhere it can affect
the output, probabilities floored to 6 decimals. Standard library only,
no network, no model calls.
"""

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------- constants

SEED = 20261112
N_SIMS = 4000            # Monte Carlo paths per coin
MAX_CANDIDATES = 220     # coins simulated (largest by last observed cap)
MAX_NAMED_PER_PLACE = 90  # schema allows 100 rows per place; keep headroom
TAU_MOMENTUM_H = 1.5     # momentum half-life-ish scale, hours
BASE_DRIFT_POOL = -0.10  # log10 cap per hour, graduated coins
BASE_DRIFT_CURVE = -0.16  # log10 cap per hour, coins still on the curve
VOL_FLOOR = 0.15         # log10 per sqrt-hour
VOL_CAP = 0.60
VOL_PRIOR_POOL = 0.32
VOL_PRIOR_CURVE = 0.26
LOG_CAP_FLOOR = 0.0      # 1 SOL; below this a coin cannot matter
GRAD_CAP_MIN = 2.15      # log10 cap where graduation becomes plausible
GRAD_JUMP_MEAN = -0.30
GRAD_JUMP_SD = 0.50
SIZE_FRESH_MIX = 0.40    # weight of the fresh-leader component in sizes
SIZE_FRESH_SD = 0.45
SIZE_FRESH_DECAY = 0.03  # log10 per hour the fresh-leader centre drifts down
SIZE_NAME_CAP = 0.97     # leave 0.03 of each place's size mass unnamed
MIN_NAMED_P = 0.0005


def parse_ts(s):
    """ISO 8601 UTC -> epoch seconds."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def log_cap_sol(lamports):
    """log10 of a market cap in SOL, as the scorer takes it."""
    v = int(lamports)
    if v <= 0:
        return LOG_CAP_FLOOR
    return math.log10(v) - 9.0


def floor6(x):
    """Floor to 6 decimals (keeps place sums <= 1)."""
    return math.floor(x * 1_000_000) / 1_000_000


# ---------------------------------------------------------------- input

def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    marks = []
    with open(os.path.join(input_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    return task, marks


def build_series(marks):
    """Per-mint series of (mark_time, log_cap, venue, complete).

    Returns (series, last_mark_time, last_rows) where series maps mint ->
    dict with 'points' (list, oldest first), 'created_at' (epoch or None),
    'last_rank' (rank in the last mark it appears in).
    """
    series = {}
    last_mark_time = None
    last_rows = []
    for m in marks:
        t = parse_ts(m["at"])
        last_mark_time = t
        last_rows = m["rows"]
        for r in m["rows"]:
            mint = r["mint"]
            entry = series.setdefault(
                mint,
                {"points": [], "created_at": None, "last_rank": None},
            )
            entry["points"].append(
                (t, log_cap_sol(r["market_cap_lamports"]), r.get("venue", "curve"),
                 bool(r.get("complete", False)))
            )
            entry["last_rank"] = r["rank"]
            ca = r.get("created_at")
            if ca:
                try:
                    entry["created_at"] = parse_ts(ca)
                except ValueError:
                    pass
    return series, last_mark_time, last_rows


# ---------------------------------------------------------------- parameters

def estimate_globals(series):
    """Global 10-minute log-return statistics, by venue.

    Used as the shrinkage prior for per-coin volatility. Returns a dict
    venue -> (mean, sd) in log10 per 10 minutes, plus the overall sd.
    """
    rets = {"pool": [], "curve": [], "migrating": []}
    for entry in series.values():
        pts = entry["points"]
        for (t1, c1, v1, _), (t2, c2, _, _) in zip(pts, pts[1:]):
            dt_min = (t2 - t1) / 60.0
            if 5.0 <= dt_min <= 20.0:  # consecutive marks on the 10-min grid
                rets.setdefault(v1, []).append((c2 - c1) * (10.0 / dt_min))
    out = {}
    all_rets = []
    for venue, xs in rets.items():
        if len(xs) >= 8:
            mean = sum(xs) / len(xs)
            var = sum((x - mean) ** 2 for x in xs) / len(xs)
            out[venue] = (mean, math.sqrt(var))
            all_rets.extend(xs)
    if all_rets:
        mean = sum(all_rets) / len(all_rets)
        var = sum((x - mean) ** 2 for x in all_rets) / len(all_rets)
        out["__all__"] = (mean, math.sqrt(var))
    return out


def coin_params(entry, last_time, globals_):
    """(y0, momentum, vol_per_sqrt_hour, venue, is_curve) for one coin.

    y0 is the last observed log-cap, decayed for the gap since its last
    sighting. momentum is the recent slope in log10/hour, shrunk toward 0.
    vol is the per-coin 10-minute sd scaled to per sqrt-hour, shrunk
    toward the venue prior.
    """
    pts = entry["points"]
    t_last, y_last, venue_last, _ = pts[-1]
    is_curve = venue_last in ("curve", "migrating")
    # Decay the last observation for staleness (a coin unseen for hours
    # has usually faded; its last rank already reflects that).
    gap_h = max(0.0, (last_time - t_last) / 3600.0)
    base = BASE_DRIFT_CURVE if is_curve else BASE_DRIFT_POOL
    y0 = y_last + base * gap_h

    # Recent slope over the last up to 6 points, time-weighted.
    recent = pts[-6:]
    momentum = 0.0
    n_ret = 0
    if len(recent) >= 2:
        t0, y0r = recent[0][0], recent[0][1]
        t1, y1r = recent[-1][0], recent[-1][1]
        span_h = (t1 - t0) / 3600.0
        if span_h > 0:
            raw = (y1r - y0r) / span_h
            n_ret = len(recent) - 1
            momentum = raw * n_ret / (n_ret + 2.0)  # shrink toward 0
            momentum = max(-0.6, min(0.6, momentum))
    # Per-coin volatility from its own 10-minute returns.
    rets = []
    for (t1, c1, _, _), (t2, c2, _, _) in zip(pts, pts[1:]):
        dt_min = (t2 - t1) / 60.0
        if 5.0 <= dt_min <= 20.0:
            rets.append((c2 - c1) * (10.0 / dt_min))
    prior = VOL_PRIOR_CURVE if is_curve else VOL_PRIOR_POOL
    # Prefer the venue's measured 10-minute sd from this history when the
    # history is long enough to estimate it (the final's 66 marks are).
    venue_key = "curve" if is_curve else "pool"
    if venue_key in globals_:
        prior = globals_[venue_key][1] * math.sqrt(6.0)
        prior = max(VOL_FLOOR, min(VOL_CAP, prior))
    if len(rets) >= 3:
        mean = sum(rets) / len(rets)
        var = sum((x - mean) ** 2 for x in rets) / len(rets)
        own = math.sqrt(var) * math.sqrt(6.0)  # 10-min sd -> per sqrt-hour
        vol = (own * len(rets) + prior * 4.0) / (len(rets) + 4.0)
    else:
        vol = prior
    vol = max(VOL_FLOOR, min(VOL_CAP, vol))
    return y0, momentum, vol, venue_last, is_curve


# ---------------------------------------------------------------- simulation

def simulate(cands, horizons_h, rng):
    """Monte Carlo over log-cap paths.

    cands: list of dicts with keys mint, y0, momentum, vol, is_curve.
    horizons_h: ascending hours from the cut to each answer read.
    Returns (place_counts, size_values): place_counts[k][mint] counts the
    simulations where mint holds place k (k 0-based), aggregated over reads;
    size_values[k] is the list of simulated log-caps at place k.
    """
    n_reads = len(horizons_h)
    n_places = 5
    place_counts = [dict() for _ in range(n_places)]
    size_values = [[] for _ in range(n_places)]
    n = len(cands)
    # Pre-extract arrays for speed.
    y0s = [c["y0"] for c in cands]
    moms = [c["momentum"] for c in cands]
    vols = [c["vol"] for c in cands]
    bases = [BASE_DRIFT_CURVE if c["is_curve"] else BASE_DRIFT_POOL for c in cands]
    curves = [c["is_curve"] for c in cands]
    mints = [c["mint"] for c in cands]
    # Read-to-read hour gaps (first gap is from the cut to read 1).
    gaps = []
    prev = 0.0
    for h in horizons_h:
        gaps.append(max(0.0, h - prev))
        prev = h
    tau = TAU_MOMENTUM_H
    for _ in range(N_SIMS):
        ys = list(y0s)
        graduated = [not c for c in curves]  # pool coins already graduated
        elapsed = 0.0
        for ri, gap in enumerate(gaps):
            if gap > 0:
                elapsed += gap
                for i in range(n):
                    # Drift over this interval: momentum decays toward base.
                    # drift(t) = base + (mom - base) * exp(-t/tau); the
                    # integral over [elapsed-gap, elapsed] is analytic.
                    b = bases[i]
                    m = moms[i]
                    t_end = elapsed
                    t_start = elapsed - gap
                    integ = b * gap + (m - b) * tau * (
                        math.exp(-t_start / tau) - math.exp(-t_end / tau)
                    )
                    ys[i] += integ + vols[i] * math.sqrt(gap) * rng.gauss(0.0, 1.0)
                    # Graduation: a curve coin near the threshold may
                    # graduate during this interval; its cap jumps.
                    if not graduated[i] and ys[i] > GRAD_CAP_MIN:
                        p_grad = min(0.45, (ys[i] - GRAD_CAP_MIN) * 1.2 * gap)
                        if rng.random() < p_grad:
                            graduated[i] = True
                            ys[i] += rng.gauss(GRAD_JUMP_MEAN, GRAD_JUMP_SD)
                            bases[i] = BASE_DRIFT_POOL
                    if ys[i] < LOG_CAP_FLOOR:
                        ys[i] = LOG_CAP_FLOOR
            # Rank this read: largest cap first, ties to mint bytes ascending.
            order = sorted(range(n), key=lambda i: (-ys[i], mints[i]))
            for k in range(min(n_places, n)):
                mint = mints[order[k]]
                d = place_counts[k]
                d[mint] = d.get(mint, 0) + 1
                size_values[k].append(ys[order[k]])
    return place_counts, size_values


# ---------------------------------------------------------------- allocation

def allocate_place(counts, n_sims, n_reads, chart_size, unnamed_reserve):
    """Greedy probability allocation for one place.

    counts: mint -> simulations where it held this place (over all reads).
    Names coins in descending simulated probability while naming beats the
    unnamed share the coin would otherwise take, keeping unnamed_reserve
    of the place unnamed for newcomers and unmodeled movers.
    Returns a list of (mint, p) with p floored to 6 decimals.
    """
    total = n_sims * n_reads
    # The simulation only ranks history coins, so its probabilities are
    # conditional on a history coin holding this place. Rescale by the
    # share a history coin holds (1 - the newborn reserve) so the named
    # probabilities are unconditional; the reserve stays unnamed.
    scale = 1.0 - unnamed_reserve
    ranked = sorted(
        ((cnt / total * scale, mint) for mint, cnt in counts.items()),
        key=lambda x: (-x[0], x[1]),
    )
    named = []
    named_sum = 0.0
    max_named = 1.0 - unnamed_reserve
    for p, mint in ranked:
        if len(named) >= MAX_NAMED_PER_PLACE:
            break
        if named_sum + p > max_named:
            break
        # Unnamed share per unnamed chart coin if we stop naming here.
        denom = max(1, chart_size - len(named))
        unnamed_share = (1.0 - named_sum) / denom
        if p <= unnamed_share:
            break
        if p < MIN_NAMED_P:
            break
        named.append((mint, p))
        named_sum += p
    out = []
    for mint, p in named:
        pf = floor6(p)
        if pf > 0:
            out.append((mint, pf))
    return out


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def allocate_sizes(values, edges, fresh_centre, fresh_sd):
    """Bin probabilities for one place's market cap.

    values: simulated log-caps at this place (all reads). Mixed 1-mix with
    a broad normal around fresh_centre (the fresh-leader component), then
    binned on the task's edges. Bins totalling SIZE_NAME_CAP are named;
    the rest is left unnamed for the scorer to spread.
    """
    n = len(values)
    nbins = len(edges) + 1
    if n == 0:
        return []
    counts = [0] * nbins
    for v in values:
        b = 0
        while b < len(edges) and v >= edges[b]:
            b += 1
        counts[b] += 1
    probs = []
    for b in range(nbins):
        sim_p = counts[b] / n
        lo = -math.inf if b == 0 else edges[b - 1]
        hi = math.inf if b == len(edges) else edges[b]
        fresh_p = normal_cdf((hi - fresh_centre) / fresh_sd) - normal_cdf(
            (lo - fresh_centre) / fresh_sd
        )
        probs.append((1.0 - SIZE_FRESH_MIX) * sim_p + SIZE_FRESH_MIX * fresh_p)
    # Name the largest bins up to the cap.
    order = sorted(range(nbins), key=lambda b: (-probs[b], b))
    named = []
    total = 0.0
    for b in order:
        if probs[b] <= 0:
            continue
        if total + probs[b] > SIZE_NAME_CAP and named:
            continue
        pf = floor6(probs[b])
        if pf > 0:
            named.append((b, pf))
            total += pf
    named.sort()
    return named


# ---------------------------------------------------------------- main

def fallback_forecast(task):
    """Even spread over nothing known: no named coins, sizes unnamed.

    Only used when the history holds no usable mark at all; the scorer
    spreads every place evenly over the chart and every size over the bins.
    top5 still needs 5 distinct mint-shaped strings.
    """
    top5 = [
        "11111111111111111111111111111111",
        "11111111111111111111111111111112",
        "11111111111111111111111111111113",
        "11111111111111111111111111111114",
        "11111111111111111111111111111115",
    ]
    return {"probabilities": [], "top5": top5[: task.get("places", 5)], "sizes": []}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, marks = load_input(input_dir)
    places = int(task.get("places", 5))
    chart_size = int(task.get("chart_size", 100))
    edges = [float(e) for e in task.get("size", {}).get("edges", [])]
    if not edges:
        edges = [2.0 + 0.05 * i for i in range(81)]
    output_name = task.get("output", "forecast.json")
    reads = task.get("reads", [])
    read_times = [parse_ts(r) for r in reads if r]

    series, last_time, last_rows = build_series(marks)
    if not series or last_time is None or not read_times:
        forecast = fallback_forecast(task)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, output_name), "w", encoding="utf-8") as f:
            f.write(json.dumps(forecast) + "\n")
        return 0

    horizons_h = sorted((t - last_time) / 3600.0 for t in read_times)
    horizons_h = [max(0.05, h) for h in horizons_h]
    median_h = horizons_h[len(horizons_h) // 2]

    globals_ = estimate_globals(series)

    # Candidate coins: every coin in the last mark, plus any coin that was
    # top-30 in any mark within the last hour (a recent leader that slipped
    # out of the last mark can come back). Cap by last observed cap.
    last_mints = {r["mint"] for r in last_rows}
    extra = set()
    hour_ago = last_time - 3600.0
    for m in marks:
        if parse_ts(m["at"]) < hour_ago:
            continue
        for r in m["rows"][:30]:
            extra.add(r["mint"])
    cand_mints = sorted(last_mints | extra)
    cands = []
    for mint in cand_mints:
        entry = series[mint]
        y0, momentum, vol, venue, is_curve = coin_params(entry, last_time, globals_)
        cands.append(
            {
                "mint": mint,
                "y0": y0,
                "momentum": momentum,
                "vol": vol,
                "is_curve": is_curve,
            }
        )
    cands.sort(key=lambda c: (-c["y0"], c["mint"]))
    cands = cands[:MAX_CANDIDATES]

    rng = random.Random(SEED)
    place_counts, size_values = simulate(cands, horizons_h, rng)

    # Unnamed reserve: longer horizons mean more of the top 5 will be coins
    # born after the cut (in the practice, 2 of the top 5 at the last read
    # were). Reserve grows from ~0.30 (short horizon) toward ~0.42.
    unnamed_reserve = 0.30 + 0.12 * (1.0 - math.exp(-median_h / 6.0))

    probabilities = []
    place_probs = []
    for k in range(places):
        named = allocate_place(
            place_counts[k], N_SIMS, len(horizons_h), chart_size, unnamed_reserve
        )
        place_probs.append(dict(named))
        for mint, p in named:
            probabilities.append({"mint": mint, "place": k + 1, "p": p})

    # top5: the most likely coin at each place, in order, distinct.
    top5 = []
    used = set()
    for k in range(places):
        best = None
        for mint, p in sorted(
            place_probs[k].items(), key=lambda kv: (-kv[1], kv[0])
        ):
            if mint not in used:
                best = mint
                break
        if best is None:
            # Fall back to the largest simulated coins not already used.
            for c in cands:
                if c["mint"] not in used:
                    best = c["mint"]
                    break
        if best is not None:
            top5.append(best)
            used.add(best)
    # Pad defensively (should not happen with a real history).
    for c in cands:
        if len(top5) >= places:
            break
        if c["mint"] not in used:
            top5.append(c["mint"])
            used.add(c["mint"])
    # With fewer than `places` coins in the whole history, pad with
    # distinct placeholder mints so top5 is always exactly `places` long;
    # placeholders hold no place in the agreement, but the file must be
    # of the right shape.
    pad = 0
    while len(top5) < places:
        ph = "1111111111111111111111111111111%d" % pad
        pad += 1
        if ph not in used:
            top5.append(ph)
            used.add(ph)

    # Sizes: fresh-leader centre per place from the last mark's caps at
    # that place, decayed with the median horizon.
    last_sorted = sorted(
        (log_cap_sol(r["market_cap_lamports"]) for r in last_rows), reverse=True
    )
    sizes = []
    for k in range(places):
        centre = last_sorted[k] if k < len(last_sorted) else (
            last_sorted[-1] if last_sorted else 2.0
        )
        centre -= SIZE_FRESH_DECAY * median_h
        named = allocate_sizes(size_values[k], edges, centre, SIZE_FRESH_SD)
        for b, p in named:
            sizes.append({"place": k + 1, "bin": b, "p": p})

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

