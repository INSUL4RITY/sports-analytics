"""Broncos 2025: luck or skill?

Reproduces every number, table, chart and the web page for this project from
public nflverse data (via the nflreadpy package).

    python analysis.py            # uses cached data in data/raw/ when present
    python analysis.py --refresh  # re-downloads everything

Outputs:
    outputs/*.csv                         tables behind every finding
    outputs/findings.json                 every headline number
    figures/*.png                         charts (also copied to the web page)
    README.md (findings section)          rewritten between FINDINGS markers
    ../docs/broncos-luck-or-skill/        the GitHub Pages web page
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
SITE = ROOT.parent / "docs" / "broncos-luck-or-skill"

TEAM = "DEN"
SEASON = 2025                    # the season under the microscope
LIVE = 2026                      # the follow-up season (in progress)
ONE_SCORE = 8                    # a one-score game is decided by 8 points or fewer
PYTH_EXP = 2.37                  # standard NFL Pythagorean exponent
HIST_FIRST = 2002                # first season of the 32-team era
FIT = range(2010, 2025)          # seasons used to fit every model (Denver 2025 is out of sample)
PBP_FIRST = 2010
FTN_FIRST = 2022                 # FTN charting (interception-worthy throws) starts in 2022
WP_TRAIN = range(2016, 2025)     # seasons used to train the win-probability stand-in model
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}   # relocated teams keep one code
N_SIMS = 20_000
RNG = np.random.default_rng(2025)

PBP_COLS = [
    "game_id", "play_id", "season", "season_type", "week", "home_team", "away_team",
    "posteam", "defteam", "posteam_type", "qtr", "down", "ydstogo", "yardline_100",
    "quarter_seconds_remaining", "half_seconds_remaining", "game_seconds_remaining",
    "score_differential", "posteam_score", "defteam_score",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining", "home_opening_kickoff",
    "play_type", "desc", "yards_gained", "air_yards", "pass", "rush", "qb_dropback",
    "pass_attempt", "sack", "special_teams_play", "epa", "wp", "wpa", "home_wp", "away_wp",
    "home_wp_post", "away_wp_post", "interception", "fumble", "fumble_lost",
    "fumble_out_of_bounds", "fumbled_1_team", "fumble_recovery_1_team",
    "field_goal_attempt", "field_goal_result", "kick_distance", "extra_point_attempt",
    "extra_point_result", "fourth_down_converted", "fourth_down_failed", "touchdown",
    "td_team", "sp", "qb_kneel", "qb_spike", "penalty",
]


# ---------------------------------------------------------------------------
# Data loading (cached as slim parquet files in data/raw/, which git ignores)
# ---------------------------------------------------------------------------
def cached(name: str, loader, refresh: bool, max_age_hours: float | None = None) -> pd.DataFrame:
    path = RAW / f"{name}.parquet"
    fresh = path.exists() and (
        max_age_hours is None or time.time() - path.stat().st_mtime < max_age_hours * 3600
    )
    if fresh and not refresh:
        return pd.read_parquet(path)
    df = loader()
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_all(refresh: bool):
    live_age = 12  # the live season changes weekly, so re-download it twice a day
    sched = cached(
        "schedules",
        lambda: nfl.load_schedules(list(range(HIST_FIRST, LIVE + 1))).to_pandas(),
        refresh, live_age,
    )
    pbp = []
    for s in range(PBP_FIRST, LIVE + 1):
        pbp.append(cached(
            f"pbp_{s}",
            lambda s=s: nfl.load_pbp(s).to_pandas()[PBP_COLS],
            refresh, live_age if s == LIVE else None,
        ))
        print(f"  play-by-play {s} loaded", end="\r")
    print()
    pbp = pd.concat(pbp, ignore_index=True)
    ftn = pd.concat([
        cached(
            f"ftn_{s}",
            lambda s=s: nfl.load_ftn_charting(s).to_pandas()[
                ["nflverse_game_id", "nflverse_play_id", "is_interception_worthy"]],
            refresh, live_age if s == LIVE else None,
        )
        for s in range(FTN_FIRST, LIVE + 1)
    ], ignore_index=True)
    for col in ["home_team", "away_team"]:
        sched[col] = sched[col].replace(FRANCHISE)
    for col in ["home_team", "away_team", "posteam", "defteam", "fumbled_1_team",
                "fumble_recovery_1_team", "td_team"]:
        pbp[col] = pbp[col].replace(FRANCHISE)
    return sched, pbp, ftn


# ---------------------------------------------------------------------------
# Game- and season-level tables from scores
# ---------------------------------------------------------------------------
def team_games(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per team per completed game."""
    g = sched.dropna(subset=["home_score", "away_score"])
    cols = ["game_id", "season", "game_type", "week", "gameday", "overtime"]
    home = g[cols].assign(team=g.home_team, opp=g.away_team, home=1,
                          pf=g.home_score, pa=g.away_score)
    away = g[cols].assign(team=g.away_team, opp=g.home_team, home=0,
                          pf=g.away_score, pa=g.home_score)
    tg = pd.concat([home, away], ignore_index=True)
    tg["margin"] = tg.pf - tg.pa
    tg["win"] = np.select([tg.margin > 0, tg.margin < 0], [1.0, 0.0], 0.5)  # ties = half a win
    tg["one_score"] = tg.margin.abs() <= ONE_SCORE
    return tg


def team_seasons(tg: pd.DataFrame) -> pd.DataFrame:
    reg = tg[tg.game_type == "REG"]
    ts = reg.groupby(["team", "season"]).agg(
        games=("win", "size"), wins=("win", "sum"), pf=("pf", "sum"), pa=("pa", "sum"))
    close = reg[reg.one_score].groupby(["team", "season"]).agg(
        os_games=("win", "size"), os_wins=("win", "sum"))
    ts = ts.join(close).fillna({"os_games": 0, "os_wins": 0}).reset_index()
    ts["win_pct"] = ts.wins / ts.games
    ts["os_win_pct"] = ts.os_wins / ts.os_games.replace(0, np.nan)
    ts["pyth_pct"] = ts.pf ** PYTH_EXP / (ts.pf ** PYTH_EXP + ts.pa ** PYTH_EXP)
    ts["pyth_wins"] = ts.pyth_pct * ts.games
    return ts


def record(w: float, g: int) -> str:
    losses = g - w
    ties = 0
    if w % 1:  # a tie shows up as half a win
        ties, w, losses = 1, w - 0.5, losses - 0.5
    return f"{int(w)}-{int(losses)}" + (f"-{ties}" if ties else "")


# ---------------------------------------------------------------------------
# Play-by-play building blocks
# ---------------------------------------------------------------------------
def scrimmage(pbp: pd.DataFrame) -> pd.DataFrame:
    """Standard nflverse filter for offensive plays: passes and runs with an EPA value."""
    return pbp[((pbp["pass"] == 1) | (pbp["rush"] == 1)) & pbp.epa.notna()]


def epa_team_seasons(pbp: pd.DataFrame) -> pd.DataFrame:
    s = scrimmage(pbp[pbp.season_type == "REG"])
    off = s.groupby(["posteam", "season"]).epa.mean().rename("off_epa")
    dfn = s.groupby(["defteam", "season"]).epa.mean().rename("def_epa")
    off.index.names = dfn.index.names = ["team", "season"]
    out = pd.concat([off, dfn], axis=1).reset_index()
    out["net_epa"] = out.off_epa - out.def_epa
    return out


def ols(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """Weighted least squares line. Returns intercept, slope, R-squared."""
    w = np.ones_like(x) if w is None else w
    X = np.column_stack([np.ones_like(x), x])
    W = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    resid = y - X @ beta
    ybar = np.average(y, weights=w)
    r2 = 1 - np.sum(w * resid ** 2) / np.sum(w * (y - ybar) ** 2)
    return float(beta[0]), float(beta[1]), float(r2)


def binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def den_wp(df: pd.DataFrame, team: str = TEAM, post: bool = False) -> pd.Series:
    h, a = ("home_wp_post", "away_wp_post") if post else ("home_wp", "away_wp")
    return np.where(df.home_team == team, df[h], df[a])


def elapsed_seconds(df: pd.DataFrame) -> pd.Series:
    reg = (df.qtr.clip(upper=4) - 1) * 900 + (900 - df.quarter_seconds_remaining)
    ot = 3600 + (600 - df.quarter_seconds_remaining)
    return np.where(df.qtr >= 5, ot, reg)


# ---------------------------------------------------------------------------
# 1. Expected wins vs actual
# ---------------------------------------------------------------------------
def expected_wins(ts: pd.DataFrame, epa: pd.DataFrame):
    m = ts.merge(epa, on=["team", "season"])
    fit = m[m.season.isin(FIT)]
    a, b, r2 = ols(fit.net_epa.values, fit.win_pct.values)
    m["epa_wins"] = (a + b * m.net_epa) * m.games
    # Expected one-score win % for a team of this efficiency (weighted by close games played)
    fo = fit[fit.os_games > 0]
    a2, b2, r2_os = ols(fo.net_epa.values, fo.os_win_pct.values, fo.os_games.values)
    m["exp_os_pct"] = a2 + b2 * m.net_epa
    cur = m[m.season == SEASON].copy()
    cur["wins_over_pyth"] = cur.wins - cur.pyth_wins
    cur["wins_over_epa"] = cur.wins - cur.epa_wins
    cur["os_wins_over_exp"] = cur.os_wins - cur.exp_os_pct * cur.os_games
    models = {
        "epa_win_model": {"intercept": a, "slope": b, "r2": r2, "n": len(fit)},
        "epa_os_model": {"intercept": a2, "slope": b2, "r2": r2_os, "n": len(fo),
                         "games": int(fo.os_games.sum())},
    }
    return m, cur.sort_values("wins_over_epa", ascending=False), models


def pyth_exponent_check(ts: pd.DataFrame, den: pd.Series) -> dict:
    """Does the standard 2.37 exponent fit recent seasons? Try 1.5-3.5 on the model-fitting seasons."""
    fit = ts[ts.season.isin(FIT)]

    def rmse_wins(k):  # typical miss, in wins per 17 games
        p = fit.pf ** k / (fit.pf ** k + fit.pa ** k)
        return float(np.sqrt(np.mean((p - fit.win_pct) ** 2)) * 17)

    def den_wins(k):
        return float(den.games * den.pf ** k / (den.pf ** k + den.pa ** k))

    ks = np.round(np.arange(1.5, 3.501, 0.01), 2)
    errs = [rmse_wins(k) for k in ks]
    best = float(ks[int(np.argmin(errs))])
    return {"best": best, "rmse_best": min(errs), "rmse_std": rmse_wins(PYTH_EXP), "n": len(fit),
            "den_best": den_wins(best), "den_2": den_wins(2.0), "den_3": den_wins(3.0)}


# ---------------------------------------------------------------------------
# 2. History: teams with 9+ one-score wins, and what happened next
# ---------------------------------------------------------------------------
def history(ts: pd.DataFrame):
    nxt = ts[["team", "season", "os_games", "os_wins", "os_win_pct", "wins", "games"]].copy()
    nxt["season"] -= 1
    pairs = ts.merge(nxt, on=["team", "season"], suffixes=("", "_next"))
    pairs = pairs[pairs.season.between(HIST_FIRST, SEASON - 1)]
    big = pairs[pairs.os_wins >= 9].sort_values(["os_wins", "season"], ascending=[False, True])
    both = pairs.dropna(subset=["os_win_pct", "os_win_pct_next"])
    r_os = float(np.corrcoef(both.os_win_pct, both.os_win_pct_next)[0, 1])
    r_win = float(np.corrcoef(pairs.win_pct, pairs.wins_next / pairs.games_next)[0, 1])
    allseasons = ts[ts.season.between(HIST_FIRST, SEASON)]
    rank = int((allseasons.os_wins > allseasons.query("team == @TEAM and season == @SEASON")
                .os_wins.iloc[0]).sum()) + 1
    summary = {
        "n_big": len(big),
        "big_os_pct": float(big.os_wins.sum() / big.os_games.sum()),
        "big_os_pct_next": float(big.os_wins_next.sum() / big.os_games_next.sum()),
        "big_wins": float((big.wins / big.games * 17).mean()),   # per 17 games
        "big_wins_next": float((big.wins_next / big.games_next * 17).mean()),
        "big_fewer_wins_next": int((big.wins_next / big.games_next < big.wins / big.games).sum()),
        "big_os_pct_next_below_60": int((big.os_win_pct_next < 0.6).sum()),
        "r_os": r_os, "n_pairs_os": len(both),
        "r_win": r_win, "n_pairs_win": len(pairs),
        "den_os_wins_rank": rank,
        "n_team_seasons": len(allseasons),
        "n_11plus": int((allseasons.os_wins >= 11).sum()),
        "list_11plus": allseasons[allseasons.os_wins >= 11]
        .sort_values("season")[["team", "season", "os_wins", "os_games"]].to_dict("records"),
    }
    return pairs, big, summary


# ---------------------------------------------------------------------------
# 3. The one-score games, play by play
# ---------------------------------------------------------------------------
BUCKETS = ["Turnovers", "Special teams & kicks", "Q4/OT offence", "Q4/OT defence", "Q1–Q3 plays"]


def bucket_epa(plays: pd.DataFrame, team: str = TEAM) -> pd.DataFrame:
    """Split each play's EPA (from the team's point of view) into five non-overlapping buckets."""
    p = plays[plays.epa.notna() & plays.posteam.notna()].copy()
    p["team_epa"] = np.where(p.posteam == team, p.epa, -p.epa)
    st = (p.special_teams_play == 1) | p.play_type.isin(["field_goal", "extra_point", "punt", "kickoff"])
    to = (p.interception == 1) | (p.fumble_lost == 1)
    late = p.qtr >= 4
    p["bucket"] = np.select(
        [to, st, late & (p.posteam == team), late & (p.defteam == team)],
        BUCKETS[:4], BUCKETS[4])
    return p


def close_games(tg: pd.DataFrame, pbp: pd.DataFrame, season: int, game_type="REG"):
    games = tg[(tg.team == TEAM) & (tg.season == season) & tg.one_score]
    if game_type == "REG":
        games = games[games.game_type == "REG"]
    else:
        games = games[games.game_type != "REG"]
    rows, timelines, buckets, fourth = [], [], [], []
    for g in games.sort_values("gameday").itertuples():
        p = pbp[pbp.game_id == g.game_id].sort_values("play_id")
        wp = pd.Series(den_wp(p), index=p.index)
        wp_post = pd.Series(den_wp(p, post=True), index=p.index)
        label = f"Wk {g.week} {'vs' if g.home else '@'} {g.opp}"
        b = bucket_epa(p)
        bsum = b.groupby("bucket").team_epa.sum().reindex(BUCKETS, fill_value=0.0)
        buckets.append(bsum.rename(label))
        den_off, opp_off = p.posteam == TEAM, p.posteam == g.opp
        giveaways = int(((p.interception == 1) | (p.fumble_lost == 1))[den_off].sum())
        takeaways = int(((p.interception == 1) | (p.fumble_lost == 1))[opp_off].sum())
        fum = p[(p.fumble == 1) & p.fumble_recovery_1_team.notna()]
        fga = p[p.field_goal_attempt == 1]
        xpa = p[p.extra_point_attempt == 1]
        go = p[(p.down == 4) & p.play_type.isin(["pass", "run"])]
        for r in go[go.posteam == TEAM].itertuples():
            fourth.append({"game": label, "qtr": r.qtr, "ydstogo": r.ydstogo,
                           "yardline_100": r.yardline_100,
                           "converted": int(r.fourth_down_converted == 1),
                           "den_wpa_pts": round(100 * r.wpa, 1)})
        # The single play that swung the game most towards Denver
        dwpa = np.where(p.posteam == TEAM, p.wpa, -p.wpa)
        top = p.iloc[int(np.nanargmax(dwpa))]
        rows.append({
            "game_id": g.game_id, "game": label, "week": g.week, "opp": g.opp,
            "home": g.home, "den_pts": int(g.pf), "opp_pts": int(g.pa),
            "margin": int(g.margin), "result": {1.0: "W", 0.0: "L"}.get(g.win, "T"),
            "overtime": int(g.overtime == 1),
            "min_wp": float(np.nanmin(np.r_[wp.values, wp_post.values])),
            "max_wp": float(np.nanmax(np.r_[wp.values, wp_post.values])),
            "to_margin": takeaways - giveaways,
            "den_fumbles_recovered": int((fum.fumble_recovery_1_team == TEAM).sum()),
            "fumbles_total": len(fum),
            "den_fg": f"{int((fga[fga.posteam == TEAM].field_goal_result == 'made').sum())}/"
                      f"{int((fga.posteam == TEAM).sum())}",
            "opp_fg": f"{int((fga[fga.posteam == g.opp].field_goal_result == 'made').sum())}/"
                      f"{int((fga.posteam == g.opp).sum())}",
            "opp_xp_missed": int(((xpa.posteam == g.opp) & (xpa.extra_point_result != "good")).sum()),
            "den_4th_go": int((go.posteam == TEAM).sum()),
            "den_4th_conv": int(((go.posteam == TEAM) & (go.fourth_down_converted == 1)).sum()),
            "q4_off_epa": float(bsum["Q4/OT offence"]), "q4_def_epa": float(bsum["Q4/OT defence"]),
            "net_epa": float(bsum.sum()),
            "top_play_wpa": float(np.nanmax(dwpa)),
            "top_play": f"Q{int(top.qtr)}: " + str(top.desc)[:140],
        })
        t = pd.DataFrame({"game": label, "t": elapsed_seconds(p), "wp": wp.values}).dropna()
        end = 3600 if not g.overtime else t.t.max()
        t = pd.concat([t, pd.DataFrame({"game": [label], "t": [end], "wp": [float(g.win)]})])
        timelines.append(t)
    games_df = pd.DataFrame(rows)
    return (games_df, pd.concat(timelines, ignore_index=True) if timelines else pd.DataFrame(),
            pd.DataFrame(buckets), pd.DataFrame(fourth))


# ---------------------------------------------------------------------------
# 4. Skill signals vs luck signals (and how much each repeats year to year)
# ---------------------------------------------------------------------------
def fg_expected(pbp: pd.DataFrame) -> pd.DataFrame:
    """Each field-goal attempt with the league make rate at that distance (5-yard band, same season)."""
    k = pbp[(pbp.field_goal_attempt == 1) & pbp.kick_distance.notna()].copy()
    k["made"] = (k.field_goal_result == "made").astype(float)
    k["band"] = (k.kick_distance.clip(18, 64) // 5) * 5
    k["p_make"] = k.groupby(["season", "band"]).made.transform("mean")
    return k


def skill_signals(pbp: pd.DataFrame, ts: pd.DataFrame, epa: pd.DataFrame, fg: pd.DataFrame):
    reg = pbp[pbp.season_type == "REG"]
    s = scrimmage(reg)
    q4d = s[s.qtr >= 4].groupby(["defteam", "season"]).epa.mean().rename("q4_def_epa")
    q4d.index.names = ["team", "season"]
    lc = s[(s.qtr >= 4) & (s.score_differential.abs() <= ONE_SCORE) & (s.qb_dropback == 1)]
    qb = lc.groupby(["posteam", "season"]).epa.mean().rename("late_close_dropback_epa")
    qb.index.names = ["team", "season"]
    fgr = fg[fg.season_type == "REG"].assign(oe=lambda d: d.made - d.p_make)
    own = fgr.groupby(["posteam", "season"]).oe.mean().rename("fg_over_exp")
    opp = fgr.groupby(["defteam", "season"]).oe.mean().rename("opp_fg_over_exp")
    own.index.names = opp.index.names = ["team", "season"]
    fum = reg[(reg.fumble == 1) & reg.fumble_recovery_1_team.notna()]
    long = pd.concat([fum[["season", "posteam", "fumble_recovery_1_team"]].rename(columns={"posteam": "team"}),
                      fum[["season", "defteam", "fumble_recovery_1_team"]].rename(columns={"defteam": "team"})])
    long["rec"] = (long.team == long.fumble_recovery_1_team).astype(float)
    fr = long.groupby(["team", "season"]).rec.mean().rename("fumble_recovery_rate")
    gv = reg.assign(t=(reg.interception == 1) | (reg.fumble_lost == 1))
    give = gv[gv.t].groupby(["posteam", "season"]).size()
    take = gv[gv.t].groupby(["defteam", "season"]).size()
    give.index.names = take.index.names = ["team", "season"]
    m = (ts.set_index(["team", "season"])[["games", "os_win_pct"]]
         .join([epa.set_index(["team", "season"]).net_epa, q4d, qb, own, opp, fr]))
    m["to_margin_pg"] = (take.reindex(m.index, fill_value=0) - give.reindex(m.index, fill_value=0)) / m.games
    m = m.reset_index()
    m = m[m.season >= PBP_FIRST]

    meta = {  # label, higher is better for the team?, what it is
        "net_epa": ("Overall efficiency (net EPA per play)", True),
        "q4_def_epa": ("4th-quarter defence (EPA allowed per play)", False),
        "late_close_dropback_epa": ("QB late & close (EPA per dropback, Q4/OT, within 8)", True),
        "fg_over_exp": ("Own field goals vs expected (per attempt)", True),
        "fumble_recovery_rate": ("Share of all fumbles recovered", True),
        "opp_fg_over_exp": ("Opponents' field goals vs expected (per attempt)", False),
        "to_margin_pg": ("Turnover margin per game", True),
        "os_win_pct": ("One-score win %", True),
    }
    nxt = m.copy()
    nxt["season"] -= 1
    pairs = m.merge(nxt, on=["team", "season"], suffixes=("", "_n"))
    pairs = pairs[pairs.season.isin(FIT) & pairs.season.add(1).isin(FIT)]
    cur = m[m.season == SEASON]
    den = cur[cur.team == TEAM].iloc[0]
    rows = []
    for col, (label, higher) in meta.items():
        pp = pairs[[col, col + "_n"]].dropna()
        r = float(np.corrcoef(pp[col], pp[col + "_n"])[0, 1])
        better = (cur[col] > den[col]) if higher else (cur[col] < den[col])
        rows.append({
            "metric": col, "label": label, "den_2025": float(den[col]),
            "league_avg_2025": float(cur[col].mean()), "den_rank_2025": int(better.sum()) + 1,
            "teams": int(cur[col].notna().sum()), "year_to_year_r": r, "n_pairs": len(pp),
            "repeats": "Repeats" if r >= 0.3 else ("Weakly" if r >= 0.15 else "Barely / not"),
        })
    return m, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. High-leverage luck
# ---------------------------------------------------------------------------
# nflverse publishes win probability (WP) for what actually happened. For the outcome
# that did not happen we need the WP of a hypothetical game state, so we train a
# stand-in model to reproduce nflverse's WP from the game state, then score the
# hypothetical states with it.
WP_FEATURES = ["score_diff", "game_secs", "half_secs", "is_ot", "yardline_100", "down",
               "ydstogo", "to_own", "to_opp", "home", "rec_2h", "diff_per_min"]


def wp_features(score_diff, game_secs, half_secs, qtr, yardline_100, down, ydstogo,
                to_own, to_opp, home, home_opening_kickoff):
    f = pd.DataFrame({
        "score_diff": np.asarray(score_diff, float), "game_secs": np.asarray(game_secs, float),
        "half_secs": np.asarray(half_secs, float), "is_ot": (np.asarray(qtr) >= 5).astype(float),
        "yardline_100": np.clip(np.asarray(yardline_100, float), 1, 99),
        "down": np.asarray(down, float), "ydstogo": np.asarray(ydstogo, float),
        "to_own": np.asarray(to_own, float), "to_opp": np.asarray(to_opp, float),
        "home": np.asarray(home, float),
    })
    # The team that kicked off to start the game receives the second-half kickoff
    rec = (f.home == 1) == (np.asarray(home_opening_kickoff) == 1)
    f["rec_2h"] = np.where(np.asarray(qtr) <= 2, rec.astype(float), 0.0)
    f["diff_per_min"] = f.score_diff / (f.game_secs / 60 + 1)
    return f[WP_FEATURES]


def features_from_plays(p: pd.DataFrame) -> pd.DataFrame:
    return wp_features(p.score_differential, p.game_seconds_remaining, p.half_seconds_remaining,
                       p.qtr, p.yardline_100, p.down, p.ydstogo, p.posteam_timeouts_remaining,
                       p.defteam_timeouts_remaining, p.posteam_type == "home", p.home_opening_kickoff)


def train_wp_model(pbp: pd.DataFrame):
    ok = pbp.down.notna() & pbp.wp.notna() & pbp.posteam.notna() & pbp.yardline_100.notna() \
        & pbp.game_seconds_remaining.notna() & pbp.ydstogo.notna()
    train = pbp[ok & pbp.season.isin(WP_TRAIN)]
    test = pbp[ok & (pbp.season == SEASON)]
    cst = [1 if c in ("score_diff", "diff_per_min") else (-1 if c == "yardline_100" else 0)
           for c in WP_FEATURES]
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
                                          monotonic_cst=cst, random_state=0)
    model.fit(features_from_plays(train), train.wp)
    err = np.abs(model.predict(features_from_plays(test)) - test.wp)
    return model, {"mae_pts": float(100 * err.mean()), "p90_err_pts": float(100 * np.quantile(err, 0.9)),
                   "n_train": len(train), "n_test": len(test)}


def next_scrimmage(pbp: pd.DataFrame, plays: pd.DataFrame) -> pd.DataFrame:
    """Attach the next play with a down (same game) to each row in `plays`."""
    nxt = pbp[pbp.down.notna() & pbp.posteam.notna()][
        ["game_id", "play_id", "posteam", "yardline_100"]].rename(
        columns={"play_id": "nxt_play_id", "posteam": "nxt_posteam", "yardline_100": "nxt_yl"})
    left = plays.sort_values("play_id")
    out = pd.merge_asof(left, nxt.sort_values("nxt_play_id"), left_on="play_id",
                        right_on="nxt_play_id", by="game_id", direction="forward",
                        allow_exact_matches=False)
    return out.set_index(left.index)


def chance_plays(pbp: pd.DataFrame, ftn: pd.DataFrame, fg: pd.DataFrame, wp_model) -> pd.DataFrame:
    """Every high-leverage chance play with: P(good for offense), WP if good, WP if bad (offense view)."""
    hist = pbp[pbp.season.isin(FIT)]
    seasons = range(FTN_FIRST, LIVE + 1)
    p = pbp[pbp.season.isin(seasons) & pbp.posteam.notna() & pbp.wp.notna()].copy()
    p["wp_post_off"] = np.where(p.posteam == p.home_team, p.home_wp_post, p.away_wp_post)
    p = p[p.wp_post_off.notna()]
    p["secs_after"] = (p.game_seconds_remaining - 5).clip(lower=0)
    p["half_after"] = (p.half_seconds_remaining - 5).clip(lower=0)

    def wp_for(rows, holder_is_off, score_diff_off, yl_holder, down=1, togo=10, secs=None):
        """WP for the offense if `holder` has the ball in this state."""
        hold = np.asarray(holder_is_off, bool)
        f = wp_features(
            np.where(hold, score_diff_off, -np.asarray(score_diff_off, float)),
            rows.secs_after if secs is None else secs, rows.half_after, rows.qtr, yl_holder,
            np.broadcast_to(down, len(rows)),
            np.minimum(np.broadcast_to(togo, len(rows)), np.asarray(yl_holder, float)),
            np.where(hold, rows.posteam_timeouts_remaining, rows.defteam_timeouts_remaining),
            np.where(hold, rows.defteam_timeouts_remaining, rows.posteam_timeouts_remaining),
            np.where(hold, rows.posteam_type == "home", rows.posteam_type != "home"),
            rows.home_opening_kickoff)
        w = wp_model.predict(f).clip(0, 1)
        return np.where(hold, w, 1 - w)

    # Average field position (from the receiving team's 100-yard line) after a kickoff
    ko = next_scrimmage(pbp, pbp[pbp.play_type == "kickoff"])
    ko_start = ko[ko.nxt_yl.notna()].groupby(ko.season).nxt_yl.median()

    out = []

    # (a) Fumbles: does the fumbling side keep the ball?
    fhist = hist[(hist.fumble == 1) & hist.fumble_recovery_1_team.notna() & hist.fumbled_1_team.notna()]
    fhist = fhist.assign(keep=fhist.fumble_recovery_1_team == fhist.fumbled_1_team,
                         st=fhist.special_teams_play == 1)
    p_keep = fhist.groupby("st").keep.mean()
    f = p[(p.fumble == 1) & p.fumble_recovery_1_team.notna() & p.fumbled_1_team.notna()
          & (p.fumble_out_of_bounds != 1)].copy()
    f = next_scrimmage(pbp, f)
    f["fumbler_is_off"] = f.fumbled_1_team == f.posteam
    f["kept"] = f.fumble_recovery_1_team == f.fumbled_1_team
    f["p_keep"] = np.where(f.special_teams_play == 1, p_keep[True], p_keep[False])
    f["recovered_by_off"] = f.fumble_recovery_1_team == f.posteam
    # Ball spot (offense's yards to goal): next scrimmage play if no score, else the end of the run
    spot_next = np.where(f.nxt_posteam == f.posteam, f.nxt_yl, 100 - f.nxt_yl)
    spot_fallback = f.yardline_100 - f.yards_gained.fillna(0)
    f["spot_off"] = np.where((f.sp != 1) & f.nxt_yl.notna(), spot_next, spot_fallback).clip(1, 99)
    # Hypothetical: the *other* side recovers at the same spot, score as before the play
    hyp_holder_off = ~f.recovered_by_off
    f["wp_hyp"] = wp_for(f, hyp_holder_off, f.score_differential,
                         np.where(hyp_holder_off, f.spot_off, 100 - f.spot_off))
    p_good_off = np.where(f.fumbler_is_off, f.p_keep, 1 - f.p_keep)  # good for offense = offense recovers
    out.append(pd.DataFrame({
        "type": "Fumble recovery", "good_off": f.recovered_by_off, "p_good_off": p_good_off,
        "wp_actual": f.wp_post_off, "wp_hyp": f.wp_hyp, "who": "both"}, index=f.index))

    # (b) Opponents' field goals and extra points vs league make rate
    k = fg[fg.season.isin(seasons)][["p_make"]]
    kk = p[p.field_goal_attempt == 1].join(k, how="inner")
    made = kk.field_goal_result == "made"
    pre = kk.score_differential
    recv_yl = kk.season.map(ko_start).fillna(70)
    # Made -> hypothetical miss: other team's ball at the spot of the kick (or its own 20)
    miss_yl = 100 - np.maximum(kk.yardline_100 + 7, 20)
    wp_miss = wp_for(kk, np.zeros(len(kk), bool), pre, miss_yl)
    wp_make = wp_for(kk, np.zeros(len(kk), bool), pre + 3, recv_yl)
    out.append(pd.DataFrame({
        "type": "Opponent kick", "good_off": made, "p_good_off": kk.p_make,
        "wp_actual": kk.wp_post_off, "wp_hyp": np.where(made, wp_miss, wp_make), "who": "def"},
        index=kk.index))
    xp = p[(p.extra_point_attempt == 1) & p.extra_point_result.notna()].copy()
    xp["made"] = xp.extra_point_result == "good"
    xp_rate = pbp[(pbp.extra_point_attempt == 1) & pbp.extra_point_result.notna()] \
        .assign(m=lambda d: d.extra_point_result == "good").groupby("season").m.mean()
    recv_xp = xp.season.map(ko_start).fillna(70)
    # score_differential on the try is the offense lead after the touchdown, before the kick
    wp_x_make = wp_for(xp, np.zeros(len(xp), bool), xp.score_differential + 1, recv_xp)
    wp_x_miss = wp_for(xp, np.zeros(len(xp), bool), xp.score_differential, recv_xp)
    out.append(pd.DataFrame({
        "type": "Opponent kick", "good_off": xp.made, "p_good_off": xp.season.map(xp_rate),
        "wp_actual": xp.wp_post_off, "wp_hyp": np.where(xp.made, wp_x_miss, wp_x_make), "who": "def"},
        index=xp.index))

    # (c) Interceptions vs interception-worthy throws (FTN charting)
    ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    ftn["play_id"] = ftn.play_id.astype(float)
    # Plays with an accepted penalty are left out: the flag, not chance, settled them
    throws = p[(p.pass_attempt == 1) & (p.sack != 1) & (p.play_type == "pass") & (p.penalty != 1)]
    throws = throws.reset_index().merge(ftn[["game_id", "play_id", "is_interception_worthy"]],
                                        on=["game_id", "play_id"], how="inner").set_index("index")
    throws["worthy"] = throws.is_interception_worthy.astype(bool)
    th = throws[throws.season < SEASON]
    p_int = th.groupby("worthy").interception.mean()
    throws["p_int"] = throws.worthy.map(p_int)
    picked = throws.interception == 1
    # Not picked -> hypothetical pick at the target spot; picked -> hypothetical incompletion
    spot = (throws.yardline_100 - throws.air_yards.fillna(10)).clip(upper=99)
    int_yl = np.where(spot <= 0, 80, 100 - spot)
    wp_int = wp_for(throws, np.zeros(len(throws), bool), throws.score_differential, int_yl)
    on_downs = throws.down == 4
    wp_inc = np.where(
        on_downs,
        wp_for(throws, np.zeros(len(throws), bool), throws.score_differential, 100 - throws.yardline_100),
        wp_for(throws, np.ones(len(throws), bool), throws.score_differential, throws.yardline_100,
               down=(throws.down + 1).clip(upper=4), togo=throws.ydstogo))
    out.append(pd.DataFrame({
        "type": "Interception luck", "good_off": ~picked, "p_good_off": 1 - throws.p_int,
        "wp_actual": throws.wp_post_off, "wp_hyp": np.where(picked, wp_inc, wp_int),
        "worthy": throws.worthy, "who": "both"}, index=throws.index))

    # (d) 4th-down attempts vs league conversion rate at that distance
    def togo_band(y):
        return pd.cut(y, [0, 1, 2, 3, 5, 10, 99], labels=["1", "2", "3", "4-5", "6-10", "11+"])

    def is_go(d):
        return (d.down == 4) & d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) \
            & (d.qb_spike != 1) & (d.fumble != 1) & (d.interception != 1) & (d.penalty != 1) \
            & ((d.fourth_down_converted == 1) | (d.fourth_down_failed == 1))

    gh = hist[is_go(hist)]
    conv_rate = gh.assign(b=togo_band(gh.ydstogo)).groupby("b", observed=True).fourth_down_converted.mean()
    go = p[is_go(p)].copy()
    go["p_conv"] = togo_band(go.ydstogo).map(conv_rate).astype(float)
    conv = go.fourth_down_converted == 1
    goal = go.ydstogo >= go.yardline_100
    recv_go = go.season.map(ko_start).fillna(70)
    wp_fail = wp_for(go, np.zeros(len(go), bool), go.score_differential, 100 - go.yardline_100)
    wp_conv = np.where(
        goal, wp_for(go, np.zeros(len(go), bool), go.score_differential + 7, recv_go),
        wp_for(go, np.ones(len(go), bool), go.score_differential, go.yardline_100 - go.ydstogo))
    out.append(pd.DataFrame({
        "type": "4th-down conversion", "good_off": conv, "p_good_off": go.p_conv,
        "wp_actual": go.wp_post_off, "wp_hyp": np.where(conv, wp_fail, wp_conv), "who": "both"},
        index=go.index))

    c = pd.concat(out)
    c = c[c.p_good_off.notna() & c.wp_hyp.notna()]
    c = c.join(p[["game_id", "season", "season_type", "week", "posteam", "defteam", "qtr", "desc"]])
    c["good_off"] = c.good_off.astype(bool)
    # Luck (offense view) = actual WP - probability-weighted average of the two outcomes
    p_actual = np.where(c.good_off, c.p_good_off, 1 - c.p_good_off)
    c["swing_off"] = np.where(c.good_off, c.wp_actual - c.wp_hyp, c.wp_hyp - c.wp_actual)
    c["luck_off"] = (1 - p_actual) * (c.wp_actual - c.wp_hyp)
    rates = {"p_keep_fumble_scrimmage": float(p_keep[False]), "p_keep_fumble_st": float(p_keep[True]),
             "p_int_worthy": float(p_int[True]), "p_int_not_worthy": float(p_int[False]),
             "n_worthy_hist": int((th.worthy).sum()),
             "fourth_conv_rate": {str(k_): float(v) for k_, v in conv_rate.items()},
             "ko_start_2025": float(ko_start.get(SEASON, np.nan))}
    return c, rates


def team_luck(c: pd.DataFrame, team: str) -> pd.DataFrame:
    """Chance plays from one team's point of view: luck, P(good), swing, actual outcome."""
    keep = ["type", "game_id", "season", "season_type", "week", "qtr", "desc"]
    off = c[(c.posteam == team) & (c.who == "both")]
    dfn = c[(c.defteam == team)]
    a = off[keep].assign(luck=off.luck_off, q=off.p_good_off, swing=off.swing_off, good=off.good_off)
    b = dfn[keep].assign(luck=-dfn.luck_off, q=1 - dfn.p_good_off, swing=dfn.swing_off,
                         good=~dfn.good_off)
    return pd.concat([a, b]).assign(team=team)


def luck_tables(c: pd.DataFrame, games: pd.DataFrame, tg: pd.DataFrame):
    teams = sorted(set(c.posteam.dropna()))
    allt = pd.concat([team_luck(c, t) for t in teams])
    reg = allt[allt.season_type == "REG"]
    by_season = reg.groupby(["team", "season"]).luck.sum().rename("luck").reset_index()
    nxt = by_season.assign(season=by_season.season - 1)
    pr = by_season.merge(nxt, on=["team", "season"], suffixes=("", "_next"))
    pr = pr[pr.season + 1 < LIVE]
    r = float(np.corrcoef(pr.luck, pr.luck_next)[0, 1])
    den = allt[allt.team == TEAM]
    per_game = (den[den.game_id.isin(games.game_id)].groupby(["game_id", "type"]).luck.sum()
                .unstack(fill_value=0.0))
    g = games.set_index("game_id").join(per_game).fillna(0.0)
    types = [t for t in ["Fumble recovery", "Opponent kick", "Interception luck", "4th-down conversion"]
             if t in g.columns]
    g["luck_total"] = g[types].sum(axis=1)
    result = (g.result == "W").astype(float)
    g["luck_adj_win_chance"] = (result - g.luck_total).clip(0, 1)
    g["decided_by_luck"] = (g.result == "W") & (g.luck_adj_win_chance < 0.5)

    # Simulation: re-draw every chance play in the close games at its normal rate
    plays = den[den.game_id.isin(games.game_id)]
    sims = np.zeros((N_SIMS, len(g)))
    for j, gid in enumerate(g.index):
        pl = plays[plays.game_id == gid]
        draws = RNG.random((N_SIMS, len(pl))) < pl.q.values
        shift = ((draws.astype(float) - pl.good.values.astype(float)) * pl.swing.values).sum(axis=1)
        chance = np.clip(result.iloc[j] + shift, 0, 1)
        sims[:, j] = RNG.random(N_SIMS) < chance
    wins = sims.sum(axis=1)
    sim = {"expected_wins": float(wins.mean()), "n_games": len(g),
           "p10": float(np.percentile(wins, 10)), "p90": float(np.percentile(wins, 90)),
           "p_11_or_more": float((wins >= 11).mean()),
           "dist": {int(k): float(v) for k, v in pd.Series(wins).value_counts(normalize=True).sort_index().items()}}
    cur = by_season[by_season.season == SEASON].sort_values("luck", ascending=False).reset_index(drop=True)
    den_season = den[den.season_type == "REG"]
    summary = {
        "den_luck_2025": float(cur.loc[cur.team == TEAM, "luck"].iloc[0]),
        "den_luck_rank_2025": int(cur.index[cur.team == TEAM][0]) + 1,
        "league_luck_sd_2025": float(cur.luck.std()),
        "den_luck_by_type_2025": den_season[den_season.season == SEASON].groupby("type").luck.sum().to_dict(),
        "den_luck_close_games": float(g.luck_total.sum()),
        "n_decided_by_luck": int(g.decided_by_luck.sum()),
        "luck_year_to_year_r": r, "luck_pairs": len(pr),
        "n_chance_plays_den_2025": int(((den.season == SEASON) & (den.season_type == "REG")).sum()),
        "simulation": sim,
    }
    top = plays.sort_values("luck", key=np.abs, ascending=False).head(12)
    return g.reset_index(), by_season, summary, top, allt


# ---------------------------------------------------------------------------
# 6. Denver across seasons (same coach and QB since 2024)
# ---------------------------------------------------------------------------
def poisson_binomial_tail(k: int, ps) -> float:
    """P(at least k successes) when each trial has its own success probability."""
    dist = np.array([1.0])
    for p in ps:
        dist = np.convolve(dist, [1 - p, p])
    return float(dist[k:].sum())


def denver_seasons(m: pd.DataFrame) -> pd.DataFrame:
    rank = m.groupby("season").net_epa.rank(ascending=False)
    d = m[(m.team == TEAM) & (m.season >= 2023)].copy()
    d["net_epa_rank"] = rank.loc[d.index].astype(int)
    d["exp_os_wins"] = d.exp_os_pct * d.os_games
    d["record"] = [record(w, g) for w, g in zip(d.wins, d.games)]
    d["os_record"] = [record(w, g) for w, g in zip(d.os_wins, d.os_games)]
    return d[["season", "games", "record", "pf", "pa", "pyth_wins", "epa_wins", "net_epa",
              "net_epa_rank", "os_games", "os_wins", "os_record", "exp_os_pct", "exp_os_wins"]]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
import textwrap                                         # noqa: E402
import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402
from matplotlib.patches import Rectangle                # noqa: E402

ORANGE, BLUE, NAVY = "#FB4F14", "#2f5ea8", "#002244"   # Broncos orange; Broncos blue lifted for charts
YELLOW, AQUA = "#eda100", "#1baf7a"                     # checked for colour-blind separation with the two above
GRAY, GRID, INK, MUTED, SURFACE = "#b4b3ad", "#e6e5e0", "#1d1d1b", "#5f5e5a", "#ffffff"
BUCKET_COLORS = dict(zip(BUCKETS, [YELLOW, AQUA, ORANGE, BLUE, GRAY]))

# Every chart is drawn twice: a desktop version (1400px wide) and a phone version (~800px wide,
# single column where needed, no text smaller than 10.5pt). The web page picks one with <picture>.
LAYOUTS = {
    "desktop": dict(phone=False, width=7.0, dpi=200, title_fs=15.5, title_wrap=46, sub_wrap=80, suffix=""),
    "phone": dict(phone=True, width=3.8, dpi=210, title_fs=14.0, title_wrap=27, sub_wrap=46, suffix="_phone"),
}
L = dict(LAYOUTS["desktop"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12.5, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
})


def fs(size: float) -> float:
    """Font size, with a floor on phone versions so nothing shrinks below ~13px on screen."""
    return max(size, 10.5) if L["phone"] else size


def game_label(g: str, r) -> str:
    if L["phone"]:  # "Wk 13 @ WAS" -> "Wk13 WAS"
        g = g.replace("Wk ", "Wk").replace(" vs ", " ").replace(" @ ", " ")
    return f"{g}  {r.result} {r.den_pts}-{r.opp_pts}"


def frame(plot_h: float, title: str, subtitle: str, nrows=1, ncols=1, **kw):
    """Figure with a left-aligned takeaway title, a subtitle and a data credit."""
    t = textwrap.fill(title, L["title_wrap"])
    s = textwrap.fill(subtitle, L["sub_wrap"])
    tl, sl = t.count("\n") + 1, s.count("\n") + 1
    th = L["title_fs"] / 72 * 1.2
    head = th * tl + 0.22 * sl + 0.45
    foot = 0.75
    h = head + plot_h + foot
    fig, axes = plt.subplots(nrows, ncols, figsize=(L["width"], h), squeeze=False, **kw)
    fig.subplots_adjust(top=1 - head / h, bottom=foot / h, left=0.08, right=0.96)
    fig.text(0.03, 1 - 0.15 / h, t, fontsize=L["title_fs"], fontweight="bold", color=NAVY, va="top",
             linespacing=1.15)
    fig.text(0.03, 1 - (0.25 + th * tl) / h, s, fontsize=10.5, color=MUTED, va="top", linespacing=1.3)
    fig.text(0.03, 0.12 / h, "Data: nflverse (nflreadpy)", fontsize=fs(9), color=MUTED)
    return fig, axes


def save(fig, name: str):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}{L['suffix']}.png", dpi=L["dpi"], facecolor=SURFACE)
    plt.close(fig)


def chart_expected(den: pd.Series):
    n = int(den.os_games)
    if L["phone"]:
        labels = ["Wins: from points", "Wins: from efficiency", f"Close wins: efficiency ({n} games)",
                  f"Close wins: coin flips ({n} games)"]
    else:
        labels = ["Season wins expected from points scored & allowed",
                  "Season wins expected from efficiency (EPA per play)",
                  f"Close wins expected for a team this efficient ({n} games)",
                  f"Close wins expected from coin flips ({n} games)"]
    vals = [(den.pyth_wins, den.wins), (den.epa_wins, den.wins), (den.exp_os_pct * n, den.os_wins),
            (0.5 * n, den.os_wins)]
    rows = [(lab, e, a) for lab, (e, a) in zip(labels, vals)]
    gap = den.wins - (den.pyth_wins + den.epa_wins) / 2
    fig, ax = frame(4.0, f"Denver won about {gap:.0f} more games than their points and efficiency predicted",
                    "Gray = expected wins, orange = actual wins. 2025 regular season (17 games).")
    ax = ax[0, 0]
    for i, (label, exp, act) in enumerate(rows[::-1]):
        ax.plot([exp, act], [i, i], color=GRAY, lw=2, solid_capstyle="round", zorder=1)
        ax.scatter(exp, i, s=110, color=GRAY, edgecolor=SURFACE, linewidth=2, zorder=2,
                   label="Expected" if i == 0 else None)
        ax.scatter(act, i, s=110, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label="Actual" if i == 0 else None)
        ax.text(exp - 0.45, i, f"{exp:.1f}", ha="right", va="center", fontsize=11, color=MUTED)
        ax.text(act + 0.45, i, f"{act:.0f}" + ("" if L["phone"] else f" (+{act - exp:.1f})"), ha="left",
                va="center", fontsize=11, color=INK, fontweight="bold")
        ax.text(0, i + 0.36, label, fontsize=10.5, color=INK, va="center")
    ax.set_yticks([])
    ax.set_xlim(0, 17.9)
    ax.set_xticks([0, 5, 10, 15] if L["phone"] else [0, 5, 10, 15, 17])
    ax.set_ylim(-0.5, len(rows) - 0.35)
    ax.set_xlabel("Wins")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", bbox_to_anchor=(1.02, 0.98), ncol=2, fontsize=fs(10), handletextpad=0.1,
              columnspacing=0.8)
    fig.subplots_adjust(left=0.04)
    save(fig, "1_expected_vs_actual")


def chart_timelines(games: pd.DataFrame, tl: pd.DataFrame):
    wins = games[games.result == "W"].reset_index(drop=True)
    below = int((wins.min_wp < 0.25).sum())
    ncols = 1 if L["phone"] else 2
    nrows = int(np.ceil(len(wins) / ncols))
    fig, axes = frame((1.35 if L["phone"] else 1.5) * nrows,
                      f"Denver's chance of winning fell below 25% in {below} of their {len(wins)} close wins",
                      "Denver's win probability (nflverse model) through each one-score win. "
                      "Orange dot = the lowest point. Gray line = 50%.",
                      nrows, ncols, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.6, wspace=0.1, left=0.12 if L["phone"] else 0.07)
    xmax = 70 if (tl.t > 3600).any() else 60
    for k, ax in enumerate(axes.flat):
        ax.grid(False)
        if k >= len(wins):
            ax.axis("off")
            continue
        g = wins.iloc[k]
        t = tl[tl.game == g.game].sort_values("t", kind="stable")
        x, y = (t.t / 60).values, (100 * t.wp).values
        ax.axhline(50, color=GRAY, lw=1, zorder=0)
        ax.fill_between(x, y, color=BLUE, alpha=0.10, lw=0)
        ax.plot(x, y, color=BLUE, lw=1.8, solid_joinstyle="round")
        i = int(np.argmin(y))
        ax.scatter(x[i], y[i], s=46, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        low = y[i] < 35
        ax.text(min(max(x[i], 8), xmax - 8), y[i] + (9 if low else -9), f"low {y[i]:.0f}%", fontsize=fs(9.5),
                ha="center", va="bottom" if low else "top", color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
        ax.set_title(f"{g.game}, {g.den_pts}-{g.opp_pts}" + (" (OT)" if g.overtime else ""),
                     loc="left", fontsize=10.5, pad=4)
        ax.set_ylim(0, 104)
        ax.set_xlim(0, xmax)
        ax.set_yticks([0, 50, 100], ["0", "50", "100%"], fontsize=fs(9))
        ax.set_xticks([0, 15, 30, 45, 60], ["Q1", "Q2", "Q3", "Q4", "OT"], fontsize=fs(9))
        ax.tick_params(labelbottom=True)
    save(fig, "2_win_probability_close_wins")


def chart_history(pairs: pd.DataFrame, big: pd.DataFrame, hs: dict, den_now: pd.Series, live: pd.Series):
    both = pairs.dropna(subset=["os_win_pct", "os_win_pct_next"])
    jit = np.random.default_rng(1).uniform(-1.2, 1.2, (2, len(both)))
    nxt = 100 * hs["big_os_pct_next"]
    fig, ax = frame(5.0 if L["phone"] else 5.6,
                    f"Close-game success barely carries over: teams with 9+ close wins won {nxt:.0f}% of close "
                    "games the next year",
                    f"Every team-season {HIST_FIRST}–{SEASON - 1}: one-score win % that season vs the next "
                    f"(n = {hs['n_pairs_os']} pairs, correlation r = {hs['r_os']:.2f}). Points are jittered slightly.")
    ax = ax[0, 0]
    ax.scatter(100 * both.os_win_pct + jit[0], 100 * both.os_win_pct_next + jit[1], s=14, color=GRAY,
               alpha=0.45, lw=0, label="All teams")
    ax.scatter(100 * big.os_win_pct, 100 * big.os_win_pct_next, s=70, color=BLUE, edgecolor=SURFACE,
               linewidth=2, zorder=3, label=f"9+ close wins ({hs['n_big']} teams)")
    for r in big[big.os_wins >= 11].itertuples():
        ax.annotate(f"{r.team} {r.season}", (100 * r.os_win_pct, 100 * r.os_win_pct_next),
                    xytext=(-8, -6 if L["phone"] else 0), textcoords="offset points", ha="right",
                    va="top" if L["phone"] else "center", fontsize=fs(10))
    x, y = 100 * den_now.os_win_pct, 100 * live.os_win_pct
    ax.scatter(x, y, s=130, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=4,
               label=f"Denver 2025 → {LIVE} so far")
    ax.annotate(f"DEN 2025 → {LIVE}: {int(live.os_wins)}-{int(live.os_games - live.os_wins)} in\n"
                f"close games ({record(live.wins, int(live.games))} overall)", (x, y), xytext=(-12, -2),
                textcoords="offset points", ha="right", va="top", fontsize=fs(10), fontweight="bold")
    ax.axhline(50, color=MUTED, lw=1)
    ax.plot([70, 101], [nxt] * 2, color=BLUE, lw=2, label=f"9+ teams, next year: {nxt:.0f}%" if L["phone"] else None)
    if not L["phone"]:
        ax.text(69, nxt, f"9+ teams, next year: {nxt:.0f}%", ha="right", va="center", fontsize=10, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.9))
    ax.set_xlim(-4, 104)
    ax.set_ylim(-4, 108)
    ticks, tl_ = [0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"]
    if L["phone"]:
        ticks, tl_ = [0, 50, 100], ["0%", "50%", "100%"]
    ax.set_xticks(ticks, tl_)
    ax.set_yticks(ticks, tl_)
    ax.set_xlabel("One-score win % this season")
    ax.set_ylabel("Next season")
    if L["phone"]:
        fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.02, 0.045), ncol=1,
                   fontsize=fs(10), handletextpad=0.4, handlelength=1.2)
        fig.subplots_adjust(left=0.2, bottom=fig.subplotpars.bottom + 0.16)
    else:
        ax.legend(loc="upper left", fontsize=10, markerscale=0.9, handletextpad=0.2)
        fig.subplots_adjust(left=0.13)
    save(fig, "3_history_close_wins_next_season")


def chart_decided(games: pd.DataFrame, bk: pd.DataFrame, q4: float, lost: float):
    order = games.game.tolist()[::-1]
    b = bk.loc[order]
    fig, ax = frame(6.6, f"Fourth quarters won the close wins (+{q4:.0f} EPA); turnovers and kicks "
                         f"cost Denver {abs(lost):.0f}",
                    "Denver's expected points added (EPA) in each one-score game, split into five parts that "
                    "add up to the net (black dot). Right of zero helped Denver.")
    ax = ax[0, 0]
    y = np.arange(len(b))
    for sign in (1, -1):
        left = np.zeros(len(b))
        for col in BUCKETS:
            v = b[col].values
            v = np.where(np.sign(v) == sign, v, 0)
            ax.barh(y, v, left=left, height=0.62, color=BUCKET_COLORS[col], edgecolor=SURFACE,
                    linewidth=1.5, label=col if sign == 1 else None)
            left += v
    ax.scatter(b.sum(axis=1), y, s=34, color=INK, zorder=4, edgecolor=SURFACE, linewidth=1.2, label="Net EPA")
    res = games.set_index("game").loc[order]
    ax.set_yticks(y, [game_label(g, r) for g, r in res.iterrows()], fontsize=10.5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("EPA for Denver (≈ points)")
    ax.grid(axis="y", visible=False)
    legend_h = 0.11 if L["phone"] else 0.065
    fig.subplots_adjust(left=0.42 if L["phone"] else 0.34, top=fig.subplotpars.top - legend_h)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left",
               bbox_to_anchor=(0.02, fig.subplotpars.top + legend_h), ncol=2 if L["phone"] else 3,
               fontsize=fs(9.5), handlelength=1, columnspacing=0.8, handletextpad=0.4)
    save(fig, "4_what_decided_each_close_game")


def chart_luck(g: pd.DataFrame):
    order = g.game.tolist()[::-1]
    d = g.set_index("game").loc[order]
    v = 100 * d.luck_total
    best = v[d.result == "W"].max()
    fig, ax = frame(5.8, "No close win hinged on lucky bounces: chance plays never added more than "
                         f"{best:+.0f} points of win probability",
                    "Win probability Denver gained (orange) or lost (blue) on chance plays (fumble recoveries, "
                    "opponents' kicks, interception-worthy throws, 4th-down tries) compared with their "
                    "normal rates. 100 points is about 1 win.")
    ax = ax[0, 0]
    y = np.arange(len(d))
    ax.barh(y, v, height=0.55, color=np.where(v >= 0, ORANGE, BLUE), edgecolor=SURFACE, linewidth=1.5)
    for yi, val in zip(y, v):
        ax.text(val + (2 if val >= 0 else -2), yi, f"{val:+.0f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=fs(10), color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    for x in (50, -50):
        ax.axvline(x, color=GRAY, lw=1.2)
    ax.text(52, -0.4, "Decides\na win" if L["phone"] else "Enough luck to\nhave decided\na win",
            fontsize=fs(9.5), color=MUTED, va="bottom")
    if L["phone"]:
        ax.text(-78, -0.4, "…or a\nloss", fontsize=fs(9.5), color=MUTED, va="bottom", ha="left")
    else:
        ax.text(-52, -0.4, "…or a\nloss", fontsize=fs(9.5), color=MUTED, va="bottom", ha="right")
    ax.set_yticks(y, [game_label(gm, r) for gm, r in d.iterrows()], fontsize=10.5)
    ax.set_xlim(-80, 108 if L["phone"] else 85)
    if L["phone"]:
        ax.set_xticks([-50, 0, 50])
    ax.set_xlabel("WP points vs expected" if L["phone"] else "Win-probability points: actual minus expected")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.42 if L["phone"] else 0.34)
    save(fig, "5_luck_per_close_game")


def chart_repeat(sig: pd.DataFrame):
    s = sig.sort_values("year_to_year_r")
    fig, ax = frame(5.6 if L["phone"] else 4.8, "Only overall efficiency carries over well from one season to the next",
                    f"How well each team stat in one season predicts the same stat the next season "
                    f"(correlation r, {FIT.start}–{FIT.stop - 1}, n = {int(s.n_pairs.iloc[0])} pairs). "
                    "Right: Denver's 2025 rank out of 32.")
    ax = ax[0, 0]
    y = np.arange(len(s))
    ax.barh(y, s.year_to_year_r, height=0.55, color=[ORANGE if m == "net_epa" else BLUE for m in s.metric])
    for yi, r in zip(y, s.itertuples()):
        ax.text(r.year_to_year_r + 0.01, yi, f"{r.year_to_year_r:.2f}", va="center", fontsize=fs(10))
        ax.text(0.66, yi, f"#{r.den_rank_2025}", va="center", ha="right", fontsize=10.5, fontweight="bold")
    ax.text(0.66, len(s) - 0.4, "DEN rank", ha="right", fontsize=fs(9.5), color=MUTED)
    ax.set_yticks(y, [textwrap.fill(lbl.split(" (")[0], 16 if L["phone"] else 28) for lbl in s.label],
                  fontsize=10.5 if L["phone"] else 10)
    ax.set_xlim(0, 0.67)
    ax.set_xticks([0, 0.2, 0.4])
    ax.set_xlabel("Year-to-year r" if L["phone"] else "Year-to-year correlation r")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.46 if L["phone"] else 0.38)
    save(fig, "6_what_repeats")


VERDICT_COLS = [("Points to SKILL", BLUE), ("Points to LUCK", ORANGE)]
LUCK_NOTE = "“Luck” here means results that don’t repeat from one season to the next."


def chart_verdict(verdict: str, skill: list[str], luck: list[str]):
    phone = L["phone"]
    wrap = 40 if phone else 33
    vt = textwrap.fill(verdict, 27 if phone else 48)
    vfs = 14.5 if phone else 16
    note = textwrap.fill(LUCK_NOTE, 46 if phone else 80)
    line = 0.19

    def col_h(items):
        return sum((line * (textwrap.fill(i, wrap).count("\n") + 1) + 0.2) for i in items) + 0.45

    head = 0.55 + vfs / 72 * 1.2 * (vt.count("\n") + 1) + 0.25 * (note.count("\n") + 1) + 0.55
    body = col_h(skill) + col_h(luck) if phone else max(col_h(skill), col_h(luck))
    h = head + body + 0.5
    fig = plt.figure(figsize=(L["width"], h))
    fig.text(0.04, 1 - 0.25 / h, "VERDICT", fontsize=11, color=MUTED, va="top", fontweight="bold")
    fig.text(0.04, 1 - 0.55 / h, vt, fontsize=vfs, fontweight="bold", color=NAVY, va="top", linespacing=1.2)
    fig.text(0.04, 1 - (head - 0.55) / h, note, fontsize=fs(10), color=MUTED, va="bottom", style="italic")
    top = 1 - head / h
    for k, ((title, color), items) in enumerate(zip(VERDICT_COLS, (skill, luck))):
        x, w = (0.04, 0.92) if phone else (0.04 + 0.48 * k, 0.44)
        if phone and k == 1:
            top -= col_h(skill) / h
        fig.patches.append(Rectangle((x, top + 0.02 / h), w, 0.05 / h, color=color, transform=fig.transFigure))
        fig.text(x, top + 0.12 / h, title, fontsize=13, fontweight="bold", color=INK)
        yy = top - 0.12 / h
        for it in items:
            txt = textwrap.fill(it, wrap)
            fig.text(x, yy, "●", fontsize=8, color=color, va="top")
            fig.text(x + (0.05 if phone else 0.03), yy + 0.02 / h, txt, fontsize=10.5, color=INK, va="top",
                     linespacing=1.3)
            yy -= (line * (txt.count("\n") + 1) + 0.2) / h
    fig.text(0.04, 0.2 / h, "Data: nflverse (nflreadpy)." + ("" if phone else " Every number is computed in analysis.py."),
             fontsize=fs(9), color=MUTED)
    save(fig, "7_verdict")


def chart_card(verdict: str, tiles: list[tuple[str, str]]):
    """1200x630 link-preview image (Open Graph / Twitter) built from the verdict."""
    fig = plt.figure(figsize=(6.0, 3.15))
    fig.patches.append(Rectangle((0, 0.965), 1, 0.035, color=ORANGE, transform=fig.transFigure))
    fig.text(0.05, 0.89, "BRONCOS 2025: LUCK OR SKILL?", fontsize=10, color=ORANGE, fontweight="bold", va="top")
    fig.text(0.05, 0.80, textwrap.fill(verdict, 52), fontsize=12.5, color=NAVY, fontweight="bold", va="top",
             linespacing=1.2)
    for k, (v, lab) in enumerate(tiles):
        x = 0.05 + k * 0.31
        fig.patches.append(Rectangle((x, 0.13), 0.28, 0.27, color="#f6f5f2", transform=fig.transFigure))
        fig.text(x + 0.02, 0.36, v, fontsize=15, color=NAVY, fontweight="bold", va="top")
        fig.text(x + 0.02, 0.235, textwrap.fill(lab, 24), fontsize=7.5, color=MUTED, va="top", linespacing=1.2)
    fig.text(0.05, 0.045, "Data: nflverse · Python analysis", fontsize=7, color=MUTED)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "7_verdict_card.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Write-up helpers
# ---------------------------------------------------------------------------
def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def one_in(p: float) -> str:
    return f"about 1 in {round(1 / p):,}" if p > 0 else "effectively zero"


def fmt_metric(metric: str, v: float) -> str:
    if metric in ("fg_over_exp", "opp_fg_over_exp"):
        return f"{100 * v:+.1f} pts"
    if metric in ("fumble_recovery_rate", "os_win_pct"):
        return pct(v)
    if metric == "to_margin_pg":
        return f"{v:+.2f}"
    return f"{v:+.3f}"


def md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False))


def html_table(df: pd.DataFrame, num_from: int = 1) -> str:
    th = "".join(f"<th{' class=num' if i >= num_from else ''}>{c}</th>" for i, c in enumerate(df.columns))
    rows = "".join("<tr>" + "".join(f"<td{' class=num' if i >= num_from else ''}>{x}</td>"
                                    for i, x in enumerate(r)) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div>'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download all data")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    print("Loading nflverse data ...")
    sched, pbp, ftn = load_all(args.refresh)
    tg = team_games(sched)
    ts = team_seasons(tg)
    epa = epa_team_seasons(pbp)

    # 0. Check the figures we started from
    den_all = tg[tg.team == TEAM]
    reg = den_all[(den_all.season == SEASON) & (den_all.game_type == "REG")]
    post = den_all[(den_all.season == SEASON) & (den_all.game_type != "REG")]
    live = den_all[(den_all.season == LIVE) & (den_all.game_type == "REG")]

    def playoff(opp):
        r = post[post.opp == opp].iloc[0]
        return f"{'W' if r.win == 1 else 'L'} {int(max(r.pf, r.pa))}-{int(min(r.pf, r.pa))}"

    checks = pd.DataFrame([
        ("2025 regular-season record", "14-3", record(reg.win.sum(), len(reg))),
        ("2025 one-score record (8 points or fewer)", "11-2", record(reg[reg.one_score].win.sum(), int(reg.one_score.sum()))),
        ("2025 point differential", "+90", f"{int(reg.margin.sum()):+d}"),
        ("Divisional round vs Buffalo", "W 33-30", playoff("BUF")),
        ("AFC Championship vs New England", "L 10-7", playoff("NE")),
        (f"{LIVE} record so far", "4-2", record(live.win.sum(), len(live))),
        (f"{LIVE} one-score record so far", "4-0", record(live[live.one_score].win.sum(), int(live.one_score.sum()))),
    ], columns=["Figure", "Stated", "From the data"])
    checks["Match"] = np.where(checks.Stated == checks["From the data"], "yes", "NO")
    print(checks.to_string(index=False))

    # 1. Expected wins
    m, cur, models = expected_wins(ts, epa)
    den = cur[cur.team == TEAM].iloc[0]
    pyth_check = pyth_exponent_check(ts, den)
    rank_net = int((cur.net_epa > den.net_epa).sum()) + 1
    rank_over = int((cur.wins_over_epa > den.wins_over_epa).sum()) + 1
    p_hat = float(den.exp_os_pct)
    n_os, k_os = int(den.os_games), int(den.os_wins)
    binom = {"coin": binom_tail(k_os, n_os, 0.5), "expected": binom_tail(k_os, n_os, p_hat), "p_hat": p_hat}

    # 2. History
    pairs, big, hs = history(ts)

    # 3. The close games
    games, tl, bk, fourth = close_games(tg, pbp, SEASON)
    po_games = close_games(tg, pbp, SEASON, "POST")[0]
    win_ids = games.result == "W"
    tot = bk.loc[games[win_ids].game].sum()
    q4_total = float(tot["Q4/OT offence"] + tot["Q4/OT defence"])
    lost_total = float(tot["Turnovers"] + tot["Special teams & kicks"])
    games["biggest_help"] = [f"{bk.loc[g].idxmax()} ({bk.loc[g].max():+.1f})" for g in games.game]

    # 4. Skill signals
    fg = fg_expected(pbp)
    sm, sig = skill_signals(pbp, ts, epa, fg)
    sg = sig.set_index("metric")

    # 5. Luck
    print("Training the win-probability stand-in model ...")
    wp_model, wp_stats = train_wp_model(pbp)
    chance, rates = chance_plays(pbp, ftn, fg, wp_model)
    g_luck, luck_by_season, luck, top_luck, all_luck = luck_tables(chance, games, tg)
    games = games.merge(g_luck[["game_id", "luck_total", "luck_adj_win_chance", "decided_by_luck"]], on="game_id")

    # 6. Denver across seasons + live check
    seasons = denver_seasons(m)
    lv = seasons[seasons.season == LIVE].iloc[0]
    live_ts = ts[(ts.team == TEAM) & (ts.season == LIVE)].iloc[0]
    games26, tl26, bk26, _ = close_games(tg, pbp, LIVE)
    g26, _, luck26, _, _ = luck_tables(chance, games26, tg)
    s3 = seasons[seasons.season >= 2024]
    ps = np.concatenate([[p] * int(n) for p, n in zip(s3.exp_os_pct, s3.os_games)])
    span = {"os_wins": float(s3.os_wins.sum()), "os_games": int(s3.os_games.sum()),
            "exp": float(s3.exp_os_wins.sum()),
            "p_at_least": poisson_binomial_tail(int(s3.os_wins.sum()), ps)}
    live_p = {"p_all_close_wins": float(lv.exp_os_pct ** int(lv.os_games)) if lv.os_wins == lv.os_games
              else binom_tail(int(lv.os_wins), int(lv.os_games), float(lv.exp_os_pct))}
    # Data cut-off: the last league week with every game played (not Denver's last game date)
    wk = sched[(sched.season == LIVE) & (sched.game_type == "REG")].groupby("week")
    done = wk.home_score.apply(lambda s: s.notna().all())
    last_week = int(done[done].index.max())
    last_game = pd.Timestamp(wk.gameday.max()[last_week])
    pulled = pd.Timestamp((RAW / "schedules.parquet").stat().st_mtime, unit="s")
    cutoff = (f"{LIVE} season through week {last_week} (last game {last_game:%d %b %Y}); "
              f"data pulled {pulled:%d %b %Y}")

    # ---- Charts
    s24 = seasons[seasons.season == 2024].iloc[0]
    verdict = (f"A good team that won more close games than it can expect to repeat: about {den.epa_wins:.0f} "
               f"wins of quality, plus about {den.wins - den.epa_wins:.0f} from late-game results that history "
               "says don't carry over.")
    skill = [
        f"Top-{rank_net} efficiency (net EPA per play {den.net_epa:+.3f}): good enough for about "
        f"{den.epa_wins:.0f} wins, and to expect {pct(p_hat)} of close games, not 50%.",
        f"Elite 4th-quarter defence: {sg.loc['q4_def_epa', 'den_2025']:+.3f} EPA per play allowed, "
        f"#{sg.loc['q4_def_epa', 'den_rank_2025']} in the NFL.",
        f"Not lucky bounces: chance plays cost Denver {abs(luck['den_luck_2025']):.1f} wins over the season "
        f"(#{luck['den_luck_rank_2025']} of 32) and decided {word(luck['n_decided_by_luck']).replace('zero', 'none')} of the {int(win_ids.sum())} close wins.",
        f"Still winning close games in {LIVE}: {lv.os_record} so far, {lv.record} overall (though chance plays helped by "
        f"{luck26['den_luck_close_games']:+.1f} wins this time).",
    ]
    luck_pts = [
        f"{record(den.os_wins, n_os)} in one-score games is "
        f"{den.os_wins_over_exp:.1f} more close wins than a team this efficient would expect; "
        f"no team since {HIST_FIRST} has won more one-score games.",
        f"Chance of {k_os}-{n_os - k_os} or better: {pct(binom['coin'], 1)} if each game were a coin flip, "
        f"{pct(binom['expected'], 1)} at Denver's expected rate.",
        f"Close-game win % barely repeats (r = {hs['r_os']:.2f}). The {hs['n_big']} teams with 9+ close wins "
        f"fell from {pct(hs['big_os_pct'])} to {pct(hs['big_os_pct_next'])} the next year.",
        f"Same coach and QB went {s24.os_record} in close games in 2024 at similar efficiency "
        f"({s24.net_epa:+.3f} vs {den.net_epa:+.3f}).",
        f"Even the late-game edges barely repeat: 4th-quarter defence r = {sg.loc['q4_def_epa', 'year_to_year_r']:.2f}, "
        f"QB late & close r = {sg.loc['late_close_dropback_epa', 'year_to_year_r']:.2f}.",
    ]
    print("Drawing charts ...")
    for layout in LAYOUTS.values():
        L.update(layout)
        chart_expected(den)
        chart_timelines(games, tl)
        chart_history(pairs, big, hs, ts[(ts.team == TEAM) & (ts.season == SEASON)].iloc[0], live_ts)
        chart_decided(games, bk, q4_total, lost_total)
        chart_luck(g_luck)
        chart_repeat(sig)
        chart_verdict(verdict, skill, luck_pts)
    L.update(LAYOUTS["desktop"])
    chart_card(verdict, [
        (record(den.wins, int(den.games)), "2025 regular-season record"),
        (f"{k_os}-{n_os - k_os}", "in one-score games (8 points or fewer)"),
        (f"{den.epa_wins:.1f}", "wins expected from efficiency"),
    ])

    # ---- Tables and findings
    games_out = games.assign(
        lowest_win_chance=lambda d: (100 * d.min_wp).round(0).astype(int).astype(str) + "%",
        chance_play_luck_pts=lambda d: (100 * d.luck_total).round(0).astype(int),
        luck_adj_win_chance_pct=lambda d: (100 * d.luck_adj_win_chance).round(0).astype(int))
    games_out.drop(columns=["timeline"], errors="ignore").to_csv(OUT / "close_games_2025.csv", index=False)
    po_games.to_csv(OUT / "close_games_2025_playoffs.csv", index=False)
    bk.round(2).to_csv(OUT / "close_games_epa_split_2025.csv")
    fourth.to_csv(OUT / "fourth_downs_close_games_2025.csv", index=False)
    big.to_csv(OUT / "history_9plus_one_score_wins.csv", index=False)
    cur.round(3).to_csv(OUT / "expected_wins_2025_all_teams.csv", index=False)
    sig.round(4).to_csv(OUT / "skill_vs_luck_signals.csv", index=False)
    seasons.round(3).to_csv(OUT / "denver_by_season.csv", index=False)
    luck_by_season.round(3).to_csv(OUT / "chance_play_luck_by_team_season.csv", index=False)
    top_luck.drop(columns=["team"]).round(3).to_csv(OUT / "biggest_chance_plays_close_games_2025.csv", index=False)
    g26.to_csv(OUT / f"close_games_{LIVE}.csv", index=False)
    checks.to_csv(OUT / "figure_checks.csv", index=False)

    F = {
        "cutoff": cutoff, "last_week": last_week, "last_game": f"{last_game:%Y-%m-%d}",
        "data_pulled": f"{pulled:%Y-%m-%d}", "verdict": verdict,
        "checks": checks.to_dict("records"),
        "denver_2025": {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                        for k, v in den.items()},
        "net_epa_rank": rank_net, "wins_over_epa_rank": rank_over, "models": models, "pyth_check": pyth_check,
        "binomial": binom, "history": {k: v for k, v in hs.items()},
        "q4_total_epa_close_wins": q4_total, "to_st_total_epa_close_wins": lost_total,
        "signals": sig.to_dict("records"), "wp_model": wp_stats, "chance_rates": rates,
        "luck": luck, "luck_2026_close_games": luck26, "span_2024_2026": span, "live": live_p,
        "denver_by_season": seasons.to_dict("records"),
    }
    (OUT / "findings.json").write_text(json.dumps(F, indent=2, default=float))

    S = build_sections(F, den, games_out, sig, seasons, hs, binom, luck, luck26, span, live_p, lv, verdict,
                       skill, luck_pts, po_games, q4_total, lost_total, big, wp_stats, rates, fourth, bk, top_luck, g26)
    method = method_md(wp_stats, pyth_check, float(den.pyth_wins))
    write_readme(S, F, method)
    write_page(S, F, den, binom, method)
    print(f"Done. Figures in {FIG}, tables in {OUT}, page in {SITE}")


# ---------------------------------------------------------------------------
# Write-up: one set of sections, rendered to the README and to the web page
# ---------------------------------------------------------------------------
import re           # noqa: E402
import html as _h   # noqa: E402

REPO_URL = "https://github.com/INSUL4RITY/sports-analytics"
PAGE_URL = "https://insul4rity.github.io/sports-analytics/broncos-luck-or-skill/"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def word(n: int) -> str:
    return ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen"][n] if 0 <= n <= 13 else str(n)


def build_sections(F, den, games, sig, seasons, hs, binom, luck, luck26, span, live_p, lv, verdict,
                   skill, luck_pts, po_games, q4_total, lost_total, big, wp_stats, rates, fourth, bk, top_luck, g26):
    S = []
    mo = F["models"]
    pc = F["pyth_check"]
    n_os, k_os = int(den.os_games), int(den.os_wins)
    wins = games[games.result == "W"]
    sg = sig.set_index("metric")

    S.append(dict(id="verdict", title="The short answer", paras=[
        f"**{verdict}**",
        f"Denver were a good team in 2025: {ordinal(F['net_epa_rank'])} of 32 in efficiency. But winning "
        f"{k_os} of {n_os} close games was well beyond what that efficiency predicts, and history shows "
        "close-game records don't carry over. The extra wins did not come from lucky bounces like fumbles or "
        "missed kicks. They came from winning fourth quarters, which also tends not to repeat from one season "
        "to the next.",
        "**What “luck” means here:** results that don't repeat from one season to the next. It doesn't "
        "mean the players did nothing right; winning a fourth quarter takes real plays. Section 6 measures the "
        "narrowest kind of luck, chance plays such as fumble recoveries and missed kicks, and those actually "
        "went against Denver.",
    ], img=("7_verdict.png", "Verdict panel listing the evidence for skill and for luck")))

    chk = pd.DataFrame(F["checks"])
    S.append(dict(id="checks", title="0. Checking the starting figures", paras=[
        "Before any analysis, every figure in the question was recomputed from nflverse schedules. "
        "A one-score game is one decided by 8 points or fewer (regular season; playoffs reported separately; "
        "a tie would count as half a win)."
        + (" All figures match." if (chk.Match == "yes").all() else " **Some figures differ: the data's values are used.**"),
    ], table=chk))

    S.append(dict(id="expected", title=f"1. Denver won about {den.wins - den.epa_wins:.0f} more games than expected", paras=[
        f"There are two standard ways to estimate how many games a team should have won. **From points**, "
        f"using the Pythagorean formula (points scored^2.37 ÷ (scored^2.37 + allowed^2.37)): a team that scores "
        f"{den.pf:.0f} and allows {den.pa:.0f} usually wins **{den.pyth_wins:.1f}** of 17. "
        f"**From efficiency**: a straight-line fit of win % on net EPA per play (offence minus defence) across "
        f"{mo['epa_win_model']['n']} team-seasons from 2010 to 2024 (R² = {mo['epa_win_model']['r2']:.2f}) "
        f"gives **{den.epa_wins:.1f}**.",
        f"Denver won {den.wins:.0f}, which is **{den.wins - den.pyth_wins:+.1f}** above the points estimate and "
        f"**{den.wins - den.epa_wins:+.1f}** above the efficiency estimate. That was the "
        f"{'' if F['wins_over_epa_rank'] == 1 else ordinal(F['wins_over_epa_rank']) + '-'}largest gap in the league in 2025. Their efficiency ranked "
        f"{ordinal(F['net_epa_rank'])}, so the team was good; the record just ran ahead of how they played.",
        f"**Checking the {PYTH_EXP}.** {PYTH_EXP} is the standard NFL exponent (from Daryl Morey's original work), "
        f"not one fitted here. Testing every exponent from 1.5 to 3.5 on the same {pc['n']} team-seasons, the best "
        f"fit is **{pc['best']:.2f}** (typical error {pc['rmse_best']:.2f} wins per season, vs {pc['rmse_std']:.2f} "
        f"for {PYTH_EXP}). With {pc['best']:.2f}, Denver's expected wins rise to {pc['den_best']:.1f}, so they still "
        f"won about {den.wins - pc['den_best']:.0f} more than their points predicted. Any exponent from 2 to 3 gives "
        f"{pc['den_2']:.1f}–{pc['den_3']:.1f} expected wins, so the finding doesn't depend on this choice.",
        f"**Close games.** Fitting one-score win % on net EPA per play (same team-seasons, "
        f"{mo['epa_os_model']['games']:,} one-score games, weighted by games played) shows that a team as "
        f"efficient as Denver typically wins **{pct(binom['p_hat'])}** of close games. That is better than 50%, "
        f"but efficiency explains only a small part of close results (R² = {mo['epa_os_model']['r2']:.2f}). "
        f"That means {binom['p_hat'] * n_os:.1f} expected close wins out of {n_os}. Denver won {k_os}: "
        f"**{den.os_wins_over_exp:+.1f} above expectation**.",
        "*How to read the chart:* each row is one estimate. The gray dot is what was expected, the orange dot "
        "is what happened, and the gap between them is what efficiency can't explain.",
    ], img=("1_expected_vs_actual.png", "Dumbbell chart of expected versus actual wins")))

    prev11 = [r for r in hs["list_11plus"] if not (r["team"] == TEAM and r["season"] == SEASON)]
    nxt = []
    for r in prev11:
        b = big[(big.team == r["team"]) & (big.season == r["season"])].iloc[0]
        nxt.append(f"{r['team']} {r['season']} ({int(b.os_wins)}-{int(b.os_games - b.os_wins)} in close games) "
                   f"went {int(b.os_wins_next)}-{int(b.os_games_next - b.os_wins_next)} in close games and "
                   f"{record(b.wins_next, int(b.games_next))} overall the next year")
    S.append(dict(id="binomial", title=f"2. How unlikely is {k_os}-{n_os - k_os}?", paras=[
        f"If every one-score game were a 50/50 coin flip, the chance of winning {k_os} or more of {n_os} is "
        f"**{pct(binom['coin'], 1)}** ({one_in(binom['coin'])}). At Denver's efficiency-based rate of "
        f"{pct(binom['p_hat'])} per game, it is **{pct(binom['expected'], 1)}** ({one_in(binom['expected'])}).",
        f"That is rare for any one team, but with 32 teams every season, someone usually ends up with an unusual "
        f"close-game record. Since {HIST_FIRST}, {hs['n_11plus']} of {hs['n_team_seasons']} team-seasons won 11 or more "
        f"one-score games. Denver 2025 is one of them, and nobody won more. The others: " + "; ".join(nxt) + ".",
        "In plain words: even for a good team, 11-2 in close games is unusual. That is the first sign that "
        "something other than skill was involved.",
    ]))

    bt = big.assign(
        Team=big.team, Season=big.season,
        **{"Close games": [record(w, int(g)) for w, g in zip(big.os_wins, big.os_games)],
           "Record": [record(w, int(g)) for w, g in zip(big.wins, big.games)],
           "Next yr close": [record(w, int(g)) for w, g in zip(big.os_wins_next, big.os_games_next)],
           "Next yr record": [record(w, int(g)) for w, g in zip(big.wins_next, big.games_next)]},
    )[["Team", "Season", "Close games", "Record", "Next yr close", "Next yr record"]]
    S.append(dict(id="history", title="3. Close-game records don't repeat", paras=[
        f"{hs['n_big']} teams since {HIST_FIRST} won 9 or more one-score games in a season. Together they won "
        f"{pct(hs['big_os_pct'])} of their close games. The next season the same teams won "
        f"**{pct(hs['big_os_pct_next'])}**, slightly worse than a coin flip. Their wins fell from "
        f"{hs['big_wins']:.1f} to {hs['big_wins_next']:.1f} per 17 games, and {hs['big_fewer_wins_next']} of "
        f"{hs['n_big']} won fewer games.",
        f"Across all {hs['n_pairs_os']} pairs of consecutive team-seasons, one season's one-score win % predicts the next "
        f"with a correlation of just **r = {hs['r_os']:.2f}** (0 means no relationship, 1 means it repeats exactly). "
        f"Overall win % carries over much more (r = {hs['r_win']:.2f}).",
        "*How to read the chart:* each gray dot is one team-season. If close-game records repeated, the dots "
        "would rise from bottom-left to top-right. Instead they form a flat cloud around 50%. Denver's 2026 "
        "point (orange) is based on only a few games so far.",
    ], img=("3_history_close_wins_next_season.png", "Scatter of one-score win % this season vs next season"),
        table=bt))

    below = wins[wins.min_wp < 0.25]
    below10 = wins[wins.min_wp < 0.10]
    gt = games.assign(**{
        "Game": games.game, "Result": [f"{r} {a}-{b}" for r, a, b in zip(games.result, games.den_pts, games.opp_pts)],
        "Low win %": games.lowest_win_chance, "TO margin": games.to_margin.map(lambda x: f"{x:+d}" if x else "0"),
        "Q4 off EPA": games.q4_off_epa.map(lambda x: f"{x:+.1f}"),
        "Q4 def EPA": games.q4_def_epa.map(lambda x: f"{x:+.1f}"),
        "Luck (WP pts)": games.chance_play_luck_pts.map(lambda x: f"{x:+d}"),
    })[["Game", "Result", "Low win %", "TO margin", "Q4 off EPA", "Q4 def EPA", "Luck (WP pts)"]]
    po_txt = "; ".join(f"{r.game}: {r.result} {r.den_pts}-{r.opp_pts}{' (OT)' if r.overtime else ''}, "
                       f"lowest win chance {pct(r.min_wp)}, turnover margin {r.to_margin:+d}"
                       for r in po_games.itertuples())
    S.append(dict(id="games", title=f"4. The {len(wins)} close wins, one by one", paras=[
        "Win probability is nflverse's play-by-play estimate of each team's chance of winning, based on the "
        f"score, time left, field position, down and distance. In **{len(below)} of the {len(wins)} close wins** "
        f"Denver's chance fell below 25% at some point, and in {len(below10)} it fell below 10% "
        f"({', '.join(f'{g} at {pct(w)}' for g, w in zip(below10.game, below10.min_wp))}).",
        "*How to read the chart:* the line is Denver's chance of winning from kick-off to the final whistle. "
        "A line that dips far below the gray 50% mark and still finishes at 100% is a comeback.",
        f"Playoffs (also one-score games, shown separately): {po_txt}.",
    ], img=("2_win_probability_close_wins.png", "Small multiples of Denver's win probability in each close win"),
        table=gt))

    btot = bk.loc[wins.game].sum()
    go = fourth
    S.append(dict(id="decided", title="5. What decided the close games: fourth quarters, not takeaways", paras=[
        "EPA (expected points added) measures how much each play changed the number of points a team can "
        "expect to score. Each play goes into exactly one of five buckets: turnovers first, then special teams "
        "and kicks, then fourth-quarter/overtime offence and defence, then everything else. So the five parts "
        "add up to Denver's net EPA for the game.",
        f"Across the {len(wins)} close wins: Q4/OT defence **{btot['Q4/OT defence']:+.1f}**, Q4/OT offence "
        f"**{btot['Q4/OT offence']:+.1f}**, Q1–Q3 plays {btot['Q1–Q3 plays']:+.1f}, turnovers "
        f"{btot['Turnovers']:+.1f}, special teams and kicks {btot['Special teams & kicks']:+.1f}. Denver didn't win "
        "these games with takeaways or kicking. They won them late: the defence got stops and the offence "
        "made plays when it counted.",
        f"4th downs: across all 13 one-score games Denver went for it {len(go)} times and converted "
        f"{int(go.converted.sum())}. Together those attempts changed Denver's win probability by "
        f"{go.den_wpa_pts.sum():+.0f} points (about {abs(go.den_wpa_pts.sum()) / 100:.1f} wins)"
        + (", so going for it was a modest plus." if go.den_wpa_pts.sum() > 20 else
           (", so it cost them." if go.den_wpa_pts.sum() < -20 else ", so it was not a big factor either way.")),
        "*How to read the chart:* each row is one game. Coloured pieces to the right of zero helped Denver and "
        "pieces to the left hurt them. The black dot is the net.",
    ], img=("4_what_decided_each_close_game.png", "Stacked bar chart of EPA by phase for each close game")))

    by_type = luck["den_luck_by_type_2025"]
    sim = luck["simulation"]
    kick = top_luck[top_luck.type == "Opponent kick"]
    example = ""
    if len(kick):
        k = kick.iloc[0]
        example = (f"Example: in week {int(k.week)}, an opponent's field goal (\"{str(k.desc).split(', Center')[0]}\") was a make "
                   f"{pct(1 - k.q)} of the time at that distance. It {'missed' if k.good else 'went in'}. The swing "
                   f"between a make and a miss was {100 * k.swing:.0f} points of win probability, so Denver's luck on "
                   f"that one play was {100 * k.luck:+.0f} points.")
    S.append(dict(id="luck", title="6. High-leverage luck: did the bounces go Denver's way? No.", paras=[
        "Some plays are mostly chance. This measure covers four kinds: **fumble recoveries** (who falls on the "
        "ball), **opponents' field goals and extra points** (compared with the league make rate at that distance), "
        "**interception-worthy throws** (FTN charting flags throws that should have been picked off; which of "
        "them actually are is largely chance), and **4th-down tries** (compared with the league conversion rate "
        "at that distance).",
        "For each play: **luck = actual change in win probability − expected change**. The expected change "
        "weights the good and bad outcomes by how often each normally happens. 100 points of win probability "
        "is about one win. " + example,
        f"**Season total: {luck['den_luck_2025']:+.2f} wins** from chance plays, "
        f"{ordinal(luck['den_luck_rank_2025'])} of 32 (the spread between teams is about "
        f"±{luck['league_luck_sd_2025']:.2f} wins). By type: fumbles {by_type.get('Fumble recovery', 0):+.2f}, "
        f"opponents' kicks {by_type.get('Opponent kick', 0):+.2f}, interception-worthy throws "
        f"{by_type.get('Interception luck', 0):+.2f}, 4th downs {by_type.get('4th-down conversion', 0):+.2f}. "
        f"In the 13 close games combined: {100 * luck['den_luck_close_games']:+.0f} points. "
        f"**{word(luck['n_decided_by_luck']).replace('zero', 'none').capitalize()} of the {len(wins)} close wins** would flip to a likely "
        "loss if the chance plays had gone at normal rates.",
        f"This kind of luck does not repeat. Across all teams, one season's chance-play luck predicts the next with "
        f"r = {luck['luck_year_to_year_r']:.2f} (2022–2025, n = {luck['luck_pairs']} pairs).",
        f"**Simulation.** Replaying the 13 close games {N_SIMS:,} times with every chance play redrawn at its "
        f"normal rate, and everything else kept the same, gives an average of **{sim['expected_wins']:.1f} close "
        f"wins** (middle 80% of replays: {sim['p10']:.0f} to {sim['p90']:.0f}). {k_os} or more happens in "
        f"{pct(sim['p_11_or_more'])} of replays. The average is a little below {k_os} even though chance plays went "
        "slightly against Denver. That's because a game can't be more than won: when replayed bounces go "
        "Denver's way in a game they already won, nothing changes, but when they go the other way a win can "
        "turn into a loss.",
        "*How to read the chart:* each bar is the win probability Denver gained (orange) or lost (blue) from "
        "chance plays in that game. A win would only count as \"decided by luck\" if its bar passed the "
        "+50 line.",
    ], img=("5_luck_per_close_game.png", "Bar chart of chance-play luck per close game")))

    stab = sig.assign(**{
        "Stat": sig.label, "Denver 2025": [fmt_metric(m_, v) for m_, v in zip(sig.metric, sig.den_2025)],
        "League avg": [fmt_metric(m_, v) for m_, v in zip(sig.metric, sig.league_avg_2025)],
        "DEN rank": sig.den_rank_2025.map(lambda r: f"#{r}"),
        "Year-to-year r": sig.year_to_year_r.map(lambda r: f"{r:.2f}"), "Repeats?": sig.repeats,
    })[["Stat", "Denver 2025", "League avg", "DEN rank", "Year-to-year r", "Repeats?"]]
    S.append(dict(id="signals", title="7. Skill signals vs luck signals", paras=[
        "A stat only counts as a skill signal if it carries over from one season to the next. For every stat "
        f"below, the year-to-year correlation is measured across all teams from {FIT.start} to {FIT.stop - 1} "
        "(r ≥ 0.30 = repeats; 0.15–0.30 = weakly; below 0.15 = barely or not). Ranks are out of 32, where "
        "1 is best for Denver.",
        f"Denver's 2025 strengths were concentrated late in games: 4th-quarter defence ranked "
        f"#{sg.loc['q4_def_epa', 'den_rank_2025']}. But late-game stats barely repeat for any team "
        f"(r = {sg.loc['q4_def_epa', 'year_to_year_r']:.2f} for Q4 defence, "
        f"{sg.loc['late_close_dropback_epa', 'year_to_year_r']:.2f} for QB play late in close games). Overall "
        f"efficiency is the one that holds up (r = {sg.loc['net_epa', 'year_to_year_r']:.2f}).",
        f"The classic luck stats did **not** favour Denver. They recovered only "
        f"{fmt_metric('fumble_recovery_rate', sg.loc['fumble_recovery_rate', 'den_2025'])} of fumbles "
        f"(#{sg.loc['fumble_recovery_rate', 'den_rank_2025']}), and opponents made "
        f"{100 * sg.loc['opp_fg_over_exp', 'den_2025']:.1f} more field goals per 100 "
        f"attempts than expected against them. Denver's own kicking was about average "
        f"({100 * sg.loc['fg_over_exp', 'den_2025']:+.1f} makes per 100 attempts vs expected).",
    ], img=("6_what_repeats.png", "Bar chart of year-to-year correlation for each stat"), table=stab))

    se = seasons.assign(**{
        "Season": seasons.season.astype(str) + np.where(seasons.season == LIVE, f" (thru wk {F['last_week']})", ""),
        "Record": seasons.record, "One-score": seasons.os_record,
        "Expected one-score wins": [f"{e:.1f} of {int(g)}" for e, g in zip(seasons.exp_os_wins, seasons.os_games)],
        "Net EPA/play (rank)": [f"{v:+.3f} (#{r})" for v, r in zip(seasons.net_epa, seasons.net_epa_rank)],
    })[["Season", "Record", "One-score", "Expected one-score wins", "Net EPA/play (rank)"]]
    s24 = seasons[seasons.season == 2024].iloc[0]
    S.append(dict(id="live", title=f"8. Denver across seasons, and the {LIVE} live check", paras=[
        f"The cleanest comparison is Denver against itself. In 2024, the same coach and quarterback went "
        f"**{s24.os_record}** in one-score games with almost the same efficiency ({s24.net_epa:+.3f} net EPA per play "
        f"vs {den.net_epa:+.3f} in 2025). The team's quality barely changed, but the close-game record swung from "
        f"{s24.os_record} to {record(den.os_wins, n_os)}.",
        f"**{LIVE} so far (through week {F['last_week']}):** {lv.record} overall and **{lv.os_record} in "
        f"one-score games**. Efficiency has dropped to {lv.net_epa:+.3f} (#{lv.net_epa_rank} of 32), so a team like "
        f"this would expect to win about {pct(lv.exp_os_pct)} of close games. Winning all {int(lv.os_games)} has a "
        f"{pct(live_p['p_all_close_wins'], 1)} chance. And unlike 2025, the chance plays in those games went "
        f"Denver's way: **{100 * luck26['den_luck_close_games']:+.0f} points of win probability** in total "
        f"(about {abs(luck26['den_luck_close_games']):.1f} wins), positive in "
        f"{int((g26.luck_total > 0).sum())} of {len(g26)} games, though no single game passed the ±50 line "
        f"({word(luck26['n_decided_by_luck']).replace('zero', 'none')} decided by luck). Replaying them with chance "
        f"plays at normal rates gives about {luck26['simulation']['expected_wins']:.1f} wins out of {len(g26)}.",
        f"Across 2024–{LIVE} (same coach and QB), Denver are {record(span['os_wins'], span['os_games'])} in one-score "
        f"games against {span['exp']:.1f} expected. The chance of {span['os_wins']:.0f} or more at their expected "
        f"rates is {pct(span['p_at_least'], 1)}.",
        "**What the live check means:** the 2026 start is the best evidence on the skill side. Some real, "
        "repeatable late-game edge that efficiency stats miss can't be ruled out. But it is only "
        f"{int(lv.os_games)} games, chance plays have helped this time, the 2024 record points the other way, and "
        "history says to expect close-game records to drift back toward what efficiency predicts. Worth re-running "
        "at the end of the season.",
    ], table=se))
    return S


def method_md(wp_stats: dict, pc: dict, pyth_wins: float) -> str:
    return f"""### Method, assumptions and limits

**Data.** nflverse schedules ({HIST_FIRST}–{LIVE}), play-by-play ({PBP_FIRST}–{LIVE}) and FTN charting
({FTN_FIRST}–{LIVE}), loaded with `nflreadpy`. Regular season unless stated. Relocated teams (OAK→LV, SD→LAC,
STL→LA) keep one code so "next season" follows the franchise. All models are fitted on {FIT.start}–{FIT.stop - 1},
so Denver 2025 is never used to fit the model that judges it.

**Definitions.** One-score game: final margin of 8 points or fewer. EPA per play: passes and runs only
(the standard nflverse filter). Net EPA per play = offence EPA/play − defence EPA/play allowed.
Pythagorean exponent {PYTH_EXP}: the standard NFL value from Daryl Morey's original work, not fitted here. Checked
against {FIT.start}–{FIT.stop - 1} ({pc['n']} team-seasons, exponents 1.5–3.5): the best fit is {pc['best']:.2f}
(typical error {pc['rmse_best']:.2f} wins per season vs {pc['rmse_std']:.2f} for {PYTH_EXP}), which moves Denver's
expected wins by {pc['den_best'] - pyth_wins:+.1f}.

**High-leverage luck — assumptions.**
- nflverse publishes win probability (WP) only for what actually happened. For the outcome that didn't happen
  (e.g. the fumble that the *other* team recovered), the WP comes from a stand-in model trained to reproduce
  nflverse's WP from the game state ({WP_TRAIN.start}–{WP_TRAIN.stop - 1} plays, gradient boosting). Tested on
  every 2025 play (n = {wp_stats['n_test']:,}), it differs from nflverse's own WP by {wp_stats['mae_pts']:.1f}
  percentage points on average (90% of plays within {wp_stats['p90_err_pts']:.1f}).
- The alternative outcome is placed simply: a lost fumble gives the other team the ball at the same spot;
  an interception happens at the target spot with no return; a missed kick gives the ball back at the spot of
  the kick; a made kick or a failed/converted 4th down is followed by a standard kickoff or change of
  possession. Returns, penalties and clock effects on the hypothetical branch are ignored.
- "Normal rates" come from the league: fumble recovery split by scrimmage vs special teams ({PBP_FIRST}–2024),
  field-goal make rate by 5-yard distance band in the same season, extra-point rate in the same season,
  interception rate for interception-worthy vs other throws (FTN, {FTN_FIRST}–2024), 4th-down conversion rate by
  yards to go ({FIT.start}–{FIT.stop - 1}).
- Only *opponents'* kicks count as luck for Denver (Denver's own kicking is treated as skill). Plays with an
  accepted penalty are left out of the interception and 4th-down groups because the flag settled them.
- Adding up win-probability changes treats each chance play as independent and assumes the rest of the game would
  have gone the same way. In reality teams change their play-calling after a big swing. The game simulation
  shares this assumption.

**Limits.**
- Small samples everywhere: 13 close games in 2025, {LIVE} still in progress. The history group is only teams with
  9+ close wins.
- "Luck" here means *doesn't repeat*, not *random*. Winning fourth quarters takes real players making real plays;
  the point is that it doesn't carry over from year to year, so it can't be relied on.
- FTN's "interception-worthy" label is a human charting judgement. Starting kickoff field position is a
  league median. 4th-down conversion rates are league-wide, not adjusted for opponent.
- Efficiency doesn't capture everything a coach controls (clock management, timeouts, late-game play-calling).
  Those may be real skills the models miss.
"""


def bold_md_to_html(t: str) -> str:
    t = _h.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)


def write_readme(S, F, method: str):
    body = [f"*Generated by `analysis.py` from nflverse data: {F['cutoff']}. Web page: {PAGE_URL}*\n"]
    for s in S:
        body.append(f"### {s['title']}\n")
        body += [p + "\n" for p in s["paras"]]
        if s.get("img"):
            body.append(f"![{s['img'][1]}](figures/{s['img'][0]})\n")
        if s.get("table") is not None:
            body.append(md_table(s["table"]) + "\n")
    body.append(method)
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- FINDINGS:START -->", "<!-- FINDINGS:END -->"
    new = start + "\n" + "\n".join(body) + "\n" + end
    text = re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: new, text, flags=re.S)
    readme.write_text(text, encoding="utf-8")


PAGE_CSS = """
:root{--bg:#ffffff;--ink:#1d1d1b;--muted:#5f5e5a;--navy:#002244;--orange:#FB4F14;--blue:#2f5ea8;
--line:#e6e5e0;--soft:#f6f5f2;color-scheme:light}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,
"Helvetica Neue",Arial,sans-serif}
main{max-width:760px;margin:0 auto;padding:28px 16px 64px}
.kicker{font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:var(--orange);font-weight:700;margin:0}
h1{font-size:clamp(30px,7vw,42px);line-height:1.1;color:var(--navy);margin:6px 0 12px}
h2{font-size:22px;line-height:1.25;color:var(--navy);margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
p{margin:0 0 14px}
.lede{font-size:19px;color:var(--ink)}
.byline{color:var(--muted);font-size:14px;margin-bottom:22px}
.tiles{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:22px 0}
@media(min-width:640px){.tiles{grid-template-columns:repeat(4,1fr)}}
.tile{background:var(--soft);border-radius:10px;padding:12px 14px}
.tile .v{font-size:28px;font-weight:700;color:var(--navy);line-height:1.1}
.tile .l{font-size:13px;color:var(--muted);line-height:1.35;margin-top:4px}
.verdict{border-left:4px solid var(--orange);background:var(--soft);padding:14px 16px;border-radius:0 10px 10px 0;
font-size:18px;margin:8px 0 18px}
figure{margin:18px 0 22px}
figure a{display:block}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:8px}
figcaption{font-size:13px;color:var(--muted);margin-top:6px}
.note{font-size:15.5px;background:var(--soft);border-radius:10px;padding:12px 14px}
.tbl{overflow-x:auto;margin:10px 0 22px;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:12.5px}
td.num,th.num{text-align:right}
ul{padding-left:20px}
li{margin-bottom:6px}
.small{font-size:14.5px;color:var(--muted)}
details{margin:12px 0 18px}
summary{cursor:pointer;color:var(--blue);font-weight:600}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--line);font-size:14px;color:var(--muted)}
a{color:var(--blue)}
figure a[data-full]{cursor:zoom-in}
html:has(dialog.lightbox[open]){overflow:hidden}
dialog.lightbox{padding:0;border:0;margin:0;width:100vw;height:100vh;max-width:none;max-height:none;
background:rgba(12,14,18,.94);overflow:hidden}
dialog.lightbox::backdrop{background:rgba(0,0,0,.6)}
.lb-inner{height:100%;overflow:auto;display:flex;justify-content:center;align-items:flex-start;
padding:60px 12px 24px;-webkit-overflow-scrolling:touch}
.lb-inner img{width:100%;max-width:1100px;height:auto;border-radius:8px;background:#fff;cursor:default}
@media(min-width:700px){.lb-inner{padding:64px 7vw 40px}}
.lb-close{position:fixed;top:10px;right:12px;width:44px;height:44px;border:0;border-radius:22px;background:#fff;
color:var(--navy);font-size:28px;line-height:1;cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.lb-hint{position:fixed;top:20px;left:16px;right:70px;margin:0;color:#fff;font-size:13px;opacity:.85}
"""


def md_block_to_html(md: str) -> str:
    """Tiny converter for the method block (headings, paragraphs, bullet and numbered lists)."""
    out, para, lst = [], [], None

    def flush():
        nonlocal para, lst
        if para:
            out.append(f"<p>{bold_md_to_html(' '.join(para))}</p>")
            para = []
        if lst:
            tag, items = lst
            out.append(f"<{tag}>" + "".join(f"<li>{bold_md_to_html(i)}</li>" for i in items) + f"</{tag}>")
            lst = None

    for line in md.splitlines():
        s = line.strip()
        if not s:
            flush()
        elif s.startswith("### "):
            flush()
            out.append(f"<h2>{bold_md_to_html(s[4:])}</h2>")
        elif s.startswith("- ") or re.match(r"^\d+\. ", s):
            tag = "ul" if s.startswith("- ") else "ol"
            if para:
                flush()
            if lst is None or lst[0] != tag:
                flush()
                lst = (tag, [])
            lst[1].append(re.sub(r"^(- |\d+\. )", "", s))
        elif lst is not None:
            lst[1][-1] += " " + s
        else:
            para.append(s)
    flush()
    return "\n".join(out)


def chart_html(name: str, alt: str, lazy: bool = True) -> str:
    """A chart that opens full size in an overlay (plain link without JS); phone version on narrow screens."""
    phone = name.replace(".png", "_phone.png")
    load = ' loading="lazy"' if lazy else ""
    a = f'href="img/{name}" data-full data-alt="{_h.escape(alt)}"'
    return (f'<figure><a {a} aria-label="Enlarge chart"><picture>'
            f'<source media="(max-width: 600px)" srcset="img/{phone}">'
            f'<img src="img/{name}" alt="{_h.escape(alt)}"{load}></picture></a>'
            f'<figcaption><a {a}>Enlarge chart</a></figcaption></figure>')


# Full-size chart overlay: closes on a tap outside the chart, the × button, Esc, or the browser back button
LIGHTBOX = """<dialog class="lightbox" id="lb" aria-label="Chart, full size">
<div class="lb-inner"><img alt=""></div>
<p class="lb-hint">Pinch or scroll to zoom · tap outside to close</p>
<button type="button" class="lb-close" aria-label="Close">×</button>
</dialog>
<script>
(() => {
  const d = document.getElementById("lb");
  if (!d || typeof d.showModal !== "function") return;
  const img = d.querySelector("img");
  // Browsers with CloseWatcher (Chrome, Edge, Android) already close a modal dialog on Esc and the
  // back button; adding a history entry there would leave a stray one behind. Elsewhere, add one so
  // the back button closes the chart instead of leaving the page.
  const useHistory = !("CloseWatcher" in window);
  let pushed = false;
  function open(src, alt) {
    img.src = src;
    img.alt = alt;
    d.showModal();
    d.querySelector(".lb-inner").scrollTop = 0;
    if (useHistory) { history.pushState({ lb: 1 }, ""); pushed = true; }
  }
  function dropHistoryEntry() {
    if (pushed) { pushed = false; if (history.state && history.state.lb) history.back(); }
  }
  function shut() { if (d.open) d.close(); dropHistoryEntry(); }
  d.addEventListener("close", dropHistoryEntry);  // Esc and other native closes
  window.addEventListener("popstate", () => { if (d.open) { pushed = false; d.close(); } });
  d.addEventListener("click", (e) => { if (e.target !== img) shut(); });
  document.querySelectorAll("a[data-full]").forEach((a) => a.addEventListener("click", (e) => {
    if (e.button || e.metaKey || e.ctrlKey || e.shiftKey) return;
    e.preventDefault();
    open(a.href, a.dataset.alt || "");
  }));
})();
</script>"""


def write_page(S, F, den, binom, method: str):
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "img").mkdir(exist_ok=True)
    for f in FIG.glob("*.png"):
        shutil.copy(f, SITE / "img" / f.name)
    n_os, k_os = int(den.os_games), int(den.os_wins)
    tiles = [
        (f"{record(den.wins, int(den.games))}", "2025 regular-season record"),
        (f"{k_os}-{n_os - k_os}", "in one-score games (8 points or fewer)"),
        (f"{den.epa_wins:.1f}", "wins expected from efficiency"),
        (pct(binom["expected"], 1), f"chance of {k_os}-{n_os - k_os} or better at Denver's expected close-game rate"),
    ]
    parts = []
    for s in S:
        if s["id"] == "verdict":
            continue
        parts.append(f'<section id="{s["id"]}"><h2>{bold_md_to_html(s["title"])}</h2>')
        parts += [f"<p>{bold_md_to_html(p)}</p>" for p in s["paras"]]
        if s.get("img"):
            parts.append(chart_html(*s["img"]))
        if s.get("table") is not None:
            tbl = html_table(s["table"].astype(str).map(_h.escape))
            parts.append(tbl if len(s["table"]) <= 8 else f"<details><summary>Show the table</summary>{tbl}</details>")
        parts.append("</section>")
    verdict = S[0]
    title = "Broncos 2025: Luck or Skill?"
    desc = _h.escape(F["verdict"])
    card = f"{PAGE_URL}img/7_verdict_card.png"
    card_alt = "Verdict: " + desc
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="canonical" href="{PAGE_URL}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Sports analytics portfolio">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{desc}">
<meta property="og:url" content="{PAGE_URL}">
<meta property="og:image" content="{card}">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{card_alt}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{desc}">
<meta name="twitter:image" content="{card}">
<meta name="twitter:image:alt" content="{card_alt}">
<style>{PAGE_CSS}</style></head>
<body><main>
<p class="kicker">NFL data analysis · Python</p>
<h1>Broncos 2025: luck or skill?</h1>
<p class="lede">The 2025 Denver Broncos went {record(den.wins, int(den.games))} and won {k_os} of {n_os} games decided by
8 points or fewer. Were they really that good in close games, or did those games break their way?</p>
<p class="byline">Data: nflverse · {F['cutoff']} ·
<a href="{REPO_URL}/tree/main/broncos-luck-or-skill">Code and data on GitHub</a></p>
<div class="tiles">{''.join(f'<div class="tile"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in tiles)}</div>
<div class="verdict">{bold_md_to_html(verdict['paras'][0])}</div>
<p>{bold_md_to_html(verdict['paras'][1])}</p>
<p class="note">{bold_md_to_html(verdict['paras'][2])}</p>
{chart_html(*verdict['img'], lazy=False)}
{''.join(parts)}
<section id="method">{md_block_to_html(method)}</section>
<footer>Data: <a href="https://github.com/nflverse">nflverse</a> (schedules, play-by-play, FTN charting) via the
nflreadpy Python package. Analysis, charts and page are reproduced by <code>analysis.py</code>.
<a href="{REPO_URL}/tree/main/broncos-luck-or-skill">Source on GitHub</a>.</footer>
</main>
{LIGHTBOX}
</body></html>
"""
    (SITE / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
