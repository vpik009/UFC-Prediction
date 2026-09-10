"""Turns scraped fights/fighters into a leakage-free, time-ordered feature matrix.

Every feature used to predict a fight is computed strictly from information
available *before* that fight happened (Elo ratings, rolling per-minute
output, layoff, experience, ...). Static bio attributes (height/reach/DOB/
stance) don't change over a career except age, which is computed relative
to the fight date, so they're leakage-free too.

Fights are listed on ufcstats.com with the winner always in the "fighter1"
slot, so naively training on (fighter1 - fighter2) features would leak the
label through position. To cancel that out, every fight produces two mirror
training rows: (winner-perspective, label=1) and (loser-perspective,
label=0), with all diff features negated between them.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from . import elo

FEATURE_NAMES = [
    "elo_diff",
    "height_diff",
    "reach_diff",
    "age_diff",
    "experience_diff",
    "win_streak_diff",
    "layoff_diff",
    "kd_per_min_diff",
    "str_per_min_diff",
    "td_per_min_diff",
    "sub_per_min_diff",
    "southpaw_diff",
    "prime_distance_diff",
    "decline_risk_diff",
    "travel_diff",
    # style matchup: raw offense/defense efficiency plus an explicit
    # "wrestling pressure vs opponent's takedown defense" interaction
    "td_acc_diff",
    "td_def_diff",
    "str_acc_diff",
    "str_def_diff",
    "style_matchup_diff",
    # cardio/pace: does this fighter's output drop in later rounds, scaled
    # by how many rounds this specific fight is scheduled for
    "cardio_diff",
    # activity: fights in the trailing 2 years, distinct from raw layoff
    "activity_diff",
    # finishing ability/power, split out instead of one generic finish_rate
    "ko_rate_diff",
    "sub_rate_diff",
    "power_diff",
]

# Athletic prime window (years old) used by prime_distance()/decline_risk()
# below — a fighter younger than PRIME_START is still physically/technically
# maturing (a mild disadvantage, not the pure asset raw age_diff implies),
# and one older than PRIME_END is aging out.
PRIME_START = 27.0
PRIME_END = 33.0

# Rough UFC-wide baseline rates, used only as a neutral prior before a
# fighter has any attempts on record (gets swamped by real data fast).
NEUTRAL_TD_ACC = 0.40
NEUTRAL_TD_DEF = 0.65
NEUTRAL_STR_ACC = 0.45
NEUTRAL_STR_DEF = 0.55

ACTIVITY_WINDOW_DAYS = 730.0  # trailing 2 years


@dataclass
class FighterState:
    name: str
    url: str
    elo: float = elo.BASE_RATING
    n_fights: int = 0
    n_wins: int = 0
    n_finishes: int = 0
    n_ko_wins: int = 0
    n_sub_wins: int = 0
    streak: int = 0  # positive = win streak, negative = loss streak
    last_fight_date: datetime | None = None
    fight_dates: list = field(default_factory=list)
    cum_minutes: float = 0.0
    cum_kd: float = 0.0
    cum_str: float = 0.0
    cum_td: float = 0.0
    cum_sub: float = 0.0
    height_in: float | None = None
    reach_in: float | None = None
    dob: datetime | None = None
    stance: str | None = None
    # count of prior fights held in each country, e.g. {"USA": 9, "Brazil": 2}
    # — a proxy for "how unfamiliar is this trip" since ufcstats.com doesn't
    # publish fighter nationality/hometown at all
    country_history: dict = field(default_factory=lambda: defaultdict(int))

    # --- from each fight's individual stats page (attempts, not just landed) ---
    cum_sig_str_landed_off: float = 0.0
    cum_sig_str_att_off: float = 0.0
    cum_sig_str_landed_def: float = 0.0  # landed AGAINST this fighter
    cum_sig_str_att_def: float = 0.0
    cum_td_landed_off: float = 0.0
    cum_td_att_off: float = 0.0
    cum_td_landed_def: float = 0.0
    cum_td_att_def: float = 0.0
    # round-1 vs last-round output, restricted to fights that reached round 2+
    # (a 1-round fight has no "later round" to compare against)
    cum_round1_str: float = 0.0
    cum_round1_minutes: float = 0.0
    cum_lastround_str: float = 0.0
    cum_lastround_minutes: float = 0.0

    def age_at(self, date: datetime) -> float | None:
        if self.dob is None:
            return None
        return (date - self.dob).days / 365.25

    def finish_rate(self) -> float:
        return self.n_finishes / self.n_wins if self.n_wins else 0.0

    def per_min(self, total: float) -> float:
        return total / self.cum_minutes if self.cum_minutes > 0 else 0.0

    def layoff_days(self, date: datetime, default: float = 365.0) -> float:
        if self.last_fight_date is None:
            return default
        return (date - self.last_fight_date).days

    def is_southpaw(self) -> float:
        if not self.stance:
            return 0.0
        return 1.0 if "southpaw" in self.stance.lower() else 0.0

    def prime_distance(self, date: datetime) -> float:
        """Years outside the athletic prime window — 0 while in it, positive
        on either side (too green, or aging out)."""
        age = self.age_at(date)
        if age is None:
            return 0.0
        if age < PRIME_START:
            return PRIME_START - age
        if age > PRIME_END:
            return age - PRIME_END
        return 0.0

    def decline_risk(self, date: datetime) -> float:
        """Years past prime, but only counted while currently on a losing
        streak — captures "aging AND fading" as worse than either alone."""
        if self.streak >= 0:
            return 0.0
        age = self.age_at(date)
        if age is None:
            return 0.0
        years_past = max(0.0, age - PRIME_END)
        return years_past * (-self.streak)

    def country_familiarity(self, country: str | None) -> float:
        """Fraction of this fighter's prior fights held in `country`. 0.5
        (neutral) if we don't know the fight location or they have no
        history yet — not a claim they're "half familiar", just "unknown"."""
        if not country:
            return 0.5
        total = sum(self.country_history.values())
        if total == 0:
            return 0.5
        return self.country_history.get(country, 0) / total

    def td_accuracy(self) -> float:
        if self.cum_td_att_off <= 0:
            return NEUTRAL_TD_ACC
        return self.cum_td_landed_off / self.cum_td_att_off

    def td_defense(self) -> float:
        """Fraction of takedowns attempted AGAINST this fighter that failed."""
        if self.cum_td_att_def <= 0:
            return NEUTRAL_TD_DEF
        return 1.0 - self.cum_td_landed_def / self.cum_td_att_def

    def str_accuracy(self) -> float:
        if self.cum_sig_str_att_off <= 0:
            return NEUTRAL_STR_ACC
        return self.cum_sig_str_landed_off / self.cum_sig_str_att_off

    def str_defense(self) -> float:
        if self.cum_sig_str_att_def <= 0:
            return NEUTRAL_STR_DEF
        return 1.0 - self.cum_sig_str_landed_def / self.cum_sig_str_att_def

    def takedown_pressure_per_min(self) -> float:
        """How often this fighter attempts takedowns — a "wrestler-ness"
        intent signal, independent of whether they land them."""
        return self.per_min(self.cum_td_att_off)

    def style_edge_over(self, defender: "FighterState") -> float:
        """How much this fighter's grappling pressure should exploit the
        given opponent's takedown defense — the wrestler-vs-striker term."""
        return self.takedown_pressure_per_min() * (1.0 - defender.td_defense())

    def fade_rate(self) -> float:
        """Per-minute output in round 1 minus per-minute output in this
        fighter's last round, averaged over fights that reached round 2+.
        Positive = output drops later in fights (cardio concern); negative
        = fighter tends to pick up the pace. 0 if there's no multi-round
        history yet."""
        if self.cum_round1_minutes <= 0 or self.cum_lastround_minutes <= 0:
            return 0.0
        round1_rate = self.cum_round1_str / self.cum_round1_minutes
        lastround_rate = self.cum_lastround_str / self.cum_lastround_minutes
        return round1_rate - lastround_rate

    def ko_rate(self) -> float:
        return self.n_ko_wins / self.n_wins if self.n_wins else 0.0

    def sub_rate(self) -> float:
        return self.n_sub_wins / self.n_wins if self.n_wins else 0.0

    def power(self) -> float:
        """Knockdowns per significant strike landed — power independent of
        output volume (kd_per_min conflates the two)."""
        if self.cum_sig_str_landed_off <= 0:
            return 0.0
        return self.cum_kd / self.cum_sig_str_landed_off

    def recent_activity(self, date: datetime, window_days: float = ACTIVITY_WINDOW_DAYS) -> int:
        return sum(1 for d in self.fight_dates if 0 <= (date - d).days <= window_days)


def _fight_minutes(round_: int | None, time_str: str | None, round_len: float = 5.0) -> float:
    if round_ is None or not time_str or ":" not in str(time_str):
        return round_len  # fallback: assume ~1 round happened
    try:
        mm, ss = str(time_str).split(":")
        minutes_in_last_round = int(mm) + int(ss) / 60.0
    except ValueError:
        minutes_in_last_round = 0.0
    return max((round_ - 1) * round_len + minutes_in_last_round, 0.5)


def estimate_scheduled_rounds(is_title_fight: bool, method: str | None = None, actual_round: int | None = None) -> int:
    """ufcstats.com doesn't publish scheduled round count directly. A
    decision always goes the full distance, so its round IS the schedule.
    Otherwise fall back to the title-fight flag (5 rounds) vs 3 — this
    undercounts non-title 5-round main events, which aren't otherwise
    flagged in the data."""
    if method and str(method).upper().startswith("DEC") and actual_round:
        return actual_round
    return 5 if is_title_fight else 3


def pair_features(
    x: FighterState,
    y: FighterState,
    fight_date: datetime,
    country: str | None = None,
    scheduled_rounds: int = 3,
) -> dict:
    def safe_diff(a, b, default=0.0):
        if a is None or b is None:
            return default
        return a - b

    return {
        "elo_diff": x.elo - y.elo,
        "height_diff": safe_diff(x.height_in, y.height_in),
        "reach_diff": safe_diff(x.reach_in, y.reach_in),
        "age_diff": safe_diff(x.age_at(fight_date), y.age_at(fight_date)),
        "experience_diff": np.log1p(x.n_fights) - np.log1p(y.n_fights),
        "win_streak_diff": x.streak - y.streak,
        "layoff_diff": np.log1p(x.layoff_days(fight_date)) - np.log1p(y.layoff_days(fight_date)),
        "kd_per_min_diff": x.per_min(x.cum_kd) - y.per_min(y.cum_kd),
        "str_per_min_diff": x.per_min(x.cum_str) - y.per_min(y.cum_str),
        "td_per_min_diff": x.per_min(x.cum_td) - y.per_min(y.cum_td),
        "sub_per_min_diff": x.per_min(x.cum_sub) - y.per_min(y.cum_sub),
        "southpaw_diff": x.is_southpaw() - y.is_southpaw(),
        "prime_distance_diff": y.prime_distance(fight_date) - x.prime_distance(fight_date),
        "decline_risk_diff": y.decline_risk(fight_date) - x.decline_risk(fight_date),
        "travel_diff": x.country_familiarity(country) - y.country_familiarity(country),
        "td_acc_diff": x.td_accuracy() - y.td_accuracy(),
        "td_def_diff": x.td_defense() - y.td_defense(),
        "str_acc_diff": x.str_accuracy() - y.str_accuracy(),
        "str_def_diff": x.str_defense() - y.str_defense(),
        "style_matchup_diff": x.style_edge_over(y) - y.style_edge_over(x),
        "cardio_diff": scheduled_rounds * (y.fade_rate() - x.fade_rate()),
        "activity_diff": x.recent_activity(fight_date) - y.recent_activity(fight_date),
        "ko_rate_diff": x.ko_rate() - y.ko_rate(),
        "sub_rate_diff": x.sub_rate() - y.sub_rate(),
        "power_diff": x.power() - y.power(),
    }


def _init_states(fighters_df: pd.DataFrame) -> dict[str, FighterState]:
    states: dict[str, FighterState] = {}
    has_pre_ufc = "pre_ufc_wins" in fighters_df.columns
    for row in fighters_df.itertuples():
        dob = row.dob if pd.notna(row.dob) else None
        if has_pre_ufc:
            pre_w = row.pre_ufc_wins if pd.notna(row.pre_ufc_wins) else 0
            pre_l = row.pre_ufc_losses if pd.notna(row.pre_ufc_losses) else 0
            seed = elo.seed_rating(pre_w, pre_l)
        else:
            seed = elo.BASE_RATING
        states[row.url] = FighterState(
            name=row.name,
            url=row.url,
            elo=seed,
            height_in=row.height_in if pd.notna(row.height_in) else None,
            reach_in=row.reach_in if pd.notna(row.reach_in) else None,
            dob=dob,
            stance=row.stance if isinstance(row.stance, str) else None,
        )
    return states


FIGHT_STATS_COLUMNS = [
    "f1_sig_str_landed", "f1_sig_str_att", "f2_sig_str_landed", "f2_sig_str_att",
    "f1_td_landed", "f1_td_att", "f2_td_landed", "f2_td_att",
    "n_rounds",
    "f1_round1_str_landed", "f1_round1_str_att", "f2_round1_str_landed", "f2_round1_str_att",
    "f1_lastround_str_landed", "f1_lastround_str_att", "f2_lastround_str_landed", "f2_lastround_str_att",
]


def _method_is_ko(method: str | None) -> bool:
    return bool(method) and "KO" in str(method).upper()


def _method_is_sub(method: str | None) -> bool:
    return bool(method) and "SUB" in str(method).upper()


def build_training_frame(
    fights_df: pd.DataFrame, fighters_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, FighterState]]:
    """Returns (feature_df with columns FEATURE_NAMES + ['label','event_date'], final states).

    `fights_df` may optionally carry the FIGHT_STATS_COLUMNS (left-joined
    from fight_stats.csv via dataset.load_fights_merged()) and
    `is_title_fight` — both used for the style/cardio features. Their
    absence degrades gracefully (those features just read as 0/neutral)."""
    states = _init_states(fighters_df)
    fights_df = fights_df.dropna(subset=["event_date"]).sort_values("event_date").reset_index(drop=True)
    stat_cols = ["kd1", "kd2", "str1", "str2", "td1", "td2", "sub1", "sub2"]
    fights_df[stat_cols] = fights_df[stat_cols].fillna(0.0)

    has_fight_stats = "f1_sig_str_landed" in fights_df.columns
    if has_fight_stats:
        fights_df[FIGHT_STATS_COLUMNS] = fights_df[FIGHT_STATS_COLUMNS].fillna(0.0)
    has_title_flag = "is_title_fight" in fights_df.columns

    rows = []
    for f in fights_df.itertuples():
        x = states.setdefault(f.fighter1_url, FighterState(name=f.fighter1, url=f.fighter1_url))
        y = states.setdefault(f.fighter2_url, FighterState(name=f.fighter2, url=f.fighter2_url))
        date = f.event_date
        country = f.event_country if isinstance(f.event_country, str) else None
        is_title = bool(f.is_title_fight) if has_title_flag else False
        scheduled_rounds = estimate_scheduled_rounds(is_title, f.method, f.round)

        if f.winner in ("fighter1", "fighter2"):
            feats = pair_features(x, y, date, country, scheduled_rounds)
            feats["label"] = 1.0 if f.winner == "fighter1" else 0.0
            feats["event_date"] = date
            feats["weight_class"] = f.weight_class
            rows.append(feats)

            mirror = {k: (-v if k in FEATURE_NAMES else v) for k, v in feats.items()}
            mirror["label"] = 1.0 - feats["label"]
            rows.append(mirror)

        # --- update state post-fight ---
        minutes = _fight_minutes(f.round, f.time)
        for state, kd, st, td, sub in (
            (x, f.kd1, f.str1, f.td1, f.sub1),
            (y, f.kd2, f.str2, f.td2, f.sub2),
        ):
            state.cum_minutes += minutes
            state.cum_kd += kd or 0
            state.cum_str += st or 0
            state.cum_td += td or 0
            state.cum_sub += sub or 0
            if country:
                state.country_history[country] += 1
            state.fight_dates.append(date)

        if has_fight_stats:
            n_rounds = int(f.n_rounds)
            for off_state, def_state, sl, sa, tl, ta in (
                (x, y, f.f1_sig_str_landed, f.f1_sig_str_att, f.f1_td_landed, f.f1_td_att),
                (y, x, f.f2_sig_str_landed, f.f2_sig_str_att, f.f2_td_landed, f.f2_td_att),
            ):
                off_state.cum_sig_str_landed_off += sl
                off_state.cum_sig_str_att_off += sa
                off_state.cum_td_landed_off += tl
                off_state.cum_td_att_off += ta
                def_state.cum_sig_str_landed_def += sl
                def_state.cum_sig_str_att_def += sa
                def_state.cum_td_landed_def += tl
                def_state.cum_td_att_def += ta
            if n_rounds >= 2:
                r1_len = 5.0
                last_len = _fight_minutes(f.round, f.time) - (n_rounds - 1) * 5.0
                for state, r1s, r1a, rls, rla in (
                    (x, f.f1_round1_str_landed, f.f1_round1_str_att, f.f1_lastround_str_landed, f.f1_lastround_str_att),
                    (y, f.f2_round1_str_landed, f.f2_round1_str_att, f.f2_lastround_str_landed, f.f2_lastround_str_att),
                ):
                    state.cum_round1_str += r1s
                    state.cum_round1_minutes += r1_len
                    state.cum_lastround_str += rls
                    state.cum_lastround_minutes += max(last_len, 0.5)

        if f.winner in ("fighter1", "fighter2"):
            score_x = 1.0 if f.winner == "fighter1" else 0.0
            k = elo.k_factor(x.n_fights, y.n_fights, f.method, f.round)
            new_x, new_y = elo.update(x.elo, y.elo, score_x, k)
            x.elo, y.elo = new_x, new_y
            is_finish = bool(f.method) and not str(f.method).upper().startswith("DEC")
            is_ko = _method_is_ko(f.method)
            is_sub = _method_is_sub(f.method)
            for state, won in ((x, score_x == 1.0), (y, score_x == 0.0)):
                state.n_fights += 1
                if won:
                    state.n_wins += 1
                    state.streak = state.streak + 1 if state.streak >= 0 else 1
                    if is_finish:
                        state.n_finishes += 1
                        if is_ko:
                            state.n_ko_wins += 1
                        elif is_sub:
                            state.n_sub_wins += 1
                else:
                    state.streak = state.streak - 1 if state.streak <= 0 else -1
                state.last_fight_date = date
        else:
            # draw / no-contest: no elo change, but counts as an outing
            for state in (x, y):
                state.n_fights += 1
                state.last_fight_date = date

    df = pd.DataFrame(rows)
    return df, states
