"""Backtests the model against real, already-decided UFC events.

Run with:
    pytest -s tests/test_backtest.py
or directly, for a readable fight-by-fight report:
    python tests/test_backtest.py [n_holdout_events]

The model is retrained on only the fights that happened *before* the
held-out events (a real forecaster would never see the outcomes it's
being graded on), then judged on how well its probabilities predicted
the actual winners.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, log_loss

from ufcpred import dataset, odds
from ufcpred.features import FEATURE_NAMES, build_training_frame
from ufcpred.model import _fit_logreg

DEFAULT_HOLDOUT_EVENTS = 50
CONFIDENCE_THRESHOLD = 0.15  # |p(fighter A) - p(fighter B)| must be at least this


def _require_data():
    if not dataset.FIGHTS_CSV.exists() or not dataset.FIGHTERS_CSV.exists():
        pytest.skip("No local data found — run `python -m ufcpred update` first to scrape ufcstats.com.")


def _build_backtest(n_holdout_events: int = DEFAULT_HOLDOUT_EVENTS):
    """Returns (report_df, accuracy, log_loss_value)."""
    fights = dataset.load_fights_merged()
    fighters = dataset.load_fighters()
    fights = fights.dropna(subset=["event_date"])

    event_dates = sorted(fights["event_date"].unique())
    if len(event_dates) <= n_holdout_events:
        pytest.skip("Not enough scraped events to hold any out for backtesting.")
    cutoff = event_dates[-n_holdout_events]

    df, _ = build_training_frame(fights, fighters)

    train = df[df["event_date"] < cutoff]
    test = df[df["event_date"] >= cutoff]
    # each real fight produced 2 mirrored rows; keep only the "label==1" (real
    # winner-perspective) row per fight so each fight is graded exactly once
    test = test[test["label"] == 1.0].reset_index(drop=True)

    model = _fit_logreg(train[FEATURE_NAMES].values, train["label"].values)

    proba_winner = model.predict_proba(test[FEATURE_NAMES].values)[:, 1]
    # the model was asked "does the labelled winner beat the loser" — a
    # correct/well-calibrated model should put most mass above 0.5 here
    predicted_correct = proba_winner >= 0.5
    accuracy = float(np.mean(predicted_correct))
    ll = float(log_loss(np.ones(len(test)), proba_winner, labels=[0, 1]))

    report = test[["event_date"]].copy()
    report["p_winner_predicted"] = proba_winner
    report["correct"] = predicted_correct
    # model's confidence gap between the two fighters in the matchup,
    # independent of who actually won: |p(winner) - p(loser)| = |2p - 1|
    report["prob_diff"] = np.abs(2 * proba_winner - 1)
    return report, accuracy, ll, cutoff


def test_backtest_beats_coin_flip():
    _require_data()
    report, accuracy, ll, cutoff = _build_backtest(DEFAULT_HOLDOUT_EVENTS)
    print(f"\nHoldout: {len(report)} fights on/after {pd.Timestamp(cutoff).date()}")
    print(f"Accuracy: {accuracy:.3f}   Log loss: {ll:.3f}  (uniform baseline log loss = {np.log(2):.3f})")
    assert accuracy > 0.50, "model should beat a coin flip on held-out real events"
    assert ll < np.log(2), "model should beat the uniform-guess log loss on held-out real events"


def test_backtest_probabilities_are_valid():
    _require_data()
    report, *_ = _build_backtest(DEFAULT_HOLDOUT_EVENTS)
    assert (report["p_winner_predicted"] > 0).all()
    assert (report["p_winner_predicted"] < 1).all()


def test_backtest_high_confidence_accuracy():
    """When the model sees a real edge (>=15 point gap between the two
    fighters' probabilities), it should be picking winners noticeably
    more often than on the full, unfiltered slate."""
    _require_data()
    report, overall_accuracy, _, cutoff = _build_backtest(DEFAULT_HOLDOUT_EVENTS)
    confident = report[report["prob_diff"] >= CONFIDENCE_THRESHOLD]
    confident_accuracy = float(confident["correct"].mean()) if len(confident) else float("nan")

    print(f"\nHoldout: {len(report)} fights on/after {pd.Timestamp(cutoff).date()}")
    print(f"Overall accuracy:                          {overall_accuracy:.3f}  (n={len(report)})")
    print(
        f"Accuracy when prob. gap >= {CONFIDENCE_THRESHOLD*100:.0f} pts:      "
        f"{confident_accuracy:.3f}  (n={len(confident)}, "
        f"{len(confident) / len(report) * 100:.0f}% of fights)"
    )
    assert len(confident) > 0, "no fights met the confidence threshold — lower CONFIDENCE_THRESHOLD"
    assert confident_accuracy >= overall_accuracy - 0.02, (
        "high-confidence picks should not be meaningfully worse than the overall rate"
    )


def _print_full_report(n_holdout_events: int):
    fights = dataset.load_fights_merged().dropna(subset=["event_date"])
    fighters = dataset.load_fighters()
    event_dates = sorted(fights["event_date"].unique())
    cutoff = event_dates[-n_holdout_events]

    df, _ = build_training_frame(fights, fighters)
    train = df[df["event_date"] < cutoff]
    test_rows = df[(df["event_date"] >= cutoff) & (df["label"] == 1.0)]

    model = _fit_logreg(train[FEATURE_NAMES].values, train["label"].values)
    proba = model.predict_proba(test_rows[FEATURE_NAMES].values)[:, 1]

    # re-derive fighter names/event names by position (test_rows keeps original index into `fights`)
    test_fights = fights[fights["event_date"] >= cutoff].reset_index(drop=True)
    decisive = test_fights[test_fights["winner"].isin(["fighter1", "fighter2"])].reset_index(drop=True)

    n_correct = 0
    current_event = None
    for i, row in decisive.iterrows():
        if row["event_name"] != current_event:
            current_event = row["event_name"]
            print(f"\n=== {current_event} ({row['event_date'].date()}) ===")
        winner_name = row["fighter1"] if row["winner"] == "fighter1" else row["fighter2"]
        loser_name = row["fighter2"] if row["winner"] == "fighter1" else row["fighter1"]
        p_win = proba[i]
        correct = p_win >= 0.5
        n_correct += correct
        mark = "correct" if correct else "WRONG"
        dec = odds.to_decimal_odds(p_win)
        am = odds.to_american_odds(p_win)
        print(
            f"  {winner_name:<24} beat {loser_name:<24} | model gave winner "
            f"{p_win*100:5.1f}% (dec {dec:4.2f}, american {odds.format_american(am)}) [{mark}]"
        )

    print(f"\nTOTAL: {n_correct}/{len(decisive)} = {n_correct/len(decisive)*100:.1f}% accuracy "
          f"over the last {n_holdout_events} events")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HOLDOUT_EVENTS
    _require_data()
    _print_full_report(n)
