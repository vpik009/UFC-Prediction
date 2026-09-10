from __future__ import annotations

import argparse
import sys
from datetime import datetime

from . import dataset, odds
from .features import FEATURE_NAMES, FighterState, build_training_frame, estimate_scheduled_rounds, pair_features
from .http_client import UfcStatsClient
from .model import _fit_logreg, load_model, predict_proba, save_model, train_and_select
from .scrape import parse_event, parse_fighter_bio, EventRef


def cmd_update(args):
    print("Scraping ufcstats.com for fighters/events/fights (this can take a few minutes on a cold cache)...")
    dataset.build_dataset(max_workers=args.workers, use_cache=not args.refresh)
    print("\nTraining model...")
    fights = dataset.load_fights_merged()
    fighters = dataset.load_fighters()
    df, _ = build_training_frame(fights, fighters)
    model, name, results = train_and_select(df)
    save_model(model, name, results)
    print(f"\nModel trained and saved ({name}).")


def _ensure_data():
    if not dataset.FIGHTS_CSV.exists() or not dataset.FIGHTERS_CSV.exists():
        print("No local data found. Run `ufcpred update` first to scrape ufcstats.com.")
        sys.exit(1)


def _load_states():
    fights = dataset.load_fights_merged()
    fighters = dataset.load_fighters()
    _, states = build_training_frame(fights, fighters)
    return states


def _get_or_fetch_state(client: UfcStatsClient, states: dict, name: str, url: str) -> FighterState:
    state = states.get(url)
    if state is not None:
        return state
    print(f"  (new fighter, fetching bio: {name})")
    try:
        bio = parse_fighter_bio(client, name, url)
        return FighterState(
            name=name, url=url, height_in=bio.height_in, reach_in=bio.reach_in,
            dob=bio.dob, stance=bio.stance,
        )
    except Exception:
        return FighterState(name=name, url=url)


def _fmt_name_elo(state: FighterState) -> str:
    return f"{state.name} ({state.elo:.0f})"


def _print_matchup(x: FighterState, y: FighterState, p_x: float, weight_class: str):
    p_y = 1 - p_x
    dec_x, dec_y = odds.to_decimal_odds(p_x), odds.to_decimal_odds(p_y)
    am_x, am_y = odds.to_american_odds(p_x), odds.to_american_odds(p_y)
    fav = x if p_x >= p_y else y

    print(f"  [{weight_class}]")
    print(f"    {_fmt_name_elo(x):<36} {p_x*100:5.1f}%   dec {dec_x:5.2f}   american {odds.format_american(am_x)}")
    print(f"    {_fmt_name_elo(y):<36} {p_y*100:5.1f}%   dec {dec_y:5.2f}   american {odds.format_american(am_y)}")
    print(f"    -> favorite: {fav.name}\n")


def cmd_predict(args):
    _ensure_data()
    model, model_name, feature_names = load_model()
    print(f"Loaded model: {model_name}\n")

    client = UfcStatsClient()
    print("Fetching upcoming events...")
    events = dataset.load_upcoming_events()
    if not events:
        print("No upcoming events found.")
        return
    top = events[: args.n]

    print(f"\nUpcoming UFC events:")
    for i, e in enumerate(top, 1):
        date_str = e["date"][:10] if e["date"] else "TBD"
        print(f"  {i}. {e['name']}  ({date_str})")

    if args.event:
        choice = args.event
    else:
        raw = input(f"\nSelect an event [1-{len(top)}]: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Invalid selection.")
            return
    if not (1 <= choice <= len(top)):
        print("Invalid selection.")
        return

    chosen = top[choice - 1]
    event_date = datetime.fromisoformat(chosen["date"]) if chosen["date"] else datetime.now()
    print(f"\nFetching card for: {chosen['name']}...")
    ev = EventRef(name=chosen["name"], url=chosen["url"], date=event_date)
    fights = parse_event(client, ev, use_cache=False)
    if not fights:
        print("Could not find any fights for this event yet (card may not be announced).")
        return

    print("\nLoading fighter ratings/history...")
    states = _load_states()

    print(f"\n=== {chosen['name']} ({event_date.date()}) — Predictions ===\n")
    for f in fights:
        x = _get_or_fetch_state(client, states, f.fighter1, f.fighter1_url)
        y = _get_or_fetch_state(client, states, f.fighter2, f.fighter2_url)
        scheduled_rounds = estimate_scheduled_rounds(f.is_title_fight)
        feats = pair_features(x, y, event_date, f.event_country, scheduled_rounds)
        p_x = predict_proba(model, feats)
        _print_matchup(x, y, p_x, f.weight_class)


def _find_fighter(states: dict, fighters_df, query: str):
    query_l = query.strip().lower()
    names_l = fighters_df["name"].str.lower()

    exact = fighters_df[names_l == query_l]
    if len(exact) >= 1:
        if len(exact) > 1:
            print(f"  (note: multiple fighters named {query!r}, using the first match)")
        return states.get(exact.iloc[0]["url"])

    contains = fighters_df[names_l.str.contains(query_l, regex=False)]
    if len(contains) == 1:
        return states.get(contains.iloc[0]["url"])
    if len(contains) == 0:
        print(f"No fighter found matching {query!r}. Use their exact ufcstats.com name, "
              f"or run `ufcpred update` first if they're new.")
        return None

    print(f"Multiple fighters match {query!r}, please be more specific:")
    for n in contains["name"].tolist()[:10]:
        print(f"  - {n}")
    return None


def cmd_compare(args):
    _ensure_data()
    model, model_name, _ = load_model()
    print(f"Loaded model: {model_name}")

    fighters_df = dataset.load_fighters()
    states = _load_states()

    x = _find_fighter(states, fighters_df, args.fighter_a)
    y = _find_fighter(states, fighters_df, args.fighter_b)
    if x is None or y is None:
        sys.exit(1)

    fight_date = datetime.fromisoformat(args.date) if args.date else datetime.now()

    try:
        dec_book_x = odds.parse_book_odds(args.odds_a)
        dec_book_y = odds.parse_book_odds(args.odds_b)
    except ValueError as e:
        print(f"Could not parse odds: {e}")
        sys.exit(1)

    feats = pair_features(x, y, fight_date, args.country, args.rounds)
    p_x = predict_proba(model, feats)
    p_y = 1 - p_x

    implied_x = odds.decimal_to_implied_prob(dec_book_x)
    implied_y = odds.decimal_to_implied_prob(dec_book_y)
    vig = implied_x + implied_y - 1.0
    fair_x, fair_y = odds.devig_two_way(implied_x, implied_y)

    print(f"\n=== {_fmt_name_elo(x)} vs {_fmt_name_elo(y)} ({fight_date.date()}) ===\n")
    header = f"  {'':<36}{'model p':>10}{'book odds':>12}{'book implied':>14}{'fair market p':>16}{'edge':>9}"
    print(header)
    for name_elo, p_model, dec_book, implied, fair in (
        (_fmt_name_elo(x), p_x, dec_book_x, implied_x, fair_x),
        (_fmt_name_elo(y), p_y, dec_book_y, implied_y, fair_y),
    ):
        am_book = round((dec_book - 1) * 100) if dec_book >= 2 else round(-100 / (dec_book - 1))
        edge = p_model - fair
        print(
            f"  {name_elo:<36}{p_model*100:9.1f}%{odds.format_american(am_book):>12}"
            f"{implied*100:13.1f}%{fair*100:15.1f}%{edge*100:+8.1f}%"
        )

    print(f"\n  Bookmaker vig on this line: {vig*100:.1f}%")

    best_name, best_p_model, best_fair, best_dec = (
        (x.name, p_x, fair_x, dec_book_x) if (p_x - fair_x) >= (p_y - fair_y) else (y.name, p_y, fair_y, dec_book_y)
    )
    edge = best_p_model - best_fair
    if edge <= 0:
        print(f"\n  No edge found on either side — the model doesn't disagree enough with the market to bet.")
        return

    full_kelly = odds.kelly_fraction(best_p_model, best_dec)
    stake_kelly = full_kelly * args.kelly_fraction
    print(f"\n  Model favors {best_name} by {edge*100:.1f} points over the fair market price.")
    print(f"  Full Kelly stake:                {full_kelly*100:5.1f}% of bankroll")
    print(f"  Suggested stake ({args.kelly_fraction:.2f}x Kelly): {stake_kelly*100:5.1f}% of bankroll")
    if args.bankroll:
        print(f"  -> ${stake_kelly * args.bankroll:,.2f} on a ${args.bankroll:,.2f} bankroll")


def cmd_history(args):
    _ensure_data()
    fights_all = dataset.load_fights_merged().dropna(subset=["event_date"])
    fighters_df = dataset.load_fighters()

    events = (
        fights_all[["event_name", "event_date"]]
        .drop_duplicates()
        .sort_values("event_date", ascending=False)
        .reset_index(drop=True)
    )
    top = events.head(args.n)

    print("Past UFC events:")
    for i, row in enumerate(top.itertuples(), 1):
        print(f"  {i}. {row.event_name}  ({row.event_date.date()})")

    if args.event:
        choice = args.event
    else:
        raw = input(f"\nSelect an event [1-{len(top)}]: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            print("Invalid selection.")
            return
    if not (1 <= choice <= len(top)):
        print("Invalid selection.")
        return

    chosen = top.iloc[choice - 1]
    event_name, event_date = chosen["event_name"], chosen["event_date"]

    print(f"\nRetraining on only fights before {event_date.date()} "
          f"(so these are genuinely 'what the model would have said at the time')...")
    train_fights = fights_all[fights_all["event_date"] < event_date]
    df, states = build_training_frame(train_fights, fighters_df)
    model = _fit_logreg(df[FEATURE_NAMES].values, df["label"].values)

    card = fights_all[fights_all["event_name"] == event_name]

    print(f"\n=== {event_name} ({event_date.date()}) — model's predictions at the time vs. what happened ===\n")
    n_correct = 0
    n_graded = 0
    for f in card.itertuples():
        x = states.get(f.fighter1_url) or FighterState(name=f.fighter1, url=f.fighter1_url)
        y = states.get(f.fighter2_url) or FighterState(name=f.fighter2, url=f.fighter2_url)
        scheduled_rounds = estimate_scheduled_rounds(bool(f.is_title_fight), f.method, f.round)
        feats = pair_features(
            x, y, event_date, f.event_country if isinstance(f.event_country, str) else None, scheduled_rounds
        )
        p_x = predict_proba(model, feats)
        p_y = 1 - p_x
        picked_fighter1 = p_x >= 0.5

        print(f"  [{f.weight_class}]")
        for state, p, is_fighter1 in ((x, p_x, True), (y, p_y, False)):
            dec, am = odds.to_decimal_odds(p), odds.to_american_odds(p)
            won = (f.winner == "fighter1") == is_fighter1 if f.winner in ("fighter1", "fighter2") else False
            tag = "  <- won" if won else ""
            print(f"    {_fmt_name_elo(state):<36} {p*100:5.1f}%   dec {dec:5.2f}   american {odds.format_american(am)}{tag}")

        if f.winner in ("fighter1", "fighter2"):
            correct = (picked_fighter1 and f.winner == "fighter1") or (not picked_fighter1 and f.winner == "fighter2")
            n_graded += 1
            n_correct += correct
            print(f"    -> model {'correctly' if correct else 'incorrectly'} favored the winner\n")
        else:
            print(f"    -> no official winner (draw/no-contest)\n")

    if n_graded:
        print(f"Overall: {n_correct}/{n_graded} correct on this card ({n_correct/n_graded*100:.0f}%)")


def build_parser():
    parser = argparse.ArgumentParser(prog="ufcpred", description="UFC fight outcome predictor")
    sub = parser.add_subparsers(dest="command")

    p_update = sub.add_parser("update", help="Scrape ufcstats.com and (re)train the model")
    p_update.add_argument("--refresh", action="store_true", help="Ignore local page cache and re-download everything")
    p_update.add_argument("--workers", type=int, default=10, help="Concurrent scraping threads")
    p_update.set_defaults(func=cmd_update)

    p_predict = sub.add_parser("predict", help="Pick an upcoming event and get predictions")
    p_predict.add_argument("-n", type=int, default=3, help="How many upcoming events to list")
    p_predict.add_argument("--event", type=int, default=None, help="Pick event number non-interactively")
    p_predict.set_defaults(func=cmd_predict)

    p_compare = sub.add_parser(
        "compare", help="Compare the model's probability to bookmaker odds for a specific matchup"
    )
    p_compare.add_argument("fighter_a", help="First fighter's name (as listed on ufcstats.com)")
    p_compare.add_argument("fighter_b", help="Second fighter's name")
    p_compare.add_argument("--odds-a", required=True, help="Bookmaker odds for fighter A: American (-150, +130) or decimal (1.67)")
    p_compare.add_argument("--odds-b", required=True, help="Bookmaker odds for fighter B")
    p_compare.add_argument("--date", default=None, help="Fight date as YYYY-MM-DD (default: today)")
    p_compare.add_argument(
        "--country", default=None,
        help="Country the fight is held in (e.g. 'Brazil'), to factor in travel/away disadvantage",
    )
    p_compare.add_argument(
        "--rounds", type=int, default=3, choices=[3, 5],
        help="Scheduled rounds (5 for a title fight), affects the cardio/pace feature (default 3)",
    )
    p_compare.add_argument("--bankroll", type=float, default=None, help="Bankroll in dollars, to size a suggested stake")
    p_compare.add_argument(
        "--kelly-fraction", type=float, default=0.25,
        help="Fraction of full Kelly to suggest staking (default 0.25 = quarter Kelly)",
    )
    p_compare.set_defaults(func=cmd_compare)

    p_history = sub.add_parser(
        "history", help="Pick a past event and see the model's odds at the time vs. what actually happened"
    )
    p_history.add_argument("-n", type=int, default=15, help="How many recent past events to list")
    p_history.add_argument("--event", type=int, default=None, help="Pick event number non-interactively")
    p_history.set_defaults(func=cmd_history)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        args = parser.parse_args(["predict"])
    args.func(args)


if __name__ == "__main__":
    main()
