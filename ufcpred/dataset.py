"""Build and load the local fights/fighters dataset scraped from ufcstats.com."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .http_client import UfcStatsClient
from .scrape import (
    list_completed_events,
    list_upcoming_events,
    scrape_events_concurrently,
    scrape_fight_stats_concurrently,
    scrape_fighter_bios_concurrently,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGHTS_CSV = DATA_DIR / "fights.csv"
FIGHTERS_CSV = DATA_DIR / "fighters.csv"
FIGHT_STATS_CSV = DATA_DIR / "fight_stats.csv"
UPCOMING_JSON = DATA_DIR / "upcoming_events.json"


def _add_pre_ufc_record(fights_df: pd.DataFrame, fighters_df: pd.DataFrame) -> pd.DataFrame:
    """pro_wins/pro_losses (from ufcstats.com's header) cover a fighter's whole
    career; subtracting their UFC-only wins/losses isolates the pre-UFC
    (regional/other-promotion) component of that record."""
    ufc_wins = pd.concat(
        [
            fights_df.loc[fights_df.winner == "fighter1", "fighter1_url"],
            fights_df.loc[fights_df.winner == "fighter2", "fighter2_url"],
        ]
    ).value_counts()
    ufc_losses = pd.concat(
        [
            fights_df.loc[fights_df.winner == "fighter2", "fighter1_url"],
            fights_df.loc[fights_df.winner == "fighter1", "fighter2_url"],
        ]
    ).value_counts()

    fighters_df = fighters_df.set_index("url")
    ufc_w = ufc_wins.reindex(fighters_df.index).fillna(0)
    ufc_l = ufc_losses.reindex(fighters_df.index).fillna(0)
    fighters_df["pre_ufc_wins"] = (fighters_df["pro_wins"].fillna(0) - ufc_w).clip(lower=0)
    fighters_df["pre_ufc_losses"] = (fighters_df["pro_losses"].fillna(0) - ufc_l).clip(lower=0)
    return fighters_df.reset_index()


# Events/fights/fighter records within this many days of "now" are always
# re-fetched live, even if a cached copy exists — ufcstats.com can list an
# event as "completed" before results/stats are fully posted (we've seen
# this happen), and a fighter's total pro record on their bio page grows
# every time they fight. Older, settled history is safe to cache forever.
RECENCY_BUFFER_DAYS = 14


def build_dataset(max_workers: int = 10, use_cache: bool = True) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = UfcStatsClient()
    now = datetime.now()

    print("Fetching list of completed events...")
    events = list_completed_events(client)
    print(f"  {len(events)} completed events found")

    recent_event_urls = {
        e.url for e in events if e.date is None or (now - e.date).days <= RECENCY_BUFFER_DAYS
    }
    if recent_event_urls:
        print(f"  {len(recent_event_urls)} event(s) within {RECENCY_BUFFER_DAYS} days — always re-fetched live")

    print("Scraping fight results for every event (cached pages are skipped, recent ones aren't)...")
    fights = scrape_events_concurrently(
        client, events, max_workers=max_workers, use_cache=use_cache, force_refresh_urls=recent_event_urls
    )
    print(f"  {len(fights)} total fights scraped")

    fights_df = pd.DataFrame([f.__dict__ for f in fights])
    fights_df.to_csv(FIGHTS_CSV, index=False)
    print(f"  saved -> {FIGHTS_CSV}")

    recent_fight_urls = {
        f.fight_url
        for f in fights
        if f.fight_url and (f.event_date is None or (now - f.event_date).days <= RECENCY_BUFFER_DAYS)
    }
    fight_urls = fights_df["fight_url"].dropna().unique().tolist()
    print(f"Scraping per-fight stats (attempts, round-by-round) for {len(fight_urls)} fights...")
    fight_stats = scrape_fight_stats_concurrently(
        client, fight_urls, max_workers=max_workers, use_cache=use_cache, force_refresh_urls=recent_fight_urls
    )
    fight_stats_df = pd.DataFrame([s.__dict__ for s in fight_stats])
    fight_stats_df.to_csv(FIGHT_STATS_CSV, index=False)
    print(f"  {len(fight_stats_df)} fights' stats saved -> {FIGHT_STATS_CSV}")

    # keyed by URL, not name: multiple distinct fighters can share a name
    # (e.g. several UFC roster members named "Bruno Silva"), and URL is the
    # only thing ufcstats.com guarantees is unique per fighter
    unique_fighters: dict[str, str] = {}
    recent_fighter_urls: set[str] = set()
    for f in fights:
        unique_fighters[f.fighter1_url] = f.fighter1
        unique_fighters[f.fighter2_url] = f.fighter2
        if f.event_date is None or (now - f.event_date).days <= RECENCY_BUFFER_DAYS:
            recent_fighter_urls.add(f.fighter1_url)
            recent_fighter_urls.add(f.fighter2_url)
    print(f"Scraping bio pages for {len(unique_fighters)} unique fighters "
          f"({len(recent_fighter_urls)} refreshed for up-to-date records)...")
    bios = scrape_fighter_bios_concurrently(
        client, unique_fighters, max_workers=max_workers, use_cache=use_cache, force_refresh_urls=recent_fighter_urls
    )
    fighters_df = pd.DataFrame([b.__dict__ for b in bios])
    fighters_df = _add_pre_ufc_record(fights_df, fighters_df)
    fighters_df.to_csv(FIGHTERS_CSV, index=False)
    print(f"  saved -> {FIGHTERS_CSV}")

    print("Fetching upcoming events...")
    upcoming = list_upcoming_events(client)
    upcoming_payload = [
        {"name": e.name, "url": e.url, "date": e.date.isoformat() if e.date else None}
        for e in upcoming
    ]
    UPCOMING_JSON.write_text(json.dumps(upcoming_payload, indent=2))
    print(f"  {len(upcoming)} upcoming events -> {UPCOMING_JSON}")


def load_fights() -> pd.DataFrame:
    df = pd.read_csv(FIGHTS_CSV, parse_dates=["event_date"])
    return df


def load_fighters() -> pd.DataFrame:
    df = pd.read_csv(FIGHTERS_CSV, parse_dates=["dob"])
    return df


def load_fight_stats() -> pd.DataFrame:
    return pd.read_csv(FIGHT_STATS_CSV)


def load_fights_merged() -> pd.DataFrame:
    """fights.csv left-joined with fight_stats.csv (attempts, round-by-round)
    on fight_url. Fights with no matching stats (scrape failure, or no
    fight_url at all for very old cards) simply carry NaN in those columns —
    features.build_training_frame() fills them to 0/neutral."""
    fights = load_fights()
    if not FIGHT_STATS_CSV.exists():
        return fights
    stats = load_fight_stats()
    return fights.merge(stats, on="fight_url", how="left")


def load_upcoming_events() -> list[dict]:
    return json.loads(UPCOMING_JSON.read_text())
