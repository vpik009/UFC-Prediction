"""Parsers + orchestration for pulling fighter/event/fight data from ufcstats.com."""
from __future__ import annotations

import re
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

from .http_client import UfcStatsClient

BASE = "http://ufcstats.com"
EVENTS_COMPLETED_URL = f"{BASE}/statistics/events/completed?page=all"
EVENTS_UPCOMING_URL = f"{BASE}/statistics/events/upcoming?page=all"
FIGHTERS_INDEX_URL = f"{BASE}/statistics/fighters?char={{letter}}&page=all"


@dataclass
class EventRef:
    name: str
    url: str
    date: Optional[datetime]
    location: Optional[str] = None


@dataclass
class FightRow:
    event_name: str
    event_date: Optional[datetime]
    event_country: Optional[str]
    fighter1: str
    fighter1_url: str
    fighter2: str
    fighter2_url: str
    winner: Optional[str]  # 'fighter1', 'fighter2', or None (draw/nc)
    weight_class: str
    method: str
    round: Optional[int]
    time: Optional[str]
    fight_url: Optional[str] = None
    is_title_fight: bool = False
    kd1: Optional[int] = None
    kd2: Optional[int] = None
    str1: Optional[int] = None
    str2: Optional[int] = None
    td1: Optional[int] = None
    td2: Optional[int] = None
    sub1: Optional[int] = None
    sub2: Optional[int] = None


@dataclass
class FightStats:
    """Extra per-fight detail not on the event page: strike/takedown
    *attempts* (not just landed, so accuracy/defense is computable),
    control time, and round-1-vs-last-round output for a cardio/pace
    signal. Pulled from each fight's own ufcstats.com page."""

    fight_url: str
    n_rounds: int = 0
    f1_sig_str_landed: int = 0
    f1_sig_str_att: int = 0
    f2_sig_str_landed: int = 0
    f2_sig_str_att: int = 0
    f1_td_landed: int = 0
    f1_td_att: int = 0
    f2_td_landed: int = 0
    f2_td_att: int = 0
    f1_ctrl_sec: float = 0.0
    f2_ctrl_sec: float = 0.0
    f1_round1_str_landed: int = 0
    f1_round1_str_att: int = 0
    f2_round1_str_landed: int = 0
    f2_round1_str_att: int = 0
    f1_lastround_str_landed: int = 0
    f1_lastround_str_att: int = 0
    f2_lastround_str_landed: int = 0
    f2_lastround_str_att: int = 0


@dataclass
class FighterBio:
    name: str
    url: str
    height_in: Optional[float] = None
    reach_in: Optional[float] = None
    stance: Optional[str] = None
    dob: Optional[datetime] = None
    # entire professional MMA record (not just UFC), as shown on the fighter's
    # ufcstats.com header, e.g. "Record: 17-3-0" or "Record: 17-3-0 (1 NC)"
    pro_wins: Optional[int] = None
    pro_losses: Optional[int] = None
    pro_draws: Optional[int] = None


def _parse_date(text: str) -> Optional[datetime]:
    text = text.strip()
    if not text or text == "--":
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _int_or_none(text: str) -> Optional[int]:
    text = text.strip()
    if not text or text == "--":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def list_completed_events(client: UfcStatsClient) -> list[EventRef]:
    # never cached: this listing grows every time an event finishes, and is
    # the only mechanism by which `update` discovers newly-completed events
    html = client.get(EVENTS_COMPLETED_URL, use_cache=False)
    soup = BeautifulSoup(html, "lxml")
    events = []
    for row in soup.select("tbody tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        date_span = row.select_one("span.b-statistics__date")
        if not link:
            continue
        events.append(
            EventRef(
                name=link.get_text(strip=True),
                url=link["href"],
                date=_parse_date(date_span.get_text(strip=True)) if date_span else None,
            )
        )
    return events


def list_upcoming_events(client: UfcStatsClient) -> list[EventRef]:
    html = client.get(EVENTS_UPCOMING_URL, use_cache=False)
    soup = BeautifulSoup(html, "lxml")
    events = []
    for row in soup.select("tbody tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        date_span = row.select_one("span.b-statistics__date")
        if not link:
            continue
        events.append(
            EventRef(
                name=link.get_text(strip=True),
                url=link["href"],
                date=_parse_date(date_span.get_text(strip=True)) if date_span else None,
            )
        )
    events.sort(key=lambda e: e.date or datetime.max)
    return events


def _split_two_line(cell) -> tuple[str, str]:
    ps = cell.find_all("p")
    if len(ps) >= 2:
        return ps[0].get_text(strip=True), ps[1].get_text(strip=True)
    text = cell.get_text(" ", strip=True)
    parts = text.split()
    if len(parts) == 2:
        return parts[0], parts[1]
    return text, ""


def parse_event(client: UfcStatsClient, event: EventRef, use_cache: bool = True) -> list[FightRow]:
    html = client.get(event.url, use_cache=use_cache)
    soup = BeautifulSoup(html, "lxml")
    if event.date is None or event.location is None:
        info_items = soup.select("li.b-list__box-list-item")
        for item in info_items:
            text = item.get_text(" ", strip=True)
            if event.date is None and text.lower().startswith("date:"):
                event.date = _parse_date(text.split(":", 1)[1])
            elif event.location is None and text.lower().startswith("location:"):
                event.location = text.split(":", 1)[1].strip() or None

    country = event.location.split(",")[-1].strip() if event.location else None

    fights = []
    for row in soup.select("tbody tr.b-fight-details__table-row"):
        cells = row.select("td.b-fight-details__table-col")
        if len(cells) < 10:
            continue
        wl_cell, fighter_cell, kd_cell, str_cell, td_cell, sub_cell, wc_cell, method_cell, round_cell, time_cell = cells[:10]

        fighter_links = fighter_cell.select("a")
        if len(fighter_links) < 2:
            continue
        f1_name, f2_name = fighter_links[0].get_text(strip=True), fighter_links[1].get_text(strip=True)
        f1_url, f2_url = fighter_links[0]["href"], fighter_links[1]["href"]

        wl_flags = [f.get_text(strip=True).lower() for f in wl_cell.select("i.b-flag__text")]
        winner = None
        if wl_flags:
            if wl_flags[0] == "win":
                winner = "fighter1"
            elif len(wl_flags) > 1 and wl_flags[1] == "win":
                winner = "fighter2"
            # 'draw' / 'nc' -> winner stays None

        kd1, kd2 = (_int_or_none(x) for x in _split_two_line(kd_cell))
        str1, str2 = (_int_or_none(x) for x in _split_two_line(str_cell))
        td1, td2 = (_int_or_none(x) for x in _split_two_line(td_cell))
        sub1, sub2 = (_int_or_none(x) for x in _split_two_line(sub_cell))

        weight_class = wc_cell.get_text(" ", strip=True)
        is_title_fight = wc_cell.find("img", src=lambda s: s and "belt" in s) is not None
        method = method_cell.get_text(" ", strip=True)
        round_ = _int_or_none(round_cell.get_text(strip=True))
        time_ = time_cell.get_text(strip=True) or None
        fight_url = row.get("data-link") or None

        fights.append(
            FightRow(
                event_name=event.name,
                event_date=event.date,
                event_country=country,
                fighter1=f1_name,
                fighter1_url=f1_url,
                fighter2=f2_name,
                fighter2_url=f2_url,
                winner=winner,
                weight_class=weight_class,
                method=method,
                round=round_,
                time=time_,
                fight_url=fight_url,
                is_title_fight=is_title_fight,
                kd1=kd1, kd2=kd2,
                str1=str1, str2=str2,
                td1=td1, td2=td2,
                sub1=sub1, sub2=sub2,
            )
        )
    return fights


_HEIGHT_RE = re.compile(r"(\d+)'\s*(\d+)")


def _parse_height(text: str) -> Optional[float]:
    m = _HEIGHT_RE.search(text)
    if not m:
        return None
    feet, inches = int(m.group(1)), int(m.group(2))
    return feet * 12 + inches


def _parse_reach(text: str) -> Optional[float]:
    text = text.strip().replace('"', "")
    if not text or text == "--":
        return None
    try:
        return float(text)
    except ValueError:
        return None


_RECORD_RE = re.compile(r"Record:\s*(\d+)-(\d+)-(\d+)")


def parse_fighter_bio(client: UfcStatsClient, name: str, url: str, use_cache: bool = True) -> FighterBio:
    html = client.get(url, use_cache=use_cache)
    soup = BeautifulSoup(html, "lxml")
    bio = FighterBio(name=name, url=url)

    record_el = soup.select_one("span.b-content__title-record")
    if record_el:
        m = _RECORD_RE.search(record_el.get_text(" ", strip=True))
        if m:
            bio.pro_wins, bio.pro_losses, bio.pro_draws = int(m.group(1)), int(m.group(2)), int(m.group(3))

    for item in soup.select("li.b-list__box-list-item"):
        text = item.get_text(" ", strip=True)
        if ":" not in text:
            continue
        key, _, val = text.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key == "height":
            bio.height_in = _parse_height(val)
        elif key == "reach":
            bio.reach_in = _parse_reach(val)
        elif key == "stance":
            bio.stance = val or None
        elif key == "dob":
            bio.dob = _parse_date(val)
    return bio


def scrape_all_fighter_urls(client: UfcStatsClient, max_workers: int = 8) -> dict[str, str]:
    """Fallback / completeness pass: enumerate the full fighters index (a-z)."""
    urls: dict[str, str] = {}

    def fetch_letter(letter: str):
        html = client.get(FIGHTERS_INDEX_URL.format(letter=letter))
        soup = BeautifulSoup(html, "lxml")
        out = {}
        for a in soup.select("td.b-statistics__table-col a.b-link"):
            out[a.get_text(strip=True)] = a["href"]
        return out

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fetch_letter, l): l for l in string.ascii_lowercase}
        for fut in as_completed(futs):
            urls.update(fut.result())
    return urls


def scrape_events_concurrently(
    client: UfcStatsClient,
    events: list[EventRef],
    max_workers: int = 8,
    use_cache: bool = True,
    force_refresh_urls: set[str] | None = None,
) -> list[FightRow]:
    """force_refresh_urls: event URLs to always re-fetch live even if cached
    (e.g. events too recent to trust a cached, possibly-pre-results copy)."""
    all_fights: list[FightRow] = []
    force_refresh_urls = force_refresh_urls or set()

    def work(ev: EventRef):
        try:
            ev_use_cache = use_cache and ev.url not in force_refresh_urls
            return parse_event(client, ev, use_cache=ev_use_cache)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed to parse event {ev.name!r}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(work, ev): ev for ev in events}
        done = 0
        total = len(events)
        for fut in as_completed(futs):
            all_fights.extend(fut.result())
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  events parsed: {done}/{total}")
    return all_fights


def scrape_fighter_bios_concurrently(
    client: UfcStatsClient,
    fighters: dict[str, str],
    max_workers: int = 8,
    use_cache: bool = True,
    force_refresh_urls: set[str] | None = None,
) -> list[FighterBio]:
    """fighters: {url: name} — keyed by URL since names aren't unique.
    force_refresh_urls: fighter URLs to always re-fetch live (their total
    pro record on the bio page grows every time they fight, so a fighter
    who's fought recently needs a fresh copy, not last run's cache)."""
    bios: list[FighterBio] = []
    force_refresh_urls = force_refresh_urls or set()

    def work(url_name):
        url, name = url_name
        try:
            f_use_cache = use_cache and url not in force_refresh_urls
            return parse_fighter_bio(client, name, url, use_cache=f_use_cache)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed to parse fighter {name!r}: {e}")
            return FighterBio(name=name, url=url)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(work, item): item for item in fighters.items()}
        done = 0
        total = len(fighters)
        for fut in as_completed(futs):
            bios.append(fut.result())
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  fighters parsed: {done}/{total}")
    return bios


_LANDED_ATT_RE = re.compile(r"(\d+)\s+of\s+(\d+)")


def _parse_landed_att(text: str) -> tuple[int, int]:
    m = _LANDED_ATT_RE.search(text or "")
    if not m:
        return 0, 0
    return int(m.group(1)), int(m.group(2))


def _parse_ctrl_seconds(text: str) -> float:
    text = (text or "").strip()
    if not text or text == "--" or ":" not in text:
        return 0.0
    try:
        mm, ss = text.split(":")
        return int(mm) * 60 + int(ss)
    except ValueError:
        return 0.0


def _parse_fight_stat_row(row) -> Optional[tuple]:
    """cell order: Fighter, KD, Sig.str, Sig.str%, Total str, Td, Td%, Sub att, Rev, Ctrl."""
    cells = row.select("td")
    if len(cells) < 10:
        return None
    s1_text, s2_text = _split_two_line(cells[2])
    t1_text, t2_text = _split_two_line(cells[5])
    c1_text, c2_text = _split_two_line(cells[9])
    s1l, s1a = _parse_landed_att(s1_text)
    s2l, s2a = _parse_landed_att(s2_text)
    t1l, t1a = _parse_landed_att(t1_text)
    t2l, t2a = _parse_landed_att(t2_text)
    return s1l, s1a, s2l, s2a, t1l, t1a, t2l, t2a, _parse_ctrl_seconds(c1_text), _parse_ctrl_seconds(c2_text)


def parse_fight_stats(client: UfcStatsClient, url: str, use_cache: bool = True) -> Optional[FightStats]:
    html = client.get(url, use_cache=use_cache)
    soup = BeautifulSoup(html, "lxml")

    totals_table = None
    round_table = None
    for t in soup.find_all("table"):
        theads = t.select("thead")
        if not theads:
            continue
        header_text = theads[0].get_text(" ", strip=True)
        if "Sig. str." in header_text and "KD" in header_text:
            if len(theads) > 1:
                round_table = t
            elif totals_table is None:
                totals_table = t

    if totals_table is None:
        return None
    totals_row = totals_table.select_one("tbody tr")
    if totals_row is None:
        return None
    parsed = _parse_fight_stat_row(totals_row)
    if parsed is None:
        return None
    s1l, s1a, s2l, s2a, t1l, t1a, t2l, t2a, c1, c2 = parsed

    stats = FightStats(
        fight_url=url,
        f1_sig_str_landed=s1l, f1_sig_str_att=s1a,
        f2_sig_str_landed=s2l, f2_sig_str_att=s2a,
        f1_td_landed=t1l, f1_td_att=t1a,
        f2_td_landed=t2l, f2_td_att=t2a,
        f1_ctrl_sec=c1, f2_ctrl_sec=c2,
    )

    round_rows = round_table.select("tbody tr") if round_table is not None else []
    if round_rows:
        stats.n_rounds = len(round_rows)
        r1 = _parse_fight_stat_row(round_rows[0])
        if r1:
            stats.f1_round1_str_landed, stats.f1_round1_str_att = r1[0], r1[1]
            stats.f2_round1_str_landed, stats.f2_round1_str_att = r1[2], r1[3]
        rlast = _parse_fight_stat_row(round_rows[-1])
        if rlast:
            stats.f1_lastround_str_landed, stats.f1_lastround_str_att = rlast[0], rlast[1]
            stats.f2_lastround_str_landed, stats.f2_lastround_str_att = rlast[2], rlast[3]
    else:
        # no separate round-by-round table means the whole fight was 1 round
        stats.n_rounds = 1
        stats.f1_round1_str_landed = stats.f1_lastround_str_landed = s1l
        stats.f1_round1_str_att = stats.f1_lastround_str_att = s1a
        stats.f2_round1_str_landed = stats.f2_lastround_str_landed = s2l
        stats.f2_round1_str_att = stats.f2_lastround_str_att = s2a

    return stats


def scrape_fight_stats_concurrently(
    client: UfcStatsClient,
    fight_urls: list[str],
    max_workers: int = 10,
    use_cache: bool = True,
    force_refresh_urls: set[str] | None = None,
) -> list[FightStats]:
    """force_refresh_urls: fight URLs to always re-fetch live (a fight from
    a recent event may have been cached before ufcstats finished posting
    its stats)."""
    results: list[FightStats] = []
    force_refresh_urls = force_refresh_urls or set()

    def work(url: str):
        try:
            u_use_cache = use_cache and url not in force_refresh_urls
            return parse_fight_stats(client, url, use_cache=u_use_cache)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed to parse fight stats {url!r}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(work, url): url for url in fight_urls}
        done = 0
        total = len(fight_urls)
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                results.append(res)
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  fight stats parsed: {done}/{total}")
    return results
