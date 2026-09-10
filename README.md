# ufcpred

Predicts UFC fight winners as a probability, and converts that into fair
American and European (decimal) odds. Data is scraped directly from
ufcstats.com; nothing is hand-entered.

## Setup

```
pip install -r requirements.txt
```

## Usage

Pull all historical fighters/events/fights from ufcstats.com and train the
model (takes a few minutes on a cold cache; re-running later is fast, since
old/settled pages are served from `data/raw_cache/` instead of re-fetched):

```
python -m ufcpred update
```

Re-running `update` genuinely refreshes, not just appends: the completed-
events list itself is always re-fetched live (that's how new events get
discovered at all), and anything dated within the last 14 days — recent
events, their fights, and the bio pages of fighters who fought in them —
is always re-fetched live too rather than served from a stale cache, since
ufcstats.com can list an event as "completed" before its stats finish
posting, and a fighter's total pro record on their bio page grows every
time they fight. Only comfortably-settled history (older than 14 days)
is cached. `--refresh` ignores the cache entirely and re-downloads
everything, for when you don't trust it.

Pick from the next few upcoming UFC events and get predictions:

```
python -m ufcpred predict
```

Each matchup prints the model's win probability for both fighters plus the
equivalent fair (no-vig) decimal and American odds, e.g.:

```
  [Flyweight]
    Joshua Van (1679)             71.0%   dec  1.41   american -244
    Alexandre Pantoja (1698)      29.0%   dec  3.44   american +244
    -> favorite: Joshua Van
```

The number in parentheses is the fighter's current Elo rating (starts
everyone at 1500; `compare` and `history` show it too — `history` shows
the rating as of just before that event, not the current one).

Options: `python -m ufcpred predict -n 5` to list more upcoming events,
`--event 2` to pick one non-interactively.

Compare the model against a real bookmaker line to see if there's an edge:

```
python -m ufcpred compare "Joshua Van" "Alexandre Pantoja" --odds-a -200 --odds-b +170 --bankroll 1000
```

`--odds-a`/`--odds-b` accept American (`-200`, `+170`) or decimal (`1.5`)
odds. It prints the model's probability next to the bookmaker's line
(devigged into a fair market probability), the edge, and a suggested
stake sized with a fractional Kelly criterion (`--kelly-fraction`,
default 0.25 = quarter Kelly). No edge, no stake suggested.

Add `--country "Brazil"` to factor in travel (see below), and `--rounds 5`
for a title-fight-length cardio assumption (default 3) — both for a
hypothetical matchup; `predict` and `history` pick these up automatically
from the real event.

## How it works

- **Scraping** (`ufcpred/scrape.py`, `http_client.py`): ufcstats.com gates
  every request behind a small JS proof-of-work challenge; the client
  solves it once per session and caches every fetched page to disk.
- **Rating/features** (`ufcpred/elo.py`, `features.py`): every fighter gets
  an Elo rating updated after each fight (bonus K-factor for finishes and
  early finishes, provisional K-factor for fighters with <5 UFC fights),
  plus rolling per-minute output (knockdowns, significant strikes,
  takedowns, submission attempts), win streak, layoff time, and bio diffs
  (height, reach, age, stance). All of it is computed strictly from fights
  that happened *before* the fight being predicted, so there's no leakage.

  Fighters don't all start their UFC Elo track at a flat 1500: ufcstats.com's
  fighter header shows a full professional record (e.g. "17-3-0"), bigger
  than their UFC-only fight count because it includes pre-UFC (regional/
  other-promotion) fights we have no fight-by-fight data for. We can't
  compute a real recursive Elo for those (their opponents' ratings are
  unknown), so the starting rating is nudged by pre-UFC win rate, shrunk
  toward 0 for small samples and capped in magnitude — calibrated by grid
  search against real held-out UFC-debut fights, where it took debut-fight
  prediction accuracy from barely-a-coin-flip (52%) to 59%, and also lifted
  overall backtest accuracy (63.0% -> 64.3%). See `ufcpred/elo.py` for the
  exact formula and calibration numbers.

  Three more features go beyond simple linear age/experience:
  - **Prime distance**: rather than assuming "younger is always better",
    fighters outside a 27-33 "prime" window (too green *or* aging out)
    are penalized relative to one inside it.
  - **Decline risk**: an explicit interaction, not just two separate
    linear terms — years past the prime window are only counted as risk
    *while the fighter is currently on a losing streak*, so an aging
    fighter who's still winning isn't penalized, but aging + fading
    compounds.
  - **Travel**: ufcstats.com doesn't publish fighter nationality or
    hometown anywhere, so this isn't true "flew from home country"
    modeling — it's a proxy built from which countries each fighter has
    actually *fought* in before. A fighter walking into a country they've
    never fought in gets treated as more of an unknown/away disadvantage
    than one who's fought there repeatedly. Because it's fight-history-
    based rather than nationality-based, it can look "backwards" for a
    fighter whose entire UFC career happens to have been in the US even
    though they're a foreign national (e.g. it doesn't know Alex Pereira
    is Brazilian, only that all 13 of his UFC fights were in the USA).

  All three were grid-searched/validated against real backtest data before
  being kept — adding them moved overall accuracy from 64.3% to 64.8% and
  log loss from 0.646 to 0.643 on the last-50-events holdout.

  A second batch of features needed genuinely new data: each fight's
  individual ufcstats.com page (not the event page, which only has
  landed counts) has strike/takedown *attempts* and a round-by-round
  breakdown. ~8,900 of those pages were scraped to build:
  - **Style matchup**: takedown accuracy/defense and strike accuracy/
    defense (main effects), plus an explicit interaction —
    `style_matchup_diff` — for how much a fighter's takedown-attempt
    volume ("wrestler pressure") should exploit the opponent's actual
    takedown defense, not just whether either fighter grapples a lot.
  - **Cardio/pace**: per-fighter round-1-vs-last-round output rate
    ("fade"), scaled by how many rounds *this* fight is scheduled for
    (5 for a title fight, else 3 — ufcstats.com doesn't publish
    scheduled rounds directly, so decisions use their actual round
    count and finishes fall back to the title-fight flag, which
    undercounts non-title 5-round main events).
  - **Activity**: fights in the trailing 2 years, as a feature distinct
    from raw layoff time — a fighter who's fought once in 3 years reads
    differently from one who's fought 6 times in 3 years even with the
    same days-since-last-fight.
  - **Finishing power**: `ko_rate`/`sub_rate` split out instead of one
    generic finish rate, plus `power` — knockdowns per significant
    strike landed, a power signal independent of output volume (which
    `kd_per_min` alone conflates).

  Same discipline as the Elo work: these were validated on the real
  holdout, not assumed useful. `style_matchup_diff` and `activity_diff`
  turned out to carry real signal (correlate 0.08-0.10 with actual fight
  outcomes); `cardio_diff` and `power_diff` individually correlate close
  to zero — noisy in isolation — but the *full* feature set, backtested
  together, still moved overall accuracy from 64.8% to 65.8% and log loss
  from 0.643 to 0.639 on the last-50-events holdout, so all of them stayed
  in. (Mid-scrape, ufcstats.com started returning HTTP 429 "Too Many
  Requests" under the original 15-worker load, and a caching bug briefly
  saved those error pages as if they were real content — both are now
  fixed: `http_client.py` backs off on 429/non-200 instead of caching
  them, and the corrupted cache entries were purged and re-scraped.)
- **Model** (`ufcpred/model.py`): fights are listed on ufcstats.com winner
  first, so every fight is turned into two mirrored training rows
  (winner-perspective label=1, loser-perspective label=0) to cancel that
  ordering out. A regularized logistic regression and a gradient-boosted
  tree are both fit and compared on a time-ordered holdout (train on
  older fights, test on the most recent slice); whichever has lower log
  loss is kept and refit on the full dataset. Logistic regression usually
  wins — it's the standard, well-calibrated choice for this size of
  sports dataset.
- **Odds** (`ufcpred/odds.py`): straightforward probability -> decimal
  (`1/p`) and American odds conversion, no vig added.

See the model's odds for any past event and grade them against what
actually happened:

```
python -m ufcpred history
python -m ufcpred history -n 30 --event 5   # non-interactive
```

Lists the most recent past events, and for whichever one you pick,
retrains the model on only the fights that happened *before* it — so
what you see is genuinely what the model would have predicted at the
time, not something with the benefit of hindsight — then prints each
matchup's odds next to who actually won.

## Testing against real past events

```
pytest -s tests/test_backtest.py
```

trains the model only on fights before the last 50 events, then checks its
predictions against what actually happened (asserts it beats a coin flip,
the uniform-guess log loss, and that high-confidence picks — >=15-point
probability gap — do at least as well as the overall rate).

For a readable fight-by-fight report instead of a pass/fail:

```
python tests/test_backtest.py 6   # last 6 events
```
