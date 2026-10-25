# ufcpred

Predicts the probability that each fighter wins a UFC fight and converts it
into fair American and European (decimal) odds. All data is scraped from
ufcstats.com; nothing is entered by hand.

## Setup

```
pip install -r requirements.txt
```

## Usage

Pull all historical fighters, events and fights from ufcstats.com and train
the model. The first run takes a few minutes. Later runs are fast because
old, settled pages are served from `data/raw_cache/` instead of being
fetched again:

```
python -m ufcpred update
```

Running `update` again refreshes recent data as well as adding new events.
The completed-events list is always fetched live, since that is how new
events get discovered. Anything dated within the last 14 days is also
fetched live: recent events, their fights, and the bio pages of fighters
who fought in them. ufcstats.com can list an event as completed before its
stats are posted, and a fighter's pro record on their bio page grows with
every fight, so only history older than 14 days is cached. `--refresh`
ignores the cache and downloads everything again, for when you don't trust
it.

Pick one of the next few upcoming UFC events and get predictions:

```
python -m ufcpred predict
```

Each matchup prints both fighters' win probabilities with the matching fair
(no-vig) decimal and American odds:

```
  [Flyweight]
    Joshua Van (1679)             71.0%   dec  1.41   american -244
    Alexandre Pantoja (1698)      29.0%   dec  3.44   american +244
    -> favorite: Joshua Van
```

The number in parentheses is the fighter's current Elo rating. Everyone
starts at 1500. `compare` and `history` show the rating too, though
`history` shows it as of just before that event rather than today.

`python -m ufcpred predict -n 5` lists more upcoming events, and `--event 2`
picks one without the prompt.

To check for an edge, compare the model against a real bookmaker line:

```
python -m ufcpred compare "Joshua Van" "Alexandre Pantoja" --odds-a -200 --odds-b +170 --bankroll 1000
```

`--odds-a` and `--odds-b` accept American (`-200`, `+170`) or decimal
(`1.5`) odds. The output shows the model's probability next to the
bookmaker's line (devigged into a fair market probability), the edge, and a
suggested stake sized with fractional Kelly (`--kelly-fraction`, default
0.25, i.e. quarter Kelly). If there is no edge, it suggests no stake.

For a hypothetical matchup, add `--country "Brazil"` to account for travel
(see below) and `--rounds 5` to assume a title-fight length for cardio
(default 3). `predict` and `history` take both from the real event.

To see the model's odds for a past event next to the actual results:

```
python -m ufcpred history
python -m ufcpred history -n 30 --event 5   # non-interactive
```

This lists recent past events. For the one you pick, it retrains the model
on only the fights before that event, so the odds are what the model would
have predicted at the time, without hindsight. Each matchup's odds are
printed next to who won.

## How it works

A prediction goes through four steps:

1. Scrape every past UFC fight from ufcstats.com.
2. Replay those fights in date order, keeping a running profile of each
   fighter (Elo rating, output rates, record, and so on).
3. Describe a matchup as the difference between the two fighters'
   profiles on 25 features and feed that to a trained classifier, which
   returns the probability that fighter A wins. Fighter B gets 1 minus
   that.
4. Convert the probability into fair odds.

### Data

`ufcpred/scrape.py` and `http_client.py` handle scraping. ufcstats.com puts
every request behind a small JavaScript proof-of-work challenge. The client
solves it once per session and caches every fetched page on disk. On HTTP
429 or any other non-200 response it backs off and does not cache the page.

`update` writes three tables to `data/`:

- `fighters.csv`: one row per fighter from their bio page, with height,
  reach, date of birth, stance and full pro record.
- `fights.csv`: one row per fight from the event pages, with date, country,
  weight class, title-fight flag, winner, method, round, time, and each
  fighter's knockdowns, significant strikes, takedowns and submission
  attempts.
- `fight_stats.csv`: one row per fight from the fight's own page, with
  significant-strike and takedown attempts as well as landed counts,
  control time, and round 1 and last-round strike numbers. Control time is
  scraped but not used as a feature yet. The event page only has landed
  counts, so about 8,900 fight pages were scraped for this.

### Building the features

`ufcpred/features.py` walks through every fight from oldest to newest.
Before each fight it computes the features from both fighters' profiles as
they stood at that moment, and only then updates the profiles with the
result. Every feature therefore comes only from fights that happened before
the one being predicted, so nothing leaks from the future. `predict` uses
the profiles as of today, and `history` rebuilds them up to the chosen
event.

Each feature is fighter A's value minus fighter B's (for the age-related
ones, B's minus A's, so a positive number always favors A):

| Feature | What it measures |
|---|---|
| `elo_diff` | Elo rating (see below) |
| `height_diff`, `reach_diff` | Height and reach in inches |
| `age_diff` | Age on fight day |
| `experience_diff` | Number of UFC fights, log-scaled |
| `win_streak_diff` | Current streak: positive for wins, negative for losses |
| `layoff_diff` | Days since last fight, log-scaled (365 for a debut) |
| `kd_per_min_diff`, `str_per_min_diff`, `td_per_min_diff`, `sub_per_min_diff` | Knockdowns, significant strikes, takedowns and submission attempts per minute of cage time |
| `southpaw_diff` | Southpaw stance |
| `prime_distance_diff` | Years outside a 27 to 33 age window |
| `decline_risk_diff` | Years past 33 multiplied by the current losing streak |
| `travel_diff` | Share of past fights held in this fight's country |
| `td_acc_diff`, `td_def_diff` | Takedown accuracy and takedown defense |
| `str_acc_diff`, `str_def_diff` | Significant-strike accuracy and defense |
| `style_matchup_diff` | Takedown attempts per minute times the opponent's takedown failure rate against them |
| `cardio_diff` | Drop in strike rate from round 1 to the last round, scaled by scheduled rounds |
| `activity_diff` | Fights in the last 2 years |
| `ko_rate_diff`, `sub_rate_diff` | Share of wins by KO/TKO and by submission |
| `power_diff` | Knockdowns per significant strike landed |

When a fighter has no data for a feature yet (a debut, no recorded
attempts), the feature falls back to a neutral value: 0 for rates, 0.5 for
travel, and 40% takedown accuracy, 65% takedown defense, 45% strike
accuracy and 55% strike defense.

Draws and no contests don't produce training rows and don't move Elo, but
they still count as a fight for experience and layoff.

#### Elo

`ufcpred/elo.py` gives every fighter an Elo rating that updates after each
fight. The base K-factor is 32. It is multiplied by 1.25 for a KO/TKO or
submission, by 1.2 for a finish in round 1 or 1.1 in round 2, and by 1.5
if either fighter has fewer than 5 UFC fights (provisional ratings).

UFC debuts don't all start at 1500. The fighter header on ufcstats.com
shows a full pro record (e.g. "17-3-0"), which includes regional and
other-promotion fights we have no fight-by-fight data for. Without those
opponents' ratings there is no way to compute a proper Elo for them, so
the starting rating is adjusted by pre-UFC win rate instead: up to 150
points either way, shrunk toward 1500 for short records. The adjustment
was tuned by grid search against held-out UFC debuts. It raised debut
prediction accuracy from 52% to 59% and overall backtest accuracy from
63.0% to 64.3%.

#### Age, decline and travel

The age-related features don't assume younger is always better. Fighters
outside the 27 to 33 window, either too green or aging out, are penalized
relative to one inside it. Decline risk only counts years past 33 while the
fighter is on a losing streak, so an aging fighter who keeps winning isn't
penalized, but age and a losing run compound.

ufcstats.com doesn't publish nationality or hometown, so travel is a proxy
built from the countries each fighter has fought in before. A fighter in a
country they've never fought in is treated as more of an unknown, closer to
an away fighter, than one who has fought there repeatedly. Because the
proxy uses fight history, it can be wrong for a foreign fighter whose whole
UFC career has been in the US. It doesn't know Alex Pereira is Brazilian,
only that all 13 of his UFC fights were in the USA.

These three were grid-searched and validated on backtest data before being
kept. Together they moved accuracy from 64.3% to 64.8% and log loss from
0.646 to 0.643 on the last-50-events holdout.

#### Style, cardio, activity and power

These come from `fight_stats.csv`. `style_matchup_diff` is meant to catch a
wrestler facing someone who can't stop takedowns, which the separate
takedown stats don't capture on their own.

ufcstats.com doesn't publish scheduled rounds, so `cardio_diff` estimates
them: decisions use their actual round count, and finishes fall back to the
title-fight flag (5 rounds for a title fight, otherwise 3). That
undercounts non-title 5-round main events. `predict` and `history` apply
the same rule, and `compare` takes `--rounds`.

`activity_diff` is separate from layoff because a fighter with one fight in
3 years and one with six fights in 3 years can have the same days since
their last fight. `power` is separate from `kd_per_min` because the latter
mixes power with output volume.

These were also validated on the holdout. `style_matchup_diff` and
`activity_diff` correlate 0.08 to 0.10 with fight outcomes. `cardio_diff`
and `power_diff` correlate close to zero on their own, but the full set
backtested together moved accuracy from 64.8% to 65.8% and log loss from
0.643 to 0.639 on the last-50-events holdout, so all of them stayed in.

### Model

ufcstats.com lists the winner first in every fight, so each fight becomes
two mirrored training rows: A's view with label 1 and every feature as is,
and B's view with label 0 and every feature negated. Otherwise the model
could learn that "fighter 1" always wins.

`ufcpred/model.py` trains two candidates:

- logistic regression on standardized features, with L2 regularization
  (`C=0.5`)
- a gradient-boosted tree (scikit-learn `HistGradientBoostingClassifier`,
  depth 3, learning rate 0.05, 200 iterations)

Both are trained on the oldest 85% of fights by date and scored on the
newest 15%. The one with lower log loss on that slice is refit on all
fights and saved to `data/model/model.joblib`, with the holdout accuracy
and log loss of both candidates in `data/model/metrics.json`. Logistic
regression usually wins; it is the standard, well-calibrated choice for a
sports dataset of this size.

### Odds

`ufcpred/odds.py` converts probability to decimal odds (`1/p`) and to
American odds, with no vig added. For `compare`, it also removes the vig
from the bookmaker's two prices to get the market's probability, and sizes
the stake with fractional Kelly.

## Testing against past events

```
pytest -s tests/test_backtest.py
```

This trains the model only on fights before the last 50 events, then checks
its predictions against the results. It asserts that the model beats a coin
flip and the uniform-guess log loss, and that high-confidence picks (a gap
of at least 15 points in probability) do at least as well as the overall
rate.

For a fight-by-fight report instead of pass/fail:

```
python tests/test_backtest.py 6   # last 6 events
```
