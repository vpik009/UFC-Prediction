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

### Scraping

`ufcpred/scrape.py` and `http_client.py` handle scraping. ufcstats.com puts
every request behind a small JavaScript proof-of-work challenge. The client
solves it once per session and caches every fetched page on disk. On HTTP
429 or any other non-200 response it backs off and does not cache the page.

### Ratings and features

`ufcpred/elo.py` and `features.py` give every fighter an Elo rating that
updates after each fight. Finishes and early finishes get a higher K-factor,
and fighters with fewer than 5 UFC fights get a provisional K-factor. The
other features are rolling per-minute output (knockdowns, significant
strikes, takedowns, submission attempts), win streak, layoff time, and
differences in height, reach, age and stance. Every feature is computed
only from fights that happened before the one being predicted, so nothing
leaks from the future.

UFC debuts don't all start at a flat 1500. The fighter header on
ufcstats.com shows a full pro record (e.g. "17-3-0"), which includes
regional and other-promotion fights we have no fight-by-fight data for.
Without those opponents' ratings there is no way to compute a proper Elo
for them, so the starting rating is adjusted by pre-UFC win rate instead,
shrunk toward 0 for small samples and capped. The adjustment was tuned by
grid search against held-out UFC debuts. It raised debut prediction
accuracy from 52% to 59% and overall backtest accuracy from 63.0% to 64.3%.
`ufcpred/elo.py` has the formula and calibration numbers.

Three features model age, experience and location in a nonlinear way:

- Prime distance. Fighters outside a 27 to 33 prime window, either too
  green or aging out, are penalized relative to one inside it.
- Decline risk. Years past the prime window count as risk only while the
  fighter is on a losing streak. An aging fighter who keeps winning isn't
  penalized, but age and a losing run compound.
- Travel. ufcstats.com doesn't publish nationality or hometown, so this is
  a proxy built from the countries each fighter has fought in before. A
  fighter in a country they've never fought in is treated as more of an
  unknown, closer to an away fighter, than one who has fought there
  repeatedly. Because the proxy uses fight history, it can be wrong for a
  foreign fighter whose whole UFC career has been in the US. It doesn't
  know Alex Pereira is Brazilian, only that all 13 of his UFC fights were
  in the USA.

All three were grid-searched and validated on backtest data before being
kept. Together they moved accuracy from 64.3% to 64.8% and log loss from
0.646 to 0.643 on the last-50-events holdout.

Another set of features needed new data. Each fight's own page on
ufcstats.com has strike and takedown attempts and a round-by-round
breakdown; the event page only has landed counts. About 8,900 fight pages
were scraped to build these:

- Style matchup: takedown accuracy and defense, strike accuracy and
  defense, plus an interaction term, `style_matchup_diff`. It measures how
  well a fighter's takedown-attempt volume (wrestling pressure) matches up
  against the opponent's actual takedown defense.
- Cardio and pace: each fighter's output in round 1 versus the last round
  (fade), scaled by the scheduled length of this fight (5 rounds for a
  title fight, otherwise 3). ufcstats.com doesn't publish scheduled rounds,
  so decisions use their actual round count and finishes fall back to the
  title-fight flag. That undercounts non-title 5-round main events.
- Activity: fights in the last 2 years, separate from layoff time. A
  fighter with one fight in 3 years and one with six fights in 3 years can
  have the same days since their last fight.
- Finishing power: `ko_rate` and `sub_rate` as separate features, plus
  `power`, knockdowns per significant strike landed. `kd_per_min` mixes
  power with output volume; `power` doesn't.

These were also validated on the holdout. `style_matchup_diff` and
`activity_diff` correlate 0.08 to 0.10 with fight outcomes. `cardio_diff`
and `power_diff` correlate close to zero on their own, but the full set
backtested together moved accuracy from 64.8% to 65.8% and log loss from
0.643 to 0.639 on the last-50-events holdout, so all of them stayed in.

### Model

ufcstats.com lists the winner first in every fight, so `ufcpred/model.py`
turns each fight into two mirrored training rows (winner's view with
label 1, loser's view with label 0) to cancel out that ordering. It fits a
regularized logistic regression and a gradient-boosted tree and compares
them on a time-ordered holdout: train on older fights, test on the most
recent slice. The one with lower log loss is refit on the full dataset.
Logistic regression usually wins; it is the standard, well-calibrated
choice for a sports dataset of this size.

### Odds

`ufcpred/odds.py` converts probability to decimal odds (`1/p`) and to
American odds, with no vig added.

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
