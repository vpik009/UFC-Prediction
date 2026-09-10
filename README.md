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
model (takes a few minutes; re-running later only re-fetches what's new
because pages are cached in `data/raw_cache/`):

```
python -m ufcpred update
```

Pick from the next few upcoming UFC events and get predictions:

```
python -m ufcpred predict
```

Each matchup prints the model's win probability for both fighters plus the
equivalent fair (no-vig) decimal and American odds, e.g.:

```
  [Flyweight]
    Joshua Van                    70.9%   dec  1.41   american -244
    Alexandre Pantoja             29.1%   dec  3.44   american +244
    -> favorite: Joshua Van
```

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

trains the model only on fights before the last 15 events, then checks its
predictions against what actually happened (asserts it beats a coin flip
and the uniform-guess log loss).

For a readable fight-by-fight report instead of a pass/fail:

```
python tests/test_backtest.py 6   # last 6 events
```
