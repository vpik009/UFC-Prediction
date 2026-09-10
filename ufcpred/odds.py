"""Convert a win probability into 'fair' (no-vig) betting odds."""
from __future__ import annotations


def to_decimal_odds(prob: float) -> float:
    """European/decimal odds: what you'd need to see, vig-free, for this probability."""
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    return 1.0 / prob


def to_american_odds(prob: float) -> int:
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    if prob >= 0.5:
        odds = -100.0 * prob / (1.0 - prob)
    else:
        odds = 100.0 * (1.0 - prob) / prob
    return int(round(odds))


def format_american(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def parse_book_odds(text: str) -> float:
    """Parse a bookmaker odds string in either American ("-150", "+130") or
    decimal ("1.67", "2.30") notation and return it as decimal odds."""
    text = text.strip()
    if "." in text:
        value = float(text)
        if value <= 1.0:
            raise ValueError(f"decimal odds must be > 1.0, got {text!r}")
        return value
    value = int(text)
    if value == 0:
        raise ValueError("odds cannot be 0")
    return value / 100.0 + 1.0 if value > 0 else 100.0 / abs(value) + 1.0


def decimal_to_implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def devig_two_way(implied_a: float, implied_b: float) -> tuple[float, float]:
    """Normalize two implied probabilities (which sum to >1 because of the
    house's vig) down to a fair, no-vig pair that sums to 1."""
    total = implied_a + implied_b
    return implied_a / total, implied_b / total


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Optimal bankroll fraction to stake, per the Kelly criterion. 0 if
    there's no edge (the bookmaker's odds are as good or better than fair)."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(f, 0.0)
