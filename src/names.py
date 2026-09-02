"""One canonical player-name normalizer, imported everywhere a name join happens.

Three files used to each roll their own (keeping spaces / dropping only some suffixes /
stripping all non-alpha), so the same player produced three different keys and stars with
generational suffixes (James Cook III, Michael Pittman Jr.) silently missed ADP/role joins.
This is the single source of truth: lowercase, drop punctuation and generational suffixes,
collapse to bare letters.
"""
import re

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(n):
    s = str(n).lower().replace(".", " ").replace("'", "").replace("-", " ")
    toks = [t for t in s.split() if t not in _SUFFIXES]
    return re.sub(r"[^a-z]", "", "".join(toks))
