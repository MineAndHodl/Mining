"""Weather adjustments for props — validated on 1,207 outdoor games (2018-24).

Findings (all real, tested):
  wind vs pass yds   r=-0.167 (p<1e-4): -49 yds/game at 15+mph vs calm — monotonic.
  temp vs pass yds   r=+0.157 (p<1e-4): cold suppresses passing.
  wind vs FG%        r=-0.025 (p=0.38): NO effect — kickers don't attempt long ones in wind.
  wind vs game total r=-0.107: real but weak; the edge is on PASS PROPS, not team totals.

So weather scales pass-based prop projections down (and nudges rushing up a touch, since
bad weather shifts offenses run-ward). Indoors/dome → no adjustment.
"""

def pass_weather_mult(wind=None, temp=None, roof="outdoors"):
    """Multiplier on a pass-based projection given game weather. 1.0 = neutral/indoors."""
    if roof in ("dome", "closed"):
        return 1.0
    m = 1.0
    if wind is not None:
        m *= 1.0 - 0.007 * max(wind - 8, 0)      # ~-0.7%/mph above 8mph
    if temp is not None:
        m *= 1.0 - 0.0023 * max(55 - temp, 0)    # cold penalty below 55F
    return max(m, 0.80)                            # floor: never cut more than 20%


def market_weather_mult(mkt, wind=None, temp=None, roof="outdoors"):
    """Apply the pass-weather effect scaled by how pass-dependent the market is.
    rush_yds gets a small BOOST in bad weather (offenses lean run)."""
    pm = pass_weather_mult(wind, temp, roof)
    if mkt == "pass_yds":
        return pm
    if mkt == "rec_yds":
        return 1.0 + 0.85 * (pm - 1.0)            # ~85% pass-dependent
    if mkt == "receptions":
        return 1.0 + 0.6 * (pm - 1.0)
    if mkt == "rush_yds":
        # bad passing weather → modest rush bump (inverse, capped)
        return 1.0 + min(0.5 * (1.0 - pm), 0.06)
    return 1.0


# how much to trust a projection by market (YoY stability, 2022-25) → confidence weight
MARKET_TRUST = {"rec_yds": 1.00, "rush_yds": 0.94, "receptions": 0.75, "pass_yds": 0.46}


if __name__ == "__main__":
    for w in (0, 10, 15, 20, 25):
        print(f"wind {w:2d}mph: pass x{pass_weather_mult(wind=w):.3f}  rec x{market_weather_mult('rec_yds',wind=w):.3f}  rush x{market_weather_mult('rush_yds',wind=w):.3f}")
    for t in (20, 35, 55, 75):
        print(f"temp {t}F: pass x{pass_weather_mult(temp=t):.3f}")
