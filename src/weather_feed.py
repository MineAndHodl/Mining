"""Kickoff weather forecast — the feed the validated weather module (weather.py) was missing.

Open-Meteo (free, no key): hourly wind / gusts / temperature / precipitation probability at
the stadium for the kickoff hour. Domes and closed roofs are neutral. Written to
data/weather_forecast.json {game_string: {wind, gust, temp, precip_prob, roof, kickoff_utc}}
and applied by prop_board (market_weather_mult: passing markets down with wind above ~8 mph
and cold, a small rushing bump) and by the scorecard's Weather lever.

    python3 src/weather_feed.py          # forecast for every game in the next 10 days
"""
import os, sys, json, datetime
import urllib.request
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = f"{DATA}/weather_forecast.json"

# stadium coordinates by HOME TEAM (domestic) — retractables listed with their usual state
TEAM_SITE = {
    "ARI": (33.5276, -112.2626, "retractable"), "ATL": (33.7554, -84.4010, "retractable"), "BAL": (39.2780, -76.6227, "outdoors"),
    "BUF": (42.7738, -78.7870, "outdoors"), "CAR": (35.2258, -80.8528, "outdoors"), "CHI": (41.8623, -87.6167, "outdoors"),
    "CIN": (39.0955, -84.5161, "outdoors"), "CLE": (41.5061, -81.6995, "outdoors"), "DAL": (32.7473, -97.0945, "retractable"),
    "DEN": (39.7439, -105.0201, "outdoors"), "DET": (42.3400, -83.0456, "dome"), "GB": (44.5013, -88.0622, "outdoors"),
    "HOU": (29.6847, -95.4107, "retractable"), "IND": (39.7601, -86.1639, "retractable"), "JAX": (30.3239, -81.6373, "outdoors"),
    "KC": (39.0489, -94.4839, "outdoors"), "LV": (36.0909, -115.1833, "dome"), "LAC": (33.9535, -118.3392, "dome"),
    "LA": (33.9535, -118.3392, "dome"), "MIA": (25.9580, -80.2389, "outdoors"), "MIN": (44.9735, -93.2575, "dome"),
    "NE": (42.0909, -71.2643, "outdoors"), "NO": (29.9511, -90.0812, "dome"), "NYG": (40.8135, -74.0745, "outdoors"),
    "NYJ": (40.8135, -74.0745, "outdoors"), "PHI": (39.9008, -75.1675, "outdoors"), "PIT": (40.4468, -80.0158, "outdoors"),
    "SF": (37.4032, -121.9698, "outdoors"), "SEA": (47.5952, -122.3316, "outdoors"), "TB": (27.9759, -82.5033, "outdoors"),
    "TEN": (36.1665, -86.7713, "outdoors"), "WAS": (38.9076, -76.8645, "outdoors"),
}
# neutral / international venues by the schedule's stadium name
VENUE_SITE = {
    "Melbourne Cricket Ground": (-37.8200, 144.9834, "outdoors"), "Bernabeu": (40.4531, -3.6883, "retractable"),
    "Estadio Banorte": (19.3029, -99.1505, "outdoors"), "FC Bayern Munich Stadium": (48.2188, 11.6247, "outdoors"),
    "Maracana Stadium": (-22.9121, -43.2302, "outdoors"), "Stade de France": (48.9245, 2.3602, "outdoors"),
    "Tottenham Hotspur Stadium": (51.6042, -0.0662, "outdoors"), "Wembley Stadium": (51.5560, -0.2796, "outdoors"),
    "Croke Park": (53.3607, -6.2511, "outdoors"), "Deutsche Bank Park": (50.0686, 8.6455, "outdoors"),
}
FULL = {"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills","CAR":"Carolina Panthers",
"CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions",
"GB":"Green Bay Packers","HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
"LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers","LA":"Los Angeles Rams","MIA":"Miami Dolphins","MIN":"Minnesota Vikings",
"NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles",
"PIT":"Pittsburgh Steelers","SF":"San Francisco 49ers","SEA":"Seattle Seahawks","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans",
"WAS":"Washington Commanders"}


def kickoff_utc(gameday, gametime):
    from zoneinfo import ZoneInfo
    hh, mm = (str(gametime) or "13:00").split(":")[:2]
    local = datetime.datetime.strptime(str(gameday), "%Y-%m-%d").replace(hour=int(hh), minute=int(mm), tzinfo=ZoneInfo("America/New_York"))
    return local.astimezone(datetime.timezone.utc)


def _site(row):
    st = row.get("stadium")
    if st in VENUE_SITE: return VENUE_SITE[st]
    return TEAM_SITE.get(row.get("home_team"))


def _roof(row, site_roof):
    r = row.get("roof")
    if isinstance(r, str) and r: return r
    return "closed" if site_roof == "retractable" else site_roof


def fetch_point(lat, lon, when_utc, tries=3):
    for i in range(tries):
        try:
            return _fetch_point(lat, lon, when_utc)
        except Exception as e:
            if i == tries - 1: raise
            import time; time.sleep(1.5)


def _fetch_point(lat, lon, when_utc):
    day = when_utc.strftime("%Y-%m-%d")
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&hourly=wind_speed_10m,wind_gusts_10m,temperature_2m,precipitation_probability"
           f"&wind_speed_unit=mph&temperature_unit=fahrenheit&timezone=UTC&start_date={day}&end_date={day}")
    d = json.load(urllib.request.urlopen(url, timeout=20))
    h = d["hourly"]; key = when_utc.strftime("%Y-%m-%dT%H:00")
    i = h["time"].index(key) if key in h["time"] else min(range(len(h["time"])), key=lambda j: abs(int(h["time"][j][11:13]) - when_utc.hour))
    return {"wind": h["wind_speed_10m"][i], "gust": h["wind_gusts_10m"][i], "temp": h["temperature_2m"][i],
            "precip_prob": (h.get("precipitation_probability") or [None] * len(h["time"]))[i]}


def forecast(days=12, season=None, write=True):
    s = pd.read_parquet(f"{DATA}/schedules.parquet")
    today = datetime.datetime.now(datetime.timezone.utc)
    season = season or (today.year if today.month >= 3 else today.year - 1)
    s = s[(s.season == season) & (s.game_type == "REG")].copy()
    out = {}
    for _, r in s.iterrows():
        try:
            ko = kickoff_utc(r["gameday"], r["gametime"])
        except Exception:
            continue
        if not (today - datetime.timedelta(hours=4) <= ko <= today + datetime.timedelta(days=days)): continue
        site = _site(r)
        if not site: continue
        lat, lon, site_roof = site
        roof = _roof(r, site_roof)
        gid = f"{FULL.get(r['away_team'], r['away_team'])} @ {FULL.get(r['home_team'], r['home_team'])}"
        rec = {"roof": roof, "kickoff_utc": ko.isoformat(timespec="minutes"), "stadium": r.get("stadium")}
        if roof in ("dome", "closed", "indoors"):
            rec.update({"wind": 0.0, "gust": 0.0, "temp": 70.0, "precip_prob": 0, "note": "indoors — neutral"})
        else:
            try:
                rec.update(fetch_point(lat, lon, ko))
            except Exception as e:
                rec["error"] = str(e)[:80]
        out[gid] = rec
    if write:
        json.dump({"taken_at": today.isoformat(timespec="minutes"), "games": out}, open(OUT, "w"), indent=1)
    return out


def load():
    try:
        return json.load(open(OUT)).get("games", {})
    except Exception:
        return {}


def weather_note(w):
    if not w or w.get("error"): return None
    if w.get("roof") in ("dome", "closed", "indoors"): return "indoors"
    bits = []
    if w.get("wind") is not None: bits.append(f"wind {w['wind']:.0f} mph" + (f" (gusts {w['gust']:.0f})" if w.get("gust") and w["gust"] >= 20 else ""))
    if w.get("temp") is not None: bits.append(f"{w['temp']:.0f}°F")
    if w.get("precip_prob") is not None and w["precip_prob"] >= 40: bits.append(f"{int(w['precip_prob'])}% rain")
    return ", ".join(bits)


if __name__ == "__main__":
    f = forecast()
    for g, w in f.items():
        print(f"  {g:48s} {w.get('kickoff_utc','')[:16]}  {weather_note(w) or w.get('error')}")
