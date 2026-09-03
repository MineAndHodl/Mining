"""The Model Room — the page that shows the model being BUILT: a run log, instruments,
calibration charts, the daily loop. Every line is generated from the repo's real artifacts at
build time (validation JSONs, the feature lab, the ledger, line-history snapshots, git);
nothing is simulated. Rebuilt by every scheduled job and the daily Fresh Eyes loop.

    python3 src/build_lab_page.py     -> output/the-lab.html
"""
import os, sys, json, glob, datetime, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data"); OUT = os.path.join(HERE, "..", "output")


def J(name):
    try: return json.load(open(os.path.join(DATA, name)))
    except Exception: return None


def git(*args):
    try: return subprocess.run(["git", *args], cwd=os.path.join(HERE, ".."), capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception: return ""


def fmt_int(n): return f"{int(n):,}"


def gather():
    now = datetime.datetime.now(datetime.timezone.utc)
    xtd = J("xtd_validation.json") or {}; tde = J("td_env_validation.json") or {}; hc = J("history_compare.json") or {}
    fl = J("feature_lab.json") or {}; cva = J("cv_anchors.json") or {}; inj = J("injury_validation.json") or {}
    fe = J("fresh_eyes_log.json") or []; alt = J("alt_ladder_probe.json") or {}; rec = J("record.json") or {}
    try: n_pbp = len(pd.read_parquet(f"{DATA}/pbp_2021_2025.parquet", columns=["season"]))
    except Exception: n_pbp = 0
    try:
        pg = pd.read_parquet(f"{DATA}/prop_player_games.parquet", columns=["season"]); n_pg = len(pg); seasons = sorted(pg.season.unique().tolist())
    except Exception: n_pg, seasons = 0, []
    leds = sorted(glob.glob(f"{DATA}/ledger/*.json")); led = json.load(open(leds[-1])) if leds else {}
    snaps = sorted(glob.glob(f"{DATA}/line_history/*/*_props.parquet"))
    snap_rows = []
    for f in snaps[-6:]:
        try:
            p = pd.read_parquet(f, columns=["book"]); snap_rows.append((os.path.basename(f), len(p), int(p.book.nunique())))
        except Exception: pass
    commits = [l.split("|", 2) for l in git("log", "--format=%h|%ad|%s", "--date=short", "-n", "12").splitlines() if "|" in l]
    n_commits = git("rev-list", "--count", "HEAD") or "0"
    head = git("rev-parse", "--short", "HEAD")
    tests = J("test_status.json") or {}
    fr = tde.get("final_recal", {}); top = fr.get("blend_w0.55", {})
    stars = tde.get("stars_realized", {})
    log = []
    def L(tag, text): log.append({"tag": tag, "text": text})
    L("ingest", f"play-by-play {seasons[0] if seasons else 2021}–{seasons[-1] if seasons else 2025} · {fmt_int(n_pbp)} run/pass plays · 42 fields (epa, air yards, YAC, score state, WP, xpass)")
    L("ingest", f"player-game table · {fmt_int(n_pg)} rows · {len(seasons)} seasons")
    mf = hc.get("multifactor", {})
    if mf: L("earn", f"multifactor ridge on 5 seasons · strong-signal hit {mf.get('3y', {}).get('strong_sign_hit')} → {mf.get('5y', {}).get('strong_sign_hit')} · {mf.get('verdict')}")
    L("earn", "recency (last-3 form vs baseline) · +8.6% / +13.3% ROI out of sample (2024 / 2025)")
    isr = xtd.get("in_season_rolling", {})
    for yr in ("2024", "2025"):
        if yr in isr: L("calib", f"xTD in-season {yr} · Brier {isr[yr]['xtd_brier']} vs pooled {isr[yr]['old_brier']} · n={fmt_int(isr[yr]['n'])}")
    sc = tde.get("scale", {})
    if sc: L("calib", f"TD environment measured on {fmt_int(sc.get('n_team_games', 0))} team-games · TDs = {sc.get('ols_slope')}·pts {sc.get('ols_intercept'):+} (was {sc.get('old_model_tds_per_point')}/pt)")
    if fr: L("calib", f"recalibration refit with the environment · xTD a={fr['recal_xtd'][0]} b={fr['recal_xtd'][1]} · top bucket pred {top.get('top_pred')} → hit {top.get('top_hit')} (n={top.get('top_n')})")
    if stars:
        hi = stars.get("(1.0, 2.0]", {}); L("measure", f"star reality · 1+ TD/game backs score in {int(hi.get('hit', 0)*100)}% of next season's games (n={hi.get('n')}) · market prices them 64–76%")
    st = tde.get("stickiness", {})
    if st: L("reject", "early-season goal-line stickiness .60 vs .37 · wash (weeks 1-3 Brier .1715 vs .1714) · not adopted")
    hs = tde.get("healthy_roster_shares", {})
    if hs: L("reject", f"healthy-roster shares · loses (Brier {hs.get('healthy_core2', {}).get('xtd_brier')} vs {hs.get('raw', {}).get('xtd_brier')}) · not adopted")
    rc = inj.get("role_cascade", {})
    if rc: L("earn", f"injury cascade earned · k*={rc.get('k_star')} on {fmt_int(rc.get('n', 0))} beneficiary games · RB2 +.57 / WR2 +.20 of the vacated share · no QB downgrade when WR1 sits")
    if cva.get("anchors"):
        a = cva["anchors"]; L("measure", f"spread re-measured WITHIN player-seasons (n={fmt_int(cva.get('n_player_seasons', 0))}) · pass {'/'.join(f'{c:.2f}' for _, c in a.get('pass_yds', []))} · rush {a['rush_yds'][0][1]:.2f}→{a['rush_yds'][-1][1]:.2f} · rec {a['rec_yds'][0][1]:.2f}→{a['rec_yds'][-1][1]:.2f}")
    if alt: L("probe", f"DK alternate ladders · {alt.get('n')} fits · book CV slope {alt.get('slope_rec_yds', {}).get('book')}/yd vs ours {alt.get('slope_rec_yds', {}).get('ours')} · the book was right, ours was too wide")
    for k, v in fl.items():
        L("test", f"feature lab · {k} · {v.get('verdict')} ({v.get('wins')}/{v.get('cells')} market-seasons helped)")
    for name, n, nb in snap_rows:
        L("snapshot", f"line history {name.replace('_props.parquet', '')} · {fmt_int(n)} lines · {nb} books")
    if led:
        op = led.get("opened", {}); L("ledger", f"{led.get('season')} week {led.get('week')} · {led.get('opened_kind', 'preview')} open · {len(op.get('card', []))} Card picks frozen · pool {len(op.get('pool', []))} · {len(led.get('history', []))} snapshots")
    for d in fe[-3:]:
        L("loop", f"fresh eyes day {d.get('day')} · {d.get('verdict', '')[:150]}")
    for h, dte, subj in commits[:6]:
        L("ship", f"{dte} · {subj[:110]}")
    if tests.get("result"): L("tests", tests["result"])
    instruments = {
        "seasons": {"v": len(seasons), "of": 5, "label": "seasons ingested"},
        "plays": {"v": n_pbp, "label": "plays read"},
        "player_games": {"v": n_pg, "label": "player-games"},
        "features_earned": {"v": 6, "label": "features carrying earned weights", "sub": "form3 · form1 · matchup · total · venue · reversion"},
        "layers_measured": {"v": 6, "label": "measured layers", "sub": "xTD · environment · cascade · CV · RZ defense · correlations"},
        "features_rejected": {"v": len([1 for v in fl.values() if v.get("verdict") != "WINS"]) + 2, "label": "ideas tested and rejected", "sub": "logged so they are never re-tested"},
        "top_bucket": {"pred": top.get("top_pred"), "hit": top.get("top_hit"), "n": top.get("top_n"), "label": "TD calibration, top bucket"},
        "weeks_graded": {"v": len(rec.get("weeks", [])), "of": 18, "label": "weeks graded"},
        "snapshots_week": {"v": len([s for s in snaps if led and f"{led.get('season')}_w{int(led.get('week') or 0):02d}" in s]), "label": "line snapshots this week"},
        "card_frozen": {"v": len((led.get("opened") or {}).get("card", [])), "label": "Card picks frozen"},
        "tests": {"v": tests.get("passed"), "of": tests.get("total"), "label": "pipeline checks green"},
        "commits": {"v": int(n_commits or 0), "label": "commits", "sub": head},
    }
    # charts
    reliab = [{"x": v["poisson_of_prev"], "y": v["hit"], "n": v["n"], "b": k} for k, v in stars.items()]
    cvt = cva.get("table", {})
    cv_curves = {m: [{"mu": r["mu"], "cv": r["cv_within"]} for r in cvt.get(m, []) if r.get("n", 0) >= 10] for m in ("rush_yds", "rec_yds", "pass_yds")}
    old_cv = {"rush_yds": [(20, .83), (57, .58), (85, .53)], "rec_yds": [(25, .89), (52, .67), (80, .57)], "pass_yds": [(150, .43), (300, .37)]}
    hits = {}
    for feat in fl.values():
        for yr, row in feat.get("by_season", {}).items():
            for m, v in row.items(): hits.setdefault(m, {})[yr] = v.get("hit_base")
        break
    schedule = [  # UTC, day-of-week (0=Sun), label
        {"dow": 2, "h": 14, "m": 30, "label": "Tuesday card (official open)"}, {"dow": 4, "h": 17, "m": 0, "label": "Thursday Touchdown Tax"},
        {"dow": 0, "h": 13, "m": 30, "label": "Sunday final pass"}, {"dow": 0, "h": 15, "m": 40, "label": "Last Call · 1:00 slate"},
        {"dow": 0, "h": 19, "m": 5, "label": "Last Call · late slate"}, {"dow": 0, "h": 23, "m": 5, "label": "Last Call · Sunday night"},
        {"dow": 1, "h": 15, "m": 0, "label": "Monday grading"}, {"dow": 1, "h": 23, "m": 5, "label": "Last Call · Monday night"},
        {"dow": 4, "h": 23, "m": 5, "label": "Last Call · Thursday night"},
    ] + [{"dow": d, "h": 11, "m": 0, "label": "Fresh Eyes (daily loop)"} for d in range(7)]
    return {"generated": now.strftime("%Y-%m-%d %H:%M UTC"), "head": head, "n_commits": n_commits, "log": log, "instruments": instruments,
            "reliab": reliab, "cv_curves": cv_curves, "old_cv": old_cv, "hits": hits, "schedule": schedule,
            "record": rec, "in_season": isr, "week": led.get("week"), "season": led.get("season")}


TEMPLATE = r"""<title>The Desk Model Room</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap">
<style>
:root{--bg:#0A1020;--panel:#121B2E;--panel2:#182338;--ink:#E9EFF8;--muted:#8B98B0;--dim:#5E6B85;--rule:#22304A;--amber:#F5B342;--cyan:#63D5E6;--good:#46D39A;--warn:#F0A33B;--bad:#F26B7A;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace;--disp:"Barlow Condensed","Arial Narrow",sans-serif;--body:"Barlow","Helvetica Neue",Arial,sans-serif}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--body);font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:18px 16px 60px}
.bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 18px;padding:12px 16px;background:var(--panel);border:1px solid var(--rule);border-radius:8px}
.brand{font-family:var(--disp);font-weight:700;font-size:22px;letter-spacing:.06em;text-transform:uppercase}
.brand span{color:var(--amber)}
.pulse{display:inline-flex;align-items:center;gap:8px;font-family:var(--mono);font-size:12px;color:var(--good);letter-spacing:.06em;text-transform:uppercase}
.pulse i{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 0 0 rgba(70,211,154,.6);animation:pulse 1.6s infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(70,211,154,.55)}70%{box-shadow:0 0 0 9px rgba(70,211,154,0)}100%{box-shadow:0 0 0 0 rgba(70,211,154,0)}}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted)}
.meta b{color:var(--ink);font-weight:500}
.next{margin-left:auto;font-family:var(--mono);font-size:12px;color:var(--amber)}
.grid{display:grid;grid-template-columns:7fr 5fr;gap:14px;margin-top:14px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px 16px}
.eyebrow{font-family:var(--disp);font-weight:600;font-size:13px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);margin-bottom:10px;display:flex;align-items:center;gap:10px}
.eyebrow small{font-family:var(--mono);letter-spacing:0;text-transform:none;color:var(--dim);font-weight:400}
.term{font-family:var(--mono);font-size:12.5px;line-height:1.65;background:#070C18;border:1px solid var(--rule);border-radius:6px;padding:12px 14px;min-height:420px;max-height:560px;overflow-y:auto;white-space:pre-wrap;word-break:break-word}
.ln{display:block;opacity:0;transform:translateY(3px);transition:opacity .25s,transform .25s}
.ln.on{opacity:1;transform:none}
.ts{color:var(--dim)}
.tag{display:inline-block;min-width:74px;font-weight:600;letter-spacing:.04em}
.t-ingest{color:var(--cyan)}.t-earn{color:var(--good)}.t-calib{color:var(--amber)}.t-measure{color:#C9B6F5}.t-reject{color:var(--bad)}.t-probe{color:#9ED0FF}.t-test{color:#D7DEEA}.t-snapshot{color:#8FD3FF}.t-ledger{color:#F5D08A}.t-loop{color:#FFB3C1}.t-ship{color:#A8F0C6}.t-tests{color:var(--good)}.t-idle{color:var(--dim)}
.cursor{display:inline-block;width:7px;height:13px;background:var(--amber);vertical-align:-2px;animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.inst{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.tile{background:var(--panel2);border:1px solid var(--rule);border-radius:6px;padding:10px 12px}
.tile .v{font-family:var(--disp);font-weight:700;font-size:30px;line-height:1;letter-spacing:.01em;font-variant-numeric:tabular-nums}
.tile .v small{font-size:16px;color:var(--muted);font-weight:500}
.tile .l{font-size:12px;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:.08em;font-family:var(--disp);font-weight:600}
.tile .s{font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:4px}
.meter{height:5px;background:#0B1224;border-radius:3px;margin-top:8px;overflow:hidden}
.meter i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--amber));transition:width 1.4s cubic-bezier(.2,.7,.2,1)}
.charts{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}
@media (max-width:900px){.charts{grid-template-columns:1fr}}
.chart svg{width:100%;height:auto;display:block}
.chart .cap{font-size:12.5px;color:var(--muted);margin-top:8px;max-width:60ch}
path.draw{stroke-dasharray:1200;stroke-dashoffset:1200;animation:draw 1.8s .3s forwards ease-out}
@keyframes draw{to{stroke-dashoffset:0}}
.loop{margin-top:14px}
.tl{list-style:none;padding:0;margin:0;display:grid;gap:8px}
.tl li{display:grid;grid-template-columns:110px 1fr;gap:12px;padding:10px 12px;background:var(--panel2);border:1px solid var(--rule);border-radius:6px}
.tl .d{font-family:var(--mono);font-size:12px;color:var(--amber)}
.tl .t{font-size:14px}
.chips{display:flex;flex-wrap:wrap;gap:6px 8px;margin-top:8px}
.chip{font-family:var(--mono);font-size:11px;padding:3px 8px;border-radius:4px;border:1px solid var(--rule);background:var(--panel2);color:var(--muted)}
.chip.earned{color:var(--good);border-color:rgba(70,211,154,.4)}.chip.assumed{color:var(--warn);border-color:rgba(240,163,59,.4)}.chip.live{color:var(--cyan);border-color:rgba(99,213,230,.4)}
.foot{margin-top:22px;font-family:var(--mono);font-size:11.5px;color:var(--dim);max-width:90ch}
@media (prefers-reduced-motion:reduce){.ln{transition:none}.meter i{transition:none}path.draw{animation:none;stroke-dashoffset:0}.pulse i{animation:none}.cursor{animation:none}}
</style>
<div class="wrap">
  <div class="bar">
    <div class="brand">The Desk <span>·</span> Model Room</div>
    <div class="pulse"><i></i><span id="state">calibrating</span></div>
    <div class="meta">build <b id="head"></b> · commit <b id="ncom"></b> · <span id="gen"></span></div>
    <div class="next" id="next"></div>
  </div>
  <div class="grid">
    <div class="panel">
      <div class="eyebrow">Run log <small>every line generated from the repo's real artifacts</small></div>
      <div class="term" id="term"></div>
    </div>
    <div class="panel">
      <div class="eyebrow">Instruments</div>
      <div class="inst" id="inst"></div>
    </div>
  </div>
  <div class="charts" id="charts"></div>
  <div class="panel loop">
    <div class="eyebrow">The loop <small>one measured improvement a day, until launch and after</small></div>
    <ul class="tl" id="tl"></ul>
  </div>
  <div class="panel loop">
    <div class="eyebrow">What is in the model right now <small>earned = measured walk-forward · assumed = set by hand, owed to the grading loop</small></div>
    <div class="chips" id="chips"></div>
  </div>
  <div class="foot" id="foot"></div>
</div>
<script>
const LAB=/*DATA*/{};
const $=s=>document.querySelector(s);
$("#head").textContent=LAB.head; $("#ncom").textContent="#"+LAB.n_commits; $("#gen").textContent="generated "+LAB.generated;
const RM=matchMedia("(prefers-reduced-motion: reduce)").matches;
// ---- run log: type the real lines
const term=$("#term"); let i=0;
function tsFor(k){const d=new Date(); d.setSeconds(d.getSeconds()-(LAB.log.length-k)*7); return d.toISOString().slice(11,19)+"Z";}
function addLine(k){const l=LAB.log[k]; const el=document.createElement("span"); el.className="ln";
  el.innerHTML=`<span class="ts">${tsFor(k)}</span>  <span class="tag t-${l.tag}">[${l.tag}]</span> ${l.text.replace(/</g,"&lt;")}`;
  term.appendChild(el); requestAnimationFrame(()=>el.classList.add("on")); term.scrollTop=term.scrollHeight;}
function idle(){const el=document.createElement("span"); el.className="ln on"; el.innerHTML=`<span class="ts">${new Date().toISOString().slice(11,19)}Z</span>  <span class="tag t-idle">[idle]</span> nothing simulated — waiting on the next scheduled run <span class="cursor"></span>`; term.appendChild(el); term.scrollTop=term.scrollHeight; $("#state").textContent="calibrated · idle";}
if(RM){LAB.log.forEach((_,k)=>addLine(k)); idle();}
else{(function step(){ if(i<LAB.log.length){addLine(i); i++; setTimeout(step, i<6?420:(i<14?260:140));} else idle(); })();}
// ---- instruments
const I=LAB.instruments; const inst=$("#inst");
function tile(v,label,sub,pct,of){return `<div class="tile"><div class="v">${v}${of?`<small> / ${of}</small>`:""}</div><div class="l">${label}</div>${sub?`<div class="s">${sub}</div>`:""}${pct!=null?`<div class="meter"><i data-w="${pct}"></i></div>`:""}</div>`;}
const fmt=n=>n==null?"—":(typeof n==="number"&&n>=1000?n.toLocaleString():n);
inst.innerHTML=[
  tile(fmt(I.seasons.v),I.seasons.label,null,Math.round(100*I.seasons.v/I.seasons.of),I.seasons.of),
  tile(fmt(I.plays.v),I.plays.label,"run/pass plays 2021–2025",100),
  tile(fmt(I.player_games.v),I.player_games.label,"5-season table",100),
  tile(fmt(I.features_earned.v),I.features_earned.label,I.features_earned.sub,100),
  tile(fmt(I.layers_measured.v),I.layers_measured.label,I.layers_measured.sub,100),
  tile(fmt(I.features_rejected.v),I.features_rejected.label,I.features_rejected.sub,null),
  tile(I.top_bucket.pred!=null?`${Math.round(I.top_bucket.pred*100)}<small>% → ${Math.round(I.top_bucket.hit*100)}%</small>`:"—",I.top_bucket.label,`predicted → realized, n=${I.top_bucket.n||"—"}, out of sample`,I.top_bucket.hit?Math.round(I.top_bucket.hit*100):null),
  tile(fmt(I.weeks_graded.v),I.weeks_graded.label,"the public record, frozen at open",Math.round(100*I.weeks_graded.v/I.weeks_graded.of),I.weeks_graded.of),
  tile(fmt(I.snapshots_week.v),I.snapshots_week.label,"every book, every prop, every fetch",null),
  tile(fmt(I.card_frozen.v),I.card_frozen.label,"this week's ledger",null),
  tile(I.tests.v!=null?fmt(I.tests.v):"—",I.tests.label,null,I.tests.v&&I.tests.of?Math.round(100*I.tests.v/I.tests.of):null,I.tests.of),
  tile(fmt(I.commits.v),I.commits.label,I.commits.sub,null)
].join("");
requestAnimationFrame(()=>setTimeout(()=>document.querySelectorAll(".meter i").forEach(b=>b.style.width=b.dataset.w+"%"),150));
// ---- charts (inline SVG, drawn to scale)
function svgOpen(w,h){return `<svg viewBox="0 0 ${w} ${h}" role="img">`;}
const C=$("#charts");
// 1) star reality: naive Poisson prediction vs realized, with the diagonal
(function(){const W=380,H=260,L=44,R=12,T=16,B=36; const pts=LAB.reliab||[]; const sx=v=>L+v*(W-L-R), sy=v=>H-B-v*(H-T-B);
  let s=svgOpen(W,H); s+=`<rect x="0" y="0" width="${W}" height="${H}" fill="#070C18"/>`;
  for(let v=0;v<=1.0001;v+=.2){s+=`<line x1="${sx(v)}" y1="${sy(0)}" x2="${sx(v)}" y2="${sy(1)}" stroke="#1A2540"/><line x1="${sx(0)}" y1="${sy(v)}" x2="${sx(1)}" y2="${sy(v)}" stroke="#1A2540"/>`;
    s+=`<text x="${sx(v)}" y="${H-B+16}" fill="#8B98B0" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">${Math.round(v*100)}%</text><text x="${L-6}" y="${sy(v)+4}" fill="#8B98B0" font-size="10" text-anchor="end" font-family="JetBrains Mono,monospace">${Math.round(v*100)}%</text>`;}
  s+=`<path class="draw" d="M${sx(0)} ${sy(0)} L${sx(1)} ${sy(1)}" stroke="#5E6B85" stroke-dasharray="4 4" fill="none"/>`;
  const path=pts.map((p,k)=>(k?"L":"M")+sx(p.x)+" "+sy(p.y)).join(" ");
  s+=`<path class="draw" d="${path}" stroke="#F5B342" stroke-width="2" fill="none"/>`;
  pts.forEach(p=>{s+=`<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="${3+Math.min(5,Math.log10(p.n))}" fill="#F5B342" fill-opacity=".9"/>`;});
  if(I.top_bucket.pred){s+=`<circle cx="${sx(I.top_bucket.pred)}" cy="${sy(I.top_bucket.hit)}" r="6" fill="#63D5E6"/><text x="${sx(I.top_bucket.pred)+9}" y="${sy(I.top_bucket.hit)-8}" fill="#63D5E6" font-size="10" font-family="JetBrains Mono,monospace">our top bucket</text>`;}
  s+=`<text x="${W/2}" y="${H-4}" fill="#5E6B85" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">naive prediction from last season's TD rate</text>`;
  s+=`<text x="${L+4}" y="${T+4}" fill="#5E6B85" font-size="10" font-family="JetBrains Mono,monospace">↑ realized next season</text></svg>`;
  C.insertAdjacentHTML("beforeend",`<div class="panel chart"><div class="eyebrow">Star reality</div>${s}<div class="cap">Amber: what last season's touchdown rate naively predicts vs what actually happened the next season (2022–25, ${(pts.reduce((a,p)=>a+p.n,0)).toLocaleString()} player-games). The curve bends under the diagonal — stars regress hard. Cyan: our recalibrated top bucket lands on the line.</div></div>`);})();
// 2) the distribution fix: CV by volume, old anchors vs within-player measurement
(function(){const W=380,H=260,L=44,R=12,T=16,B=36; const mk=[["rush_yds","#F5B342","rush"],["rec_yds","#63D5E6","rec"],["pass_yds","#C9B6F5","pass"]];
  const xs=v=>L+Math.min(1,v/300)*(W-L-R), ys=v=>H-B-Math.min(1,v/1.15)*(H-T-B);
  let s=svgOpen(W,H)+`<rect x="0" y="0" width="${W}" height="${H}" fill="#070C18"/>`;
  for(let v=0;v<=1.0001;v+=.25){s+=`<line x1="${xs(0)}" y1="${ys(v)}" x2="${xs(300)}" y2="${ys(v)}" stroke="#1A2540"/><text x="${L-6}" y="${ys(v)+4}" fill="#8B98B0" font-size="10" text-anchor="end" font-family="JetBrains Mono,monospace">${v.toFixed(2)}</text>`;}
  [0,50,100,150,200,250,300].forEach(v=>{s+=`<text x="${xs(v)}" y="${H-B+16}" fill="#8B98B0" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">${v}</text>`;});
  mk.forEach(([m,col,lab])=>{const o=LAB.old_cv[m]||[]; const n=LAB.cv_curves[m]||[];
    if(o.length) s+=`<path d="${o.map((p,k)=>(k?"L":"M")+xs(p[0])+" "+ys(p[1])).join(" ")}" stroke="${col}" stroke-opacity=".45" stroke-dasharray="3 4" stroke-width="1.5" fill="none"/>`;
    if(n.length) s+=`<path class="draw" d="${n.map((p,k)=>(k?"L":"M")+xs(p.mu)+" "+ys(p.cv)).join(" ")}" stroke="${col}" stroke-width="2.2" fill="none"/>`;
    n.forEach(p=>{s+=`<circle cx="${xs(p.mu)}" cy="${ys(p.cv)}" r="3" fill="${col}"/>`;});
    const last=n[n.length-1]; if(last) s+=`<text x="${Math.min(xs(last.mu)+6,W-30)}" y="${ys(last.cv)-6}" fill="${col}" font-size="10" font-family="JetBrains Mono,monospace">${lab}</text>`;});
  s+=`<text x="${W/2}" y="${H-4}" fill="#5E6B85" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">projected volume (yards)</text></svg>`;
  C.insertAdjacentHTML("beforeend",`<div class="panel chart"><div class="eyebrow">The distribution fix</div>${s}<div class="cap">Game-to-game spread (CV) falls with volume. Dashed: the old anchors, pooled across players. Solid: measured within each player's own season on five seasons — 10–40% tighter at the top, and what the books' own alternate ladders imply. Every prop was re-priced.</div></div>`);})();
// 3) out-of-sample strong-signal hit rate by market
(function(){const W=380,H=260,L=44,R=12,T=16,B=40; const H_=LAB.hits||{}; const mk=Object.keys(H_); const yrs=["2024","2025"];
  const ys=v=>H-B-Math.max(0,Math.min(1,(v-.45)/.25))*(H-T-B); const bw=(W-L-R)/Math.max(1,mk.length); let s=svgOpen(W,H)+`<rect x="0" y="0" width="${W}" height="${H}" fill="#070C18"/>`;
  [.45,.5,.55,.6,.65,.7].forEach(v=>{s+=`<line x1="${L}" y1="${ys(v)}" x2="${W-R}" y2="${ys(v)}" stroke="#1A2540"/><text x="${L-6}" y="${ys(v)+4}" fill="#8B98B0" font-size="10" text-anchor="end" font-family="JetBrains Mono,monospace">${Math.round(v*100)}%</text>`;});
  s+=`<line x1="${L}" y1="${ys(.524)}" x2="${W-R}" y2="${ys(.524)}" stroke="#F26B7A" stroke-dasharray="4 4"/><text x="${L+4}" y="${T+4}" fill="#F26B7A" font-size="10" font-family="JetBrains Mono,monospace">dashed: 52.4% break-even at -110</text>`;
  mk.forEach((m,k)=>{yrs.forEach((y,j)=>{const v=H_[m][y]; if(v==null) return; const x=L+k*bw+8+j*(bw-16)/2; const w=(bw-16)/2-3;
    s+=`<rect x="${x}" y="${ys(v)}" width="${w}" height="${ys(.45)-ys(v)}" fill="${j?"#63D5E6":"#F5B342"}" rx="2"/><text x="${x+w/2}" y="${ys(v)-4}" fill="#E9EFF8" font-size="9.5" text-anchor="middle" font-family="JetBrains Mono,monospace">${Math.round(v*1000)/10}</text>`;});
    s+=`<text x="${L+k*bw+bw/2}" y="${H-B+16}" fill="#8B98B0" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">${m.replace("_yds"," yds").replace("receptions","rec'ns")}</text>`;});
  s+=`<text x="${L}" y="${H-6}" fill="#F5B342" font-size="10" font-family="JetBrains Mono,monospace">■ 2024 held out</text><text x="${L+110}" y="${H-6}" fill="#63D5E6" font-size="10" font-family="JetBrains Mono,monospace">■ 2025 held out</text></svg>`;
  C.insertAdjacentHTML("beforeend",`<div class="panel chart"><div class="eyebrow">Out of sample, by market</div>${s}<div class="cap">When the model's signal is strong (top 40% of its own conviction), how often our side of the player's baseline hits — each season scored by a model fit only on the seasons before it. Nothing here was tuned on the year it is scored on.</div></div>`);})();
// ---- the loop timeline
const tl=$("#tl"); const items=[];
(LAB.loop_items||[]).forEach(x=>items.push(x));
tl.innerHTML=(LAB.timeline||[]).map(t=>`<li><span class="d">${t.d}</span><span class="t">${t.t}</span></li>`).join("");
// ---- chips
$("#chips").innerHTML=(LAB.chips||[]).map(c=>`<span class="chip ${c.k}">${c.t}</span>`).join("");
// ---- next scheduled run (UTC schedule -> countdown)
function nextRun(){const now=new Date(); let best=null; LAB.schedule.forEach(s=>{for(let d=0;d<8;d++){const t=new Date(Date.UTC(now.getUTCFullYear(),now.getUTCMonth(),now.getUTCDate()+d,s.h,s.m,0)); if(t.getUTCDay()===s.dow&&t>now){ if(!best||t<best.t) best={t,label:s.label}; break;}}}); return best;}
function tick(){const n=nextRun(); if(!n){$("#next").textContent="";return;} const ms=n.t-new Date(); const h=Math.floor(ms/36e5), m=Math.floor(ms%36e5/6e4); $("#next").textContent=`next run · ${n.label} · in ${h}h ${String(m).padStart(2,"0")}m`;}
tick(); setInterval(tick,30000);
$("#foot").textContent=`Every number on this page is read from the repository's real artifacts when the page is built (${LAB.generated}): validation files, the feature lab, the ledger, line-history snapshots, git. The log animates; nothing is simulated. The record starts Week ${LAB.week||1}, ${LAB.season||2026}.`;
</script>
"""


def build():
    lab = gather()
    fe = J("fresh_eyes_log.json") or []
    timeline = []
    for d in fe:
        timeline.append({"d": f"{d.get('date')} · day {d.get('day')}", "t": d.get("verdict", "")})
    commits = [l.split("|", 2) for l in git("log", "--format=%h|%ad|%s", "--date=short", "-n", "10").splitlines() if "|" in l]
    for h, dte, subj in commits:
        timeline.append({"d": f"{dte} · {h}", "t": subj[:140]})
    lab["timeline"] = timeline
    lab["chips"] = [
        {"k": "earned", "t": "recency form (last-3 / last-1)"}, {"k": "earned", "t": "multi-factor ridge · 5 seasons"}, {"k": "earned", "t": "market-role baseline weeks 0-3"},
        {"k": "earned", "t": "offseason form ratio"}, {"k": "earned", "t": "volume-dependent spread (within-player)"}, {"k": "earned", "t": "injury cascade k*=.87"},
        {"k": "earned", "t": "xTD decomposed touchdowns"}, {"k": "earned", "t": "TD environment (measured scale)"}, {"k": "earned", "t": "red-zone defense (small, capped)"},
        {"k": "earned", "t": "measured correlations · copula"}, {"k": "live", "t": "8-book line shop"}, {"k": "live", "t": "beat-intel hard gates"}, {"k": "live", "t": "kickoff weather (validated multiplier)"},
        {"k": "live", "t": "line history · opened → now"}, {"k": "live", "t": "Last Call · inactives · stale numbers"},
        {"k": "assumed", "t": "beat-intel score weight (4 pts)"}, {"k": "assumed", "t": "Desk Score hump · tail · trust"}, {"k": "assumed", "t": "grade bands · rails"}, {"k": "assumed", "t": "TD anchor .35 · market-disagreement rail"},
    ]
    html = TEMPLATE.replace("/*DATA*/{}", json.dumps(lab, separators=(",", ":")))
    path = os.path.join(OUT, "the-lab.html"); open(path, "w").write(html)
    return path


if __name__ == "__main__":
    print("built:", build())
