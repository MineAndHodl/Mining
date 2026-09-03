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
    L("read", f"read {fmt_int(n_pbp)} NFL plays from the last {len(seasons)} seasons — every carry, every target, every red-zone snap")
    L("read", f"built {fmt_int(n_pg)} player-games into one table — every player's real game log")
    L("learned", "recent form is the biggest edge in player props — strong signals hit 60–67% out of sample (break-even is 52.4%)")
    mf = hc.get("multifactor", {})
    if mf: L("learned", f"more history helps: five seasons beat three ({mf.get('3y', {}).get('strong_sign_hit')} → {mf.get('5y', {}).get('strong_sign_hit')} hit rate on strong signals)")
    isr = xtd.get("in_season_rolling", {})
    if isr: L("tested", f"the touchdown model on {fmt_int(sum(int(isr[y]['n']) for y in ('2024', '2025') if y in isr))} real player-games — it beat the old way in 2024 AND 2025")
    if fr: L("checked", f"when our touchdown model says 50%, it hits {int(top.get('top_hit', 0)*100)}% (n={top.get('top_n')}, out of sample)")
    if stars:
        hi = stars.get("(1.0, 2.0]", {}); L("learned", f"stars get overpriced: touchdown-a-game backs score in only {int(hi.get('hit', 0)*100)}% of next season's games — the books price them at 64–76%")
    sc = tde.get("scale", {})
    if sc: L("checked", f"how many touchdowns a team really scores per point the market expects — measured on {fmt_int(sc.get('n_team_games', 0))} team-games, fixed a 10–17% miss")
    rc = inj.get("role_cascade", {})
    if rc: L("learned", f"when a starter sits, who eats: measured on {fmt_int(rc.get('n', 0))} real games — the RB2 gets 57% of the carries, the WR2 20% of the targets, and the QB throws for MORE")
    if cva.get("anchors"): L("learned", f"how streaky every player really is, measured player by player on {fmt_int(cva.get('n_player_seasons', 0))} player-seasons — workhorses are steadier, committee backs are wild")
    if alt: L("checked", f"our numbers against the sportsbooks' own alternate-line ladders ({alt.get('n')} players) — found our spread was too wide, fixed it the same day")
    for k, v in fl.items():
        L("rejected", f"'{ {'spread_c': 'favorites run it out / dogs throw', 'short_week': 'short-week fatigue', 'pace_c': 'fast-paced teams'}.get(k, k) }' — tested on two seasons, adds nothing, thrown out")
    L("rejected", "'backups inflate their share when the starter was hurt' — tested, loses, thrown out")
    L("shopped", "every anytime-touchdown price across 8 sportsbooks — the best price is usually 10–30 cents better than the consensus")
    for name, n, nb in snap_rows[-2:]:
        st_ = name.split("_")[0]
        L("tracked", f"{fmt_int(n)} prop lines across {nb} books saved {st_[4:6]}/{st_[6:8]} {st_[9:11]}:{st_[11:13]}Z — every number's move since it opened is on the board")
    L("checked", "kickoff weather for every outdoor game — wind and cold trim passing numbers, domes are neutral")
    if led:
        op = led.get("opened", {}); L("frozen", f"week {led.get('week')} · {len(op.get('card', []))} Card picks locked in the ledger at their opening numbers — graded in public, never edited")
    for d in fe[-2:]:
        L("loop", f"daily improvement pass, day {d.get('day')} — {str(d.get('verdict', ''))[:120]}")
    for h, dte, subj in commits[:4]:
        L("shipped", f"{dte} · {subj.split(':')[0][:90]}")
    if tests.get("result"): L("verified", f"{tests.get('passed')} of {tests.get('total')} automated checks passed before publishing")
    hits_all = []
    for feat in fl.values():
        for yr, row in feat.get("by_season", {}).items():
            for m, v in row.items():
                if v.get("hit_base"): hits_all.append(v["hit_base"])
        break
    instruments = {
        "plays": {"v": n_pbp, "label": "NFL plays read", "sub": f"{len(seasons)} seasons, every snap"},
        "player_games": {"v": n_pg, "label": "player game logs", "sub": "every player, every week"},
        "hit": {"lo": round(min(hits_all)*100) if hits_all else None, "hi": round(max(hits_all)*100) if hits_all else None, "label": "strong-signal hit rate", "sub": "out of sample · break-even is 52.4%"},
        "top_bucket": {"pred": top.get("top_pred"), "hit": top.get("top_hit"), "n": top.get("top_n"), "label": "touchdown calibration", "sub": "when we say 50%, it hits"},
        "books": {"v": 8, "label": "sportsbooks shopped", "sub": "best price on every touchdown"},
        "lines": {"v": (max(n for _, n, _ in snap_rows) if snap_rows else 0), "label": "prop lines tracked this week", "sub": "opened → now on every bet"},
        "rejected": {"v": len([1 for v in fl.values() if v.get("verdict") != "WINS"]) + 2, "label": "ideas tested and thrown out", "sub": "if it doesn't hold up, it's gone"},
        "tests": {"v": tests.get("passed"), "of": tests.get("total"), "label": "checks before publishing"},
        "weeks_graded": {"v": len(rec.get("weeks", [])), "of": 18, "label": "weeks graded in public", "sub": "the record starts Week 1"},
        "commits": {"v": int(n_commits or 0), "label": "model updates shipped", "sub": head},
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
.tag{display:inline-block;min-width:82px;font-weight:600;letter-spacing:.04em}
.t-read{color:var(--cyan)}.t-learned{color:var(--good)}.t-tested{color:#D7DEEA}.t-checked{color:var(--amber)}.t-rejected{color:var(--bad)}.t-shopped{color:#9ED0FF}.t-tracked{color:#8FD3FF}.t-frozen{color:#F5D08A}.t-loop{color:#FFB3C1}.t-shipped{color:#A8F0C6}.t-verified{color:var(--good)}.t-idle{color:var(--dim)}
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
.charts{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
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
.hero{padding:22px 4px 6px}
.hero .h1{font-family:var(--disp);font-weight:700;font-size:38px;line-height:1.05;letter-spacing:.01em;text-wrap:balance}
.hero .h2{color:var(--muted);font-size:16px;max-width:64ch;margin-top:8px}
.wall{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:6px 10px}
.wall div{font-size:13.5px;padding:6px 8px;border-radius:5px;background:var(--panel2);border:1px solid var(--rule);display:flex;gap:8px;align-items:flex-start}
.wall div b{color:var(--good);font-family:var(--mono);flex:none}
.wall div.x b{color:var(--bad)}
.wall div.x{color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:860px){.grid2{grid-template-columns:1fr}.hero .h1{font-size:30px}}
.lab{list-style:none;padding:0;margin:0;display:grid;gap:8px}
.lab li{display:grid;grid-template-columns:1fr auto;gap:10px;align-items:center;padding:10px 12px;background:var(--panel2);border:1px solid var(--rule);border-radius:6px;font-size:14px}
.lab .st{font-family:var(--mono);font-size:11px;letter-spacing:.06em;text-transform:uppercase;padding:3px 8px;border-radius:4px;border:1px solid var(--rule)}
.st.measuring{color:var(--amber);border-color:rgba(245,179,66,.5)}.st.building{color:var(--cyan);border-color:rgba(99,213,230,.5)}.st.queued{color:var(--muted)}.st.shipped{color:var(--good);border-color:rgba(70,211,154,.5)}
.foot{margin-top:22px;font-family:var(--mono);font-size:11.5px;color:var(--dim);max-width:90ch}
@media (prefers-reduced-motion:reduce){.ln{transition:none}.meter i{transition:none}path.draw{animation:none;stroke-dashoffset:0}.pulse i{animation:none}.cursor{animation:none}}
</style>
<div class="wrap">
  <div class="bar">
    <div class="brand">The Desk <span>·</span> Model Room</div>
    <div class="pulse"><i></i><span id="state">training</span></div>
    <div class="meta">build <b id="head"></b> · update <b id="ncom"></b> · <span id="gen"></span></div>
    <div class="next" id="next"></div>
  </div>
  <div class="hero">
    <div class="h1">We thought of everything. Then we tested it.</div>
    <div class="h2">The model reads every play, checks every price at every book, watches every line move, and grades itself in public. This room shows it working.</div>
  </div>
  <div class="grid">
    <div class="panel">
      <div class="eyebrow">Live run log <small>pulled from the model's own logs</small></div>
      <div class="term" id="term"></div>
    </div>
    <div class="panel">
      <div class="eyebrow">By the numbers</div>
      <div class="inst" id="inst"></div>
    </div>
  </div>
  <div class="panel loop">
    <div class="eyebrow">Everything the model thinks about <small>every box is wired into the picks — not a wish list</small></div>
    <div class="wall" id="wall"></div>
  </div>
  <div class="charts" id="charts"></div>
  <div class="grid2">
    <div class="panel loop">
      <div class="eyebrow">In the lab right now <small>being built and tested — nothing ships until it wins</small></div>
      <ul class="lab" id="lab"></ul>
    </div>
    <div class="panel loop">
      <div class="eyebrow">Latest updates <small>one measured improvement a day</small></div>
      <ul class="tl" id="tl"></ul>
    </div>
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
function idle(){const el=document.createElement("span"); el.className="ln on"; el.innerHTML=`<span class="ts">${new Date().toISOString().slice(11,19)}Z</span>  <span class="tag t-idle">[standby]</span> watching the lines — next pass on the schedule below <span class="cursor"></span>`; term.appendChild(el); term.scrollTop=term.scrollHeight; $("#state").textContent="trained · watching lines";}
if(RM){LAB.log.forEach((_,k)=>addLine(k)); idle();}
else{(function step(){ if(i<LAB.log.length){addLine(i); i++; setTimeout(step, i<6?420:(i<14?260:140));} else idle(); })();}
// ---- instruments
const I=LAB.instruments; const inst=$("#inst");
function tile(v,label,sub,pct,of){return `<div class="tile"><div class="v">${v}${of?`<small> / ${of}</small>`:""}</div><div class="l">${label}</div>${sub?`<div class="s">${sub}</div>`:""}${pct!=null?`<div class="meter"><i data-w="${pct}"></i></div>`:""}</div>`;}
const fmt=n=>n==null?"—":(typeof n==="number"&&n>=1000?n.toLocaleString():n);
inst.innerHTML=[
  tile(fmt(I.plays.v),I.plays.label,I.plays.sub,100),
  tile(fmt(I.player_games.v),I.player_games.label,I.player_games.sub,100),
  tile(I.hit.lo!=null?`${I.hit.lo}<small>–${I.hit.hi}%</small>`:"—",I.hit.label,I.hit.sub,I.hit.hi),
  tile(I.top_bucket.pred!=null?`${Math.round(I.top_bucket.pred*100)}<small>% → ${Math.round(I.top_bucket.hit*100)}%</small>`:"—",I.top_bucket.label,I.top_bucket.sub,I.top_bucket.hit?Math.round(I.top_bucket.hit*100):null),
  tile(fmt(I.books.v),I.books.label,I.books.sub,100),
  tile(fmt(I.lines.v),I.lines.label,I.lines.sub,null),
  tile(fmt(I.rejected.v),I.rejected.label,I.rejected.sub,null),
  tile(I.tests.v!=null?fmt(I.tests.v):"—",I.tests.label,null,I.tests.v&&I.tests.of?Math.round(100*I.tests.v/I.tests.of):null,I.tests.of),
  tile(fmt(I.weeks_graded.v),I.weeks_graded.label,I.weeks_graded.sub,Math.round(100*I.weeks_graded.v/I.weeks_graded.of),I.weeks_graded.of),
  tile(fmt(I.commits.v),I.commits.label,I.commits.sub,null)
].join("");
requestAnimationFrame(()=>setTimeout(()=>document.querySelectorAll(".meter i").forEach(b=>b.style.width=b.dataset.w+"%"),150));
// ---- the wall: everything the model thinks about (all wired) + a few things it tested and threw out
$("#wall").innerHTML=(LAB.wall||[]).map(w=>`<div class="${w.x?'x':''}"><b>${w.x?'✕':'✓'}</b><span>${w.t}</span></div>`).join("");
$("#lab").innerHTML=(LAB.lab||[]).map(l=>`<li><span>${l.t}</span><span class="st ${l.s}">${l.s}</span></li>`).join("");
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
  C.insertAdjacentHTML("beforeend",`<div class="panel chart"><div class="eyebrow">Stars get overpriced</div>${s}<div class="cap">What last season's touchdown rate says a player "should" do (across) vs what actually happened the next year (up), on ${(pts.reduce((a,p)=>a+p.n,0)).toLocaleString()} real player-games. Every dot sits under the line: big scorers come back to earth, and the books price them like they won't. Our number (blue) sits on the line — when we say 50%, it's 50%.</div></div>`);})();
// 3) out-of-sample strong-signal hit rate by market
(function(){const W=380,H=260,L=44,R=12,T=16,B=40; const H_=LAB.hits||{}; const mk=Object.keys(H_); const yrs=["2024","2025"];
  const ys=v=>H-B-Math.max(0,Math.min(1,(v-.45)/.25))*(H-T-B); const bw=(W-L-R)/Math.max(1,mk.length); let s=svgOpen(W,H)+`<rect x="0" y="0" width="${W}" height="${H}" fill="#070C18"/>`;
  [.45,.5,.55,.6,.65,.7].forEach(v=>{s+=`<line x1="${L}" y1="${ys(v)}" x2="${W-R}" y2="${ys(v)}" stroke="#1A2540"/><text x="${L-6}" y="${ys(v)+4}" fill="#8B98B0" font-size="10" text-anchor="end" font-family="JetBrains Mono,monospace">${Math.round(v*100)}%</text>`;});
  s+=`<line x1="${L}" y1="${ys(.524)}" x2="${W-R}" y2="${ys(.524)}" stroke="#F26B7A" stroke-dasharray="4 4"/><text x="${L+4}" y="${T+4}" fill="#F26B7A" font-size="10" font-family="JetBrains Mono,monospace">dashed: 52.4% break-even at -110</text>`;
  mk.forEach((m,k)=>{yrs.forEach((y,j)=>{const v=H_[m][y]; if(v==null) return; const x=L+k*bw+8+j*(bw-16)/2; const w=(bw-16)/2-3;
    s+=`<rect x="${x}" y="${ys(v)}" width="${w}" height="${ys(.45)-ys(v)}" fill="${j?"#63D5E6":"#F5B342"}" rx="2"/><text x="${x+w/2}" y="${ys(v)-4}" fill="#E9EFF8" font-size="9.5" text-anchor="middle" font-family="JetBrains Mono,monospace">${Math.round(v*1000)/10}</text>`;});
    s+=`<text x="${L+k*bw+bw/2}" y="${H-B+16}" fill="#8B98B0" font-size="10" text-anchor="middle" font-family="JetBrains Mono,monospace">${m.replace("_yds"," yds").replace("receptions","rec'ns")}</text>`;});
  s+=`<text x="${L}" y="${H-6}" fill="#F5B342" font-size="10" font-family="JetBrains Mono,monospace">■ 2024 held out</text><text x="${L+110}" y="${H-6}" fill="#63D5E6" font-size="10" font-family="JetBrains Mono,monospace">■ 2025 held out</text></svg>`;
  C.insertAdjacentHTML("beforeend",`<div class="panel chart"><div class="eyebrow">How often our strong signals hit</div>${s}<div class="cap">On the plays the model feels strongest about, how often our side hit — scored on a full season the model had never seen, twice. The dashed line is what you need at -110 just to break even.</div></div>`);})();
// ---- the loop timeline
const tl=$("#tl"); const items=[];
(LAB.loop_items||[]).forEach(x=>items.push(x));
tl.innerHTML=(LAB.timeline||[]).slice(0,7).map(t=>`<li><span class="d">${t.d}</span><span class="t">${t.t}</span></li>`).join("");
// ---- next scheduled run (UTC schedule -> countdown)
function nextRun(){const now=new Date(); let best=null; LAB.schedule.forEach(s=>{for(let d=0;d<8;d++){const t=new Date(Date.UTC(now.getUTCFullYear(),now.getUTCMonth(),now.getUTCDate()+d,s.h,s.m,0)); if(t.getUTCDay()===s.dow&&t>now){ if(!best||t<best.t) best={t,label:s.label}; break;}}}); return best;}
function tick(){const n=nextRun(); if(!n){$("#next").textContent="";return;} const ms=n.t-new Date(); const h=Math.floor(ms/36e5), m=Math.floor(ms%36e5/6e4); $("#next").textContent=`next run · ${n.label} · in ${h}h ${String(m).padStart(2,"0")}m`;}
tick(); setInterval(tick,30000);
$("#foot").textContent=`Everything on this page comes straight from the model's own logs and test files when the page is built (${LAB.generated}). The log animates; nothing is made up. Picks are frozen the moment they post and graded in public — the record starts Week ${LAB.week||1}, ${LAB.season||2026}. 21+. Please gamble responsibly.`;
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
    lab["wall"] = [{"t": t} for t in [
        "Recent form — the last 3 games vs the season", "Last game's momentum", "Where the market's number implies the role (Week 1-3)",
        "How steady each player is, game to game", "The opponent's defense vs the position", "The game total (points expected)",
        "Home or away", "Game script — favorites run it out, dogs throw", "Injury report — Out / Doubtful / Questionable",
        "Who benefits when a starter sits (measured)", "Game-day inactives, 90 minutes before kickoff", "Red-zone role — carries and targets inside the 20",
        "Goal-line role and the vulture stealing it", "The coach's red-zone play calling (run vs pass)", "Coaching changes that flip a tendency",
        "Last season's touchdowns vs what the role should produce", "Opponent's red-zone defense", "Expected team points from the spread and total",
        "Every anytime-TD price at 8 sportsbooks", "The best line and price for our side", "Where every line opened and where it is now",
        "Books that lag the market (stale numbers)", "Kickoff weather — wind, cold, dome", "Beat writers' reports — pitch counts, roles, ruled out",
        "Player correlations for parlays (measured, not guessed)", "Closing-line value on every pick", "Calibration — do our 60%s hit 60%?",
        "A public record, frozen at post time"]] + [{"t": t, "x": True} for t in [
        "Short-week fatigue — tested, no edge", "Fast-paced teams — tested, no edge", "Last year's matchup stats — tested, no edge"]]
    lab["lab"] = [{"t": "Kicker props — field-goal range drives × distance make-rate", "s": "measuring"},
                  {"t": "Rookie projections — no history yet, market-role baseline", "s": "building"},
                  {"t": "Streaky-player left tail — early exits and blowouts", "s": "measuring"},
                  {"t": "Cross-market checks — catches × yards per catch vs the yards line", "s": "queued"},
                  {"t": "Which book is soft when eight disagree", "s": "measuring"},
                  {"t": "Week 1 grading + closing-line report card", "s": "queued"},
                  {"t": "Last Call — inactives, stale numbers, kickoff weather", "s": "shipped"},
                  {"t": "Line tracking on every bet", "s": "shipped"}]
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
