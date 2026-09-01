"""Branded social graphics generator: pick cards, slate cards, recap cards.
Renders HTML templates to PNG (1080x1350 for feed, 1600x900 for X) via Playwright.
Brand is configurable; 'BEAT THE CLOSE' is the placeholder identity.
"""
import os
from playwright.sync_api import sync_playwright

BRAND = os.environ.get("BRAND_NAME", "BEAT THE CLOSE")
HANDLE = os.environ.get("BRAND_HANDLE", "@beattheclose")

BASE_CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Poppins','DejaVu Sans',sans-serif;background:#0c120e;color:#eef2ee;
     width:100%;height:100vh;display:flex;flex-direction:column}
.head{display:flex;justify-content:space-between;align-items:center;
      padding:44px 56px 0}
.brand{font-family:'Poppins';font-weight:700;font-size:40px;
       letter-spacing:.06em;color:#4cc492}
.tag{font-family:'DejaVu Sans Mono';font-size:22px;color:#7d8a80}
.title{font-family:'Poppins';font-weight:700;letter-spacing:-.01em;
       padding:28px 56px 8px;font-size:68px;line-height:1.02}
.sub{padding:0 56px;color:#9aa69d;font-size:26px}
.rows{flex:1;display:flex;flex-direction:column;justify-content:center;
      gap:18px;padding:24px 56px}
.row{background:#131a15;border:1px solid #24302а;border:1px solid #24302a;
     border-radius:14px;padding:26px 34px;display:flex;
     justify-content:space-between;align-items:center}
.pick{font-family:'Poppins';font-weight:600;font-size:44px}
.meta{font-family:'DejaVu Sans Mono';font-size:22px;color:#9aa69d;margin-top:6px}
.price{font-family:'DejaVu Sans Mono';font-weight:500;font-size:40px;color:#4cc492}
.price.red{color:#e07b6f}
.badge{font-family:'DejaVu Sans Mono';font-size:20px;background:#1b2a1f;color:#4cc492;
       border-radius:6px;padding:6px 14px;margin-left:16px;vertical-align:middle}
.foot{display:flex;justify-content:space-between;align-items:center;
      padding:0 56px 40px;color:#7d8a80;font-family:'DejaVu Sans Mono';font-size:22px}
.stripe{height:10px;background:linear-gradient(90deg,#4cc492,#0e6b4a)}
.big{font-family:'Poppins';font-weight:700;font-size:190px;line-height:1;
     color:#4cc492;text-align:center}
.big.red{color:#e07b6f}
.center{flex:1;display:flex;flex-direction:column;justify-content:center;
        align-items:center;gap:10px}
"""

FONTS = ("<link rel='stylesheet' href='https://fonts.googleapis.com/css2?"
         "family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&"
         "family=IBM+Plex+Mono:wght@400;500&display=swap'>")


def _page(body, w, h):
    return f"<!doctype html><html><head>{FONTS}<style>{BASE_CSS}</style></head><body style='width:{w}px;height:{h}px'>{body}</body></html>"


def pick_card(picks, title, subtitle, footnote, out, w=1600, h=900):
    rows = "".join(
        f"<div class='row'><div><div class='pick'>{p['pick']}"
        + (f"<span class='badge'>{p['badge']}</span>" if p.get('badge') else "")
        + f"</div><div class='meta'>{p['meta']}</div></div>"
        + f"<div class='price'>{p['price']}</div></div>"
        for p in picks)
    body = (f"<div class='stripe'></div><div class='head'><div class='brand'>{BRAND}</div>"
            f"<div class='tag'>{HANDLE}</div></div>"
            f"<div class='title'>{title}</div><div class='sub'>{subtitle}</div>"
            f"<div class='rows'>{rows}</div>"
            f"<div class='foot'><div>{footnote}</div><div>every pick graded · full record in bio</div></div>")
    _render(_page(body, w, h), out, w, h)


def recap_card(headline, sub, lines, footnote, out, w=1600, h=900, red=False):
    rows = "".join(
        f"<div class='row'><div class='pick'>{l['pick']}</div>"
        f"<div class='price {'red' if l.get('loss') else ''}'>{l['result']}</div></div>"
        for l in lines)
    body = (f"<div class='stripe'></div><div class='head'><div class='brand'>{BRAND}</div>"
            f"<div class='tag'>{HANDLE}</div></div>"
            f"<div class='center'><div class='big {'red' if red else ''}'>{headline}</div>"
            f"<div class='sub' style='text-align:center;font-size:32px'>{sub}</div></div>"
            f"<div class='rows' style='flex:none'>{rows}</div>"
            f"<div class='foot'><div>{footnote}</div><div>graded automatically · receipts in bio</div></div>")
    _render(_page(body, w, h), out, w, h)


def _render(html, out, w, h):
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        pg.set_content(html, wait_until="networkidle")
        pg.evaluate("document.fonts.ready")
        pg.wait_for_timeout(600)
        pg.screenshot(path=out)
        b.close()
    print("wrote", out)
