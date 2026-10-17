#!/usr/bin/env python3
"""Draw the link-preview card for the browser simulator (docs/img/simulator-card.png, 1200 x 630).

    py -I scripts\\make_card.py

Sites such as X, Reddit and Discord show this image when someone shares the simulator's link (the page's og:image
tag). The heatmap is the simulator's own P/L grid for the README's dated example (2026-10-13 close), computed by
definedtarget/sim.py: model output, not market data. Same terminal look as the page.
"""
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from definedtarget import sim  # noqa: E402

OUT = REPO / "docs" / "img" / "simulator-card.png"
BG, FG, DIM, AMB, RED = "#000000", "#33ff66", "#1c8a3c", "#ffb000", "#ff3355"
FONT = "DejaVu Sans Mono"
LOGO = [" ██╗  ██╗██╗  ██╗", " ╚██╗██╔╝╚██╗██╔╝", "  ╚███╔╝  ╚███╔╝ ", "  ██╔██╗  ██╔██╗ ", " ██╔╝ ██╗██╔╝ ██╗",
        " ╚═╝  ╚═╝╚═╝  ╚═╝"]


def glow(ax, x, y, s, color, size, **kw):
    """Text with a soft phosphor glow (two faint wide copies under the sharp one)."""
    for w, a in ((5, 0.10), (2.5, 0.18)):
        ax.text(x, y, s, color=color, fontsize=size, family=FONT, alpha=a, zorder=2, **kw,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=w, foreground=color)])
    ax.text(x, y, s, color=color, fontsize=size, family=FONT, zorder=3, **kw)


def grid_values():
    """P/L per share on a date x S&P grid for the README example (vol solved from the price; slide +1 pt)."""
    T, r, s0, idx0, paid = 1185 / 365, 0.0486, 778.01, 7805.45, 26.51
    m = sim.Market(r, r - math.log(882.2603 / s0) / 3.2465753, T)
    fund = sim.Fund((sim.Leg("call", 1, 995.01, 0.48),), cash=0.25, fee=0.0099)
    vol = sim.solve_vol(fund, paid, s0, m)
    ratio_now, ratio_t = idx0 / s0, 9995 / 995.01
    cols = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, T]
    rows = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4]
    out = []
    for mv in rows:
        lvl = idx0 * (1 + mv)
        out.append([sim.share_value(fund, lvl / sim.ratio_at(t, ratio_now, ratio_t, T), t,
                                    sim.vol_at(vol, 0.0, 0.01, lvl, idx0), m) - paid for t in cols])
    return out, paid


def main():
    import matplotlib.patheffects  # noqa: F401  (used by glow)
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(630, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1200, 630, color=BG, zorder=0))

    # top tape
    ax.add_line(plt.Line2D([40, 1160], [58, 58], color=DIM, lw=1, zorder=1))
    ax.text(40, 42, "> XX // S&P 500 TARGET 10,000 2030 // DATE x S&P // P/L PER SHARE", color=AMB, fontsize=15,
            family=FONT, zorder=3)

    # logo and title
    for i, line in enumerate(LOGO):
        glow(ax, 40, 100 + i * 25, line, FG, 22, va="center")
    glow(ax, 470, 112, "DEFINED-TARGET-LAB", FG, 30, va="center", weight="bold")
    ax.text(470, 150, "TARGET SIM", color=DIM, fontsize=22, family=FONT, va="center", zorder=3)
    ax.text(470, 192, "What is one share of a defined-target ETF", color=FG, fontsize=17, family=FONT, va="center")
    ax.text(470, 220, "worth at any S&P level, on any date?", color=FG, fontsize=17, family=FONT, va="center")

    # heatmap
    vals, paid = grid_values()
    x0, y0, cw, ch = 40, 290, 62, 22
    for i, row in enumerate(vals):
        for j, pl in enumerate(row):
            a = (min(abs(pl) / paid, 1.0) ** 0.6) * 0.85 + 0.06
            ax.add_patch(Rectangle((x0 + j * cw, y0 + i * ch), cw - 3, ch - 3, color=FG if pl >= 0 else RED,
                                   alpha=a, lw=0, zorder=2))
    ax.add_patch(Rectangle((x0 - 6, y0 - 6), 10 * cw + 9, 11 * ch + 9, fill=False, edgecolor=DIM, lw=1, zorder=2))
    ax.text(x0, y0 + 11 * ch + 24, "rows: S&P -40% .. +60%   columns: today .. 2030-01-10", color=DIM, fontsize=12,
            family=FONT, zorder=3)

    # right column: what it does
    bullets = ["value, P/L and leverage per share", "odds of finishing above each level",
               "crash / crush volatility scenarios", "XX, a listed SPX call, or any spec",
               "runs in your browser; no data sent"]
    for i, b in enumerate(bullets):
        ax.text(712, 312 + i * 40, "> " + b, color=FG, fontsize=15, family=FONT, va="center", zorder=3)

    # footer
    ax.add_line(plt.Line2D([40, 1160], [580, 580], color=DIM, lw=1, zorder=1))
    ax.text(40, 605, "open source (MIT) // research tool, not investment advice // not affiliated with any issuer",
            color=DIM, fontsize=13, family=FONT, va="center", zorder=3)

    # CRT scanlines
    for y in range(0, 630, 3):
        ax.add_line(plt.Line2D([0, 1200], [y, y], color="#000000", alpha=0.18, lw=0.6, zorder=5))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=100, facecolor=BG, metadata={"Software": None})
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
