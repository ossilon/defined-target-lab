"""The daily fund report: one plain-English page per close, built only from tested engine outputs.

Inputs: the fund state table (reconcile.fund_state), the model table (marks.model_daily), the fund spec, and the
effective-trigger band (FINDINGS F7, from scripts/ratio_band.py). Output: Markdown, and the same page as HTML.

Daily attribution ("what moved the price today"), per option on one SPY share, using YESTERDAY's Greeks:
    change in mark ~ delta x dS + 1/2 gamma x dS^2 + vega x d(implied vol) + theta x days + rho x d(rate)
                     + delta x (SPY dividend gone ex) + residual
Because the model solves today's implied vol from today's mark, the vega term absorbs everything the other terms
do not explain; the residual is only the higher-order and cross effects. Per XX share = per option x options per share.
On an ex-dividend date SPY's price drops by the dividend, which the option never receives; the dividend term adds
that back, so the market term is not read as a loss the option did not suffer.
"""
import html
import math
import re

from . import analytics as an

# Effective S&P trigger band (FINDINGS F7, 2026-10-14). Central value and the widest/narrowest history ranges.
TRIGGER_CENTRAL = 9995.0
TRIGGER_RANGE_TEXT = "about 9,955 to 10,040, depending on how much history the estimate uses (central ~9,995)"


def attribution(prev, cur):
    """Per-option decomposition of the mark change between two model rows (dicts or pandas rows)."""
    ds = cur["spot"] - prev["spot"]
    days = (cur["as_of"] - prev["as_of"]).days
    parts = {
        "market (delta)": prev["delta"] * ds,
        "market (gamma)": 0.5 * prev["gamma"] * ds * ds,
        "volatility (vega)": prev["vega_pt"] * (cur["iv_disc_0bp"] - prev["iv_disc_0bp"]) * 100.0,
        "time (theta)": prev["theta_1d"] * days,
        "rates (rho)": prev["rho_1pct"] * (cur["zero_rate_T"] - prev["zero_rate_T"]) * 100.0,
    }
    ex_div = cur.get("ex_div_today", 0.0) if hasattr(cur, "get") else 0.0
    if ex_div == ex_div and ex_div:                                   # not NaN, not zero
        parts["SPY dividend (ex-date)"] = prev["delta"] * ex_div
    total = cur["mark"] - prev["mark"]
    parts["residual"] = total - sum(parts.values())
    return total, parts, days


def _pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def build(state, model, spec, as_of=None):
    """Markdown report for the close `as_of` (default: the latest close in both tables)."""
    st = state.set_index("as_of")
    md = model.set_index("as_of")
    days = [d for d in md.index if d in st.index]
    if not days:
        raise ValueError("no close date present in both the state and model tables")
    as_of = as_of or max(days)
    s, m = st.loc[as_of], md.loc[as_of]
    leg = spec.legs[0]
    ops = m["options_per_share"]
    cash_ps = s["cash_per_share"]
    nav, mkt = s["nav_published"], s["market_price"]
    tau = m["tau"]
    fee = float(spec.fee.value)
    spx_now = m["spx"] if m["spx"] == m["spx"] else m["spot"] * TRIGGER_CENTRAL / leg.strike.value
    ratio_at_expiry = TRIGGER_CENTRAL / leg.strike.value          # S&P per SPY at expiry, central (F7)

    L = []
    L.append(f"# {spec.ticker} daily report: close {as_of}")
    L.append("")
    L.append(f"*{spec.name.value}. A defined-target call ETF: it pays off only if {spec.index.value} finishes above "
             f"about {spec.index_target.value:,} on {spec.target_date.value}. Research tool, not investment advice.*")
    L.append("")
    L.append("## Today in one line")
    L.append("")
    L.append(f"NAV **${nav:.2f}**, market price ${mkt:.2f} ({s['premium']:+.2%} vs NAV). "
             f"SPY {m['spot']:.2f}; the option's implied volatility is **{_pct(m['iv_disc_0bp'], 2)}**; "
             f"each 1% move in SPY moves NAV about **{m['xx_elasticity']:.1f}%**.")
    L.append("")
    L.append("## What one share holds")
    L.append("")
    L.append(f"- Call options on **{ops:.2f} SPY shares** (strike {leg.strike.value}, expiry {leg.expiry.value}, "
             f"European, cash-settled); the issuer's mark of ${m['mark']:.4f} per option values them at "
             f"${ops * m['mark']:.2f}.")
    L.append(f"- Cash ${cash_ps:.2f} ({_pct(s['cash_weight'])} of NAV).")
    L.append(f"- {s['shares_outstanding']:,.0f} shares outstanding"
             + (f"; **flow day** ({s['shares_change']:+,.0f} shares)" if s["flow_day"] else "") + ".")
    L.append("")
    L.append("## What the option's price implies")
    L.append("")
    L.append(f"- Implied volatility {_pct(m['iv_disc_0bp'], 2)} (Treasury curve; with the curve +25 / +50 bp: "
             f"{_pct(m['iv_disc_25bp'], 2)} / {_pct(m['iv_disc_50bp'], 2)}; dividend growth 0% / 10%: "
             f"{_pct(m['iv_disc_0bp_g0'], 2)} / {_pct(m['iv_disc_0bp_g10'], 2)}).")
    L.append(f"- Years to the target date: {tau:.2f}. SPY forward price for that date: {m['forward']:.2f}.")
    L.append(f"- Market-implied (risk-neutral) chance SPY finishes above the strike: **{_pct(m['prob_itm_rn'])}** "
             "(a price, not a forecast).")
    L.append("")
    L.append("## How the share moves (per XX share)")
    L.append("")
    L.append("| If ... | NAV moves about |")
    L.append("|---|---|")
    L.append(f"| SPY moves 1% | {m['xx_elasticity']:.1f}% (its leverage today) |")
    L.append(f"| implied volatility moves 1 point | {_pct(m['xx_vega_pct_nav_per_pt'])} |")
    L.append(f"| interest rates move 1 percentage point | {_pct(m['xx_rho_pct_nav_per_1pct'])} |")
    decay = m["xx_theta_decay_pct_nav_per_day"]
    L.append(f"| one calendar day passes, SPY unchanged | {_pct(m['xx_theta_pct_nav_per_day'], 2)} "
             f"(time decay {_pct(decay, 2)}, carry {_pct(m['xx_theta_pct_nav_per_day'] - decay, 2)}) |")
    L.append(f"| one calendar day passes, SPY drifts along its forward price | "
             f"{_pct(m['xx_theta_fwd_pct_nav_per_day'], 2)} |")
    L.append("")
    L.append("*Time decay: the option's time value melts. Carry: with SPY flat, the forward price (what SPY is priced "
             "to reach at the target date) slips by a day of interest net of dividends. The option's price assumes "
             "SPY rises along that forward on average; the second line is one day's cost if it does exactly that.*")
    L.append("")
    L.append("## Break-even for a buyer today (hold to the target date)")
    L.append("")
    L.append(f"Assumes the fund pays its {fee:.2%}/yr fee by selling options (conservative). S&P levels use the "
             f"central finish-line ratio; the finish line itself sits at {TRIGGER_RANGE_TEXT}.")
    L.append("")
    L.append("| Buying at | Needs SPY above | ~ S&P | Price growth needed per year |")
    L.append("|---|---|---|---|")
    for label, price in (("NAV", nav), ("market price", mkt)):
        be = an.breakeven_spot(price, leg.strike.value, ops, cash_ps, fee, tau)
        L.append(f"| {label} ${price:.2f} | {be:,.2f} | {be * ratio_at_expiry:,.0f} | "
                 f"{an.required_growth(spx_now, be * ratio_at_expiry, tau):.1%} |")
    L.append(f"| (just to reach the strike) | {leg.strike.value:,.2f} | {TRIGGER_CENTRAL:,.0f} | "
             f"{an.required_growth(spx_now, TRIGGER_CENTRAL, tau):.1%} |")
    L.append("")
    L.append("## Odds and outcomes at the target date (buying at NAV)")
    L.append("")
    L.append(f"Lognormal model with volatility {_pct(m['iv_disc_0bp'], 1)} (the option's own implied vol). "
             "Nobody knows the market's future return, so read across the rows.")
    L.append("")
    rows, _, _ = an.outcome_table(nav, m["spot"], m["forward"], m["zero_rate_T"], m["iv_disc_0bp"], tau,
                                  leg.strike.value, ops, cash_ps, fee)
    L.append("| Assumption | Median path: SPY price growth/yr | Median path: total return/yr | Options worthless | "
             "Any profit | At least doubles | 10th pct | Median | 90th pct |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['assumption']} | {_pct(r['median_price_cagr'])} | {_pct(r['median_tr_cagr'])} | "
                 f"{_pct(r['p_worthless'], 0)} | {_pct(r['p_profit'], 0)} | "
                 f"{_pct(r['p_double'], 0)} | {r['multiple_p10']:.2f}x | {r['multiple_median']:.2f}x | "
                 f"{r['multiple_p90']:.2f}x |")
    L.append("")
    L.append(f"*'Mean total return' is an average that includes dividends. The typical (median) path grows more "
             f"slowly, so compare the median SPY price growth with the "
             f"{an.required_growth(spx_now, TRIGGER_CENTRAL, tau):.1%}/yr the index needs just to reach the strike "
             "(break-even table above).*")
    L.append("")
    L.append("*Each multiple divides the share's value at the target date by today's NAV. 0.01x means the options "
             "expired worthless and only the cash buffer remains: whenever 'options worthless' exceeds 50%, the "
             "median outcome is a near-total loss.*")
    L.append("")
    L.append("*The model uses one volatility for every outcome and ignores the volatility skew. On 2026-10-13, "
             "listed S&P 500 option prices put the chance of finishing above the strike about 10 points higher "
             "(one snapshot; FINDINGS F10). Hypothesis H19 tracks that gap.*")
    prev_days = [d for d in days if d < as_of]
    if prev_days:
        p = md.loc[max(prev_days)].to_dict()
        p["as_of"] = max(prev_days)
        c = m.to_dict()
        c["as_of"] = as_of
        total, parts, ndays = attribution(p, c)
        L.append("")
        L.append(f"## What moved the price since {max(prev_days)} ({ndays} calendar day{'s' if ndays != 1 else ''})")
        L.append("")
        L.append(f"Option mark {p['mark']:.4f} -> {m['mark']:.4f}; per XX share {total * ops:+.3f} "
                 f"({total / p['mark']:+.2%}).")
        L.append("")
        prev_nav = p["nav"]
        L.append("| Cause | Per XX share | % of NAV |")
        L.append("|---|---|---|")
        for k, v in parts.items():
            L.append(f"| {k} | {v * ops:+.3f} | {v * ops / prev_nav:+.2%} |")
        L.append(f"| **total** | **{total * ops:+.3f}** | **{total * ops / prev_nav:+.2%}** |")
        if s["flow_day"] or st.loc[max(prev_days)]["flow_day"]:
            L.append("")
            L.append("*A flow day falls in this window: the issuer may mark creation days differently "
                     "(hypothesis H1 tests this; FINDINGS F4).*")
    L.append("")
    L.append("## Sources")
    L.append("")
    L.append(f"- Holdings, NAV and market price: the issuer's daily files for {as_of}. SPY spot: {m['spot_source']}.")
    L.append(f"- Rates: {m['curve_source']}. Dividends: the model projects SPY's history at 5%/yr growth (ASSUMED).")
    L.append("- Model: Black-76 on the dividend-adjusted forward (FINDINGS F6-F9; audited). General information "
             "only; not investment advice; not affiliated with the fund's issuer or index provider.")
    return "\n".join(L) + "\n"


def to_html(markdown, title):
    """Small Markdown-to-HTML converter for the report's own subset (headings, tables, bullets, bold, italics)."""
    def inline(t):
        t = html.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        return re.sub(r"\*(.+?)\*", r"<em>\1</em>", t)

    out, in_table, in_list = [], False, False
    for line in markdown.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_table:
                out.append("<table>")
                in_table = True
                out.append("<tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    style = ("body{font-family:system-ui,sans-serif;max-width:860px;margin:2em auto;padding:0 16px;line-height:1.5;"
             "color:#1b1b1b;background:#fff}table{border-collapse:collapse;margin:1em 0}td,th{border:1px solid #ccc;"
             "padding:4px 10px;text-align:right}td:first-child,th:first-child{text-align:left}"
             "@media (prefers-color-scheme:dark){body{color:#e6e6e6;background:#141414}td,th{border-color:#444}}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
            f"initial-scale=1'><title>{html.escape(title)}</title><style>{style}</style></head><body>"
            + "\n".join(out) + "</body></html>\n")
