/* defined-target-lab :: browser simulator page. MIT license.
 * Page code only: every number comes from engine.js (checked against the Python engine by selftest.html).
 * Nothing leaves the browser: no requests, no storage, no trackers. Inputs can arrive after a "#" in the link.
 */
(function () {
  "use strict";
  var E = window.DTL, $ = function (id) { return document.getElementById(id); };
  var MS_YEAR = 365 * 86400000;

  // Presets. Market inputs are the dated example (README facts, close of 2026-10-13); users update them.
  var MARKET = {date: "2026-10-13", etf: 778.01, idx: 7805.45, rate: 4.86};
  var FUNDS = {
    XX: {button: "XX", etf: "SPY", per: "share", underlying: "etf", target: "2030-01-10", units: 0.48, strike: 995.01,
         cash: 0.25, fee: 0.99, finish: 9995, q: 0.99, paid: 26.51,
         title: "S&amp;P 500 TARGET <b>10,000</b> &#9472;&#9472;&gt; <b>2030-01-10</b>",
         holds: function (v) { return "1 SHARE = " + v.units + " SPY CALLS @ " + v.strike + " + CASH"; }},
    LISTED: {button: "SPX 10000C DEC-29", etf: "S&P", per: "contract", underlying: "index", target: "2029-12-21",
             units: 100, strike: 10000, cash: 0, fee: 0, finish: 10000, q: 1.08, modelVol: 0.17,
             title: "LISTED S&amp;P 500 CALL <b>10,000</b> &#9472;&#9472;&gt; <b>2029-12-21</b> (AM SETTLED)",
             holds: function () { return "1 CONTRACT = 100 x S&P 500 CALL @ 10,000 // NO FEE, NO CASH"; },
             note: "Price paid is a model value at 17% vol, not a quote: type your broker's ask x 100 (one contract)."},
    SPEC: {button: "+ SPEC", per: "share", underlying: "etf", q: 0.99, modelVol: 0.166,
           note: "Price paid is a model value at 16.6% vol: type the fund's real price."}
  };
  var SPEC_EXAMPLE = {
    name: "Hypothetical call-spread target fund", etf: "SPY", target: "2030-01-10", index_per_etf_target: 10.045,
    cash: 0.3, fee: 0.95,
    legs: [{kind: "call", sign: 1, strike: 900, units: 0.5}, {kind: "call", sign: -1, strike: 1100, units: 0.5}]
  };
  var MUS = [null, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20, 0.24];
  var VIEWS = [["pl", "P/L $"], ["plp", "P/L %"], ["val", "VALUE"], ["lev", "LEVERAGE"]];
  var PRESETS = [["F1 CRASH +3", 3, "am"], ["F2 CRASH +6", 6, "am"], ["F3 CRASH +10", 10, "am"],
                 ["F4 CRUSH -3", -3, ""], ["F5 RESET", 0, ""]];
  var S = {fund: "XX", spec: null, volTyped: false, shift: 0, slideOn: true, slide: 1.0, view: "pl", mode: "table",
           muKey: 0, customMu: 0.18, range: 40, fromLink: false, cells: null};
  var NUM_FIELDS = ["paid", "etf", "idx", "rate", "q", "vol", "units", "strike", "cash", "fee", "finish", "ratio"];

  // ---------- small helpers ----------
  function num(id) {
    var el = $(id), x = parseFloat(String(el.value).replace(/[,$%\s]/g, ""));
    el.classList.toggle("bad", !isFinite(x) && !el.disabled);
    return isFinite(x) ? x : NaN;
  }
  function fmt(x, d) {
    return isFinite(x) ? x.toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d}) : "--";
  }
  function pct(x, d) { return isFinite(x) ? (x * 100).toFixed(d === undefined ? 1 : d) + "%" : "--"; }
  function signed(x, d) {
    if (!isFinite(x)) return "--";
    if (Math.abs(x) < 0.5 * Math.pow(10, -d)) x = 0;             // no "-0.00" from rounding dust
    return (x >= 0 ? "+" : "") + fmt(x, d);
  }
  function money(x) { return Math.abs(x) >= 10000 ? signed(x / 1000, 1) + "k" : Math.abs(x) >= 100 ? signed(x, 0) : signed(x, 2); }
  function utc(s) { var p = String(s).split("-"); return p.length === 3 ? Date.UTC(+p[0], +p[1] - 1, +p[2]) : NaN; }
  function iso(ms) { return new Date(ms).toISOString().slice(0, 10); }
  function addMonths(ms, n) { var d = new Date(ms); return Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + n, d.getUTCDate()); }
  var MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];
  function colLabel(ms) { var d = new Date(ms); return MON[d.getUTCMonth()] + "<br>" + String(d.getUTCFullYear()).slice(2); }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
  function F() { return FUNDS[S.fund]; }
  function round(x, d) { var k = Math.pow(10, d); return Math.round(x * k) / k; }
  function mu() { return S.muKey === "c" ? S.customMu : MUS[S.muKey]; }

  // ---------- funds and inputs ----------
  function specFund() {                                // the loaded spec as a fund the engine understands
    var s = S.spec;
    return {legs: s.legs.map(function (L) { return {kind: L.kind, sign: L.sign, strike: +L.strike, units: +L.units}; }),
            cash: +s.cash, fee: +s.fee / 100};
  }
  function parseSpec(text) {
    var s = JSON.parse(text);
    if (!s || !Array.isArray(s.legs) || !s.legs.length) throw new Error("needs a non-empty \"legs\" list");
    s.legs.forEach(function (L, i) {
      if ((L.kind !== "call" && L.kind !== "put") || (L.sign !== 1 && L.sign !== -1) || !(L.strike > 0) || !(L.units > 0))
        throw new Error("leg " + (i + 1) + ": kind call/put, sign 1 or -1, strike > 0, units > 0");
    });
    if (!(utc(s.target) > 0)) throw new Error("\"target\" must be a date like 2030-01-10");
    if (!(s.cash >= 0) || !(s.fee >= 0)) throw new Error("\"cash\" and \"fee\" must be numbers >= 0");
    if (s.index_per_etf_target !== undefined && !(s.index_per_etf_target > 0)) throw new Error("\"index_per_etf_target\" must be > 0");
    return s;
  }
  function setMarket(f) {
    $("date").value = MARKET.date; $("rate").value = MARKET.rate; $("q").value = f.q;
    $("idx").value = MARKET.idx;
    $("etf").value = f.underlying === "index" ? MARKET.idx : MARKET.etf;
  }
  function loadFund(key, keepMarket) {
    var f = FUNDS[key];
    S.fund = key; S.volTyped = false;
    var spec = key === "SPEC";
    if (spec) {
      $("units").value = ""; $("strike").value = ""; $("cash").value = S.spec.cash; $("fee").value = S.spec.fee;
      $("target").value = S.spec.target;
    } else {
      $("units").value = f.units; $("strike").value = f.strike; $("cash").value = f.cash; $("fee").value = f.fee;
      $("target").value = f.target; $("finish").value = f.finish;
    }
    ["units", "strike", "finish"].forEach(function (id) { $(id).disabled = spec; });
    if (!keepMarket) setMarket(f);
    $("ratio").value = f.underlying === "index" ? 1 : round(num("idx") / num("etf"), 4);
    var etfName = spec ? (S.spec.etf || "ETF") : f.etf;
    $("etfLbl").textContent = etfName; $("ratioLbl").textContent = etfName;
    $("etfField").hidden = f.underlying === "index";
    $("ratioField").hidden = f.underlying === "index";
    $("specBox").hidden = !spec;
    if (!keepMarket) {
      if (f.paid) $("paid").value = f.paid;
      else {                                            // a model price, labeled as such in the note
        var v = readInputs(), M = model(v, f.modelVol);
        $("paid").value = round(E.shareValue(M.fund, v.etf, 0, f.modelVol, M.m), f.per === "contract" ? 0 : 2);
      }
    }
  }
  function fromHash() {
    var h = location.hash.replace(/^#/, "");
    if (!h) return false;
    var map = {paid: "paid", spy: "etf", etf: "etf", spx: "idx", idx: "idx", date: "date", rate: "rate", q: "q",
               vol: "vol", units: "units", strike: "strike", cash: "cash", fee: "fee", finish: "finish", target: "target",
               ratio: "ratio"};
    var got = {};
    h.split("&").forEach(function (kv) {
      var p = kv.split("="), id = map[decodeURIComponent(p[0] || "").toLowerCase()];
      if (id && p[1] !== undefined) { $(id).value = decodeURIComponent(p[1]); got[id] = true; }
    });
    if (got.vol) S.volTyped = true;
    if (got.etf && got.idx && !got.ratio) $("ratio").value = round(num("idx") / num("etf"), 4);
    else if (got.etf && !got.idx) $("idx").value = round(num("etf") * num("ratio"), 2);
    return Object.keys(got).length > 0;
  }
  function readInputs() {
    var v = {}, bad = [], f = F(), spec = S.fund === "SPEC";
    NUM_FIELDS.forEach(function (id) {
      v[id] = num(id);
      var optional = (id === "vol" && !S.volTyped) || ($(id).disabled) || (f.underlying === "index" && (id === "etf" || id === "ratio"));
      if (!isFinite(v[id]) && !optional) bad.push(id);
    });
    if (f.underlying === "index") { v.etf = v.idx; v.ratio = 1; }
    v.d0 = utc($("date").value); v.dT = utc($("target").value);
    v.T = (v.dT - v.d0) / MS_YEAR;
    if (!(v.T > 0)) bad.push("target date must be after the date");
    ["paid", "etf", "idx", "ratio"].concat(spec ? [] : ["units", "strike", "finish"]).forEach(function (id) {
      if (!(v[id] > 0)) bad.push(id);
    });
    if (v.cash < 0 || v.fee < 0) bad.push("cash and fee cannot be negative");
    if (spec) {
      var calls = S.spec.legs.filter(function (L) { return L.kind === "call"; });
      v.ratioT = S.spec.index_per_etf_target || v.ratio;
      v.finish = calls.length ? Math.min.apply(null, calls.map(function (L) { return L.strike; })) * v.ratioT : NaN;
      $("finish").value = isFinite(v.finish) ? round(v.finish, 0) : "";
    } else {
      v.ratioT = f.underlying === "index" ? 1 : v.finish / v.strike;
    }
    v.bad = bad.filter(function (b, i) { return bad.indexOf(b) === i; });
    return v;
  }

  // ---------- the model for one set of inputs ----------
  function model(v, fixedVol) {
    var fund = S.fund === "SPEC" ? specFund() :
      {legs: [{kind: "call", sign: 1, strike: v.strike, units: v.units}], cash: v.cash, fee: v.fee / 100};
    if (S.fund === "SPEC") { fund.cash = v.cash; fund.fee = v.fee / 100; }
    var m = {rate: v.rate / 100, divYield: v.q / 100, horizon: v.T};
    var vol = fixedVol || v.vol / 100;
    if (!fixedVol && !S.volTyped) {
      vol = E.solveVol(fund, v.paid, v.etf, m);
      $("vol").value = isFinite(vol) ? (vol * 100).toFixed(2) : "";
      $("vol").classList.toggle("bad", !isFinite(vol));
    }
    return {fund: fund, m: m, vol: vol, ratioNow: v.idx / v.etf, ratioT: v.ratioT};
  }

  // ---------- render ----------
  function render() {
    var v = readInputs(), note = $("inNote");
    $("volNote").innerHTML = S.volTyped ? "&lt;- TYPED (SOLVE VOL to undo)" : "&lt;- SOLVED FROM PRICE";
    $("solve").disabled = !S.volTyped;
    $("solve").title = S.volTyped ? "Solve the implied vol from the price paid again" : "The vol is already solved from the price paid";
    if (v.bad.length) { note.innerHTML = '<span class="err">check: ' + esc(v.bad.join(", ")) + "</span>"; return; }
    var M = model(v);
    if (!isFinite(M.vol) || M.vol <= 0) {
      // The inputs describe one moment: this price cannot go with this market. Say why, and offer the fair price.
      var per = F().per, lo = Infinity, hi = -Infinity, ref = S.lastVol || 0.166;
      for (var n = 0; n <= 60; n++) {
        var val = E.shareValue(M.fund, v.etf, 0, 1e-4 * Math.pow(3 / 1e-4, n / 60), M.m);
        lo = Math.min(lo, val); hi = Math.max(hi, val);
      }
      var fair = round(E.shareValue(M.fund, v.etf, 0, ref, M.m), per === "contract" ? 0 : 2);
      note.innerHTML = '<span class="err">With these market inputs one ' + per + " is worth between $" + fmt(lo, 2) + " and $" +
        fmt(hi, 2) + " at any volatility, so a price of $" + fmt(v.paid, 2) + " cannot go with them.</span> The inputs describe one " +
        "moment: change the price too, or " + '<button class="k am" type="button" data-fair="' + fair + '">USE FAIR PRICE $' +
        fmt(fair, per === "contract" ? 0 : 2) + " (AT " + pct(ref, 2) + " VOL)</button>";
      return;
    }
    if (!S.volTyped) S.lastVol = M.vol;
    note.innerHTML = S.fromLink ? "Inputs came from the link (after the #)." : (F().note ? F().note + " " : "") +
      "Market inputs: dated example (close of " + MARKET.date + "); type today's numbers to update every figure.";
    $("rangeOut").textContent = " +" + (1.5 * S.range) + "% / -" + S.range + "%";
    renderTape(v, M); renderTitle(v); renderSummary(v, M);
    if (S.mode === "graph") renderGraph(v, M); else renderGrid(v, M);
    $("grid").hidden = S.mode === "graph"; $("graph").hidden = S.mode !== "graph";
    renderOdds(v, M); renderNfo();
  }

  function renderTape(v, M) {
    var f = F(), sep = " &#9474; ", name = S.fund === "SPEC" ? "SPEC" : f.button;
    $("tape").innerHTML = "&gt; " + esc(name) + " <b>" + fmt(v.paid, 2) + "</b> PAID" + sep +
      (f.underlying === "index" ? "" : esc($("etfLbl").textContent) + " <b>" + fmt(v.etf, 2) + "</b>" + sep) +
      "S&amp;P <b>" + fmt(v.idx, 2) + "</b>" + sep + "IV <b>" + pct(M.vol, 2) + "</b>" + sep + "RATE <b>" +
      fmt(v.rate, 2) + "%</b>" + sep + "T-MINUS <b>" + v.T.toFixed(2) + "Y</b>" + sep + "AS OF " + esc($("date").value);
  }
  function renderTitle(v) {
    var f = F(), title, holds;
    if (S.fund === "SPEC") {
      title = esc(S.spec.name || "CUSTOM FUND") + " &#9472;&#9472;&gt; <b>" + esc(S.spec.target) + "</b>";
      holds = "1 SHARE = " + S.spec.legs.map(function (L) {
        return (L.sign > 0 ? "+" : "-") + L.units + " " + L.kind.toUpperCase() + "S @ " + L.strike;
      }).join(" ") + " + CASH";
    } else { title = f.title; holds = f.holds(v); }
    $("title").innerHTML = "&#9617;&#9618;&#9619; <b>DEFINED-TARGET-LAB</b> &#9619;&#9618;&#9617;  TARGET SIM v0.2\n" + title +
      "\n" + esc(holds) + "\n[ runs in your browser // no data leaves it ]";
  }

  function worstValue(fund, m) {
    var pts = [1e-6], w = Infinity;
    fund.legs.forEach(function (L) { pts.push(L.strike); pts.push(20 * L.strike); });
    pts.forEach(function (s) { w = Math.min(w, E.valueAtTarget(fund, s, m)); });
    return w;
  }

  function renderSummary(v, M) {
    var f = F(), T = v.T, etf = f.underlying === "index" ? "S&amp;P" : esc($("etfLbl").textContent);
    var loss = worstValue(M.fund, M.m) - v.paid;
    var be = E.levelForValue(M.fund, v.paid, M.m) * M.ratioT;
    var row = E.outcomeRow(M.fund, v.paid, v.etf, M.vol, M.m, mu());
    var day = E.oneDay(M.fund, v.etf, M.vol, M.m);
    var lev = E.leverage(M.fund, v.etf, 0, M.vol, M.m);
    var a = mu() === null ? "RISK-NEUTRAL" : "AT " + pct(mu(), 0) + " MEAN";
    var items = [
      ["", "PRICE PAID", "$" + fmt(v.paid, 2)],
      ["bad", "MAX LOSS", money(loss) + " (" + pct(loss / v.paid, 0) + ")"],
      ["", "BREAK-EVEN AT TARGET", isFinite(be) ? "S&amp;P " + fmt(be, 0) : "out of reach"],
      ["warn", "GROWTH NEEDED / YR", pct(E.requiredGrowth(v.idx, be, T)) + " (FINISH LINE " + pct(E.requiredGrowth(v.idx, v.finish, T)) + ")"],
      ["bad", "OPTIONS WORTHLESS", pct(row.p_worthless, 0) + " " + a],
      ["", "ANY PROFIT", pct(row.p_profit, 0) + " " + a],
      ["", "LEVERAGE NOW", isFinite(lev) ? lev.toFixed(1) + "x" : "--"],
      ["", "1 DAY, " + etf + " FLAT", pct(day[0], 2) + " OF VALUE"],
      ["", "&#9492; DECAY / CARRY", pct(day[1], 2) + " / " + pct(day[0] - day[1], 2)],
      ["", "1 DAY, " + etf + " ON ITS FORWARD", pct(day[2], 2) + " OF VALUE"]
    ];
    $("sum").innerHTML = items.map(function (it) { return '<div class="' + it[0] + '"><i>' + it[1] + "</i>" + it[2] + "</div>"; }).join("");
  }

  function columns(v) {
    var out = [{ms: v.d0, t: 0}], offs = [3, 6, 9, 12, 18, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120];
    for (var i = 0; i < offs.length; i++) {
      var ms = addMonths(v.d0, offs[i]);
      if (ms > v.dT - 30 * 86400000) break;
      out.push({ms: ms, t: (ms - v.d0) / MS_YEAR});
    }
    out.push({ms: v.dT, t: v.T});
    return out;
  }
  function cellAt(v, M, L, t) {                       // one point of the grid or graph: index level L, t years ahead
    var Sx = L / E.ratioAt(t, M.ratioNow, M.ratioT, v.T);
    var vol = E.volAt(M.vol, S.shift / 100, S.slideOn ? S.slide / 100 : 0, L, v.idx);
    var val = E.shareValue(M.fund, Sx, t, vol, M.m);
    return {L: L, S: Sx, vol: vol, val: val, pl: val - v.paid, lev: E.leverage(M.fund, Sx, t, vol, M.m)};
  }

  function renderGrid(v, M) {
    var cols = columns(v), R = S.range, step = R / 4, rows = [], f = F();
    for (var k = 6; k >= -4; k--) rows.push(k * step / 100);
    var meanST = E.forwardAtTarget(v.etf, M.m, mu());
    var etfName = f.underlying === "index" ? "" : esc($("etfLbl").textContent);
    var html = '<thead><tr><th class="lvl">S&amp;P</th><th class="mv">MOVE</th>' +
      cols.map(function (c) { return "<th>" + colLabel(c.ms) + "</th>"; }).join("") + "<th>P(&gt;)</th></tr></thead><tbody>";
    var cells = [], finDone = false;
    rows.forEach(function (mv, i) {
      var L = v.idx * (1 + mv);
      if (!finDone && L < v.finish) {
        finDone = true;
        if (i > 0) html += '<tr class="fin"><td colspan="' + (cols.length + 3) + '">&#9472;&#9472;[ FINISH LINE ~' + fmt(v.finish, 0) +
          " S&amp;P" + (etfName ? " // " + etfName + " " + fmt(v.finish / M.ratioT, 2) : "") + " ]&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;</td></tr>";
      }
      html += "<tr" + (mv === 0 ? ' class="now"' : "") + '><th class="lvl">' + fmt(L, 0) +
        (etfName ? "<small>" + fmt(L / M.ratioNow, 0) + " " + etfName + "</small>" : "") +
        '</th><th class="mv">' + (mv >= 0 ? "+" : "") + (mv * 100).toFixed(step % 1 ? 1 : 0) + "%</th>";
      cells.push([]);
      cols.forEach(function (c, j) {
        var d = cellAt(v, M, L, c.t), txt, cls, a;
        d.mv = mv; d.ms = c.ms;
        cells[i].push(d);
        if (S.view === "lev") {
          txt = isFinite(d.lev) ? d.lev.toFixed(1) + "x" : "--";
          cls = "l"; a = isFinite(d.lev) ? Math.min(Math.abs(d.lev) / 12, 1) * 0.8 + 0.05 : 0.02;
        } else {
          txt = S.view === "pl" ? money(d.pl) : S.view === "plp" ? signed(d.pl / v.paid * 100, Math.abs(d.pl / v.paid) < 0.1 ? 1 : 0) + "%" :
            "$" + (d.val >= 100 ? fmt(d.val, 0) : fmt(d.val, 2));
          cls = d.pl >= 0 ? "p" : "n"; a = Math.pow(Math.min(Math.abs(d.pl) / v.paid, 1), 0.6) * 0.85 + 0.06;
        }
        html += '<td class="c ' + cls + (a > 0.55 ? " hi" : "") + '" style="--a:' + a.toFixed(2) + '" data-i="' + i + '" data-j="' + j + '">' + txt + "</td>";
      });
      html += '<td class="odds">' + pct(E.probAbove(L / M.ratioT, meanST, M.vol, v.T), 0) + "</td></tr>";
    });
    $("grid").innerHTML = html + "</tbody>";
    S.cells = cells;
    var per = f.per.toUpperCase();
    var titles = {pl: "P/L PER " + per, plp: "P/L % PER " + per, val: "VALUE PER " + per, lev: "LEVERAGE"};
    $("gridTitle").innerHTML = "[ " + titles[S.view] + " // DATE x S&amp;P ]";
    $("gridNote").innerHTML = "Cell = one " + f.per + " on that date with the S&amp;P at that level" +
      (S.view === "lev" ? ": % move per 1% S&amp;P move." : ", minus the price paid where it says P/L.") +
      " Last column: chance the S&amp;P finishes above that level on " + esc($("target").value) + " (" +
      (mu() === null ? "risk-neutral: a price, not a forecast" : "mean total return " + pct(mu(), 0) + "/yr") +
      "; pick another in the odds table). Odds use today's implied vol; the shift and slide change the grid's values only." +
      " One volatility for every strike: listed options price the odds of finishing above the strike about 10 points higher (FINDINGS F10).";
  }

  function renderGraph(v, M) {
    var W = 64, H = 16, R = S.range / 100, lo = v.idx * (1 - R), hi = v.idx * (1 + 1.5 * R), f = F();
    var dates = [[0, "g0", ".", "TODAY"], [1, "g1", ":", "+1 YR"], [2, "g2", "+", "+2 YR"]].filter(function (d) { return d[0] < v.T - 0.25; });
    dates.push([v.T, "g3", "#", "TARGET " + esc($("target").value)]);
    var ys = dates.map(function (d) {
      var out = [];
      for (var x = 0; x < W; x++) out.push(cellAt(v, M, lo + (hi - lo) * x / (W - 1), d[0]).val);
      return out;
    });
    var ymax = Math.max.apply(null, ys.map(function (r) { return Math.max.apply(null, r); })) * 1.05 || 1;
    var grid = [], x, y;
    for (y = 0; y < H; y++) { grid.push([]); for (x = 0; x < W; x++) grid[y].push([" ", ""]); }
    function row(val) { return H - 1 - Math.max(0, Math.min(H - 1, Math.round(val / ymax * (H - 1)))); }
    var py = row(v.paid), fx = Math.round((v.finish - lo) / (hi - lo) * (W - 1));
    for (x = 0; x < W; x++) grid[py][x] = ["-", "gp"];
    if (fx >= 0 && fx < W) for (y = 0; y < H; y++) if (grid[y][fx][0] === " ") grid[y][fx] = ["|", "gf"];
    dates.forEach(function (d, k) { for (x = 0; x < W; x++) grid[row(ys[k][x])][x] = [d[2], d[1]]; });
    var unit = f.per === "contract" ? 1000 : 1, lines = [];
    for (y = 0; y < H; y++) {
      var lab = (ymax * (H - 1 - y) / (H - 1)) / unit;
      var s = (y % 3 === 0 ? (unit > 1 ? fmt(lab, 0) + "k" : fmt(lab, lab < 10 ? 1 : 0)) : ""), run = "", cls = null;
      s = new Array(Math.max(0, 8 - s.length) + 1).join(" ") + s + " |";
      grid[y].forEach(function (c) {
        if (c[1] !== cls) { if (run) s += cls ? '<span class="' + cls + '">' + esc(run) + "</span>" : esc(run); run = ""; cls = c[1]; }
        run += c[0];
      });
      s += cls ? '<span class="' + cls + '">' + esc(run) + "</span>" : esc(run);
      lines.push(s);
    }
    lines.push("        +" + new Array(W + 1).join("-"));
    var ticks = new Array(W + 9).join(" ").split("");
    [0, 0.25, 0.5, 0.75, 1].forEach(function (p) {
      var lbl = fmt(lo + (hi - lo) * p, 0), at = 9 + Math.round(p * (W - 1)) - Math.floor(lbl.length / 2);
      at = Math.max(0, Math.min(ticks.length - lbl.length, at));
      for (var i = 0; i < lbl.length; i++) ticks[at + i] = lbl[i];
    });
    lines.push(ticks.join("") + "  S&P");
    lines.push("");
    lines.push("  " + dates.map(function (d) { return '<span class="' + d[1] + '">' + d[2] + " " + d[3] + "</span>"; }).join("   ") +
      '   <span class="gp">- PRICE PAID</span>   <span class="gf">| FINISH LINE</span>   VALUE PER ' + f.per.toUpperCase() +
      (unit > 1 ? " ($ THOUSANDS)" : " ($)"));
    $("graph").innerHTML = lines.join("\n");
    $("gridTitle").innerHTML = "[ GRAPH // VALUE PER " + f.per.toUpperCase() + " vs S&amp;P ]";
    $("gridNote").innerHTML = "Value of one " + f.per + " against the S&amp;P level on each date (vol shift and slide applied, as in the table).";
    S.cells = null;
    $("readout").innerHTML = "&gt; graph mode: switch to TABLE for cell readouts";
  }

  function readout(i, j) {
    var d = S.cells && S.cells[i] && S.cells[i][j];
    if (!d) return;
    var paid = num("paid"), f = F();
    $("readout").innerHTML = "&gt; " + iso(d.ms) + " &#9474; S&amp;P " + fmt(d.L, 0) + " (" + (d.mv >= 0 ? "+" : "") + (d.mv * 100).toFixed(1) +
      "%)" + (f.underlying === "index" ? "" : " &#9474; " + esc($("etfLbl").textContent) + " " + fmt(d.S, 2)) + " &#9474; VOL " + pct(d.vol, 2) +
      " &#9474; VALUE $" + fmt(d.val, 2) + " &#9474; P/L " + money(d.pl) + " (" + signed(d.pl / paid * 100, 1) + "%) &#9474; LEVERAGE " +
      (isFinite(d.lev) ? d.lev.toFixed(1) + "x" : "--");
    var prev = document.querySelector("#grid td.sel");
    if (prev) prev.classList.remove("sel");
    var td = document.querySelector('#grid td[data-i="' + i + '"][data-j="' + j + '"]');
    if (td) td.classList.add("sel");
  }

  function renderOdds(v, M) {
    var etfName = F().underlying === "index" ? "S&amp;P" : esc($("etfLbl").textContent);
    var h = "<thead><tr><th>ASSUMPTION</th><th>MEDIAN PATH: " + etfName + " PRICE/YR</th><th>TOTAL RETURN/YR</th>" +
      "<th>WORTHLESS</th><th>PROFIT</th><th>DOUBLES</th><th>P10</th><th>MEDIAN</th><th>P90</th></tr></thead><tbody>";
    MUS.map(function (x, i) { return [i, x]; }).concat([["c", S.customMu]]).forEach(function (kv) {
      var key = kv[0], m_ = kv[1], r = E.outcomeRow(M.fund, v.paid, v.etf, M.vol, M.m, m_);
      var label = key === "c" ? "CUSTOM " + pct(m_, 1).replace(".0%", "%") : m_ === null ? "RISK-NEUTRAL (PRICE)" : "MEAN TOTAL RETURN " + pct(m_, 0);
      h += '<tr data-k="' + key + '"' + (key === S.muKey ? ' class="on"' : "") + "><td>" + label + "</td><td>" + pct(r.median_price_cagr) +
        "</td><td>" + pct(r.median_tr_cagr) + "</td><td>" + pct(r.p_worthless, 0) + "</td><td>" + pct(r.p_profit, 0) + "</td><td>" +
        pct(r.p_double, 0) + "</td><td>" + r.multiple_p10.toFixed(2) + "x</td><td>" + r.multiple_median.toFixed(2) + "x</td><td>" +
        r.multiple_p90.toFixed(2) + "x</td></tr>";
    });
    $("odds").innerHTML = h + "</tbody>";
    $("oddsNote").innerHTML = "Lognormal model at today's implied vol (" + pct(M.vol, 2) + "). 'Mean total return' is an average incl. dividends; " +
      "the median path grows more slowly, so compare its price growth with the " + pct(E.requiredGrowth(v.idx, v.finish, v.T)) +
      "/yr the S&amp;P needs just to reach the finish line. Multiples = value at the target date / price paid. Click a row to use it in the grid.";
    $("customNote").textContent = S.customMu > 0.40 ? "above the best 3¼-year stretch since 1871 (40.2%/yr incl. dividends)" :
      S.customMu > 0.27 ? "beyond history's top 5% (26.6%/yr)" : "";
  }

  function renderNfo() {
    var lines = ["model .... Black-76, continuous dividend yield, one vol",
                 "fee ...... paid by selling options (conservative)",
                 "selftest . selftest.html checks this engine vs Python",
                 "data ..... none: you type every input; nothing is sent",
                 "research tool // not investment advice // not affiliated"];
    var w = Math.max.apply(null, lines.map(function (s) { return s.length; })) + 2, title = "══[ NFO ]";
    var out = ["╔" + title + new Array(w - title.length + 1).join("═") + "╗"];
    lines.forEach(function (s) { out.push("║ " + s + new Array(w - 2 - s.length + 1).join(" ") + " ║"); });
    out.push("╚" + new Array(w + 1).join("═") + "╝");
    $("nfo").textContent = out.join("\n");
  }

  // ---------- controls ----------
  function buttons(container, items, isOn, onClick) {
    var c = $(container);
    c.innerHTML = "";
    items.forEach(function (it) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "k" + (it.cls ? " " + it.cls : "") + (isOn(it) ? " on" : "");
      b.textContent = it.label; b.setAttribute("aria-pressed", isOn(it) ? "true" : "false");
      b.addEventListener("click", function () { onClick(it); });
      c.appendChild(b);
    });
  }
  function drawControls() {
    buttons("funds", ["XX", "LISTED", "SPEC"].map(function (k) { return {label: FUNDS[k].button, key: k}; }),
            function (it) { return it.key === S.fund; },
            function (it) {
              if (it.key === "SPEC" && !S.spec) { S.spec = SPEC_EXAMPLE; $("spec").value = JSON.stringify(SPEC_EXAMPLE, null, 1); }
              S.fromLink = false; loadFund(it.key); drawControls(); render();
            });
    buttons("views", VIEWS.map(function (x) { return {label: x[1], key: x[0]}; }),
            function (it) { return it.key === S.view; }, function (it) { S.view = it.key; S.mode = "table"; drawControls(); render(); });
    buttons("modes", [{label: "TABLE", key: "table"}, {label: "GRAPH", key: "graph"}],
            function (it) { return it.key === S.mode; }, function (it) { S.mode = it.key; drawControls(); render(); });
    buttons("presets", PRESETS.map(function (p) { return {label: p[0], val: p[1], cls: p[2]}; }),
            function (it) { return it.val !== 0 && it.val === S.shift; },
            function (it) { S.shift = it.val; $("shift").value = it.val; drawControls(); render(); });
    $("shiftOut").textContent = (S.shift >= 0 ? "+" : "") + S.shift.toFixed(1) + " PTS";
    var so = $("slideOn");
    so.textContent = S.slideOn ? "ON" : "OFF"; so.classList.toggle("on", S.slideOn); so.setAttribute("aria-pressed", String(S.slideOn));
  }

  var timer = null;
  function later() { clearTimeout(timer); timer = setTimeout(render, 120); }
  function link(from) {                                // ETF and S&P fill each other through the ratio
    if (F().underlying === "index") return;
    var etf = num("etf"), idx = num("idx"), r = num("ratio");
    if (from === "etf" && isFinite(etf) && isFinite(r)) $("idx").value = round(etf * r, 2);
    if (from === "idx" && isFinite(idx) && isFinite(r)) $("etf").value = round(idx / r, 2);
    if (from === "ratio" && isFinite(etf) && isFinite(r)) $("idx").value = round(etf * r, 2);
  }

  function init() {
    loadFund("XX");
    $("slide").value = S.slide.toFixed(1);
    $("customMu").value = (S.customMu * 100).toFixed(0);
    S.fromLink = fromHash();
    NUM_FIELDS.concat(["date", "target"]).forEach(function (id) {
      $(id).addEventListener("input", function () {
        if (id === "vol") S.volTyped = true;
        if (id === "etf" || id === "idx" || id === "ratio") link(id);
        S.fromLink = false;
        later();
      });
    });
    var flashTimer = null;
    function flash(text) {                              // a short confirmation beside the buttons
      $("status").textContent = "> " + text;
      clearTimeout(flashTimer);
      flashTimer = setTimeout(function () { $("status").textContent = ""; }, 3000);
    }
    $("solve").addEventListener("click", function () {
      var typed = $("vol").value;
      S.volTyped = false; render();
      flash("VOL SOLVED FROM PRICE: " + $("vol").value + "% (WAS TYPED " + typed + "%)");
    });
    $("inNote").addEventListener("click", function (e) {      // "USE FAIR PRICE": reprice at the last good vol
      var b = e.target.closest ? e.target.closest("button[data-fair]") : null;
      if (b) { $("paid").value = b.getAttribute("data-fair"); S.volTyped = false; render(); }
    });
    $("reset").addEventListener("click", function () {
      S.fromLink = false; loadFund(S.fund); history.replaceState(null, "", location.pathname + location.search); render();
      flash("EXAMPLE LOADED (CLOSE OF " + MARKET.date + ")" + (S.fund === "SPEC" ? "; YOUR SPEC IS KEPT" : ""));
    });
    $("specApply").addEventListener("click", function () {
      try {
        S.spec = parseSpec($("spec").value);
        $("specNote").textContent = "loaded: " + S.spec.legs.length + " leg(s)";
        $("specNote").className = "note";
        loadFund("SPEC"); render();
      } catch (e) { $("specNote").textContent = "spec error: " + e.message; $("specNote").className = "note err"; }
    });
    $("shift").addEventListener("input", function () { S.shift = parseFloat($("shift").value); drawControls(); later(); });
    $("range").addEventListener("input", function () { S.range = parseFloat($("range").value); later(); });
    $("slideOn").addEventListener("click", function () { S.slideOn = !S.slideOn; drawControls(); render(); });
    $("slide").addEventListener("input", function () { var x = num("slide"); if (isFinite(x)) { S.slide = x; later(); } });
    $("customMu").addEventListener("input", function () {
      var x = num("customMu");
      if (isFinite(x) && x > -50 && x < 200) { S.customMu = x / 100; S.muKey = "c"; later(); }
    });
    var grid = $("grid");
    ["mouseover", "click"].forEach(function (ev) {
      grid.addEventListener(ev, function (e) {
        var td = e.target.closest ? e.target.closest("td.c") : null;
        if (td) readout(+td.getAttribute("data-i"), +td.getAttribute("data-j"));
      });
    });
    $("odds").addEventListener("click", function (e) {
      var tr = e.target.closest ? e.target.closest("tr[data-k]") : null;
      if (tr) { var k = tr.getAttribute("data-k"); S.muKey = k === "c" ? "c" : +k; render(); }
    });
    window.addEventListener("hashchange", function () {    // a new "#..." link on an open page: no reload happens
      if (fromHash()) { S.fromLink = true; drawControls(); render(); }
    });
    drawControls();
    render();
    window.APP = {state: S, cell: function (i, j) { return S.cells && S.cells[i] && S.cells[i][j]; }, render: render};
  }
  init();
})();
