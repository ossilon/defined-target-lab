/* defined-target-lab :: browser simulator page. MIT license.
 * Page code only: every number comes from engine.js (checked against the Python engine by selftest.html).
 * Nothing leaves the browser: no requests, no storage, no trackers. Inputs can arrive after a "#" in the link.
 */
(function () {
  "use strict";
  var E = window.DTL, $ = function (id) { return document.getElementById(id); };
  var MS_YEAR = 365 * 86400000;

  // Dated example (README facts, close of 2026-10-13). Users update these.
  var FUNDS = {
    XX: {key: "XX", button: "XX", etf: "SPY", per: "share", target: "2030-01-10", units: 0.48, strike: 995.01,
         cash: 0.25, fee: 0.99, finish: 9995, kind: "call",
         example: {date: "2026-10-13", paid: 26.51, etf: 778.01, idx: 7805.45, rate: 4.86, q: 0.99},
         title: "S&P 500 TARGET <b>10,000</b> &#9472;&#9472;&gt; <b>2030-01-10</b>",
         holds: function (u, k) { return "1 SHARE = " + u + " SPY CALLS @ " + k + " + CASH"; }}
  };
  var MUS = [null, 0.04, 0.06, 0.08, 0.10, 0.12];
  var VIEWS = [["pl", "P/L $"], ["plp", "P/L %"], ["val", "VALUE"], ["lev", "LEVERAGE"]];
  var PRESETS = [["F1 CRASH +3", 3, "am"], ["F2 CRASH +6", 6, "am"], ["F3 CRASH +10", 10, "am"],
                 ["F4 CRUSH -3", -3, ""], ["F5 RESET", 0, ""]];
  var S = {fund: "XX", volTyped: false, shift: 0, slideOn: true, slide: 1.0, view: "pl", mu: null, range: 40,
           fromLink: false, cells: null};
  var NUM_FIELDS = ["paid", "etf", "idx", "rate", "q", "vol", "units", "strike", "cash", "fee", "finish"];

  // ---------- small helpers ----------
  function num(id) {
    var el = $(id), x = parseFloat(String(el.value).replace(/[,$%\s]/g, ""));
    var ok = isFinite(x);
    el.classList.toggle("bad", !ok);
    return ok ? x : NaN;
  }
  function fmt(x, d) {
    if (!isFinite(x)) return "--";
    return x.toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});
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
  function el(tag, attrs, html) {
    var e = document.createElement(tag);
    for (var k in attrs || {}) e.setAttribute(k, attrs[k]);
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }

  // ---------- inputs ----------
  function loadFund(key, keepMarket) {
    var F = FUNDS[key], X = F.example;
    S.fund = key;
    $("units").value = F.units; $("strike").value = F.strike; $("cash").value = F.cash; $("fee").value = F.fee;
    $("target").value = F.target; $("finish").value = F.finish;
    if (!keepMarket) {
      $("paid").value = X.paid; $("etf").value = X.etf; $("idx").value = X.idx; $("date").value = X.date;
      $("rate").value = X.rate; $("q").value = X.q;
    }
    $("etfLbl").textContent = F.etf;
    S.volTyped = false;
  }
  function fromHash() {
    var h = location.hash.replace(/^#/, "");
    if (!h) return false;
    var map = {paid: "paid", spy: "etf", etf: "etf", spx: "idx", idx: "idx", date: "date", rate: "rate", q: "q",
               vol: "vol", units: "units", strike: "strike", cash: "cash", fee: "fee", finish: "finish", target: "target"};
    var any = false;
    h.split("&").forEach(function (kv) {
      var p = kv.split("="), id = map[decodeURIComponent(p[0] || "").toLowerCase()];
      if (id && p[1] !== undefined) { $(id).value = decodeURIComponent(p[1]); any = true; if (id === "vol") S.volTyped = true; }
    });
    return any;
  }
  function readInputs() {
    var v = {}, bad = [];
    NUM_FIELDS.forEach(function (id) { v[id] = num(id); if (!isFinite(v[id]) && !(id === "vol" && !S.volTyped)) bad.push(id); });
    v.d0 = utc($("date").value); v.dT = utc($("target").value);
    v.T = (v.dT - v.d0) / MS_YEAR;
    if (!(v.T > 0)) bad.push("target date must be after the date");
    ["paid", "etf", "idx", "units", "strike", "finish"].forEach(function (id) { if (!(v[id] > 0)) bad.push(id); });
    if (v.cash < 0 || v.fee < 0) bad.push("cash and fee cannot be negative");
    v.bad = bad.filter(function (b, i) { return bad.indexOf(b) === i; });
    return v;
  }

  // ---------- the model for one set of inputs ----------
  function model(v) {
    var fund = {legs: [{kind: FUNDS[S.fund].kind, sign: 1, strike: v.strike, units: v.units}], cash: v.cash, fee: v.fee / 100};
    var m = {rate: v.rate / 100, divYield: v.q / 100, horizon: v.T};
    var vol = v.vol / 100;
    if (!S.volTyped) {
      vol = E.solveVol(fund, v.paid, v.etf, m);
      $("vol").value = isFinite(vol) ? (vol * 100).toFixed(2) : "";
      $("vol").classList.toggle("bad", !isFinite(vol));
    }
    return {fund: fund, m: m, vol: vol, ratioNow: v.idx / v.etf, ratioT: v.finish / v.strike};
  }

  // ---------- render ----------
  function render() {
    var v = readInputs(), note = $("inNote");
    $("volNote").innerHTML = S.volTyped ? "&lt;- TYPED (SOLVE VOL to undo)" : "&lt;- SOLVED FROM PRICE";
    if (v.bad.length) { note.innerHTML = '<span class="err">check: ' + esc(v.bad.join(", ")) + "</span>"; return; }
    var M = model(v);
    if (!isFinite(M.vol) || M.vol <= 0) {
      note.innerHTML = '<span class="err">No volatility gives that price (it must be above the cash and below the ' +
        "price at very high volatility). Check the inputs or type a vol.</span>";
      return;
    }
    note.innerHTML = S.fromLink ? "Inputs came from the link (after the #)." :
      "Dated example (close of " + FUNDS[S.fund].example.date + "): type today's numbers to update every figure.";
    renderTape(v, M); renderTitle(v); renderSummary(v, M); renderGrid(v, M); renderOdds(v, M); renderNfo();
  }

  function renderTape(v, M) {
    var F = FUNDS[S.fund], sep = " &#9474; ";
    $("tape").innerHTML = "&gt; " + F.button + " <b>" + fmt(v.paid, 2) + "</b> PAID" + sep + F.etf + " <b>" + fmt(v.etf, 2) +
      "</b>" + sep + "S&amp;P <b>" + fmt(v.idx, 2) + "</b>" + sep + "IV <b>" + pct(M.vol, 2) + "</b>" + sep + "RATE <b>" +
      fmt(v.rate, 2) + "%</b>" + sep + "T-MINUS <b>" + v.T.toFixed(2) + "Y</b>" + sep + "AS OF " + esc($("date").value);
  }
  function renderTitle(v) {
    var F = FUNDS[S.fund];
    $("title").innerHTML = "&#9617;&#9618;&#9619; <b>DEFINED-TARGET-LAB</b> &#9619;&#9618;&#9617;  TARGET SIM v0.2\n" + F.title +
      "\n" + F.holds(v.units, v.strike) + "\n[ runs in your browser // no data leaves it ]";
  }

  function worstValue(fund, m) {
    var pts = [1e-6], w = Infinity;
    fund.legs.forEach(function (L) { pts.push(L.strike); pts.push(20 * L.strike); });
    pts.forEach(function (s) { w = Math.min(w, E.valueAtTarget(fund, s, m)); });
    return w;
  }

  function renderSummary(v, M) {
    var F = FUNDS[S.fund], T = v.T;
    var worst = worstValue(M.fund, M.m), loss = worst - v.paid;
    var be = E.levelForValue(M.fund, v.paid, M.m) * M.ratioT;
    var row = E.outcomeRow(M.fund, v.paid, v.etf, M.vol, M.m, S.mu);
    var day = E.oneDay(M.fund, v.etf, M.vol, M.m);
    var lev = E.leverage(M.fund, v.etf, 0, M.vol, M.m);
    var a = S.mu === null ? "RISK-NEUTRAL" : "AT " + pct(S.mu, 0) + " MEAN";
    var items = [
      ["", "PRICE PAID", "$" + fmt(v.paid, 2)],
      ["bad", "MAX LOSS", money(loss) + " (" + pct(loss / v.paid, 0) + ")"],
      ["", "BREAK-EVEN AT TARGET", isFinite(be) ? "S&amp;P " + fmt(be, 0) : "out of reach"],
      ["warn", "GROWTH NEEDED / YR", pct(E.requiredGrowth(v.idx, be, T)) + " (STRIKE " + pct(E.requiredGrowth(v.idx, v.finish, T)) + ")"],
      ["bad", "OPTIONS WORTHLESS", pct(row.p_worthless, 0) + " " + a],
      ["", "ANY PROFIT", pct(row.p_profit, 0) + " " + a],
      ["", "LEVERAGE NOW", isFinite(lev) ? lev.toFixed(1) + "x" : "--"],
      ["", "1 DAY, " + F.etf + " FLAT", pct(day[0], 2) + " NAV"],
      ["", "&#9492; DECAY / CARRY", pct(day[1], 2) + " / " + pct(day[0] - day[1], 2)],
      ["", "1 DAY, " + F.etf + " ON ITS FORWARD", pct(day[2], 2) + " NAV"]
    ];
    $("sum").innerHTML = items.map(function (it) {
      return '<div class="' + it[0] + '"><i>' + it[1] + "</i>" + it[2] + "</div>";
    }).join("");
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

  function renderGrid(v, M) {
    var cols = columns(v), R = S.range, step = R / 4, rows = [];
    for (var k = 6; k >= -4; k--) rows.push(k * step / 100);
    $("rangeOut").textContent = " +" + (1.5 * R) + "% / -" + R + "%";
    var slide = S.slideOn ? S.slide / 100 : 0, shift = S.shift / 100;
    var meanST = E.forwardAtTarget(v.etf, M.m, S.mu);
    var html = '<thead><tr><th class="lvl">S&amp;P</th><th class="mv">MOVE</th>' +
      cols.map(function (c) { return "<th>" + colLabel(c.ms) + "</th>"; }).join("") + "<th>P(&gt;)</th></tr></thead><tbody>";
    var cells = [], finDone = false, F = FUNDS[S.fund];
    rows.forEach(function (mv, i) {
      var L = v.idx * (1 + mv);
      if (!finDone && L < v.finish) {
        finDone = true;
        if (i > 0) html += '<tr class="fin"><td colspan="' + (cols.length + 3) + '">&#9472;&#9472;[ FINISH LINE ~' + fmt(v.finish, 0) +
          " S&amp;P // " + F.etf + " " + fmt(v.strike, 2) + " ]&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;</td></tr>";
      }
      var etfNow = L / M.ratioNow;
      html += "<tr" + (mv === 0 ? ' class="now"' : "") + '><th class="lvl">' + fmt(L, 0) + "<small>" + fmt(etfNow, 0) + " " + F.etf +
        '</small></th><th class="mv">' + (mv >= 0 ? "+" : "") + (mv * 100).toFixed(step % 1 ? 1 : 0) + "%</th>";
      cells.push([]);
      cols.forEach(function (c, j) {
        var Sx = L / E.ratioAt(c.t, M.ratioNow, M.ratioT, v.T);
        var vol = E.volAt(M.vol, shift, slide, L, v.idx);
        var val = E.shareValue(M.fund, Sx, c.t, vol, M.m);
        var lev = E.leverage(M.fund, Sx, c.t, vol, M.m);
        var pl = val - v.paid, d = {L: L, mv: mv, S: Sx, ms: c.ms, vol: vol, val: val, pl: pl, lev: lev};
        cells[i].push(d);
        var txt, cls, a;
        if (S.view === "lev") {
          txt = isFinite(lev) ? lev.toFixed(1) + "x" : "--";
          cls = "l"; a = isFinite(lev) ? Math.min(lev / 12, 1) * 0.8 + 0.05 : 0.02;
        } else {
          txt = S.view === "pl" ? money(pl) : S.view === "plp" ? signed(pl / v.paid * 100, Math.abs(pl / v.paid) < 0.1 ? 1 : 0) + "%" :
            "$" + (val >= 100 ? fmt(val, 0) : fmt(val, 2));
          cls = pl >= 0 ? "p" : "n"; a = Math.pow(Math.min(Math.abs(pl) / v.paid, 1), 0.6) * 0.85 + 0.06;
        }
        html += '<td class="c ' + cls + (a > 0.55 ? " hi" : "") + '" style="--a:' + a.toFixed(2) + '" data-i="' + i + '" data-j="' + j + '">' + txt + "</td>";
      });
      html += '<td class="odds">' + pct(E.probAbove(L / M.ratioT, meanST, M.vol, v.T), 0) + "</td></tr>";
    });
    $("grid").innerHTML = html + "</tbody>";
    S.cells = cells;
    var titles = {pl: "P/L PER " + F.per.toUpperCase(), plp: "P/L % PER " + F.per.toUpperCase(), val: "VALUE PER " + F.per.toUpperCase(),
                  lev: "LEVERAGE"};
    $("gridTitle").innerHTML = "[ " + titles[S.view] + " // DATE x S&amp;P ]";
    $("gridNote").innerHTML = "Cell = one " + F.per + " on that date with the S&amp;P at that level" +
      (S.view === "lev" ? ": % move per 1% S&amp;P move." : ", minus the price paid where it says P/L.") +
      " Last column: chance the S&amp;P finishes above that level on " + esc($("target").value) + " (" +
      (S.mu === null ? "risk-neutral: a price, not a forecast" : "mean total return " + pct(S.mu, 0) + "/yr") +
      "; pick another in the odds table). Odds use today's implied vol; the shift and slide change the grid's values only." +
      " One volatility for every strike: listed options price the odds of finishing above the strike about 10 points higher (FINDINGS F10).";
  }

  function readout(i, j) {
    var d = S.cells && S.cells[i] && S.cells[i][j];
    if (!d) return;
    var paid = num("paid"), F = FUNDS[S.fund];
    $("readout").innerHTML = "&gt; " + iso(d.ms) + " &#9474; S&amp;P " + fmt(d.L, 0) + " (" + (d.mv >= 0 ? "+" : "") + (d.mv * 100).toFixed(1) +
      "%) &#9474; " + F.etf + " " + fmt(d.S, 2) + " &#9474; VOL " + pct(d.vol, 2) + " &#9474; VALUE $" + fmt(d.val, 2) + " &#9474; P/L " +
      money(d.pl) + " (" + signed(d.pl / paid * 100, 1) + "%) &#9474; LEVERAGE " + (isFinite(d.lev) ? d.lev.toFixed(1) + "x" : "--");
    var prev = document.querySelector("#grid td.sel");
    if (prev) prev.classList.remove("sel");
    var td = document.querySelector('#grid td[data-i="' + i + '"][data-j="' + j + '"]');
    if (td) td.classList.add("sel");
  }

  function renderOdds(v, M) {
    var h = "<thead><tr><th>ASSUMPTION</th><th>MEDIAN PATH: " + FUNDS[S.fund].etf + " PRICE/YR</th><th>TOTAL RETURN/YR</th>" +
      "<th>WORTHLESS</th><th>PROFIT</th><th>DOUBLES</th><th>P10</th><th>MEDIAN</th><th>P90</th></tr></thead><tbody>";
    MUS.forEach(function (mu, i) {
      var r = E.outcomeRow(M.fund, v.paid, v.etf, M.vol, M.m, mu);
      h += '<tr data-k="' + i + '"' + (mu === S.mu ? ' class="on"' : "") + "><td>" + (mu === null ? "RISK-NEUTRAL (PRICE)" : "MEAN TOTAL RETURN " + pct(mu, 0)) +
        "</td><td>" + pct(r.median_price_cagr) + "</td><td>" + pct(r.median_tr_cagr) + "</td><td>" + pct(r.p_worthless, 0) + "</td><td>" +
        pct(r.p_profit, 0) + "</td><td>" + pct(r.p_double, 0) + "</td><td>" + r.multiple_p10.toFixed(2) + "x</td><td>" +
        r.multiple_median.toFixed(2) + "x</td><td>" + r.multiple_p90.toFixed(2) + "x</td></tr>";
    });
    $("odds").innerHTML = h + "</tbody>";
    $("oddsNote").innerHTML = "Lognormal model at today's implied vol (" + pct(M.vol, 2) + "). 'Mean total return' is an average incl. dividends; " +
      "the median path grows more slowly, so compare its price growth with the " + pct(E.requiredGrowth(v.idx, v.finish, v.T)) +
      "/yr the S&amp;P needs just to reach the finish line. Multiples = value at the target date / price paid. Click a row to use it in the grid.";
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
      var b = el("button", {"class": "k" + (it.cls ? " " + it.cls : "") + (isOn(it) ? " on" : ""), type: "button"}, esc(it.label));
      b.setAttribute("aria-pressed", isOn(it) ? "true" : "false");
      if (it.disabled) { b.disabled = true; b.title = it.disabled; }
      b.addEventListener("click", function () { onClick(it); });
      c.appendChild(b);
    });
  }
  function drawControls() {
    buttons("funds", [{label: "XX", key: "XX"}, {label: "SPX 10000C DEC-29", key: "LISTED", disabled: "coming in the next build step"},
                      {label: "+ SPEC", key: "SPEC", disabled: "coming in the next build step"}],
            function (it) { return it.key === S.fund; }, function (it) { loadFund(it.key); drawControls(); render(); });
    buttons("views", VIEWS.map(function (x) { return {label: x[1], key: x[0]}; }),
            function (it) { return it.key === S.view; }, function (it) { S.view = it.key; drawControls(); render(); });
    buttons("presets", PRESETS.map(function (p) { return {label: p[0], val: p[1], cls: p[2]}; }),
            function (it) { return it.val !== 0 && it.val === S.shift; },
            function (it) { S.shift = it.val; $("shift").value = it.val; drawControls(); render(); });
    $("shiftOut").textContent = (S.shift >= 0 ? "+" : "") + S.shift.toFixed(1) + " PTS";
    var so = $("slideOn");
    so.textContent = S.slideOn ? "ON" : "OFF"; so.classList.toggle("on", S.slideOn); so.setAttribute("aria-pressed", String(S.slideOn));
  }

  var timer = null;
  function later() { clearTimeout(timer); timer = setTimeout(render, 120); }

  function init() {
    loadFund("XX");
    $("slide").value = S.slide.toFixed(1);
    S.fromLink = fromHash();
    NUM_FIELDS.concat(["date", "target"]).forEach(function (id) {
      $(id).addEventListener("input", function () {
        if (id === "vol") S.volTyped = true;
        S.fromLink = false;
        later();
      });
    });
    $("solve").addEventListener("click", function () { S.volTyped = false; render(); });
    $("reset").addEventListener("click", function () { S.fromLink = false; loadFund(S.fund); history.replaceState(null, "", location.pathname); render(); });
    $("shift").addEventListener("input", function () { S.shift = parseFloat($("shift").value); drawControls(); later(); });
    $("range").addEventListener("input", function () { S.range = parseFloat($("range").value); later(); });
    $("slideOn").addEventListener("click", function () { S.slideOn = !S.slideOn; drawControls(); render(); });
    $("slide").addEventListener("input", function () { var x = num("slide"); if (isFinite(x)) { S.slide = x; later(); } });
    var grid = $("grid");
    ["mouseover", "click"].forEach(function (ev) {
      grid.addEventListener(ev, function (e) {
        var td = e.target.closest ? e.target.closest("td.c") : null;
        if (td) readout(+td.getAttribute("data-i"), +td.getAttribute("data-j"));
      });
    });
    $("odds").addEventListener("click", function (e) {
      var tr = e.target.closest ? e.target.closest("tr[data-k]") : null;
      if (tr) { S.mu = MUS[+tr.getAttribute("data-k")]; render(); }
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
