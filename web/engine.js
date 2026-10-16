/* defined-target-lab :: browser engine (maths only, no page code). MIT license.
 *
 * A line-for-line port of definedtarget/sim.py and definedtarget/pricing.py. selftest.html checks every function
 * against answers the Python engine wrote to golden.json (agreement to 1e-9). Change the Python first, then this.
 *
 * Plain words: a fund = option legs + cash. One share on date t (years from today), ETF at S, volatility vol:
 *   exp(-fee t) * sum(sign * units * Black-76 option value) + cash
 * Black-76 uses a continuous dividend yield: forward = S exp((r - q) tau), discount = exp(-r tau).
 */
(function (root) {
  "use strict";
  var SQRT2 = Math.SQRT2, SQRTPI = Math.sqrt(Math.PI), INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);
  var DAY = 1 / 365, VOL_FLOOR = 0.01, LN_09 = Math.log(0.9), VOL_LO = 1e-6, VOL_HI = 5.0;

  // erfc: power series below 1.5, continued fraction above (the classic CPython algorithm; ~1e-15 relative).
  function erfSeries(x) {
    var x2 = x * x, acc = 0, fk = 25.5;
    for (var i = 0; i < 25; i++) { acc = 2 + x2 * acc / fk; fk -= 1; }
    return acc * x * Math.exp(-x2) / SQRTPI;
  }
  function erfcContfrac(x) {
    if (x >= 30) return 0;
    var x2 = x * x, a = 0, da = 0.5, p = 1, pLast = 0, q = da + x2, qLast = 1, t, b;
    for (var i = 0; i < 50; i++) {
      a += da; da += 2; b = da + x2;
      t = p; p = b * p - a * pLast; pLast = t;
      t = q; q = b * q - a * qLast; qLast = t;
    }
    return p / q * x * Math.exp(-x2) / SQRTPI;
  }
  function erfc(x) {
    if (x !== x) return NaN;
    var ax = Math.abs(x);
    if (ax < 1.5) return 1 - erfSeries(x);
    var cf = erfcContfrac(ax);
    return x > 0 ? cf : 2 - cf;
  }
  function normCdf(x) { return 0.5 * erfc(-x / SQRT2); }
  function normPdf(x) { return INV_SQRT_2PI * Math.exp(-0.5 * x * x); }

  // Inverse normal: Acklam's rational start, then two Halley steps on normCdf (full double precision).
  var A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
           -3.066479806614716e+01, 2.506628277459239e+00];
  var B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
           -1.328068155288572e+01];
  var C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
           4.374664141464968e+00, 2.938163982698783e+00];
  var D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
  function normInv(p) {
    if (!(p > 0 && p < 1)) return NaN;
    if (p > 0.5) return -normInv(1 - p);               // upper tail by symmetry: refine where digits are not lost
    var q, r, x, lo = 0.02425;
    if (p < lo) {
      q = Math.sqrt(-2 * Math.log(p));
      x = (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
          ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
    } else if (p <= 1 - lo) {
      q = p - 0.5; r = q * q;
      x = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q /
          (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1);
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) /
           ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1);
    }
    for (var i = 0; i < 2; i++) {
      var e = normCdf(x) - p, u = e * Math.sqrt(2 * Math.PI) * Math.exp(0.5 * x * x);
      x = x - u / (1 + 0.5 * x * u);
    }
    return x;
  }

  // ---- Black-76 (pricing.py) ----
  function callPrice(F, K, DF, tau, sigma) {
    var sd = tau > 0 ? sigma * Math.sqrt(tau) : 0;
    if (sd <= 0) return DF * Math.max(F - K, 0);
    var d1 = (Math.log(F / K) + 0.5 * sd * sd) / sd;
    return DF * (F * normCdf(d1) - K * normCdf(d1 - sd));
  }
  function putPrice(F, K, DF, tau, sigma) { return callPrice(F, K, DF, tau, sigma) - DF * (F - K); }
  function callGreeks(F, K, DF, tau, sigma) {           // [price, deltaF, gammaF, vega (per 1.00 vol), probItm]
    var sd = tau > 0 ? sigma * Math.sqrt(tau) : 0;
    if (sd <= 0) { var itm = F > K ? 1 : 0; return [DF * Math.max(F - K, 0), DF * itm, 0, 0, itm]; }
    var d1 = (Math.log(F / K) + 0.5 * sd * sd) / sd, d2 = d1 - sd, nd1 = normCdf(d1), pdf = normPdf(d1);
    return [DF * (F * nd1 - K * normCdf(d2)), DF * nd1, DF * pdf / (F * sd), DF * F * pdf * Math.sqrt(tau), normCdf(d2)];
  }
  function impliedVol(price, F, K, DF, tau) {           // Newton inside a bisection bracket (pricing.implied_vol)
    var tol = 1e-12;
    if (tau <= 0 || !(price > DF * Math.max(F - K, 0)) || !(price < DF * F)) return NaN;
    var lo = VOL_LO, hi = VOL_HI;
    if (callPrice(F, K, DF, tau, hi) < price) return NaN;
    if (callPrice(F, K, DF, tau, lo) > price) return lo;
    var sigma = Math.sqrt(2 * Math.PI / tau) * price / (DF * F);
    sigma = Math.min(Math.max(sigma, 0.05), 1.0);
    for (var i = 0; i < 100; i++) {
      var g = callGreeks(F, K, DF, tau, sigma), diff = g[0] - price, vega = g[3];
      if (Math.abs(diff) <= tol * (price - DF * Math.max(F - K, 0))) return sigma;
      if (diff > 0) hi = sigma; else lo = sigma;
      var nw = vega > 1e-14 ? sigma - diff / vega : 0.5 * (lo + hi);
      if (!(lo < nw && nw < hi)) nw = 0.5 * (lo + hi);
      sigma = nw;
      if (hi - lo < 1e-15) return sigma;
    }
    return sigma;
  }

  // ---- the simulator model (sim.py) ----
  // fund = {legs: [{kind: "call"|"put", sign: 1|-1, strike, units}], cash, fee};  m = {rate, divYield, horizon}
  function optionValue(kind, S, K, tau, vol, m) {
    if (tau <= 0) return kind === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
    var F = S * Math.exp((m.rate - m.divYield) * tau), DF = Math.exp(-m.rate * tau);
    return kind === "call" ? callPrice(F, K, DF, tau, vol) : putPrice(F, K, DF, tau, vol);
  }
  function optionDelta(kind, S, K, tau, vol, m) {
    if (tau <= 0) return kind === "call" ? (S > K ? 1 : 0) : (S < K ? -1 : 0);
    var F = S * Math.exp((m.rate - m.divYield) * tau), DF = Math.exp(-m.rate * tau);
    var dfds = Math.exp((m.rate - m.divYield) * tau), callDelta = callGreeks(F, K, DF, tau, vol)[1] * dfds;
    return kind === "call" ? callDelta : callDelta - DF * dfds;
  }
  function shareValue(fund, S, t, vol, m) {
    var keep = Math.exp(-(fund.fee || 0) * t), tau = m.horizon - t, opts = 0;
    for (var i = 0; i < fund.legs.length; i++) {
      var L = fund.legs[i];
      opts += L.sign * L.units * optionValue(L.kind, S, L.strike, tau, vol, m);
    }
    return keep * opts + (fund.cash || 0);
  }
  function shareDelta(fund, S, t, vol, m) {
    var keep = Math.exp(-(fund.fee || 0) * t), tau = m.horizon - t, d = 0;
    for (var i = 0; i < fund.legs.length; i++) {
      var L = fund.legs[i];
      d += L.sign * L.units * optionDelta(L.kind, S, L.strike, tau, vol, m);
    }
    return keep * d;
  }
  function leverage(fund, S, t, vol, m) {
    var v = shareValue(fund, S, t, vol, m);
    return v > 0 ? shareDelta(fund, S, t, vol, m) * S / v : NaN;
  }
  function volAt(base, shift, slide, S, S0) {
    return Math.max(base + shift + slide * Math.log(S / S0) / LN_09, VOL_FLOOR);
  }
  function solveVol(fund, price, S, m) {
    var lo = 1e-4, hi = 3.0;
    if (!(shareValue(fund, S, 0, lo, m) - price < 0 && 0 < shareValue(fund, S, 0, hi, m) - price)) return NaN;
    for (var i = 0; i < 200; i++) {
      var mid = 0.5 * (lo + hi);
      if (shareValue(fund, S, 0, mid, m) < price) lo = mid; else hi = mid;
      if (hi - lo < 1e-14) break;
    }
    return 0.5 * (lo + hi);
  }
  function oneDay(fund, S, vol, m) {                    // [flat, decay, alongForward] as fractions of today's value
    var v0 = shareValue(fund, S, 0, vol, m);
    var flat = shareValue(fund, S, DAY, vol, m) - v0;
    var fwd = shareValue(fund, S * Math.exp((m.rate - m.divYield) * DAY), DAY, vol, m) - v0;
    var keep = Math.exp(-(fund.fee || 0) * DAY), tau0 = m.horizon;
    var F = S * Math.exp((m.rate - m.divYield) * tau0), DF = Math.exp(-m.rate * tau0), opts = 0;
    for (var i = 0; i < fund.legs.length; i++) {
      var L = fund.legs[i], c = callPrice(F, L.strike, DF, tau0 - DAY, vol);
      opts += L.sign * L.units * (L.kind === "call" ? c : c - DF * (F - L.strike));
    }
    var decay = keep * opts + (fund.cash || 0) - v0;
    return [flat / v0, decay / v0, fwd / v0];
  }
  function forwardAtTarget(S0, m, mu) {                 // mu = null: risk-neutral; else MEAN total return (incl. dividends)
    var growth = (mu === null || mu === undefined) ? m.rate : Math.log1p(mu);
    return S0 * Math.exp((growth - m.divYield) * m.horizon);
  }
  function probAbove(level, meanST, vol, T) {
    if (level <= 0) return 1;
    var sd = vol * Math.sqrt(T);
    return normCdf((Math.log(meanST / level) - 0.5 * sd * sd) / sd);
  }
  function quantile(meanST, vol, T, p) {
    var sd = vol * Math.sqrt(T);
    return meanST * Math.exp(-0.5 * sd * sd + sd * normInv(p));
  }
  function valueAtTarget(fund, ST, m) { return shareValue(fund, ST, m.horizon, 0, m); }
  function levelForValue(fund, target, m) {
    var lo = 1e-6, hi = 0;
    for (var i = 0; i < fund.legs.length; i++) hi = Math.max(hi, fund.legs[i].strike);
    hi *= 20;
    if (valueAtTarget(fund, hi, m) < target) return NaN;
    if (valueAtTarget(fund, lo, m) >= target) return lo;
    for (var j = 0; j < 200; j++) {
      var mid = 0.5 * (lo + hi);
      if (valueAtTarget(fund, mid, m) < target) lo = mid; else hi = mid;
      if (hi - lo < 1e-12 * hi) break;
    }
    return hi;
  }
  function pWorthless(fund, meanST, vol, T) {
    var minCall = Infinity, maxPut = -Infinity;
    for (var i = 0; i < fund.legs.length; i++) {
      var L = fund.legs[i];
      if (L.kind === "call") minCall = Math.min(minCall, L.strike); else maxPut = Math.max(maxPut, L.strike);
    }
    var belowCall = minCall < Infinity ? 1 - probAbove(minCall, meanST, vol, T) : 1;
    var belowPut = maxPut > -Infinity ? 1 - probAbove(maxPut, meanST, vol, T) : 0;
    return Math.max(belowCall - belowPut, 0);
  }
  function outcomeRow(fund, price, S0, vol, m, mu) {
    var meanST = forwardAtTarget(S0, m, mu), T = m.horizon;
    var be1 = levelForValue(fund, price, m), be2 = levelForValue(fund, 2 * price, m);
    var mult = function (p) { return valueAtTarget(fund, quantile(meanST, vol, T, p), m) / price; };
    var meanGrowth = (mu === null || mu === undefined) ? Math.exp(m.rate) : 1 + mu;
    return {
      mu: (mu === undefined ? null : mu),
      p_worthless: pWorthless(fund, meanST, vol, T),
      p_profit: be1 === be1 ? probAbove(be1, meanST, vol, T) : 0,
      p_double: be2 === be2 ? probAbove(be2, meanST, vol, T) : 0,
      multiple_p10: mult(0.10), multiple_median: mult(0.50), multiple_p90: mult(0.90),
      median_price_cagr: Math.pow(quantile(meanST, vol, T, 0.5) / S0, 1 / T) - 1,
      median_tr_cagr: meanGrowth * Math.exp(-0.5 * vol * vol) - 1,
      breakeven: be1, double_level: be2
    };
  }
  function requiredGrowth(now, level, T) { return Math.pow(level / now, 1 / T) - 1; }
  function ratioAt(t, ratioNow, ratioTarget, T) { return ratioNow + (ratioTarget - ratioNow) * (t / T); }

  var api = {
    erfc: erfc, normCdf: normCdf, normPdf: normPdf, normInv: normInv,
    callPrice: callPrice, putPrice: putPrice, callGreeks: callGreeks, impliedVol: impliedVol,
    optionValue: optionValue, optionDelta: optionDelta, shareValue: shareValue, shareDelta: shareDelta,
    leverage: leverage, volAt: volAt, solveVol: solveVol, oneDay: oneDay, forwardAtTarget: forwardAtTarget,
    probAbove: probAbove, quantile: quantile, valueAtTarget: valueAtTarget, levelForValue: levelForValue,
    pWorthless: pWorthless, outcomeRow: outcomeRow, requiredGrowth: requiredGrowth, ratioAt: ratioAt,
    DAY: DAY, VOL_FLOOR: VOL_FLOOR
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.DTL = api;
})(this);
