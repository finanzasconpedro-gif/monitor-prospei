"""
Track récord de la señal "Comprar hoy" del Monitor PROSPEI.

Mira hacia atrás 4 años, día por día, en las 3 empresas de mayor capitalización de cada sector del S&P 500,
y registra cada vez que se dio la señal. Después mide qué pasó con el precio 3, 6 y 12 meses más tarde.

Todo es "punto en el tiempo": en cada fecha solo se usa información que existía ese día
(precios hasta esa fecha y balances ya presentados ante la SEC). Así se evita mirar el futuro.

Qué se puede reconstruir y qué no:
  - La parte técnica (puntaje AT, tendencia, Golden Cross, ruptura, corrección con rebote): exacta.
  - Fecha de balance (no comprar si presenta en 7 días): exacta (Yahoo; si falta, fecha de presentación a la SEC).
  - Fundamentos: aproximados con los balances históricos de la SEC (ganancia y ventas de 12 meses).
  - Precio objetivo de analistas, ganancia esperada y valor justo: NO existen gratis hacia atrás.
    Por eso el veredicto completo no se puede reproducir y se prueban dos variantes (ver VARIANTS).

Resultado: backtest.json (lo lee la página) y un resumen en el log de GitHub Actions.
"""
import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd
import requests

import screener as S

YEARS = 4
GAP = 20                      # ruedas sin señal para contar una señal nueva (evita contar la misma compra muchas veces)
H = {"3m": 63, "6m": 126, "12m": 252}
TARGET, STOP, TS_DAYS = 0.10, -0.08, 126
GOAL = 0.70
# La SEC exige identificarse con nombre y un email de contacto; si no, rechaza el pedido (error 403).
# Se carga como secreto del repositorio: Settings → Secrets and variables → Actions → SEC_UA
SEC_UA = os.environ.get("SEC_UA") or "PROSPEI Monitor research github.com/finanzasconpedro-gif"
BROAD_N, SEED = 100, 7        # muestra amplia: 100 acciones al azar del S&P 500 (incluye las que anduvieron mal)
BRK_RATIO = {"BRK-B": 1500}   # la SEC informa la ganancia por acción clase A

FALLBACK = {   # por si no está data.json: 3 de mayor capitalización por sector (aprox.)
    "Tecnología": ["NVDA", "AAPL", "MSFT"], "Comunicaciones": ["GOOGL", "META", "NFLX"],
    "Consumo discrecional": ["AMZN", "TSLA", "HD"], "Financiero": ["BRK-B", "JPM", "V"],
    "Salud": ["LLY", "JNJ", "ABBV"], "Industrial": ["GE", "CAT", "RTX"],
    "Consumo básico": ["WMT", "COST", "PG"], "Energía": ["XOM", "CVX", "COP"],
    "Servicios públicos": ["NEE", "SO", "CEG"], "Inmobiliario": ["PLD", "AMT", "WELL"],
    "Materiales": ["LIN", "SHW", "NEM"]}

VARIANTS = {
    "A": "Técnica del monitor: puntaje técnico ≥ 65 + señal de entrada (corrección con rebote, Golden Cross o ruptura) + sin balance en 7 días",
    "B": "A + fundamentos básicos de ese momento: ganancia positiva, ventas creciendo y P/E no más de 10% arriba de su promedio de 5 años",
    "C": "A sin Golden Cross: solo corrección con rebote o ruptura de máximos",
    "D": "C + los mismos fundamentos de B (la regla propuesta para el monitor)",
}
KEYS = ("A", "B", "C", "D")
# reglas originales del monitor (la A incluye el Golden Cross; el monitor ya lo sacó en base a esta prueba)
ENTRY_A = {"Corrección con señal de rebote", "Golden Cross", "Ruptura de máximos"}


# ------------------------------------------------------------------ universo
def pick_universes():
    """top: las 3 más grandes de cada sector. broad: 100 al azar del S&P 500, repartidas por sector."""
    top = pick_universe()
    tops = {t for v in top.values() for t in v}
    broad = {}
    try:
        rows = json.load(open("data.json", encoding="utf-8"))["rows"]
        df = pd.DataFrame([{"t": r["t"], "sector": r["sector"], "region": r.get("region")} for r in rows])
        df = df[(df.region == "EE.UU.") & ~df.t.isin(tops)]
        frac = min(1.0, BROAD_N / len(df))
        for sec, g in df.groupby("sector"):
            broad[sec] = sorted(g.sample(min(len(g), max(1, round(len(g) * frac))), random_state=SEED).t)
    except Exception as e:
        print("Sin muestra amplia (falta data.json):", e)
    return {"top": top, "broad": broad}


def pick_universe():
    try:
        rows = json.load(open("data.json", encoding="utf-8"))["rows"]
        df = pd.DataFrame([{"t": r["t"], "sector": r["sector"], "mcap": r.get("mcap") or 0, "region": r.get("region")}
                           for r in rows])
        df = df[(df.region == "EE.UU.") & (df.mcap > 0)]
        out = {}
        for sec, g in df.sort_values("mcap", ascending=False).groupby("sector"):
            out[sec] = list(g.t.head(3))
        if sum(len(v) for v in out.values()) >= 25:
            return out
    except Exception as e:
        print("No pude leer data.json, uso lista fija:", e)
    return FALLBACK


# ------------------------------------------------------------------ SEC (balances punto en el tiempo)
def sec_get(url, timeout=60):
    for k in range(3):
        r = requests.get(url, headers={"User-Agent": SEC_UA, "Accept-Encoding": "gzip, deflate"}, timeout=timeout)
        if r.status_code == 200:
            return r
        if r.status_code == 403:
            raise RuntimeError("la SEC rechazó el pedido (403): falta cargar el secreto SEC_UA con nombre y email de contacto")
        time.sleep(2 + 3 * k)
    r.raise_for_status()


def sec_ciks():
    r = sec_get("https://www.sec.gov/files/company_tickers.json", 30)
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in r.json().values()}


def sec_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    r = sec_get(url)
    time.sleep(0.15)          # la SEC pide no más de 10 consultas por segundo
    return r.json()


def quarters(facts, tags, unit, adj=None):
    """Serie trimestral (fin, fecha de presentación, valor). El 4º trimestre se deriva del anual menos 3 trimestres."""
    best = {}
    for pri, tag in enumerate(tags):
        node = facts.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get(unit)
        if not node:
            continue
        ent = {}
        for e in node:
            if "start" not in e or e.get("form", "")[:4] not in ("10-Q", "10-K"):
                continue
            s, en, f = pd.Timestamp(e["start"]), pd.Timestamp(e["end"]), pd.Timestamp(e["filed"])
            k = (s, en)
            if k not in ent or f < ent[k][1]:          # la primera vez que se publicó (sin revisiones posteriores)
                v_ = float(e["val"])
                ent[k] = (adj(v_, f) if adj else v_, f)
        qs = {k: v for k, v in ent.items() if 80 <= (k[1] - k[0]).days <= 100}
        ys = {k: v for k, v in ent.items() if 350 <= (k[1] - k[0]).days <= 380}
        q = {en: (val, f) for (s, en), (val, f) in qs.items()}
        for (s, en), (val, f) in ys.items():
            ins = [v for (qs_, qe), v in qs.items() if qs_ >= s - pd.Timedelta(days=7) and qe <= en - pd.Timedelta(days=60)]
            if len(ins) == 3 and en not in q:
                q[en] = (val - sum(x[0] for x in ins), f)
        for en, v in q.items():
            if en not in best or best[en][2] > pri:
                best[en] = (v[0], v[1], pri)
    return sorted((en, f, v) for en, (v, f, _) in best.items())


def ttm_daily(qs, index):
    """Valor de 12 meses conocido en cada fecha del índice (solo trimestres ya presentados)."""
    if len(qs) < 4:
        return pd.Series(np.nan, index=index)
    out = []
    for d in index:
        known = [(en, v) for en, f, v in qs if f <= d]
        known = known[-4:]
        if len(known) == 4 and 250 <= (known[-1][0] - known[0][0]).days <= 300:
            out.append(sum(v for _, v in known))
        else:
            out.append(np.nan)
    return pd.Series(out, index=index)


def split_adj(splits):
    """La SEC publica la ganancia por acción con la cantidad de acciones de ese momento. Los precios de Yahoo
    vienen ajustados por todos los splits posteriores, así que la ganancia se divide por los splits
    ocurridos después de cada presentación (ej. NVDA 4:1 en 2021 y 10:1 en 2024)."""
    if not splits:
        return None
    def f(val, filed):
        k = 1.0
        for d, r in splits:
            if d > filed:
                k *= r
        return val / k
    return f


def fundamentals(sym, cik, close, splits=None):
    """Devuelve DataFrame diario con eps_ttm, pe, pe_med5 (mediana de 5 años previa), rev_g y fechas de presentación."""
    idx = close.index
    try:
        f = sec_facts(cik)
    except Exception as e:
        print("  SEC sin datos para", sym, e)
        return None, []
    eps_q = quarters(f, ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted", "EarningsPerShareBasic"], "USD/shares",
                     split_adj(splits))
    rev_q = quarters(f, ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                         "RevenuesNetOfInterestExpense", "RevenueFromContractWithCustomerIncludingAssessedTax"], "USD")
    # muestreo semanal para acelerar y luego se rellena
    wk = idx[::5]
    eps = ttm_daily(eps_q, wk).reindex(idx, method="ffill") / BRK_RATIO.get(sym, 1)
    rev = ttm_daily(rev_q, wk).reindex(idx, method="ffill")
    pe = (close / eps).where(eps > 0)
    pe_med5 = pe.rolling(1260, min_periods=500).median().shift(1)
    rev_g = rev / rev.shift(252) - 1
    filed = sorted({f for _, f, _ in eps_q})
    return pd.DataFrame({"eps": eps, "pe": pe, "pe_med5": pe_med5, "rev_g": rev_g}), filed


def sec_earn_dates(cik):
    """Fechas de publicación de resultados: formularios 8-K con el ítem 2.02 (Results of Operations)."""
    try:
        r = sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", 30).json()
        time.sleep(0.12)
        rec = r.get("filings", {}).get("recent", {})
        return sorted({pd.Timestamp(d) for f, d, it in zip(rec.get("form", []), rec.get("filingDate", []), rec.get("items", []))
                       if f == "8-K" and "2.02" in str(it)})
    except Exception:
        return []


def earnings_dates(sym, filed, cik=None):
    if cik:
        d = sec_earn_dates(cik)
        if len(d) >= 8:
            return d
    try:
        import yfinance as yf
        ed = yf.Ticker(sym).get_earnings_dates(limit=40)
        if ed is not None and len(ed):
            return sorted({pd.Timestamp(d).tz_localize(None).normalize() for d in ed.index})
    except Exception:
        pass
    return [pd.Timestamp(d) for d in filed]


# ------------------------------------------------------------------ puntaje técnico (idéntico al del screener)
def score_tech_pit(df, mom_pct):
    t = pd.Series(0.0, index=df.index)
    t += df.trend_n.map({2: 10, 1: 6, 0: 4, -1: 2, -2: 0}).fillna(4)
    t += np.where(df.sma50 > df.sma200, 10, 0)
    t += np.where(df.sma200_up, 10, 0)
    t += np.where(df.et_ctx, 1.5 * df.et_n, 0)
    t += df.cross.map({"Golden Cross reciente": 3, "Golden Cross en formación": 2,
                       "Death Cross reciente": -3, "Death Cross en formación": -2}).fillna(0)
    t += 25 * mom_pct.fillna(0.3)
    r = df.rsi
    t += np.select([(r >= 40) & (r <= 60), (r > 60) & (r <= 70), (r >= 30) & (r < 40), r < 30, r > 70],
                   [15, 10, 11, 7, 3], 5)
    t += np.where(df.macd_h > 0, 5, 0) + np.where(df.macd_up, 2, 0) + np.where(df.macd_cross, 3, 0)
    up = df.trend_n >= 1
    dd = df.dd
    t += np.select([up & (dd <= -0.05) & (dd >= -0.20), up & (dd > -0.05), up & (dd < -0.20)], [20, 15, 10], 4)
    return t.clip(0, 100)


# ------------------------------------------------------------------ estadísticas
def stats(rows, key):
    v = [r for r in rows if r.get(key) is not None]
    if not v:
        return {"n": 0}
    x = np.array([r[key] for r in v])
    out = {"n": len(v), "win": float((x > 0).mean()), "avg": float(x.mean()), "med": float(np.median(x)),
           "p10": float(np.percentile(x, 10))}
    b = [r[key] - r["spy_" + key] for r in v if r.get("spy_" + key) is not None]
    if b:
        out["beat"] = float((np.array(b) > 0).mean())
        out["alpha"] = float(np.mean(b))
    return out


def ts_stats(rows):
    v = [r["ts"] for r in rows if r.get("ts")]
    if not v:
        return {"n": 0}
    return {"n": len(v), "target": v.count("objetivo") / len(v), "stop": v.count("stop") / len(v),
            "open": v.count("abierta") / len(v)}


def summarize(rows):
    return {**{h: stats(rows, h) for h in H}, "ts": ts_stats(rows)}


TECH_COLS = ["p", "sma50", "sma200", "sma200_up", "trend_n", "dd", "dd3", "rsi", "macd_h", "macd_up", "macd_cross",
             "r1m", "r6m", "vol_ratio", "recovery", "rebound", "et_hl", "et_div", "et_acc", "et_brk", "et_mom"]
CROSS = [None, "Golden Cross reciente", "Golden Cross en formación", "Death Cross reciente", "Death Cross en formación"]


def run_ticker(args):
    """Corre la prueba completa para una acción (en un proceso aparte). Devuelve episodios por variante,
    muestra de "cualquier día" y los datos para el gráfico."""
    t, c, v, spy, mom, days, dates, ed, fd = args
    rows, keep = [], []
    for i in days:
        lo = max(0, i - 1100)
        cc, vv = c[lo:i + 1], v[lo:i + 1]
        ok = np.isfinite(cc)
        if ok.sum() < 260 or not np.isfinite(c[i]):
            continue
        r = S.technical(cc[ok], vv[ok])
        if not r:
            continue
        row = {k: r.get(k) for k in TECH_COLS}
        row["cross"] = r.get("cross")
        rows.append(row)
        keep.append(i)
    if not rows:
        return t, None
    df = pd.DataFrame(rows, index=keep)
    for k in TECH_COLS:
        df[k] = pd.to_numeric(df[k], errors="coerce") if k not in ("sma200_up", "macd_up", "macd_cross", "recovery", "rebound",
                                                                  "et_hl", "et_div", "et_acc", "et_brk", "et_mom") else df[k].fillna(False).astype(bool)
    ki = np.array(keep)
    spy_r1m = spy[ki] / spy[ki - 21] - 1
    df["et_rs"] = (df.r1m.values - spy_r1m) > 0
    df["et_n"] = df[["et_hl", "et_div", "et_acc", "et_brk", "et_mom", "et_rs"]].astype(bool).sum(axis=1)
    df["et_ctx"] = (df.trend_n <= 0) & (df.dd <= -0.05)
    df["at"] = score_tech_pit(df, pd.Series(mom[ki], index=df.index))
    df["af"] = 0
    df["setup"] = [S.setup(r) for r in df.itertuples()]
    n = len(c)
    eps = {"A": [], "B": [], "C": [], "D": []}
    base = []
    last = {}
    for i, r in zip(ki, df.itertuples()):
        if i + 1 >= n:
            continue
        d = np.datetime64(dates[i])
        earn = bool(len(ed) and ((ed > d) & (ed <= d + np.timedelta64(7, "D"))).any())
        a = bool(r.at >= S.V_AT and r.setup in ENTRY_A and not earn)
        b = False
        if a and fd is not None:
            e, pe, med, g = fd[0][i], fd[1][i], fd[2][i], fd[3][i]
            b = bool(np.isfinite(e) and e > 0 and np.isfinite(g) and g > 0 and np.isfinite(pe) and np.isfinite(med) and pe <= med * 1.10)
        cc_ = a and r.setup != "Golden Cross"
        sig = {"A": a, "B": b, "C": cc_, "D": b and cc_}
        new = {k: sig[k] and i - last.get(k, -999) > GAP for k in sig}
        if i % 5 == 0 or any(new.values()):
            p0 = c[i + 1]
            res = {"date": str(dates[i])[:10], "setup": r.setup, "at": round(float(r.at)), "p": round(float(p0), 2)}
            for h, hn in H.items():
                if i + 1 + hn < n and np.isfinite(c[i + 1 + hn]):
                    res[h] = float(c[i + 1 + hn] / p0 - 1)
                    res["spy_" + h] = float(spy[i + 1 + hn] / spy[i + 1] - 1)
            path = c[i + 2:i + 2 + TS_DAYS] / p0 - 1
            ht = int(np.argmax(path >= TARGET)) if (path >= TARGET).any() else None
            hs = int(np.argmax(path <= STOP)) if (path <= STOP).any() else None
            if ht is not None and (hs is None or ht < hs):
                res["ts"] = "objetivo"
            elif hs is not None:
                res["ts"] = "stop"
            elif len(path) >= TS_DAYS:
                res["ts"] = "abierta"
            res["year"] = res["date"][:4]
            if i % 5 == 0:
                base.append(res)
            for k in sig:
                if new[k]:
                    eps[k].append(dict(res, b=b, c=cc_, d=b and cc_))
        for k in sig:
            if sig[k]:
                last[k] = i
    return t, {"eps": eps, "base": base}


def by(rows, field):
    g = {}
    for r in rows:
        g.setdefault(r[field], []).append(r)
    return {k: {"n": len(v), **{h: stats(v, h) for h in ("6m", "12m")}} for k, v in sorted(g.items())}


UNI_LBL = {"top": "Las 3 más grandes de cada sector", "broad": f"Muestra amplia: {BROAD_N} acciones al azar del S&P 500"}


def all_tickers():
    """Todas las acciones del monitor (S&P 500 + ADRs) con nombre y sector."""
    try:
        rows = json.load(open("data.json", encoding="utf-8"))["rows"]
        return {r["t"]: (r.get("name") or r["t"], r.get("sector") or "", r.get("region") or "") for r in rows}
    except Exception:
        return {s: (n, S.SECTOR_ES.get(g, g), "EE.UU.") for s, n, g in S.sp500()}


def main():
    import yfinance as yf
    from concurrent.futures import ProcessPoolExecutor
    unis = pick_universes()
    meta = all_tickers()
    for u in unis.values():
        for sname, v in u.items():
            for t in v:
                meta.setdefault(t, (t, sname, "EE.UU."))
    tick = sorted(meta)
    member = {k: {t for v in u.values() for t in v} for k, u in unis.items()}
    print("Acciones a probar:", len(tick), "· universos:", {k: len(v) for k, v in member.items()})

    frames = []
    for i in range(0, len(tick), 100):
        frames.append(yf.download(tick[i:i + 100] + ["SPY"], period="10y", interval="1d", auto_adjust=True,
                                  actions=True, group_by="ticker", threads=True, progress=False))
        time.sleep(1)
    lv = lambda d: set(d.columns.get_level_values(0))
    close = pd.concat([pd.DataFrame({s: d[s]["Close"] for s in tick + ["SPY"] if s in lv(d)}) for d in frames], axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    vol = pd.concat([pd.DataFrame({s: d[s]["Volume"] for s in tick if s in lv(d)}) for d in frames], axis=1)
    vol = vol.loc[:, ~vol.columns.duplicated()].reindex(close.index)
    splits = {}
    for d in frames:
        for s_ in tick:
            if s_ in lv(d) and "Stock Splits" in d[s_].columns:
                sp_ = d[s_]["Stock Splits"].fillna(0)
                ev = [(pd.Timestamp(k).tz_localize(None) if pd.Timestamp(k).tzinfo else pd.Timestamp(k), float(v)) for k, v in sp_[sp_ > 0].items() if float(v) != 1.0]
                if ev:
                    splits[s_] = ev
    print("Acciones con splits en 10 años:", len(splits))
    close = close.ffill(limit=3)
    tick = [t for t in tick if t in close.columns and t in vol.columns and close[t].notna().sum() > 300]
    sp = [s for s, _, _ in S.sp500()]
    mom_cols = [c for c in close.columns if c in set(sp)]
    mom_pct_all = ((close[mom_cols] / close[mom_cols].shift(126)) / (close[mom_cols] / close[mom_cols].shift(21)) - 1).rank(axis=1, pct=True)

    idx = close.index
    end = idx[-1]
    start = end - pd.DateOffset(years=YEARS)
    days = [i for i, d in enumerate(idx) if d >= start]
    print(f"Período: {idx[days[0]].date()} a {end.date()} · {len(days)} ruedas · {len(tick)} acciones con precios")

    sec_msg = None
    try:
        ciks = sec_ciks()
    except Exception as e:
        sec_msg = str(e)
        print("SEC no disponible:", e)
        ciks = {}
    fund, edates = {}, {}
    for n_t, t in enumerate(tick):
        cik = ciks.get(t.upper()) or ciks.get(t.upper().replace("-", "."))
        fd, filed = fundamentals(t, cik, close[t], splits.get(t)) if cik else (None, [])
        fund[t] = None if fd is None else np.vstack([fd.eps.values, fd.pe.values, fd.pe_med5.values, fd.rev_g.values])
        edates[t] = np.array(earnings_dates(t, filed, cik), dtype="datetime64[ns]")
        if n_t % 50 == 0:
            print(f"  balances {n_t + 1}/{len(tick)}")
    sec_ok = sum(1 for t in tick if fund[t] is not None and np.isfinite(fund[t][1]).any())
    print(f"Balances de la SEC: {sec_ok} de {len(tick)} empresas" + (f" · {sec_msg}" if sec_msg else ""))

    spy = close["SPY"].values
    dates = idx.values
    jobs = []
    for t in tick:
        mom = mom_pct_all[t].values if t in mom_pct_all.columns else np.full(len(idx), np.nan)
        jobs.append((t, close[t].values, vol[t].values, spy, mom, days, dates, edates[t], fund[t]))
    res = {}
    with ProcessPoolExecutor() as ex:
        for k, (t, r) in enumerate(ex.map(run_ticker, jobs, chunksize=4)):
            res[t] = r
            if k % 50 == 0:
                print(f"  señales {k + 1}/{len(tick)}")

    def pack(u):
        bk = {k: [] for k in KEYS + ("base",)}
        for t in member[u]:
            r = res.get(t)
            if not r:
                continue
            extra = {"t": t, "sector": meta[t][1]}
            for k in KEYS:
                bk[k] += [dict(e, **extra) for e in r["eps"][k]]
            bk["base"] += [dict(e, **extra) for e in r["base"]]
        bset = sorted({f"{r['t']}|{r['date']}" for r in bk["B"]})
        return {
            "label": UNI_LBL[u], "n": len(member[u]), "tickers": unis[u],
            "variants": {k: {"label": VARIANTS[k], **summarize(bk[k])} for k in KEYS},
            "base": {"label": "Comprar cualquier día cualquiera de estas acciones (referencia)", **summarize(bk["base"])},
            "by_sector": {k: by(bk[k], "sector") for k in KEYS},
            "by_setup": {k: by(bk[k], "setup") for k in KEYS},
            "by_year": {k: by(bk[k], "year") for k in KEYS + ("base",)},
            "episodes": sorted([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("b", "c", "d")}
                                for r in bk["A"]], key=lambda r: r["date"], reverse=True),
            "b_set": bset,
        }

    out = {
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "period": {"from": str(idx[days[0]].date()), "to": str(end.date())},
        "goal": GOAL, "gap": GAP, "target": TARGET, "stop": STOP, "ts_days": TS_DAYS,
        "sec": {"ok": sec_ok, "total": len(tick), "error": sec_msg},
        "universes": {u: pack(u) for u in unis if member[u]},
    }
    with open("backtest.json", "w", encoding="utf-8") as f:
        json.dump(S.no_nan(out), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    # ---- por acción: para buscar una empresa puntual en la página de control
    wk = list(range(days[0], len(idx), 5))
    if wk[-1] != len(idx) - 1:
        wk.append(len(idx) - 1)
    per = {}
    r4 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
    for t in tick:
        r = res.get(t)
        if not r:
            continue
        sig = [{k: (r4(v) if isinstance(v, float) else v) for k, v in e.items() if k != "year"} for e in r["eps"]["A"]]
        per[t] = {"n": meta[t][0], "s": meta[t][1], "f": fund[t] is not None,
                  "w": [r4(x) for x in close[t].values[wk]],
                  "sig": sig,
                  "base": {h: stats(r["base"], h) for h in ("6m", "12m")}}
    with open("backtest_tickers.json", "w", encoding="utf-8") as f:
        json.dump(S.no_nan({"dates": [str(idx[i].date()) for i in wk], "spy": [r4(x) for x in spy[wk]], "tickers": per}),
                  f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(f"Detalle por acción: {len(per)} empresas")

    print("\n================ TRACK RÉCORD ================")
    print(f"Balances de la SEC: {sec_ok} de {len(tick)}" + (f" · {sec_msg}" if sec_msg else ""))
    for u, U in out["universes"].items():
        print(f"\n##### {U['label']} ({U['n']} acciones)")
        for k in KEYS:
            v = U["variants"][k]
            s_ = v["6m"]
            if s_.get("n"):
                print(f"  {k} 6m: {s_['n']} señales · positivas {s_['win']*100:.0f}% · le ganó al S&P {s_.get('beat',0)*100:.0f}%"
                      f" · promedio {s_['avg']*100:+.1f}% · peor 10% {s_['p10']*100:+.1f}%")
        b = U["base"]["6m"]
        if b.get("n"):
            print(f"  Referencia 6m: positivas {b['win']*100:.0f}% · promedio {b['avg']*100:+.1f}% · peor 10% {b['p10']*100:+.1f}%")


if __name__ == "__main__":
    main()
