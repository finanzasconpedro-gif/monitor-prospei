"""
Screener PROSPEI - analisis tecnico (AT) y fundamental (AF)
Universo: S&P 500 + ADRs argentinos + ADRs brasilenos.
Genera data.json, que lee la pestana "Oportunidades" del monitor.
Lo corre GitHub Actions todos los dias habiles despues del cierre de EE.UU.
"""
import json, math, time, datetime as dt
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

ADR_AR = ["GGAL", "YPF", "PAM", "BMA", "BBAR", "SUPV", "CEPU", "TGS", "EDN", "CRESY",
          "IRS", "LOMA", "TEO", "VIST", "GLOB", "MELI", "CAAP", "BIOX"]
ADR_BR = ["PBR", "VALE", "ITUB", "BBD", "ABEV", "NU", "SBS", "GGB", "SID",
          "SUZ", "UGP", "CIG", "BSBR", "XP", "STNE", "PAGS", "VIV", "TIMB", "BAK"]
TOP_N = 30
SECTOR_ETF = {"Tecnología": "XLK", "Financiero": "XLF", "Salud": "XLV", "Consumo discrecional": "XLY",
              "Consumo básico": "XLP", "Industrial": "XLI", "Energía": "XLE", "Utilities": "XLU",
              "Materiales": "XLB", "Real estate": "XLRE", "Comunicaciones": "XLC"}
BENCH = "SPY"
UA = {"User-Agent": "Mozilla/5.0 (PROSPEI screener)"}

SECTOR_ES = {
    "Technology": "Tecnología", "Information Technology": "Tecnología",
    "Financial Services": "Financiero", "Financials": "Financiero",
    "Healthcare": "Salud", "Health Care": "Salud",
    "Consumer Cyclical": "Consumo discrecional", "Consumer Discretionary": "Consumo discrecional",
    "Consumer Defensive": "Consumo básico", "Consumer Staples": "Consumo básico",
    "Industrials": "Industrial", "Energy": "Energía", "Utilities": "Utilities",
    "Basic Materials": "Materiales", "Materials": "Materiales",
    "Real Estate": "Real estate", "Communication Services": "Comunicaciones",
}


# ---------------------------------------------------------------- universo
def sp500():
    try:
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers=UA, timeout=30).text
        df = pd.read_html(StringIO(html))[0]
        return [(str(s).replace(".", "-"), str(n), str(g)) for s, n, g in
                zip(df["Symbol"], df["Security"], df["GICS Sector"])]
    except Exception as e:
        print("Wikipedia fallo, uso CSV alternativo:", e)
        df = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
        return [(str(s).replace(".", "-"), str(n), str(g)) for s, n, g in
                zip(df["Symbol"], df["Security"], df["GICS Sector"])]


# ---------------------------------------------------------------- tecnico
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def rsi(c, n=14):
    d = np.diff(c)
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    rs = up[-1] / dn[-1] if dn[-1] > 0 else np.inf
    return float(100 - 100 / (1 + rs))


def ret(c, n):
    return float(c[-1] / c[-1 - n] - 1) if len(c) > n and c[-1 - n] > 0 else None


def technical(c, v):
    """c: cierres ajustados (np.array), v: volumenes. Devuelve dict o None."""
    c = c[np.isfinite(c)]
    if len(c) < 210:
        return None
    p = float(c[-1])
    s20, s50, s200 = (float(c[-n:].mean()) for n in (20, 50, 200))
    s200_prev = float(c[-220:-20].mean())
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    hist = macd - sig
    w = c[-252:]
    hi, lo = float(w.max()), float(w.min())
    vv = v[np.isfinite(v)] if v is not None else np.array([])
    vol_ratio = float(vv[-20:].mean() / vv[-60:].mean()) if len(vv) >= 60 and vv[-60:].mean() > 0 else None
    dr = np.diff(np.log(c[-64:]))
    return {
        "p": p, "sma20": s20, "sma50": s50, "sma200": s200,
        "sma200_up": s200 > s200_prev,
        "rsi": rsi(c[-300:]),
        "macd_h": float(hist[-1]),
        "macd_cross": bool(hist[-1] > 0 and (hist[-6:-1] <= 0).any()),
        "macd_up": bool(hist[-1] > hist[-2]),
        "r1d": ret(c, 1), "r1m": ret(c, 21), "r3m": ret(c, 63), "r6m": ret(c, 126),
        "r12m": ret(c, 251) if len(c) > 251 else None,
        "hi52": hi, "lo52": lo, "dd": p / hi - 1, "from_lo": p / lo - 1,
        "vol_ratio": vol_ratio,
        "volat": float(dr.std() * math.sqrt(252)),
    }


def pct_rank(s, higher_better=True):
    """Percentil 0..1 dentro de la serie (NaN queda NaN)."""
    r = s.rank(pct=True, ascending=higher_better)
    return r


def score_tech(df):
    t = pd.Series(0.0, index=df.index)
    # Tendencia (30)
    t += np.where(df.p > df.sma200, 10, 0)
    t += np.where(df.sma50 > df.sma200, 10, 0)
    t += np.where(df.sma200_up, 10, 0)
    # Momentum relativo 6m sin el ultimo mes (25)
    mom = (1 + df.r6m) / (1 + df.r1m) - 1
    t += 25 * pct_rank(mom).fillna(0.3)
    # Timing: RSI (15) + MACD (10)
    r = df.rsi
    t += np.select([(r >= 40) & (r <= 60), (r > 60) & (r <= 70), (r >= 30) & (r < 40), r < 30, r > 70],
                   [15, 10, 11, 7, 3], 5)
    t += np.where(df.macd_h > 0, 5, 0) + np.where(df.macd_up, 2, 0) + np.where(df.macd_cross, 3, 0)
    # Posicion respecto del maximo (20)
    up = df.p > df.sma200
    dd = df.dd
    t += np.select([up & (dd <= -0.05) & (dd >= -0.20), up & (dd > -0.05), up & (dd < -0.20)],
                   [20, 15, 10], 4)
    return t.clip(0, 100)


def score_fund(df):
    """Cada pilar 0..1 por percentil; la valuacion se compara dentro del sector."""
    fpe = df.fpe.where(df.fpe > 0)
    val = fpe.groupby(df.sector).rank(pct=True, ascending=False)
    peg = df.peg.where((df.peg > 0) & (df.peg < 10))
    val = (0.7 * val + 0.3 * pct_rank(peg, False)).fillna(val).fillna(0.15)
    val = val.where(df.fpe.notna() | df.pe.notna(), 0.15)
    gro = pd.concat([pct_rank(df.rev_g), pct_rank(df.eps_g), pct_rank(df.fwd_eps_g)], axis=1).mean(axis=1).fillna(0.3)
    qua = pd.concat([pct_rank(df.roe), pct_rank(df.margin), pct_rank(df.op_margin)], axis=1).mean(axis=1).fillna(0.3)
    sol = pct_rank(df.de, False).fillna(0.5)
    has_an = df.n_an >= 3
    ana = (0.6 * pct_rank(df.upside).fillna(0.3) + 0.4 * pct_rank(df.rec, False).fillna(0.3)).where(has_an, 0.3)
    df["f_val"], df["f_gro"], df["f_qua"], df["f_sol"], df["f_ana"] = val, gro, qua, sol, ana
    return (25 * val + 25 * gro + 25 * qua + 10 * sol + 15 * ana).clip(0, 100)


# ---------------------------------------------------------------- textos
def fx(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def f_pct(x, d=0):
    return f"{x*100:+.{d}f}%".replace(".", ",")


def setup(r):
    up = r.p > r.sma200
    if up and r.dd > -0.04 and r.vol_ratio and r.vol_ratio > 1.15:
        return "Ruptura de máximos"
    if up and -0.20 <= r.dd <= -0.05 and r.rsi < 55:
        return "Corrección en tendencia alcista"
    if up and r.sma50 > r.sma200:
        return "Tendencia alcista"
    if not up and r.rsi < 35 and r.af >= 60:
        return "Sobrevendida con buenos fundamentos"
    if r.macd_cross:
        return "Giro técnico"
    return "Valor"


def reasons(r):
    pro, con, grow = [], [], []
    # Tecnico
    if r.p > r.sma200 and r.sma50 > r.sma200:
        pro.append("Tendencia alcista de largo plazo: cotiza arriba de su media de 200 ruedas y la de 50 está por encima de la de 200.")
    elif r.p > r.sma200:
        pro.append("Cotiza por encima de su media de 200 ruedas.")
    else:
        con.append(f"Está {f_pct(r.p/r.sma200-1)} por debajo de su media de 200 ruedas: la tendencia de largo plazo todavía es bajista.")
    if r.p > r.sma200 and -0.20 <= r.dd <= -0.05:
        pro.append(f"Corrigió {f_pct(r.dd)} desde su máximo de 52 semanas dentro de una tendencia alcista: posible punto de entrada.")
    if r.dd > -0.03:
        pro.append("Opera cerca de máximos de 52 semanas.")
    if 40 <= r.rsi <= 60:
        pro.append(f"RSI en {fx(r.rsi,0)}: zona neutral, sin sobrecompra.")
    elif r.rsi > 70:
        con.append(f"RSI en {fx(r.rsi,0)}: sobrecomprada en el corto plazo, conviene esperar una pausa.")
    elif r.rsi < 30:
        pro.append(f"RSI en {fx(r.rsi,0)}: sobrevendida, posible rebote.")
    if r.macd_cross:
        pro.append("El MACD cruzó al alza en los últimos días.")
    elif r.macd_h < 0:
        con.append("El MACD está debajo de su señal: el impulso de corto plazo es débil.")
    if r.vol_ratio and r.vol_ratio > 1.3:
        pro.append(f"El volumen de las últimas 4 semanas es {fx(r.vol_ratio,1)} veces su promedio: hay interés comprador o vendedor fuerte.")
    if r.mom_pct >= 0.8:
        pro.append("Su momentum de 6 meses está entre el 20% más fuerte del universo.")
    if r.volat and r.volat > 0.55:
        con.append(f"Volatilidad alta ({fx(r.volat*100,0)}% anual): movimientos bruscos en ambos sentidos.")
    # Fundamental
    if pd.notna(r.fpe) and r.fpe > 0:
        if pd.notna(r.sec_fpe) and r.fpe < r.sec_fpe * 0.85:
            pro.append(f"P/E esperado de {fx(r.fpe,1)}x contra {fx(r.sec_fpe,1)}x de la mediana de su sector: cotiza con descuento.")
        elif pd.notna(r.sec_fpe) and r.fpe > r.sec_fpe * 1.4:
            con.append(f"P/E esperado de {fx(r.fpe,1)}x, caro frente a {fx(r.sec_fpe,1)}x de su sector.")
    elif pd.notna(r.pe) and r.pe < 0 or (pd.notna(r.eps) and r.eps < 0):
        con.append("Sin ganancias en los últimos 12 meses.")
    if pd.notna(r.roe) and r.roe > 0.20:
        pro.append(f"Alta rentabilidad sobre el patrimonio (ROE {fx(r.roe*100,0)}%).")
    if pd.notna(r.margin) and r.margin > 0.20:
        pro.append(f"Margen neto de {fx(r.margin*100,0)}%.")
    elif pd.notna(r.margin) and r.margin < 0:
        con.append("Margen neto negativo.")
    if pd.notna(r.de) and r.de > 200:
        con.append(f"Endeudamiento elevado (deuda/patrimonio {fx(r.de/100,1)}x).")
    # Crecimiento
    if pd.notna(r.rev_g):
        (grow if r.rev_g >= 0 else con).append(f"Ventas: {f_pct(r.rev_g,1)} interanual en el último trimestre.")
    if pd.notna(r.eps_g):
        (grow if r.eps_g >= 0 else con).append(f"Ganancias: {f_pct(r.eps_g,1)} interanual en el último trimestre.")
    if pd.notna(r.fwd_eps_g):
        grow.append(f"Los analistas esperan que la ganancia por acción pase de {fx(r.eps,2)} a {fx(r.fwd_eps,2)} US$ ({f_pct(r.fwd_eps_g)}).")
    if r.n_an >= 3 and pd.notna(r.upside):
        grow.append(f"Precio objetivo promedio de {int(r.n_an)} analistas: US$ {fx(r.target,2)} ({f_pct(r.upside)} desde el precio actual).")
        if r.upside < 0:
            con.append("El precio ya superó el objetivo promedio de los analistas.")
    if r.n_an >= 3 and pd.notna(r.rec):
        txt = "compra fuerte" if r.rec <= 1.5 else "compra" if r.rec <= 2.5 else "mantener" if r.rec <= 3.5 else "venta"
        grow.append(f"Consenso de analistas: {txt} ({fx(r.rec,1)} en escala de 1 = compra fuerte a 5 = venta).")
    return pro, con, grow


# ---------------------------------------------------------------- datos
def fetch_info(sym):
    import yfinance as yf
    for i in range(3):
        try:
            return sym, yf.Ticker(sym).info or {}
        except Exception as e:
            time.sleep(2 + 3 * i)
    return sym, {}


def num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def build(universe, prices, infos):
    rows = []
    for sym, name, gics, region in universe:
        if sym not in prices:
            continue
        c, v = prices[sym]
        t = technical(c, v)
        if not t:
            continue
        inf = infos.get(sym, {})
        sec = SECTOR_ES.get(inf.get("sector") or gics, SECTOR_ES.get(gics, gics or "Otros"))
        eps, fwd_eps = num(inf.get("trailingEps")), num(inf.get("forwardEps"))
        target = num(inf.get("targetMeanPrice"))
        rows.append(dict(t, t=sym, name=inf.get("shortName") or name, sector=sec, region=region,
                         mcap=num(inf.get("marketCap")),
                         pe=num(inf.get("trailingPE")), fpe=num(inf.get("forwardPE")),
                         peg=num(inf.get("trailingPegRatio") or inf.get("pegRatio")),
                         pb=num(inf.get("priceToBook")),
                         rev_g=num(inf.get("revenueGrowth")), eps_g=num(inf.get("earningsGrowth")),
                         eps=eps, fwd_eps=fwd_eps,
                         fwd_eps_g=(fwd_eps / eps - 1) if eps and fwd_eps and eps > 0 else None,
                         roe=num(inf.get("returnOnEquity")), margin=num(inf.get("profitMargins")),
                         op_margin=num(inf.get("operatingMargins")), de=num(inf.get("debtToEquity")),
                         div=num(inf.get("dividendYield")),
                         target=target, upside=(target / t["p"] - 1) if target else None,
                         rec=num(inf.get("recommendationMean")),
                         n_an=num(inf.get("numberOfAnalystOpinions")) or 0,
                         closes=c[-450:]))
    df = pd.DataFrame(rows)
    num_cols = ["pe", "fpe", "peg", "pb", "rev_g", "eps_g", "eps", "fwd_eps", "fwd_eps_g", "roe", "margin",
                "op_margin", "de", "div", "target", "upside", "rec", "mcap", "vol_ratio", "r1m", "r6m"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["at"] = score_tech(df)
    df["af"] = score_fund(df)
    df["score"] = 0.5 * df["at"] + 0.5 * df["af"]
    df["mom_pct"] = pct_rank((1 + df.r6m) / (1 + df.r1m) - 1).fillna(0)
    df["sec_fpe"] = df.fpe.where(df.fpe > 0).groupby(df.sector).transform("median")
    df["setup"] = [setup(r) for r in df.itertuples()]
    # Elegibles como oportunidad: tendencia sana, o sobreventa con buenos fundamentos
    elig = (df.p > df.sma200) | ((df.rsi < 35) & (df.af >= 60))
    df["elig"] = elig
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    top = df[df.elig].head(TOP_N)
    df["rank"] = None
    for i, idx in enumerate(top.index):
        df.at[idx, "rank"] = i + 1
    return df


# ---------------------------------------------------------------- rotacion (RRG)
RRG_WEEKS = 26


def weekly(ser):
    return ser.resample("W-FRI").last().dropna()


def rrg(ser, bench_w, weeks=RRG_WEEKS):
    """RS-Ratio (fuerza relativa vs su promedio de 10 semanas) y RS-Momentum (cambio de 4 semanas del RS-Ratio).
    Ambos centrados en 100. Devuelve (puntos [[x,y],...], fechas) o (None, None)."""
    if ser is None or bench_w is None:
        return None, None
    df = pd.concat([weekly(ser), bench_w], axis=1, join="inner").dropna()
    if len(df) < weeks + 16:
        return None, None
    rs = (df.iloc[:, 0] / df.iloc[:, 1]).ewm(span=3, adjust=False).mean()
    rsr = 100 * rs / rs.rolling(10).mean()
    rsm = (100 * rsr / rsr.shift(4)).ewm(span=3, adjust=False).mean()
    out = pd.concat([rsr, rsm], axis=1).dropna().tail(weeks + 1)
    return ([[round(float(x), 2), round(float(y), 2)] for x, y in out.values],
            [d.strftime("%Y-%m-%d") for d in out.index])


def quadrant(x, y):
    if x >= 100:
        return "Liderando" if y >= 100 else "Perdiendo impulso"
    return "Mejorando" if y >= 100 else "Rezagado"


def heading(pts):
    """Cuadrante actual y hacia dónde va según las últimas 3 semanas."""
    x, y = pts[-1]
    x0, y0 = pts[-4] if len(pts) >= 4 else pts[0]
    dx, dy = x - x0, y - y0
    q = quadrant(x, y)
    if q == "Mejorando":
        h = "Rumbo a Liderando" if dx > 0 else "Mejorando sin avanzar"
    elif q == "Rezagado":
        h = "Girando hacia Mejorando" if dy > 0 else "Sigue rezagándose"
    elif q == "Liderando":
        h = "Liderazgo firme" if dy >= 0 else "Empieza a frenarse"
    else:
        h = "Retomando impulso" if dy > 0 else "Rumbo a Rezagado"
    return q, h, round(dx, 2), round(dy, 2)


def build_etfs(prices, series=None):
    """AT de los ETFs sectoriales y fase relativa contra el S&P 500 (SPY)."""
    if BENCH not in prices:
        return None, []
    spy = technical(*prices[BENCH])
    rows = []
    for sec, etf in SECTOR_ETF.items():
        if etf not in prices:
            continue
        t = technical(*prices[etf])
        if not t:
            continue
        rows.append(dict(t, sector=sec, etf=etf, closes=prices[etf][0][-450:]))
    if not rows:
        return spy, []
    df = pd.DataFrame(rows)
    # 60% técnico propio + 40% fuerza relativa contra el S&P 500 (3, 6 y 1 mes)
    rs_mix = 0.5 * (df.r3m - spy["r3m"]) + 0.3 * (df.r6m - spy["r6m"]) + 0.2 * (df.r1m - spy["r1m"])
    df["at_own"] = score_tech(df)
    df["rs_score"] = 100 * pct_rank(rs_mix).fillna(0.5)
    df["at"] = (0.6 * df.at_own + 0.4 * df.rs_score).clip(0, 100)
    out = []
    bw = weekly(series[BENCH]) if series and BENCH in series else None
    for r in df.itertuples():
        pts, dates = rrg(series.get(r.etf) if series else None, bw)
        rs3 = (r.r3m - spy["r3m"]) if r.r3m is not None else None
        rs1 = (r.r1m - spy["r1m"]) if r.r1m is not None else None
        rs6 = (r.r6m - spy["r6m"]) if r.r6m is not None else None
        if rs3 is None or rs1 is None:
            fase = "—"
        elif rs3 >= 0 and rs1 >= 0:
            fase = "Liderando"
        elif rs3 >= 0:
            fase = "Perdiendo impulso"
        elif rs1 >= 0:
            fase = "Mejorando"
        else:
            fase = "Rezagado"
        head = None
        if pts:
            fase, head, _, _ = heading(pts)
        pro, con = [], []
        if r.p > r.sma200 and r.sma50 > r.sma200:
            pro.append("Tendencia alcista de largo plazo: arriba de la media de 200 ruedas y con la de 50 por encima de la de 200.")
        elif r.p > r.sma200:
            pro.append("Cotiza arriba de su media de 200 ruedas.")
        else:
            con.append(f"Está {f_pct(r.p/r.sma200-1,1)} debajo de su media de 200 ruedas: tendencia de largo plazo bajista.")
        if rs3 is not None:
            (pro if rs3 >= 0 else con).append(f"En 3 meses rindió {f_pct(rs3,1)} contra el S&P 500.")
        if rs1 is not None:
            (pro if rs1 >= 0 else con).append(f"En el último mes rindió {f_pct(rs1,1)} contra el S&P 500.")
        if r.rsi > 70:
            con.append(f"RSI en {fx(r.rsi,0)}: sobrecomprado en el corto plazo.")
        elif r.rsi < 30:
            pro.append(f"RSI en {fx(r.rsi,0)}: sobrevendido, posible rebote.")
        else:
            pro.append(f"RSI en {fx(r.rsi,0)}: sin sobrecompra ni sobreventa.")
        if r.macd_cross:
            pro.append("El MACD cruzó al alza en los últimos días.")
        elif r.macd_h < 0:
            con.append("El MACD está debajo de su señal: impulso de corto plazo débil.")
        if -0.12 <= r.dd <= -0.04 and r.p > r.sma200:
            pro.append(f"Corrigió {f_pct(r.dd,1)} desde su máximo dentro de una tendencia alcista.")
        d = {k: r2(getattr(r, k)) for k in ["sector", "etf", "p", "r1d", "r1m", "r3m", "r6m", "r12m", "rsi", "sma50", "sma200",
                                              "hi52", "lo52", "dd", "volat", "macd_h", "macd_cross", "at", "at_own", "rs_score"]}
        d.update(rrg=pts, rrg_dates=dates, head=head)
        d.update(fase=fase, rs1=r2(rs1), rs3=r2(rs3), rs6=r2(rs6), pro=pro, con=con,
                 closes=[round(float(x), 2) for x in r.closes])
        out.append(d)
    spy_out = {k: r2(spy[k]) for k in ["p", "r1d", "r1m", "r3m", "r6m", "r12m", "rsi", "sma50", "sma200", "dd"]}
    spy_out["above200"] = bool(spy["p"] > spy["sma200"])
    return spy_out, sorted(out, key=lambda e: -(e["at"] or 0))


def r2(x, d=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.floating, np.integer)):
        return round(float(x), d)
    return x


def add_rotation(df, series):
    if not series or BENCH not in series:
        return df
    bw = weekly(series[BENCH])
    rr, qq, hh, cand, rs = [], [], [], [], []
    for r in df.itertuples():
        pts, _ = rrg(series.get(r.t), bw)
        if not pts:
            rr.append(None); qq.append(None); hh.append(None); cand.append(False); rs.append(None)
            continue
        q, h, dx, dy = heading(pts)
        good_val = (r.f_val >= 0.5) and (r.af >= 50)
        c = bool(good_val and ((q == "Mejorando" and dx > 0) or (q == "Rezagado" and dx > 0 and dy > 0)))
        rr.append(pts); qq.append(q); hh.append(h); cand.append(c)
        rs.append(round(0.5 * r.af + 50 * r.f_val, 1))
    df["rrg"], df["rot_q"], df["rot_h"], df["rot_cand"], df["rot_score"] = rr, qq, hh, cand, rs
    return df


# ---------------------------------------------------------------- valor y momentum
VM_WEEKS = 26


def fair_value(r):
    """Valor justo estimado: promedio de (EPS esperado x P/E mediano del sector) y precio objetivo de analistas."""
    vals, src = [], []
    if pd.notna(r.fwd_eps) and r.fwd_eps > 0 and pd.notna(r.sec_fpe) and r.sec_fpe > 0:
        v = r.fwd_eps * r.sec_fpe
        if 0.3 < v / r.p < 3:
            vals.append(v); src.append("P/E del sector")
    if r.n_an >= 3 and pd.notna(r.target) and 0.3 < r.target / r.p < 3:
        vals.append(r.target); src.append("analistas")
    if not vals:
        return None, None
    return float(np.mean(vals)), " + ".join(src)


def momentum_index(c):
    """Índice de momentum de -100 a +100 (serie diaria): medias 50/200, RSI y MACD."""
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi_s = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    cl = lambda x: x.clip(-1, 1)
    m = 100 * (0.35 * cl((c / s50 - 1) / 0.10) + 0.25 * cl((c / s200 - 1) / 0.20)
               + 0.25 * cl((rsi_s - 50) / 25) + 0.15 * cl(hist / (c * 0.01)))
    return m, rsi_s


def vm_zone(x, y):
    if x < 0:
        return "Compra" if y >= 0 else "Vigilar"
    return "Mantener" if y >= 0 else "Evitar"


def vm_status(pts):
    x, y = pts[-1]
    z = vm_zone(x, y)
    zp = vm_zone(*pts[-9]) if len(pts) >= 9 else vm_zone(*pts[0])
    x3, y3 = pts[-4] if len(pts) >= 4 else pts[0]
    dx, dy = x - x3, y - y3
    if z == "Compra":
        if zp in ("Vigilar", "Evitar"):
            st = "Recién entró a zona de compra"
        elif x > -0.05 and dx > 0:
            st = "Llegando a su valor justo"
        else:
            st = "En zona de compra"
    elif z == "Vigilar":
        st = "Girando: el momentum mejora" if dy > 8 else "Barata pero sigue débil"
    elif z == "Mantener":
        st = "Alcanzó su valor justo" if zp == "Compra" else "Sin margen, tendencia alcista"
    else:
        st = "Cara y perdiendo fuerza"
    return z, zp, st


def add_value_momentum(df, series):
    out = {k: [] for k in ["vm", "fair", "fair_src", "fair_up", "vm_z", "vm_zp", "vm_st"]}
    for r in df.itertuples():
        fair, src = fair_value(r)
        ser = series.get(r.t) if series else None
        pts = None
        if fair and ser is not None and len(ser) > 260:
            m, _ = momentum_index(ser)
            w = pd.DataFrame({"c": ser, "m": m}).resample("W-FRI").last().dropna().tail(VM_WEEKS + 1)
            if len(w) >= 8:
                pts = [[round(float(c / fair - 1), 4), round(float(mm), 1)] for c, mm in zip(w.c, w.m)]
        if pts:
            z, zp, st = vm_status(pts)
        else:
            z = zp = st = None
        out["vm"].append(pts); out["fair"].append(fair); out["fair_src"].append(src)
        out["fair_up"].append((fair / r.p - 1) if fair else None)
        out["vm_z"].append(z); out["vm_zp"].append(zp); out["vm_st"].append(st)
    for k, v in out.items():
        df[k] = v
    return df


def to_json(df, n_universe, spy=None, etfs=None, vm_dates=None):
    out_rows = []
    keep = ["t", "name", "sector", "region", "p", "mcap", "r1d", "r1m", "r3m", "r6m", "r12m", "rsi", "sma20", "sma50",
            "sma200", "hi52", "lo52", "dd", "volat", "vol_ratio", "macd_h", "macd_cross", "pe", "fpe", "peg", "pb",
            "rev_g", "eps_g", "eps", "fwd_eps", "fwd_eps_g", "roe", "margin", "op_margin", "de", "div", "target",
            "upside", "rec", "n_an", "at", "af", "score", "f_val", "f_gro", "f_qua", "f_sol", "f_ana", "sec_fpe",
            "setup", "elig", "rank"]
    for r in df.itertuples():
        d = {k: r2(getattr(r, k)) for k in keep}
        d["rank"] = int(r.rank) if r.rank is not None and r.rank == r.rank else None
        d["pro"], d["con"], d["grow"] = reasons(r)
        if getattr(r, "vm", None):
            d.update(vm=r.vm, fair=r2(r.fair, 2), fair_src=r.fair_src, fair_up=r2(r.fair_up), vm_z=r.vm_z,
                     vm_zp=r.vm_zp, vm_st=r.vm_st)
        if r.rank is not None:
            d["closes"] = [round(float(x), 2) for x in r.closes]
        out_rows.append(d)
    heat = []
    for sec, g in df.groupby("sector"):
        heat.append({"sector": sec, "n": int(len(g)), "score": r2(g.score.mean(), 1),
                     "r1d": r2(g.r1d.median()), "r1m": r2(g.r1m.median())})
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "universe": n_universe, "analyzed": int(len(df)), "top_n": TOP_N,
            "method": "AT 50% (tendencia, momentum, RSI/MACD, posición vs máximo) + AF 50% (valuación vs sector, crecimiento, calidad, solidez, analistas).",
            "sectors": sorted(heat, key=lambda h: -h["score"]), "spy": spy, "etfs": etfs or [],
            "rrg_dates": next((e["rrg_dates"] for e in (etfs or []) if e.get("rrg_dates")), []),
            "vm_dates": vm_dates or [],
            "rows": out_rows}


def main():
    import yfinance as yf
    # segunda clase de acciones de la misma empresa: queda solo una
    DUP = {"GOOG", "FOX", "NWS"}
    uni = [(s, n, g, "EE.UU.") for s, n, g in sp500() if s not in DUP]
    seen = {u[0] for u in uni}
    uni += [(s, s, "", "Argentina") for s in ADR_AR if s not in seen]
    uni += [(s, s, "", "Brasil") for s in ADR_BR if s not in seen]
    syms = [u[0] for u in uni]
    print("Universo:", len(syms))

    prices, series = {}, {}
    dl = syms + [BENCH] + list(SECTOR_ETF.values())
    for i in range(0, len(dl), 100):
        chunk = dl[i:i + 100]
        data = yf.download(chunk, period="2y", interval="1d", auto_adjust=True, group_by="ticker",
                           threads=True, progress=False)
        for s in chunk:
            try:
                sub = data[s] if len(chunk) > 1 else data
                c = sub["Close"].values.astype(float)
                v = sub["Volume"].values.astype(float)
                if np.isfinite(c).sum() >= 210:
                    prices[s] = (c[np.isfinite(c)], v[np.isfinite(c)])
                    series[s] = sub["Close"].dropna()
            except Exception:
                pass
        time.sleep(1)
    print("Con precios:", len(prices))
    if len(prices) < 100:
        raise SystemExit("Yahoo Finance devolvió muy pocos precios; no se actualiza data.json. Reintentá más tarde.")

    infos = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for sym, inf in ex.map(fetch_info, [s for s in prices if s in set(syms)]):
            infos[sym] = inf
    print("Con fundamentales:", sum(1 for v in infos.values() if v))

    df = build(uni, prices, infos)
    df = add_value_momentum(df, series)
    spy, etfs = build_etfs(prices, series)
    vm_dates = [d.strftime("%Y-%m-%d") for d in series[BENCH].resample("W-FRI").last().dropna().tail(VM_WEEKS + 1).index] if BENCH in series else []
    print("Zonas valor/momentum:", df.vm_z.value_counts().to_dict())
    print("ETFs sectoriales:", [(e["etf"], round(e["at"]), e["fase"]) for e in etfs])
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(to_json(df, len(syms), spy, etfs, vm_dates), f, ensure_ascii=False, separators=(",", ":"))
    print("OK data.json:", len(df), "papeles. Top:", list(df[df["rank"].notna()].t))


if __name__ == "__main__":
    main()
