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
          "IRS", "LOMA", "TEO", "VIST", "GLOB", "MELI", "DESP", "CAAP", "BIOX", "ADGO"]
ADR_BR = ["PBR", "VALE", "ITUB", "BBD", "ABEV", "NU", "SBS", "EBR", "GGB", "SID", "ERJ",
          "SUZ", "BRFS", "UGP", "CIG", "ELP", "BSBR", "XP", "STNE", "PAGS", "VIV", "TIMB", "BAK"]
TOP_N = 30
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


def r2(x, d=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.floating, np.integer)):
        return round(float(x), d)
    return x


def to_json(df, n_universe):
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
            "sectors": sorted(heat, key=lambda h: -h["score"]), "rows": out_rows}


def main():
    import yfinance as yf
    uni = [(s, n, g, "EE.UU.") for s, n, g in sp500()]
    seen = {u[0] for u in uni}
    uni += [(s, s, "", "Argentina") for s in ADR_AR if s not in seen]
    uni += [(s, s, "", "Brasil") for s in ADR_BR if s not in seen]
    syms = [u[0] for u in uni]
    print("Universo:", len(syms))

    prices = {}
    for i in range(0, len(syms), 100):
        chunk = syms[i:i + 100]
        data = yf.download(chunk, period="2y", interval="1d", auto_adjust=True, group_by="ticker",
                           threads=True, progress=False)
        for s in chunk:
            try:
                sub = data[s] if len(chunk) > 1 else data
                c = sub["Close"].values.astype(float)
                v = sub["Volume"].values.astype(float)
                if np.isfinite(c).sum() >= 210:
                    prices[s] = (c[np.isfinite(c)], v[np.isfinite(c)])
            except Exception:
                pass
        time.sleep(1)
    print("Con precios:", len(prices))
    if len(prices) < 100:
        raise SystemExit("Yahoo Finance devolvió muy pocos precios; no se actualiza data.json. Reintentá más tarde.")

    infos = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for sym, inf in ex.map(fetch_info, list(prices)):
            infos[sym] = inf
    print("Con fundamentales:", sum(1 for v in infos.values() if v))

    df = build(uni, prices, infos)
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(to_json(df, len(syms)), f, ensure_ascii=False, separators=(",", ":"))
    print("OK data.json:", len(df), "papeles. Top:", list(df[df["rank"].notna()].t))


if __name__ == "__main__":
    main()
