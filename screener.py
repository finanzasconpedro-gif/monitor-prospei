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
# ---- modelos de valuación por tipo de negocio
HOLDINGS = {"BRK-B"}                         # holdings/conglomerados: se valúan por valor libros
BOOK_INDUSTRIES = {"Banks - Diversified", "Banks - Regional", "Insurance - Diversified",
                   "Insurance - Property & Casualty", "Insurance - Life", "Insurance - Reinsurance",
                   "Insurance - Specialty", "Mortgage Finance"}
EPS_GAP = 0.40      # si la ganancia de 12 meses y la esperada difieren más de esto, el P/E no es confiable
MODEL_TXT = {"pe": "P/E (ganancias)", "book": "precio / valor libros", "reit": "dividendo y analistas (REIT)",
             "sales": "precio / ventas (sin ganancias)", "na": "sin método confiable"}
TNX_YIELD = None  # se completa en main() con el bono a 10 años
SECTOR_ETF = {"Tecnología": "XLK", "Financiero": "XLF", "Salud": "XLV", "Consumo discrecional": "XLY",
              "Consumo básico": "XLP", "Industrial": "XLI", "Energía": "XLE", "Utilities": "XLU",
              "Materiales": "XLB", "Real estate": "XLRE", "Comunicaciones": "XLC"}
BENCH = "SPY"
MACRO = {"^TNX": "Bono EE.UU. 10 años", "^VIX": "VIX", "DX-Y.NYB": "Dólar global (DXY)", "CL=F": "Petróleo WTI"}
CYCLICAL = {"Semiconductors", "Semiconductor Equipment & Materials", "Steel", "Aluminum", "Copper",
            "Oil & Gas E&P", "Oil & Gas Integrated", "Oil & Gas Refining & Marketing", "Oil & Gas Equipment & Services",
            "Oil & Gas Drilling", "Auto Manufacturers", "Auto Parts", "Specialty Chemicals", "Chemicals",
            "Marine Shipping", "Other Industrial Metals & Mining", "Agricultural Inputs", "Building Materials",
            "Residential Construction", "Airlines", "Gold", "Silver", "Coking Coal", "Thermal Coal", "Paper & Paper Products",
            "Lumber & Wood Production", "Trucking", "Railroads", "Farm & Heavy Construction Machinery"}
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
    tr = trend_state(c, p, s200, s200 > s200_prev)
    # corrección: pico reciente (3 meses) y señales de rebote
    w3 = c[-63:]
    hi3_i = int(np.argmax(w3))
    hi3 = float(w3[hi3_i])
    hi52_age = int(len(w) - 1 - int(np.argmax(w)))
    rsi_prev = rsi(c[-305:-5])
    rsi_now = rsi(c[-300:])
    rebound = bool(p > s20 and (hist[-1] > hist[-2] or rsi_now > rsi_prev + 3))
    pb_days = max(1, len(w3) - 1 - hi3_i)
    pb_vol = (float(vv[-min(pb_days, 20):].mean() / vv[-60:].mean())
              if len(vv) >= 60 and vv[-60:].mean() > 0 and pb_days >= 3 else None)
    return {**tr, **cross_state(c), **early_turn(c, vv if len(vv) == len(c) else None), "dd3": p / hi3 - 1, "hi3_age": pb_days, "hi52_age": hi52_age, "rebound": rebound,
            "pb_vol": pb_vol, "near50": bool(abs(p / s50 - 1) <= 0.02), "near200": bool(-0.01 <= p / s200 - 1 <= 0.03),
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


CONFIRM_DAYS = 10      # ruedas consecutivas del mismo lado de la media de 200 para confirmar
DEF_BAND = 0.02        # +/- 2% alrededor de la media de 200 = zona de definición
TREND_N = {"Alcista confirmada": 2, "Alcista en formación": 1, "Zona de definición": 0,
           "Bajista en formación": -1, "Bajista confirmada": -2}


def trend_state(c, p, s200, s200_up):
    """Tendencia de largo plazo con confirmación por tiempo (diario) y por cierre semanal."""
    sma = pd.Series(c).rolling(200).mean().values
    side = np.sign(c[-60:] - sma[-60:])
    side = side[np.isfinite(side)]
    cur = 1 if p > s200 else -1
    run = 0
    for x in side[::-1]:
        if x == cur:
            run += 1
        else:
            break
    wk = c[::-1][::5][::-1]                       # cierres semanales aproximados (cada 5 ruedas)
    sma40w = float(wk[-40:].mean()) if len(wk) >= 40 else None
    sma200w = float(wk[-200:].mean()) if len(wk) >= 200 else None
    w_above = (wk[-1] > sma40w) if sma40w else (cur > 0)
    dist = p / s200 - 1
    if abs(dist) < DEF_BAND and run < CONFIRM_DAYS:
        st = "Zona de definición"
    elif cur > 0:
        st = "Alcista confirmada" if (run >= CONFIRM_DAYS or dist > 0.05) and w_above else "Alcista en formación"
    else:
        st = "Bajista confirmada" if (run >= CONFIRM_DAYS or dist < -0.05) and not w_above else "Bajista en formación"
    # recuperación: cruzó arriba hace poco después de haber estado claramente debajo (>5%) en los últimos 6 meses
    look = min(126, len(c))
    was_below = bool(np.nanmin(c[-look:] / sma[-look:] - 1) < -0.05) if np.isfinite(sma[-look:]).any() else False
    recovery = bool(cur > 0 and run < 40 and was_below)
    return {"recovery": recovery, "trend": st, "trend_n": TREND_N[st], "run200": int(run * cur), "dist200": float(dist),
            "sma40w": sma40w, "sma200w": sma200w, "w_above": bool(w_above),
            "d200w": (float(p / sma200w - 1) if sma200w else None)}


def pivots(x, k=7, kind="low"):
    """Índices de mínimos (o máximos) locales confirmados: el extremo de una ventana de +/- k ruedas."""
    out = []
    for i in range(k, len(x) - k):
        w = x[i - k:i + k + 1]
        if (kind == "low" and x[i] == w.min()) or (kind == "high" and x[i] == w.max()):
            if not out or i - out[-1] > k:
                out.append(i)
    return out


def early_turn(c, v):
    """Señales tempranas de giro alcista (antes del Golden Cross). Devuelve dict con cada señal."""
    n = len(c)
    look = min(140, n)
    cw, vw = c[-look:], (v[-look:] if v is not None and len(v) >= look else None)
    ser = pd.Series(c)
    d = ser.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi_s = (100 - 100 / (1 + up / dn.replace(0, np.nan))).values[-look:]
    lows, highs = pivots(cw, 7, "low"), pivots(cw, 7, "high")
    p = float(cw[-1])
    sig = {"et_hl": False, "et_div": False, "et_acc": False, "et_brk": False, "et_mom": False}
    L1 = L2 = H = None
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        L1, L2 = float(cw[i1]), float(cw[i2])
        sig["et_hl"] = bool(L2 > L1 * 1.01 and p > L2 and look - 1 - i2 <= 60)
        sig["et_div"] = bool(L2 < L1 and np.isfinite(rsi_s[i1]) and rsi_s[i2] > rsi_s[i1] + 3 and look - 1 - i2 <= 60)
    elif len(lows) == 1:
        L2 = float(cw[lows[-1]])
    hs = [h for h in highs if h < look - 3 and (not lows or h > lows[-2] if len(lows) >= 2 else True)]
    if hs:
        H = float(cw[hs[-1]])
        sig["et_brk"] = bool(p > H)
    if vw is not None:
        dd = np.diff(cw[-21:])
        vv = vw[-20:]
        upv, dnv = vv[dd > 0].sum(), vv[dd < 0].sum()
        sig["et_acc"] = bool(dnv > 0 and upv / dnv > 1.2)
    macd = ser.ewm(span=12, adjust=False).mean() - ser.ewm(span=26, adjust=False).mean()
    rs_now, rs_10 = rsi_s[-1], rsi_s[-11]
    sig["et_mom"] = bool(macd.values[-1] > 0 or (rs_now > 50 and np.isfinite(rs_10) and rs_10 < 50))
    sig.update(et_low=L2, et_high=H)
    return sig


def cross_state(c):
    """Cruces de la media de 50 con la de 200 (Golden / Death Cross) y pendiente de la media de 200."""
    ser = pd.Series(c)
    m50, m200 = ser.rolling(50).mean().values, ser.rolling(200).mean().values
    gap = (m50 / m200 - 1)[-80:]
    gap = gap[np.isfinite(gap)]
    out = {"cross": None, "cross_days": None, "cross_eta": None, "gap50": None, "slope200": None, "slope200_pct": None}
    if len(gap) < 25:
        return out
    g = float(gap[-1])
    out["gap50"] = g
    sl = float(m200[-1] / m200[-21] - 1) if np.isfinite(m200[-21]) else 0.0
    out["slope200_pct"] = sl
    out["slope200"] = "Subiendo" if sl > 0.005 else "Bajando" if sl < -0.005 else "Plana"
    sign = np.sign(gap)
    days = None
    for k in range(1, len(sign)):
        if sign[-k] != sign[-k - 1] and sign[-k - 1] != 0:
            days = k - 1
            break
    if days is not None and days <= 20:
        out["cross"] = "Golden Cross reciente" if g > 0 else "Death Cross reciente"
        out["cross_days"] = int(days)
        return out
    speed = (g - float(gap[-11])) / 10      # cambio de la distancia por rueda
    if -0.02 < g < 0 and speed > 0:
        eta = -g / speed
        if eta <= 40:
            out["cross"], out["cross_eta"] = "Golden Cross en formación", int(round(eta))
    elif 0 < g < 0.02 and speed < 0:
        eta = g / -speed
        if eta <= 40:
            out["cross"], out["cross_eta"] = "Death Cross en formación", int(round(eta))
    return out


def cross_texts(r):
    pro, con = [], []
    st = getattr(r, "cross", None)
    if not st:
        return pro, con
    pend = {"Subiendo": "la media de 200 ya está subiendo, lo que le da más fuerza a la señal",
            "Plana": "la media de 200 está plana",
            "Bajando": "pero la media de 200 todavía baja: es una señal más débil, conviene confirmar"}.get(r.slope200, "")
    if st == "Golden Cross reciente":
        pro.append(f"Golden Cross {'hoy' if r.cross_days == 0 else 'hace ' + str(r.cross_days) + ' ruedas'}: la media de 50 cruzó por encima de la de 200; {pend}.")
    elif st == "Golden Cross en formación":
        pro.append(f"Golden Cross en formación (a vigilar): la media de 50 está {fx(abs(r.gap50)*100,1)}% debajo de la de 200 y se acerca; "
                   f"al ritmo actual cruzaría en unas {r.cross_eta} ruedas. Es una señal tardía: confirma el giro, no lo anticipa.")
    elif st == "Death Cross reciente":
        con.append(f"Death Cross {'hoy' if r.cross_days == 0 else 'hace ' + str(r.cross_days) + ' ruedas'}: la media de 50 cruzó por debajo de la de 200.")
    elif st == "Death Cross en formación":
        con.append(f"Death Cross en formación: la media de 50 está {fx(r.gap50*100,1)}% arriba de la de 200 y se acerca; "
                   f"al ritmo actual cruzaría en unas {r.cross_eta} ruedas.")
    return pro, con


def trend_text(r, short=False):
    """Frase de tendencia para A favor / A vigilar. Devuelve (es_favor, texto)."""
    n, run, d = r.trend_n, abs(int(r.run200)), r.dist200
    sem = "el cierre semanal confirma (arriba de la media de 40 semanas)" if r.w_above else "el cierre semanal todavía no confirma (debajo de la media de 40 semanas)"
    if n == 2:
        extra = " y la de 50 está por encima de la de 200" if r.sma50 > r.sma200 else ""
        return True, f"Tendencia alcista confirmada: {run}+ ruedas arriba de su media de 200 ({f_pct(d,1)}){extra}; {sem}."
    if n == 1:
        return True, f"Tendencia alcista en formación: hace {run} ruedas que cotiza arriba de su media de 200 ({f_pct(d,1)}); {sem}."
    if n == 0:
        lado = "arriba" if d >= 0 else "debajo"
        return None, f"Zona de definición: cotiza {fx(abs(d)*100,1)}% {lado} de su media de 200, sin confirmar dirección ({run} ruedas de ese lado)."
    if n == -1:
        sem2 = "el cierre semanal sigue arriba de la media de 40 semanas" if r.w_above else "el cierre semanal también está debajo de la media de 40 semanas"
        return False, f"Tendencia bajista en formación: hace {run} ruedas que cotiza debajo de su media de 200 ({f_pct(d,1)}); {sem2}."
    return False, f"Tendencia bajista confirmada: {run}+ ruedas debajo de su media de 200 ({f_pct(d,1)}) y cierre semanal debajo de la media de 40 semanas."


def pct_rank(s, higher_better=True):
    """Percentil 0..1 dentro de la serie (NaN queda NaN)."""
    r = s.rank(pct=True, ascending=higher_better)
    return r


def score_tech(df):
    t = pd.Series(0.0, index=df.index)
    # Tendencia (30)
    t += df.trend_n.map({2: 10, 1: 6, 0: 4, -1: 2, -2: 0}).fillna(4)
    t += np.where(df.sma50 > df.sma200, 10, 0)
    t += np.where(df.sma200_up, 10, 0)
    if "et_ctx" in df:
        t += np.where(df.et_ctx, 1.5 * df.et_n, 0)
    t += df.cross.map({"Golden Cross reciente": 3, "Golden Cross en formación": 2,
                       "Death Cross reciente": -3, "Death Cross en formación": -2}).fillna(0)
    # Momentum relativo 6m sin el ultimo mes (25)
    mom = (1 + df.r6m) / (1 + df.r1m) - 1
    t += 25 * pct_rank(mom).fillna(0.3)
    # Timing: RSI (15) + MACD (10)
    r = df.rsi
    t += np.select([(r >= 40) & (r <= 60), (r > 60) & (r <= 70), (r >= 30) & (r < 40), r < 30, r > 70],
                   [15, 10, 11, 7, 3], 5)
    t += np.where(df.macd_h > 0, 5, 0) + np.where(df.macd_up, 2, 0) + np.where(df.macd_cross, 3, 0)
    # Posicion respecto del maximo (20)
    up = df.trend_n >= 1
    dd = df.dd
    t += np.select([up & (dd <= -0.05) & (dd >= -0.20), up & (dd > -0.05), up & (dd < -0.20)],
                   [20, 15, 10], 4)
    return t.clip(0, 100)


def score_fund(df):
    """Cada pilar 0..1 por percentil; la valuacion se compara dentro del sector."""
    v_peer = pct_rank(df.v_peer, False)                                         # vs pares (según el modelo de cada empresa)
    v_hist = pct_rank(df.v_hist, False)                                         # vs su propia historia
    v_peg = pct_rank(df.peg.where((df.peg > 0) & (df.peg < 10)), False)
    parts = pd.concat([v_peer * 0.5, v_hist * 0.3, v_peg * 0.2], axis=1)
    wts = pd.concat([v_peer.notna() * 0.5, v_hist.notna() * 0.3, v_peg.notna() * 0.2], axis=1).sum(axis=1)
    val = (parts.sum(axis=1, skipna=True) / wts.replace(0, np.nan)).fillna(0.15)
    val = val.where(df.v_peer.notna() | df.v_hist.notna(), 0.3)
    # trampa cíclica: P/E bajo en industria cíclica cerca del pico de ganancias -> techo a la valuación
    val = val.where(~df.cyc, val.clip(upper=0.3))
    fin = df.sector == "Financiero"
    # ganancia no recurrente: se usa el crecimiento de ventas en lugar del de ganancias
    eps_g_adj = df.eps_g.where(~df.one_off, df.rev_g)
    gro = pd.concat([pct_rank(df.rev_g), pct_rank(eps_g_adj), pct_rank(df.fwd_eps_g)], axis=1).mean(axis=1).fillna(0.3)
    fcf_rank = pct_rank(df.fcf_y.where(~fin))
    qua = pd.concat([pct_rank(df.roe), pct_rank(df.margin), pct_rank(df.op_margin), fcf_rank], axis=1).mean(axis=1).fillna(0.3)
    qua = qua.where(~((df.fcf < 0) & ~fin), qua.clip(upper=0.35))
    # deuda/patrimonio: en bancos y financieras la deuda es el negocio -> neutro
    sol = pct_rank(df.de.where(~fin), False).where(~fin, 0.5).fillna(0.5)
    has_an = df.n_an >= 3
    ana = (0.6 * pct_rank(df.upside).fillna(0.3) + 0.4 * pct_rank(df.rec, False).fillna(0.3)).where(has_an, 0.3)
    df["f_val"], df["f_gro"], df["f_qua"], df["f_sol"], df["f_ana"] = val, gro, qua, sol, ana
    return (25 * val + 25 * gro + 25 * qua + 10 * sol + 15 * ana).clip(0, 100)


def days_to(ts):
    try:
        ts = float(ts)
    except Exception:
        return None
    if not ts or not math.isfinite(ts):
        return None
    d = dt.datetime.fromtimestamp(ts, dt.timezone.utc) - dt.datetime.now(dt.timezone.utc)
    return int(math.floor(d.total_seconds() / 86400))


def lights(r):
    """Semáforo: 2 verde, 1 amarillo, 0 rojo, None sin dato."""
    lv = lambda v, hi, lo: None if v is None or (isinstance(v, float) and not math.isfinite(v)) else (2 if v >= hi else 1 if v >= lo else 0)
    q = lv(r.f_qua, 0.6, 0.35)
    if r.sector != "Financiero" and pd.notna(r.fcf) and r.fcf < 0:
        q = 0
    v = {"Pico de ciclo": 0, "Cara": 0, "Barata": 2}.get(r.val_verdict, 1)
    t = lv(r.at, 65, 45)
    if pd.notna(r.earn_days) and 0 <= r.earn_days <= 7:
        e = 0
    elif pd.notna(r.earn_days) and 0 <= r.earn_days <= 21:
        e = 1
    elif pd.notna(r.earn_days) and r.earn_days >= 0:
        e = 2
    else:
        e = None
    return {"cal": q, "val": v, "tec": t, "evt": e}


# ---------------------------------------------------------------- textos
def fx(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def f_pct(x, d=0):
    return f"{x*100:+.{d}f}%".replace(".", ",")


def is_correction(r):
    """Corrección dentro de una suba confirmada: cayó 5-20% desde su pico de los últimos 3 meses."""
    return bool(r.trend_n == 2 and r.sma50 > r.sma200 and -0.20 <= r.dd3 <= -0.05 and r.dd >= -0.25 and r.rsi < 55)


def setup(r):
    up = r.trend_n >= 1
    if up and r.dd > -0.04 and r.vol_ratio and r.vol_ratio > 1.15:
        return "Ruptura de máximos"
    if r.cross == "Golden Cross reciente" and r.p > r.sma200:
        return "Golden Cross"
    if r.recovery:
        return "Recuperación"
    if is_correction(r):
        return "Corrección con señal de rebote" if r.rebound else "Corrección en curso"
    if up and r.sma50 > r.sma200:
        return "Tendencia alcista"
    if not up and r.et_ctx and r.et_n >= 4:
        return "Giro temprano"
    if not up and r.cross == "Golden Cross en formación":
        return "Golden Cross en formación"
    if not up and r.rsi < 35 and r.af >= 60:
        return "Sobrevendida con buenos fundamentos"
    if r.macd_cross:
        return "Giro técnico"
    return "Valor"


def correction_texts(r):
    pro, con = [], []
    if r.recovery:
        con.append(f"Recuperación: cruzó arriba de su media de 200 hace {abs(int(r.run200))} ruedas después de cotizar claramente por debajo. "
                   "No es una corrección dentro de una suba: falta confirmar que sostenga la media.")
    elif is_correction(r):
        cuando = f"hace {int(r.hi3_age)} ruedas"
        if r.rebound:
            pro.append(f"Corrección con señal de rebote: cayó {fx(abs(r.dd3)*100,1)}% desde su último pico ({cuando}) y ya recuperó su media "
                       "de 20 ruedas con el MACD o el RSI girando al alza: posible punto de entrada.")
        else:
            con.append(f"Corrección en curso: cayó {fx(abs(r.dd3)*100,1)}% desde su último pico ({cuando}) y todavía no muestra señal de rebote "
                       "(debajo de la media de 20 o sin giro de MACD/RSI). Conviene esperar la confirmación.")
        if r.pb_vol is not None:
            if r.pb_vol < 0.9:
                pro.append(f"La caída se dio con volumen bajo ({fx(r.pb_vol,2)} veces el promedio): parece toma de ganancias, no venta fuerte.")
            elif r.pb_vol > 1.25:
                con.append(f"La caída se dio con volumen alto ({fx(r.pb_vol,2)} veces el promedio): hay presión vendedora real.")
        if r.near200:
            pro.append("Está apoyada en su media de 200 ruedas, el soporte clave de la tendencia.")
        elif r.near50:
            pro.append("Está apoyada en su media de 50 ruedas, un soporte habitual en tendencias alcistas.")
    if r.trend_n >= 1 and -0.20 <= r.dd <= -0.05 and r.hi52_age > 150 and not is_correction(r) and not r.recovery:
        con.append(f"Su máximo de 52 semanas fue hace {int(r.hi52_age)} ruedas: más que una corrección, viene de un lateral largo.")
    return pro, con


def reasons(r):
    pro, con, grow = [], [], []
    # Tecnico
    fav, txt = trend_text(r)
    (pro if fav else con).append(txt)
    if pd.notna(r.d200w) and -0.03 <= r.d200w <= 0.05:
        pro.append(f"Está cerca de su media de 200 semanas ({f_pct(r.d200w,1)}), un soporte de ciclo largo (unos 4 años).")
    pro_c, con_c = correction_texts(r)
    pro += pro_c
    con += con_c
    if r.et_ctx and r.et_n >= 3:
        names = [t for k, t in [("et_hl", "mínimo más alto"), ("et_div", "divergencia alcista del RSI"),
                                ("et_acc", "acumulación por volumen"), ("et_brk", "rompió el último máximo"),
                                ("et_mom", "impulso girando a positivo"), ("et_rs", "le gana al S&P en el último mes")] if getattr(r, k)]
        pro.append(f"Giro temprano: {int(r.et_n)} de 6 señales ({', '.join(names)}). Son señales anticipadas: fallan más que las de confirmación.")
    pro_x, con_x = cross_texts(r)
    pro += pro_x
    con += con_x
    if r.dd > -0.03 and r.vol_ratio and r.vol_ratio < 0.9:
        con.append("Está cerca de máximos pero con volumen bajo: la suba muestra poca convicción frente a la resistencia.")
    elif r.dd > -0.03:
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
    if r.val_model == "book" and pd.notna(r.pb) and pd.notna(r.peer_pb):
        if r.pb < r.peer_pb * 0.85:
            pro.append(f"Cotiza a {fx(r.pb,2)} veces su valor libros contra {fx(r.peer_pb,2)}x de {r.peer_lbl}: descuento sobre su patrimonio.")
        elif r.pb > r.peer_pb * 1.4:
            con.append(f"Cotiza a {fx(r.pb,2)} veces su valor libros, caro frente a {fx(r.peer_pb,2)}x de {r.peer_lbl}.")
        if pd.notna(r.pb_hist):
            dv = r.pb / r.pb_hist - 1
            if dv < -0.15:
                pro.append(f"Su P/VL ({fx(r.pb,2)}x) está {fx(abs(dv)*100,0)}% debajo de su promedio histórico ({fx(r.pb_hist,2)}x).")
            elif dv > 0.25:
                con.append(f"Su P/VL ({fx(r.pb,2)}x) está {fx(dv*100,0)}% arriba de su promedio histórico ({fx(r.pb_hist,2)}x).")
    elif r.val_model == "sales" and pd.notna(r.ps) and pd.notna(r.peer_ps):
        if r.ps < r.peer_ps * 0.85:
            pro.append(f"Precio/ventas de {fx(r.ps,1)}x contra {fx(r.peer_ps,1)}x de {r.peer_lbl}: descuento por ventas.")
        elif r.ps > r.peer_ps * 1.4:
            con.append(f"Precio/ventas de {fx(r.ps,1)}x, caro frente a {fx(r.peer_ps,1)}x de {r.peer_lbl}.")
    if r.val_model == "pe" and not r.eps_gap_flag and pd.notna(r.fpe) and r.fpe > 0:
        if pd.notna(r.sec_fpe) and r.fpe < r.sec_fpe * 0.85 and not r.cyc:
            pro.append(f"P/E esperado de {fx(r.fpe,1)}x contra {fx(r.sec_fpe,1)}x de la mediana de {r.peer_lbl}: cotiza con descuento.")
        elif pd.notna(r.sec_fpe) and r.fpe > r.sec_fpe * 1.4:
            con.append(f"P/E esperado de {fx(r.fpe,1)}x, caro frente a {fx(r.sec_fpe,1)}x de {r.peer_lbl}.")
    elif pd.notna(r.pe) and r.pe < 0 or (pd.notna(r.eps) and r.eps < 0):
        con.append("Sin ganancias en los últimos 12 meses.")
    if pd.notna(r.roe) and r.roe > 0.20:
        pro.append(f"Alta rentabilidad sobre el patrimonio (ROE {fx(r.roe*100,0)}%).")
    if pd.notna(r.margin) and r.margin > 0.20:
        pro.append(f"Margen neto de {fx(r.margin*100,0)}%.")
    elif pd.notna(r.margin) and r.margin < 0:
        con.append("Margen neto negativo.")
    if r.val_model == "pe" and not r.eps_gap_flag and pd.notna(r.pe_hist) and pd.notna(r.pe) and r.pe > 0:
        dv = r.pe / r.pe_hist - 1
        if dv < -0.15:
            pro.append(f"Cotiza a P/E {fx(r.pe,1)}x contra su promedio de los últimos {int(r.hist_n or 4)} años ({fx(r.pe_hist,1)}x): {fx(abs(dv)*100,0)}% más barata que su propia norma.")
        elif dv > 0.25:
            con.append(f"Cotiza a P/E {fx(r.pe,1)}x contra su promedio de los últimos {int(r.hist_n or 4)} años ({fx(r.pe_hist,1)}x): {fx(dv*100,0)}% más cara que su propia norma.")
        elif r.v_peer is not None and pd.notna(r.v_peer) and r.v_peer < -0.15:
            con.append(f"Su P/E es bajo frente a {r.peer_lbl}, pero similar a su propio promedio histórico ({fx(r.pe_hist,1)}x): el descuento puede ser estructural, no una oportunidad.")
    if pd.notna(r.div) and pd.notna(r.div5) and r.div > 0 and r.div5 > 0 and r.div > r.div5 * 1.2:
        pro.append(f"Su dividendo rinde {fx(r.div,2)}%, por encima de su promedio de 5 años ({fx(r.div5,2)}%): el precio está bajo frente a su historia.")
    if r.cyc:
        con.append("Posible pico de ciclo: en industrias cíclicas un P/E bajo suele aparecer cerca del techo de ganancias. La valuación del puntaje está limitada.")
    if r.one_off and r.t in HOLDINGS:
        con.append("Su ganancia contable incluye la suba o baja de su cartera de inversiones: varía mucho de un trimestre a otro y no refleja el negocio operativo. Por eso se valúa por valor libros.")
    elif r.one_off:
        con.append("La ganancia del último trimestre creció mucho más que las ventas: posiblemente inflada por resultados no recurrentes. El crecimiento se mide por ventas.")
    if r.sector != "Financiero" and pd.notna(r.fcf):
        if r.fcf < 0:
            con.append("Flujo de caja libre negativo en los últimos 12 meses: la empresa consume más caja de la que genera.")
        elif pd.notna(r.fcf_y) and TNX_YIELD and r.fcf_y > TNX_YIELD:
            pro.append(f"Rendimiento del flujo de caja libre de {fx(r.fcf_y*100,1)}%, por encima del bono del Tesoro a 10 años ({fx(TNX_YIELD*100,2)}%).")
    if pd.notna(r.earn_days) and 0 <= r.earn_days <= 14:
        con.append(f"Presenta balance en {int(r.earn_days)} días: el precio puede moverse fuerte en cualquier dirección.")
    if r.sector != "Financiero" and pd.notna(r.de) and r.de > 200:
        con.append(f"Endeudamiento elevado (deuda/patrimonio {fx(r.de/100,1)}x).")
    # Crecimiento
    if pd.notna(r.rev_g):
        (grow if r.rev_g >= 0 else con).append(f"Ventas: {f_pct(r.rev_g,1)} interanual en el último trimestre.")
    if pd.notna(r.eps_g) and not r.one_off:
        (grow if r.eps_g >= 0 else con).append(f"Ganancias: {f_pct(r.eps_g,1)} interanual en el último trimestre.")
    if pd.notna(r.fwd_eps_g):
        grow.append(f"Los analistas esperan que la ganancia por acción pase de {fx(r.eps,2)} a {fx(r.fwd_eps,2)} US$ ({f_pct(r.fwd_eps_g)})."
                    + (" La de 12 meses incluye resultados extraordinarios, así que la comparación no indica una caída del negocio." if r.eps_gap_flag else ""))
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


def fetch_hist(sym, series, model="pe"):
    """Valuación histórica propia: P/E y precio/ventas al cierre de cada uno de los últimos ~4 ejercicios."""
    import yfinance as yf
    ser = series.get(sym)
    if ser is None or not len(ser):
        return sym, {}
    for i in range(3):
        try:
            st = yf.Ticker(sym).income_stmt
            break
        except Exception:
            time.sleep(2 + 3 * i)
    else:
        return sym, {}
    if st is None or st.empty:
        return sym, {}
    row = lambda *names: next((st.loc[n] for n in names if n in st.index), None)
    eps, rev, sh = row("Diluted EPS", "Basic EPS"), row("Total Revenue", "Operating Revenue"), row("Diluted Average Shares", "Basic Average Shares")
    eq = None
    if model == "book":
        try:
            bs = yf.Ticker(sym).balance_sheet
            if bs is not None and not bs.empty:
                eq = next((bs.loc[n] for n in ("Common Stock Equity", "Stockholders Equity") if n in bs.index), None)
                if sh is None:
                    sh = next((bs.loc[n] for n in ("Ordinary Shares Number", "Share Issued") if n in bs.index), None)
        except Exception:
            eq = None
    idx = ser.index.tz_localize(None) if getattr(ser.index, "tz", None) is not None else ser.index
    s2 = pd.Series(ser.values, index=idx)
    pes, pss, pbs = [], [], []
    cols = list(st.columns) + ([c for c in eq.index if c not in st.columns] if eq is not None else [])
    for d in cols:
        d = pd.Timestamp(d)
        px = s2[:d]
        if len(px) == 0 or (d - px.index[-1]).days > 10:
            continue
        price = float(px.iloc[-1])
        e = num(eps.get(d)) if eps is not None else None
        if e and e > 0 and 0 < price / e < 200:
            pes.append(price / e)
        rv, n = (num(rev.get(d)) if rev is not None else None), (num(sh.get(d)) if sh is not None else None)
        if rv and n and rv > 0:
            pss.append(price * n / rv)
        if eq is not None and n:
            e2 = num(eq.get(d))
            if e2 and e2 > 0 and 0.05 < price * n / e2 < 50:
                pbs.append(price * n / e2)
    out = {}
    if len(pbs) >= 2:
        out["pb_hist"] = float(np.mean(pbs)); out["pb_hist_n"] = len(pbs)
    if len(pes) >= 3:
        out["pe_hist"] = float(np.mean(pes)); out["pe_hist_n"] = len(pes)
    if len(pss) >= 3:
        out["ps_hist"] = float(np.mean(pss)); out["ps_hist_n"] = len(pss)
    return sym, out


def num(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def audit_and_model(df):
    """Corrige datos conocidos como problemáticos, elige el método de valuación y registra las particularidades."""
    audit = [[] for _ in range(len(df))]
    df = df.reset_index(drop=True)
    for i, r in df.iterrows():
        a = audit[i]
        # Berkshire: Yahoo a veces publica el valor libros por acción clase A (1 A = 1.500 B)
        if r.t == "BRK-B" and pd.notna(r.bvps) and r.bvps > r.p * 50:
            df.at[i, "bvps"] = r.bvps / 1500
            a.append("Valor libros corregido: Yahoo lo informa por acción clase A (1 A = 1.500 B).")
        bv = df.at[i, "bvps"]
        if pd.notna(bv) and bv > 0:
            pb_calc = r.p / bv
            if pd.notna(r.pb) and r.pb > 0 and not (0.5 < pb_calc / r.pb < 2):
                a.append(f"Precio/valor libros de Yahoo ({r.pb:.1f}x) no coincide con el calculado ({pb_calc:.1f}x): se usa el calculado.")
            df.at[i, "pb"] = pb_calc
        elif pd.notna(bv) and bv < 0:
            df.loc[i, [c for c in ["pb", "de", "roe"] if c in df.columns]] = np.nan
            a.append("Patrimonio negativo (por recompras o pérdidas acumuladas): deuda/patrimonio, ROE y P/VL no aplican.")
        # ADRs y datos en otra moneda: el P/E calculado no coincide con el informado
        adr = r.region != "EE.UU."
        if pd.notna(r.eps) and r.eps > 0 and pd.notna(r.pe) and r.pe > 0 and not (0.5 < (r.p / r.eps) / r.pe < 2):
            df.loc[i, ["eps", "pe"]] = np.nan
            a.append("Ganancia por acción en otra moneda o por acción ordinaria (no por ADR): P/E de 12 meses no comparable."
                     if adr else "La ganancia de 12 meses de Yahoo no coincide con su P/E: se descarta.")
        if pd.notna(r.fwd_eps) and r.fwd_eps > 0 and pd.notna(r.fpe) and r.fpe > 0 and not (0.5 < (r.p / r.fwd_eps) / r.fpe < 2):
            df.loc[i, [c for c in ["fwd_eps", "fpe", "peg", "fwd_eps_g"] if c in df.columns]] = np.nan
            a.append("Ganancia esperada en otra moneda: P/E esperado no comparable." if adr
                     else "La ganancia esperada de Yahoo no coincide con su P/E esperado: se descarta.")
        # valores atípicos
        if pd.notna(r.pe) and r.pe > 200:
            a.append(f"P/E histórico atípico ({r.pe:.0f}x): ganancia de 12 meses muy chica.")
        if pd.notna(r.margin) and abs(r.margin) > 1:
            df.at[i, "margin"] = np.nan
            a.append("Margen neto fuera de rango (>100%): descartado.")
        if pd.notna(r.roe) and abs(r.roe) > 2:
            df.at[i, "roe"] = np.nan
            a.append("ROE fuera de rango (>200%): descartado.")
        if r.n_an == 0:
            a.append("Sin cobertura de analistas: no hay precio objetivo de referencia.")
        elif r.n_an < 3:
            a.append(f"Solo {int(r.n_an)} analista(s): su precio objetivo no se usa.")
        elif r.n_an < 5:
            a.append(f"Solo {int(r.n_an)} analistas: el precio objetivo pesa la mitad.")
    df["audit"] = audit
    e, f = df.eps, df.fwd_eps
    gap = ((f / e - 1).abs()).where((e > 0) & (f > 0))
    df["eps_gap"] = gap
    df["eps_gap_flag"] = (gap > EPS_GAP).fillna(False).astype(bool)
    model = []
    for i, r in df.iterrows():
        if r.t in HOLDINGS or r.industry in BOOK_INDUSTRIES:
            mdl = "book" if pd.notna(r.pb) and r.pb > 0 else "na"
        elif str(r.industry).startswith("REIT"):
            mdl = "reit"
        elif not (pd.notna(r.eps) and r.eps > 0) and not (pd.notna(r.fwd_eps) and r.fwd_eps > 0):
            mdl = "sales" if pd.notna(r.ps) and r.ps > 0 else "na"
        elif not (pd.notna(r.fpe) and r.fpe > 0) and not (pd.notna(r.pe) and r.pe > 0):
            mdl = "sales" if pd.notna(r.ps) and r.ps > 0 else "na"
        else:
            mdl = "pe"
        model.append(mdl)
        if mdl != "pe":
            df.at[i, "audit"].append(("Sin método de valuación confiable" if mdl == "na" else f"Valuada por {MODEL_TXT[mdl]}") + {
                "book": ": en bancos, aseguradoras y holdings el P/E no es representativo.",
                "reit": ": en inmobiliarias la depreciación distorsiona la ganancia.",
                "sales": ": la empresa no tiene ganancias positivas o utilizables.",
                "na": ": los datos disponibles no permiten estimar un valor justo."}[mdl])
        if mdl == "book" and r.eps_gap_flag:
            df.at[i, "audit"].append(f"La ganancia de 12 meses y la esperada difieren {r.eps_gap*100:.0f}%: el P/E y el PEG no se usan.")
        if mdl == "pe" and r.eps_gap_flag:
            df.at[i, "audit"].append(f"Ganancia no comparable: la de 12 meses y la esperada difieren {r.eps_gap*100:.0f}%. "
                                     "Para pares se usa precio/ventas y el P/E no entra en el valor justo.")
    df["val_model"] = model
    return df


def build(universe, prices, infos, hist=None):
    hist = hist or {}
    rows = []
    for sym, name, gics, region in universe:
        if sym not in prices:
            continue
        c, v = prices[sym]
        t = technical(c, v)
        if not t:
            continue
        inf = infos.get(sym, {})
        # S&P 500: clasificación oficial GICS (la misma que usan los ETFs sectoriales XLK, XLI, etc.)
        # ADRs (sin GICS): clasificación de Yahoo Finance
        if gics:
            sec = SECTOR_ES.get(gics, gics)
        else:
            ys = inf.get("sector") or ""
            sec = SECTOR_ES.get(ys, ys or "Otros")
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
                         div=num(inf.get("dividendYield")), div5=num(inf.get("fiveYearAvgDividendYield")),
                         ps=num(inf.get("priceToSalesTrailing12Months")), bvps=num(inf.get("bookValue")),
                         pe_hist=hist.get(sym, {}).get("pe_hist"), ps_hist=hist.get(sym, {}).get("ps_hist"),
                         pb_hist=hist.get(sym, {}).get("pb_hist"),
                         hist_n=hist.get(sym, {}).get("pe_hist_n") or hist.get(sym, {}).get("ps_hist_n"),
                         target=target, upside=(target / t["p"] - 1) if target else None,
                         rec=num(inf.get("recommendationMean")),
                         n_an=num(inf.get("numberOfAnalystOpinions")) or 0,
                         industry=inf.get("industry") or "",
                         fcf=num(inf.get("freeCashflow")),
                         earn_days=days_to(inf.get("earningsTimestampStart") or inf.get("earningsTimestamp")),
                         closes=c[-450:]))
    df = pd.DataFrame(rows)
    num_cols = ["pe", "fpe", "peg", "pb", "rev_g", "eps_g", "eps", "fwd_eps", "fwd_eps_g", "roe", "margin",
                "op_margin", "de", "div", "target", "upside", "rec", "mcap", "vol_ratio", "r1m", "r6m", "fcf", "earn_days",
                "div5", "ps", "pe_hist", "ps_hist", "hist_n", "bvps", "pb_hist"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fcf_y"] = df.fcf / df.mcap
    df = audit_and_model(df)
    # ---- giro temprano: 6 señales; solo tiene sentido después de una caída
    spy = technical(*prices[BENCH]) if BENCH in prices else None
    df["et_rs"] = ((df.r1m - spy["r1m"]) > 0) if spy and spy.get("r1m") is not None else False
    et_cols = ["et_hl", "et_div", "et_acc", "et_brk", "et_mom", "et_rs"]
    df["et_n"] = df[et_cols].fillna(False).astype(bool).sum(axis=1)
    df["et_ctx"] = (df.trend_n <= 0) & (df.dd <= -0.05)   # viene de una caída y todavía sin tendencia alcista
    # se calcula ANTES del puntaje fundamental porque lo usa la regla cíclica
    # P/E de pares: mediana de su industria si tiene al menos 5 empresas comparables; si no, de su sector
    fpe_pos = df.fpe.where(df.fpe > 0)
    ind_n = fpe_pos.groupby(df.industry).transform("count")
    ind_med = fpe_pos.groupby(df.industry).transform("median")
    sec_med = fpe_pos.groupby(df.sector).transform("median")
    use_ind = (ind_n >= 5) & (df.industry != "")
    df["sec_fpe"] = ind_med.where(use_ind, sec_med)
    # precio/valor libros y precio/ventas de pares (para los modelos "book" y "sales")
    pb_pos = df.pb.where((df.pb > 0) & (df.val_model == "book"))
    pb_ind_n = pb_pos.groupby(df.industry).transform("count")
    df["peer_pb"] = pb_pos.groupby(df.industry).transform("median").where(pb_ind_n >= 5, float(pb_pos.median()) if pb_pos.notna().any() else np.nan)
    ps_pos = df.ps.where(df.ps > 0)
    ps_ind_n = ps_pos.groupby(df.industry).transform("count")
    df["peer_ps"] = ps_pos.groupby(df.industry).transform("median").where(ps_ind_n >= 5, ps_pos.groupby(df.sector).transform("median"))
    df["peer_lbl"] = np.where(use_ind, "su industria", "su sector")
    df["peer_grp"] = np.where(use_ind, df.industry, df.sector)
    # ---- tres miradas de valuación (negativo = más barata)
    m = df.val_model
    v_pe = df.fpe / df.sec_fpe - 1
    v_pe = v_pe.where(~df.eps_gap_flag, df.ps / df.peer_ps - 1)          # ganancia no comparable: se usa P/V
    df["v_peer"] = np.select([m == "pe", m == "book", m == "sales"],
                             [v_pe, df.pb / df.peer_pb - 1, df.ps / df.peer_ps - 1], np.nan)
    df["v_peer"] = df.v_peer.clip(-0.9, 1.5)
    pe_now = df.pe.where((df.pe > 0) & (m == "pe") & ~df.eps_gap_flag)
    div_rev = (df.div5 / df["div"] - 1).where((df["div"] > 0) & (df.div5 > 0))
    vh = pd.concat([pe_now / df.pe_hist - 1,
                    (df.ps / df.ps_hist - 1).where(m.isin(["pe", "sales"])),
                    (df.pb / df.pb_hist - 1).where(m == "book"),
                    div_rev.where(m != "sales")], axis=1).clip(-0.9, 1.5)
    df["v_hist"] = vh.mean(axis=1, skipna=True).where(vh.notna().any(axis=1))
    df["v_an"] = (-df.upside).where(df.n_an >= 3)
    df["peer_lbl"] = np.where(df.val_model == "book", np.where(pb_ind_n >= 5, "su industria", "bancos y aseguradoras"), df.peer_lbl)
    cyc_ind = df.industry.isin(CYCLICAL)
    peak_eps = (df.eps_g > 0.5) | ((df.fwd_eps < df.eps) & (df.eps > 0))
    df["cyc"] = (cyc_ind & peak_eps & (df.fpe > 0) & (df.fpe < 0.6 * df.sec_fpe)).fillna(False).astype(bool)
    df["one_off"] = ((df.eps_g > 1.0) & (df.rev_g < 0.35)).fillna(False).astype(bool)
    df["at"] = score_tech(df)
    df["af"] = score_fund(df)
    df["score"] = 0.5 * df["at"] + 0.5 * df["af"]
    cheap = pd.concat([df.v_peer < -0.15, df.v_hist < -0.10, df.v_an < -0.20], axis=1)
    dear = pd.concat([df.v_peer > 0.15, df.v_hist > 0.10, df.v_an > 0], axis=1)
    avail = pd.concat([df.v_peer.notna(), df.v_hist.notna(), df.v_an.notna()], axis=1).sum(axis=1)
    df["val_cheap"], df["val_dear"], df["val_n"] = cheap.sum(axis=1), dear.sum(axis=1), avail
    df["val_verdict"] = np.select([df.cyc, df.val_cheap >= 2, df.val_dear >= 2],
                                  ["Pico de ciclo", "Barata", "Cara"], "Neutral")
    df["mom_pct"] = pct_rank((1 + df.r6m) / (1 + df.r1m) - 1).fillna(0)
    df["setup"] = [setup(r) for r in df.itertuples()]
    # Elegibles como oportunidad: tendencia sana, o sobreventa con buenos fundamentos
    elig = (df.trend_n >= 1) | ((df.rsi < 35) & (df.af >= 60))
    df["elig"] = elig
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    top = df[df.elig].head(TOP_N)
    df["rank"] = None
    for i, idx in enumerate(top.index):
        df.at[idx, "rank"] = i + 1
    return df


def build_macro(series):
    out = {}
    for k, name in MACRO.items():
        ser = series.get(k)
        if ser is None or len(ser) < 30:
            continue
        v = float(ser.iloc[-1])
        g = lambda n: float(ser.iloc[-1] / ser.iloc[-1 - n] - 1) if len(ser) > n else None
        d = {"name": name, "v": round(v, 2), "d1": r2(g(1)), "m1": r2(g(21))}
        if k == "^TNX":  # en puntos básicos
            d["d1_bp"] = round((v - float(ser.iloc[-2])) * 100) if len(ser) > 1 else None
            d["m1_bp"] = round((v - float(ser.iloc[-22])) * 100) if len(ser) > 22 else None
        out[k] = d
    return out


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
        fav, txt = trend_text(r)
        (pro if fav else con).append(txt)
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
        pc, cc = correction_texts(r)
        px, cx = cross_texts(r)
        pro += pc + px
        con += cc + cx
        d = {k: r2(getattr(r, k)) for k in ["sector", "etf", "p", "r1d", "r1m", "r3m", "r6m", "r12m", "rsi", "sma50", "sma200",
                                              "hi52", "lo52", "dd", "volat", "macd_h", "macd_cross", "at", "at_own", "rs_score",
                                              "trend", "trend_n", "run200", "dist200", "sma40w", "sma200w", "w_above", "d200w",
                                              "cross", "cross_days", "cross_eta", "gap50", "slope200", "slope200_pct"]}
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


AN_K = 1.0   # sesgo optimista típico de los precios objetivo (se calcula cada día)


def fair_value(r):
    """Valor justo según el método de cada empresa, promediado con el precio objetivo de los analistas."""
    vals, w, src = [], [], []
    def add(v, weight, label):
        if v and pd.notna(v) and 0.3 < v / r.p < 3:
            vals.append(v); w.append(weight); src.append(label)
    m = r.val_model
    if m == "pe" and not r.eps_gap_flag:
        if r.cyc:
            cand = [x for x in [r.eps, r.fwd_eps] if pd.notna(x) and x > 0]
            base = min(cand) * 0.75 if cand else None
        else:
            base = r.fwd_eps if pd.notna(r.fwd_eps) and r.fwd_eps > 0 else None
        if base and pd.notna(r.sec_fpe) and r.sec_fpe > 0:
            add(base * r.sec_fpe, 1, f"P/E de {r.peer_lbl}" + (" con ganancia de ciclo normalizada" if r.cyc else ""))
        if base and pd.notna(r.pe_hist) and not r.cyc:
            add(base * r.pe_hist, 1, "P/E histórico propio")
    elif m == "book" and pd.notna(r.bvps) and r.bvps > 0:
        if pd.notna(r.peer_pb):
            add(r.bvps * r.peer_pb, 1, f"P/VL de {r.peer_lbl}")
        if pd.notna(r.pb_hist):
            add(r.bvps * r.pb_hist, 1, "P/VL histórico propio")
    if m in ("sales",) or (m == "pe" and r.eps_gap_flag):
        rps = r.p / r.ps if pd.notna(r.ps) and r.ps > 0 else None
        if rps and pd.notna(r.peer_ps):
            add(rps * r.peer_ps, 1, f"P/V de {r.peer_lbl}")
        if rps and pd.notna(r.ps_hist):
            add(rps * r.ps_hist, 1, "P/V histórico propio")
    if m == "reit":
        d0, d5 = getattr(r, "div", None), getattr(r, "div5", None)
        if d0 and d5 and pd.notna(d0) and pd.notna(d5) and d0 > 0 and d5 > 0:
            add(r.p * d0 / d5, 1, "dividendo vs su promedio de 5 años")
    if r.n_an >= 3 and pd.notna(r.target):
        add(r.target / AN_K, 1.0 if r.n_an >= 5 else 0.5, "analistas ajustados por optimismo" + ("" if r.n_an >= 5 else ", pocos: peso reducido"))
    if not vals:
        return None, None
    return float(np.average(vals, weights=w)), " + ".join(src)


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
        if zp == "Compra":
            st = "Perdió impulso: volvió de Compra a Vigilar"
        else:
            st = "Girando: el momentum mejora" if dy > 8 else "Barata pero sigue débil"
    elif z == "Mantener":
        st = "Alcanzó su valor justo" if zp == "Compra" else "Sin margen, tendencia alcista"
    else:
        st = "Perdió impulso: pasó de Mantener a Evitar" if zp == "Mantener" else "Cara y perdiendo fuerza"
    return z, zp, st


VM_DOUBT = -0.35   # descuento mayor a esto (ya calibrado) = valuación dudosa


def add_value_momentum(df, series):
    out = {k: [] for k in ["vm", "fair", "fair_raw", "fair_src", "fair_up", "vm_z", "vm_zp", "vm_st", "vm_doubt"]}
    # calibración: solo el precio objetivo de los analistas, que tiene un sesgo optimista sistemático
    # (la mediana del S&P suele quedar 8-12% arriba del precio). Los métodos por múltiplos ya son relativos.
    global AN_K
    us = df[(df.region == "EE.UU.") & (df.n_an >= 5) & df.target.notna()]
    ratios = (us.target / us.p).clip(0.5, 2)
    AN_K = float(ratios.median()) if len(ratios) >= 20 else 1.0
    df.attrs["vm_k"] = AN_K
    raw = [fair_value(r) for r in df.itertuples()]
    for (fair_r, src), r in zip(raw, df.itertuples()):
        fair = fair_r
        ser = series.get(r.t) if series else None
        pts = None
        if fair and ser is not None and len(ser) > 260:
            m, _ = momentum_index(ser)
            w = pd.DataFrame({"c": ser, "m": m}).resample("W-FRI").last().dropna()
            w["m"] = w.m.ewm(span=3, adjust=False).mean()
            w = w.tail(VM_WEEKS + 1)
            if len(w) >= 8:
                pts = [[round(float(c / fair - 1), 4), round(float(mm), 1)] for c, mm in zip(w.c, w.m)]
        xr = (r.p / fair - 1) if fair else None
        # dudosa: descuento extremo, o el modelo dice "barata" mientras las tres miradas dicen "cara"/pico de ciclo
        doubt = bool(xr is not None and (xr < VM_DOUBT or (xr < -0.15 and r.val_verdict in ("Cara", "Pico de ciclo"))))
        if pts:
            z, zp, st = vm_status(pts)
            if doubt:
                st = "Descuento extremo: revisar la valuación" if xr < VM_DOUBT else "Valuación contradictoria: revisar"
        else:
            z = zp = st = None
        out["vm"].append(pts); out["fair"].append(fair); out["fair_raw"].append(fair_r); out["fair_src"].append(src)
        out["vm_doubt"].append(doubt)
        out["fair_up"].append((fair / r.p - 1) if fair else None)
        out["vm_z"].append(z); out["vm_zp"].append(zp); out["vm_st"].append(st)
    for k, v in out.items():
        df[k] = v
    return df


# ---------------------------------------------------------------- veredicto
def no_nan(x):
    """Reemplaza NaN/infinito por null en todo el JSON (el navegador no acepta NaN)."""
    if isinstance(x, dict):
        return {k: no_nan(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [no_nan(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating,)):
        return no_nan(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def today_ar():
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)).date().isoformat()


PREV_DATE = None   # fecha de la rueda contra la que se comparan las novedades


def load_prev_verdicts(path="data.json"):
    """Veredictos del día hábil anterior. Si el screener ya corrió hoy (por ejemplo a mano),
    se reutiliza la base de esa corrida, así las novedades del día no se pierden al volver a correrlo."""
    global PREV_DATE
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("run_date") == today_ar() and isinstance(d.get("verdict_base"), dict):
            PREV_DATE = d.get("verdict_base_date")
            return d["verdict_base"]
        PREV_DATE = d.get("run_date") or str(d.get("generated", ""))[:10] or None
        return {r["t"]: r.get("verdict") for r in d.get("rows", []) if r.get("verdict")}
    except Exception:
        return {}


def add_verdicts(df, prev, mkt_up=True):
    vv = [verdict(r, mkt_up) for r in df.itertuples()]
    df["verdict"] = [v[0] for v in vv]
    df["verdict_why"] = [v[1] for v in vv]
    df["verdict_prev"] = [prev.get(t) for t in df.t]
    df.attrs["verdict_base"] = {k: v for k, v in prev.items() if isinstance(v, str)}
    df["verdict_new"] = [(v[0] == "Comprar" and bool(prev) and prev.get(t) != "Comprar") for v, t in zip(vv, df.t)]
    return df


V_AT, V_AF, V_UP = 65, 60, 0.10      # umbrales: técnico, fundamental, recorrido mínimo a valor justo
# El Golden Cross no habilita "Comprar": en el track récord (4 años, 100 acciones al azar) llegó tarde y
# rindió peor que comprar sin señal. Queda como dato informativo en la ficha.
ENTRY_SETUPS = {"Corrección con señal de rebote", "Ruptura de máximos"}


def verdict(r, mkt_up=True):
    L = lights(r)
    up = r.fair_up if r.fair_up is not None and pd.notna(r.fair_up) else None
    doubt = bool(getattr(r, "vm_doubt", False))
    entry = (r.vm_z == "Compra") or (r.setup in ENTRY_SETUPS)
    checks = [
        (r.at >= V_AT, f"técnico ≥ {V_AT} (tiene {r.at:.0f})"),
        (r.af >= V_AF, f"fundamental ≥ {V_AF} (tiene {r.af:.0f})"),
        (all(L[k] != 0 for k in ("cal", "val", "tec")), "ninguna luz roja en calidad, valuación o técnico"),
        (up is not None and up >= V_UP and not doubt,
         "valuación confiable (hoy está marcada como dudosa)" if doubt else
         f"recorrido a valor justo ≥ {V_UP*100:.0f}%" + (f" (tiene {up*100:+.0f}%)" if up is not None else "")),
        (entry, "zona Compra o señal de entrada (corrección con rebote o ruptura de máximos)"),
    ]
    falta = [t for ok, t in checks if not ok]
    earn_soon = L["evt"] == 0
    up_txt = f"{up*100:+.0f}%".replace(".", ",") if up is not None else "—"
    # 1) Comprar
    if not falta:
        if earn_soon:
            cuando = "hoy" if int(r.earn_days) == 0 else f"en {int(r.earn_days)} días"
            return "Vigilar", f"Cumple todo, pero presenta balance {cuando}: esperar el resultado antes de entrar."
        ent = r.setup if r.setup in ENTRY_SETUPS else "zona Compra"
        if not mkt_up:
            # validado en el track récord: con el mercado bajista las señales fallan más (2022)
            return "Comprar en partes", (f"Cumple los 5 criterios ({ent.lower()}), pero el S&P 500 está debajo de su media de 200 días: "
                                         "mercado bajista. Entrada escalonada de a tercios, con stop.")
        return "Comprar", f"Cumple los 5 criterios: técnico {r.at:.0f}, fundamental {r.af:.0f}, recorrido {up_txt}, {ent.lower()}."
    # 2) Evitar
    if r.vm_z == "Evitar" or r.cyc or (r.val_verdict == "Cara" and r.trend_n <= 0):
        why = "pico de ciclo" if r.cyc else "cara y perdiendo fuerza" if r.vm_z == "Evitar" else "cara y sin tendencia alcista"
        return "Evitar", f"Evitar por ahora: {why}."
    # 3) Mantener
    if r.vm_z == "Mantener" or (up is not None and up < V_UP and r.trend_n >= 1):
        if up is not None and up < 0:
            return "Mantener", f"Ya supera su valor justo estimado (recorrido {up_txt}): si ya se tiene, mantener; no sumar."
        return "Mantener", f"Cerca de su valor justo (recorrido {up_txt}): si ya se tiene, mantener; no sumar."
    # 4) Comprar en partes
    early = (getattr(r, "et_ctx", False) and r.et_n >= 4) or r.setup == "Golden Cross en formación"
    if (r.af >= 55 and L["cal"] != 0 and L["val"] != 0 and up is not None and up >= V_UP and not doubt
            and (early or entry) and not earn_soon):
        how = "giro temprano" if early else "señal de entrada"
        return "Comprar en partes", f"Entrada escalonada por {how}, con stop. Para compra plena falta: {', '.join(falta)}."
    # 5) Vigilar
    return "Vigilar", "Para pasar a Comprar falta: " + ", ".join(falta) + "." + (" Presenta balance en ≤7 días." if earn_soon else "")


SIGNAL_LOG = "signals.json"
SIGNAL_GAP_DAYS = 30     # la misma acción con el mismo veredicto cuenta como señal nueva recién después de 30 días


def save_signals(df, spy, path=SIGNAL_LOG):
    """Registro permanente de cada "Comprar" y "Comprar en partes" real, para medirlos más adelante."""
    try:
        with open(path, encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        log = []
    today = today_ar()
    log = [e for e in log if e.get("date") != today]          # si se corre dos veces el mismo día, se rehace
    last = {}
    for e in log:
        last[(e["t"], e["verdict"])] = max(last.get((e["t"], e["verdict"]), ""), e["date"])
    lim = (dt.date.fromisoformat(today) - dt.timedelta(days=SIGNAL_GAP_DAYS)).isoformat()
    spy_p = spy.get("p") if spy else None
    added = 0
    for r in df[df.verdict.isin(["Comprar", "Comprar en partes"])].itertuples():
        if last.get((r.t, r.verdict), "") > lim:
            continue
        log.append({"date": today, "t": r.t, "name": r.name, "sector": r.sector, "verdict": r.verdict,
                    "setup": r.setup, "p": r2(r.p, 2), "spy": r2(spy_p, 2), "fair": r2(r.fair, 2),
                    "fair_up": r2(r.fair_up), "at": r2(r.at, 0), "af": r2(r.af, 0)})
        added += 1
    with open(path, "w", encoding="utf-8") as f:
        json.dump(no_nan(log), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    print(f"Registro de señales: {added} nuevas hoy · {len(log)} en total")


def save_hist(series, syms, hilo=None, path="hist.json"):
    """Cierres semanales de 5 años de cada papel: los usa la pestaña "Mi cartera" (precio a la fecha de compra
    y máximo alcanzado desde la compra)."""
    cols = {s: series[s] for s in syms if s in series}
    if not cols:
        return
    w = pd.DataFrame(cols).resample("W-FRI").last().dropna(how="all")
    rnd = lambda arr: [None if not np.isfinite(x) else round(float(x), 2) for x in arr]
    out = {"dates": [d.strftime("%Y-%m-%d") for d in w.index], "c": {s: rnd(w[s].values) for s in w.columns}, "h": {}, "l": {}}
    hilo = hilo or {}
    for s in w.columns:
        if s in hilo:
            h_, l_ = hilo[s]
            out["h"][s] = rnd(h_.resample("W-FRI").max().reindex(w.index).values)
            out["l"][s] = rnd(l_.resample("W-FRI").min().reindex(w.index).values)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))
    print("Historial semanal:", len(out["c"]), "papeles,", len(out["dates"]), "semanas")


def audit_summary(df):
    cats = {"Valuada por precio / valor libros": "Bancos, aseguradoras y holdings (valor libros)",
            "Valuada por dividendo y analistas": "REITs (dividendo y analistas)",
            "Valuada por precio / ventas": "Sin ganancias (precio/ventas)",
            "Sin método de valuación": "Sin método de valuación confiable",
            "no coincide con su P/E": "Datos inconsistentes de Yahoo",
            "Patrimonio negativo": "Patrimonio negativo",
            "en otra moneda": "Datos en otra moneda (ADRs)",
            "Ganancia no comparable": "Ganancia no comparable",
            "P/E histórico atípico": "P/E atípico (>200x)",
            "analista": "Pocos o ningún analista",
            "corregido": "Datos corregidos",
            "no coincide": "Datos inconsistentes de Yahoo",
            "fuera de rango": "Valores fuera de rango"}
    out = {}
    for r in df.itertuples():
        for msg in (r.audit or []):
            for k, lbl in cats.items():
                if k in msg:
                    out.setdefault(lbl, []).append(r.t)
                    break
    return {k: sorted(set(v)) for k, v in out.items()}


def to_json(df, n_universe, spy=None, etfs=None, vm_dates=None, macro=None):
    out_rows = []
    keep = ["t", "name", "sector", "region", "p", "mcap", "r1d", "r1m", "r3m", "r6m", "r12m", "rsi", "sma20", "sma50",
            "sma200", "hi52", "lo52", "dd", "volat", "vol_ratio", "macd_h", "macd_cross", "pe", "fpe", "peg", "pb",
            "rev_g", "eps_g", "eps", "fwd_eps", "fwd_eps_g", "roe", "margin", "op_margin", "de", "div", "target",
            "upside", "rec", "n_an", "at", "af", "score", "f_val", "f_gro", "f_qua", "f_sol", "f_ana", "sec_fpe",
            "setup", "elig", "rank", "val_model", "eps_gap", "bvps", "pb_hist", "peer_pb", "peer_ps", "et_hl", "et_div", "et_acc", "et_brk", "et_mom", "et_rs", "et_n", "et_ctx", "et_low", "et_high", "cross", "cross_days", "cross_eta", "gap50", "slope200", "slope200_pct", "peer_lbl", "v_peer", "v_hist", "v_an", "val_verdict", "val_cheap", "val_n",
            "div5", "ps", "pe_hist", "ps_hist", "hist_n", "dd3", "hi3_age", "rebound", "pb_vol", "recovery", "trend", "trend_n", "run200", "dist200", "sma40w", "sma200w", "w_above", "d200w", "industry", "fcf", "fcf_y", "earn_days", "cyc", "one_off"]
    for r in df.itertuples():
        d = {k: r2(getattr(r, k)) for k in keep}
        d["rank"] = int(r.rank) if r.rank is not None and r.rank == r.rank else None
        d["lights"] = lights(r)
        d["audit"] = list(r.audit) if isinstance(r.audit, list) else []
        if getattr(r, "verdict", None):
            vp = r.verdict_prev if isinstance(r.verdict_prev, str) else None
            d["verdict"], d["verdict_why"], d["verdict_prev"], d["verdict_new"] = r.verdict, r.verdict_why, vp, bool(r.verdict_new)
        d["earn_days"] = int(r.earn_days) if pd.notna(r.earn_days) else None
        d["pro"], d["con"], d["grow"] = reasons(r)
        if getattr(r, "vm", None):
            d.update(vm=r.vm, fair=r2(r.fair, 2), fair_raw=r2(r.fair_raw, 2), vm_doubt=bool(r.vm_doubt), fair_src=r.fair_src, fair_up=r2(r.fair_up), vm_z=r.vm_z,
                     vm_zp=r.vm_zp, vm_st=r.vm_st)
        if r.rank is not None:
            d["closes"] = [round(float(x), 2) for x in r.closes]
        out_rows.append(d)
    heat = []
    for sec, g in df.groupby("sector"):
        heat.append({"sector": sec, "n": int(len(g)), "score": r2(g.score.mean(), 1),
                     "r1d": r2(g.r1d.median()), "r1m": r2(g.r1m.median())})
    return {"generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
            "run_date": today_ar(), "verdict_base": df.attrs.get("verdict_base", {}), "verdict_base_date": PREV_DATE,
            "universe": n_universe, "analyzed": int(len(df)), "top_n": TOP_N,
            "method": "AT 50% (tendencia, momentum, RSI/MACD, posición vs máximo) + AF 50% (valuación vs sector, crecimiento, calidad, solidez, analistas).",
            "sectors": sorted(heat, key=lambda h: -h["score"]), "spy": spy, "etfs": etfs or [],
            "rrg_dates": next((e["rrg_dates"] for e in (etfs or []) if e.get("rrg_dates")), []),
            "vm_dates": vm_dates or [],
            "audit_summary": audit_summary(df),
            "market": df.attrs.get("market", {"up": True}),
            "macro": macro or {}, "vm_k": r2(df.attrs.get("vm_k", 1.0), 3),
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

    prices, series, hilo = {}, {}, {}
    dl = syms + [BENCH] + list(SECTOR_ETF.values()) + list(MACRO)
    for i in range(0, len(dl), 100):
        chunk = dl[i:i + 100]
        data = yf.download(chunk, period="5y", interval="1d", auto_adjust=True, group_by="ticker",
                           threads=True, progress=False)
        for s in chunk:
            try:
                sub = data[s] if len(chunk) > 1 else data
                c = sub["Close"].values.astype(float)
                v = sub["Volume"].values.astype(float)
                if np.isfinite(c).sum() >= 210:
                    prices[s] = (c[np.isfinite(c)], v[np.isfinite(c)])
                    series[s] = sub["Close"].dropna()
                    if "High" in sub and "Low" in sub:
                        hilo[s] = (sub["High"], sub["Low"])
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

    global TNX_YIELD
    macro = build_macro(series)
    if macro.get("^TNX"):
        TNX_YIELD = macro["^TNX"]["v"] / 100
    df = build(uni, prices, infos)
    # ---- segunda etapa: valuación histórica propia solo para las candidatas (no duplica la carga sobre Yahoo)
    cand = set(df.sort_values("score", ascending=False).head(150).t)
    cand |= set(df[(df.v_peer < -0.10) | (df.v_an < -0.15)].sort_values("af", ascending=False).head(150).t)
    cand |= set(df[df.val_model.isin(["book", "sales"]) | df.eps_gap_flag | df.one_off | (df.v_peer.abs() > 0.5)].t)
    models = dict(zip(df.t, df.val_model))
    hist = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for sym, h in ex.map(lambda t: fetch_hist(t, series, models.get(t, "pe")), sorted(cand)):
            if h:
                hist[sym] = h
    print("Valuación histórica:", len(hist), "de", len(cand), "candidatas")
    df = build(uni, prices, infos, hist)
    us = df[df.region == "EE.UU."]
    if len(us):
        macro["breadth"] = {"name": "Amplitud S&P 500", "v": round(float((us.p > us.sma200).mean()) * 100, 1),
                            "v50": round(float((us.p > us.sma50).mean()) * 100, 1)}
    df = add_value_momentum(df, series)
    # filtro de mercado: con el S&P 500 debajo de su media de 200 días no hay "Comprar" pleno
    st = technical(*prices[BENCH]) if BENCH in prices else None
    mkt = {"up": bool(st["p"] > st["sma200"]), "p": r2(st["p"], 2), "sma200": r2(st["sma200"], 2),
           "dist": r2(st["p"] / st["sma200"] - 1)} if st else {"up": True}
    df.attrs["market"] = mkt
    print("Filtro de mercado:", "S&P arriba de su media de 200" if mkt["up"] else "S&P DEBAJO de su media de 200 (mercado bajista)")
    df = add_verdicts(df, load_prev_verdicts(), mkt["up"])
    print("Veredictos:", df.verdict.value_counts().to_dict(), "· nuevas Comprar:", list(df[df.verdict_new].t))
    spy, etfs = build_etfs(prices, series)
    vm_dates = [d.strftime("%Y-%m-%d") for d in series[BENCH].resample("W-FRI").last().dropna().tail(VM_WEEKS + 1).index] if BENCH in series else []
    print("Zonas valor/momentum:", df.vm_z.value_counts().to_dict())
    print("ETFs sectoriales:", [(e["etf"], round(e["at"]), e["fase"]) for e in etfs])
    print("Modelos de valuación:", df.val_model.value_counts().to_dict())
    print("== Auditoría de datos ==")
    for k, v in audit_summary(df).items():
        print(f"  {k} ({len(v)}): {', '.join(v[:40])}")
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(no_nan(to_json(df, len(syms), spy, etfs, vm_dates, macro)), f, ensure_ascii=False,
                  separators=(",", ":"), allow_nan=False)
    print("OK data.json:", len(df), "papeles. Top:", list(df[df["rank"].notna()].t))
    save_signals(df, spy)
    save_hist(series, list(df.t), hilo)


if __name__ == "__main__":
    main()
