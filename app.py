"""
Streamlit Stock Scanner — IA/Tech Rush Finder

App gratuite pour analyser des actions : fondamental + momentum + news + watchlist + paper trading.

Fichiers nécessaires pour Streamlit Cloud :
1) app.py                -> ce fichier
2) requirements.txt      -> voir contenu en bas de ce fichier dans les commentaires

Lancer en local :
pip install streamlit yfinance pandas numpy plotly
streamlit run app.py

Attention : ce dashboard n'est pas un conseil financier. Il sert à apprendre, filtrer et comparer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# CONFIG APP
# =========================================================

st.set_page_config(
    page_title="IA/Tech Stock Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_TICKERS = "IREN, OSS, AMD, NVDA, IONQ, SNDK, VECO, PLTR, SMCI, MU, AVGO, TSM, ASML, ARM"


# =========================================================
# HELPERS
# =========================================================

def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except Exception:
        return None


def fmt_num(value) -> str:
    value = safe_float(value)
    if value is None:
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f} T"
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} Md"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f} M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.2f} k"
    return f"{value:.2f}"


def fmt_pct(value, ratio: bool = True) -> str:
    value = safe_float(value)
    if value is None:
        return "N/A"
    if ratio:
        value *= 100
    return f"{value:.2f}%"


def parse_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.replace(";", ",").split(",") if t.strip()]


def color_score(score: int) -> str:
    if score >= 75:
        return "🟢"
    if score >= 55:
        return "🟡"
    if score >= 35:
        return "🟠"
    return "🔴"


def risk_label(score: int) -> str:
    if score >= 8:
        return "🔴 Très élevé"
    if score >= 6:
        return "🟠 Élevé"
    if score >= 4:
        return "🟡 Moyen"
    return "🟢 Modéré"


# =========================================================
# DATA
# =========================================================

@dataclass
class StockData:
    ticker: str
    info: Dict
    hist: pd.DataFrame
    news: List[Dict]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock(ticker: str, period: str = "1y") -> StockData:
    ticker = ticker.strip().upper()
    stock = yf.Ticker(ticker)

    try:
        info = stock.info or {}
    except Exception:
        info = {}

    try:
        hist = stock.history(period=period, interval="1d", auto_adjust=True)
    except Exception:
        hist = pd.DataFrame()

    try:
        news = stock.news or []
    except Exception:
        news = []

    return StockData(ticker=ticker, info=info, hist=hist, news=news)


# =========================================================
# INDICATEURS TECHNIQUES
# =========================================================

def pct_change(series: pd.Series, days: int) -> Optional[float]:
    if series is None or len(series) <= days:
        return None
    old = safe_float(series.iloc[-days - 1])
    new = safe_float(series.iloc[-1])
    if old is None or old == 0 or new is None:
        return None
    return (new / old) - 1


def moving_average(close: pd.Series, window: int) -> Optional[float]:
    if close is None or len(close) < window:
        return None
    return safe_float(close.rolling(window).mean().iloc[-1])


def volume_ratio(volume: pd.Series, window: int = 20) -> Optional[float]:
    if volume is None or len(volume) < window + 1:
        return None
    today = safe_float(volume.iloc[-1])
    avg = safe_float(volume.iloc[-window:].mean())
    if today is None or avg is None or avg == 0:
        return None
    return today / avg


def compute_rsi(close: pd.Series, period: int = 14) -> Optional[float]:
    if close is None or len(close) < period + 2:
        return None
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    last_loss = safe_float(avg_loss.iloc[-1])
    last_gain = safe_float(avg_gain.iloc[-1])
    if last_gain is None or last_loss is None:
        return None
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return 100 - (100 / (1 + rs))


def technical_metrics(hist: pd.DataFrame) -> Dict[str, Optional[float]]:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {}

    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna() if "Volume" in hist.columns else pd.Series(dtype=float)
    price = safe_float(close.iloc[-1]) if len(close) else None
    high_52w = safe_float(close.max()) if len(close) else None
    low_52w = safe_float(close.min()) if len(close) else None

    dist_high = None
    if price is not None and high_52w not in (None, 0):
        dist_high = (price / high_52w) - 1

    return {
        "price": price,
        "return_1d": pct_change(close, 1),
        "return_5d": pct_change(close, 5),
        "return_20d": pct_change(close, 20),
        "return_60d": pct_change(close, 60),
        "return_120d": pct_change(close, 120),
        "return_1y": pct_change(close, min(250, max(len(close) - 2, 1))),
        "volume_ratio": volume_ratio(volume, 20),
        "rsi": compute_rsi(close),
        "ma20": moving_average(close, 20),
        "ma50": moving_average(close, 50),
        "ma200": moving_average(close, 200),
        "high_52w": high_52w,
        "low_52w": low_52w,
        "distance_high": dist_high,
    }


# =========================================================
# SCORES
# =========================================================

def score_fundamental(info: Dict) -> Tuple[int, str, List[str]]:
    score = 0
    flags = []

    market_cap = safe_float(info.get("marketCap"))
    revenue_growth = safe_float(info.get("revenueGrowth"))
    gross_margin = safe_float(info.get("grossMargins"))
    profit_margin = safe_float(info.get("profitMargins"))
    pe = safe_float(info.get("trailingPE"))
    forward_pe = safe_float(info.get("forwardPE"))
    ps = safe_float(info.get("priceToSalesTrailing12Months"))
    cash = safe_float(info.get("totalCash"))
    debt = safe_float(info.get("totalDebt"))
    beta = safe_float(info.get("beta"))

    # Taille
    if market_cap is not None:
        if market_cap >= 100_000_000_000:
            score += 15
        elif market_cap >= 10_000_000_000:
            score += 12
        elif market_cap >= 2_000_000_000:
            score += 8
        elif market_cap >= 300_000_000:
            score += 4
            flags.append("Petite capitalisation")
        else:
            flags.append("Micro-cap")
    else:
        flags.append("Market cap indisponible")

    # Croissance
    if revenue_growth is not None:
        if revenue_growth >= 0.50:
            score += 20
        elif revenue_growth >= 0.25:
            score += 16
        elif revenue_growth >= 0.10:
            score += 10
        elif revenue_growth > 0:
            score += 5
        else:
            flags.append("Croissance faible ou négative")
    else:
        flags.append("Croissance CA indisponible")

    # Marges
    if gross_margin is not None:
        if gross_margin >= 0.60:
            score += 10
        elif gross_margin >= 0.35:
            score += 7
        elif gross_margin >= 0.15:
            score += 3
        else:
            flags.append("Marge brute faible")
    else:
        flags.append("Marge brute indisponible")

    if profit_margin is not None:
        if profit_margin >= 0.20:
            score += 15
        elif profit_margin >= 0.05:
            score += 10
        elif profit_margin > 0:
            score += 5
        else:
            flags.append("Non rentable / marge nette négative")
    else:
        flags.append("Marge nette indisponible")

    # Valorisation
    if pe is not None and pe > 0:
        if pe <= 25:
            score += 15
        elif pe <= 50:
            score += 10
        elif pe <= 90:
            score += 5
            flags.append("P/E élevé")
        else:
            flags.append("P/E très élevé")
    elif forward_pe is not None and forward_pe > 0:
        if forward_pe <= 25:
            score += 12
        elif forward_pe <= 50:
            score += 7
        else:
            flags.append("Forward P/E élevé")
    elif ps is not None and ps > 0:
        if ps <= 5:
            score += 10
        elif ps <= 12:
            score += 5
        else:
            flags.append("Price/Sales très élevé")
    else:
        flags.append("Valorisation difficile à analyser")

    # Bilan
    if cash is not None and debt is not None:
        if debt == 0 or cash >= debt:
            score += 15
        elif cash >= debt * 0.5:
            score += 8
        else:
            flags.append("Dette élevée vs cash")
    else:
        flags.append("Cash/dette indisponibles")

    # Beta
    if beta is not None:
        if beta <= 1.2:
            score += 10
        elif beta <= 1.8:
            score += 5
        elif beta >= 2.5:
            flags.append("Beta très élevé")
    else:
        flags.append("Beta indisponible")

    score = min(score, 100)
    if score >= 75:
        verdict = "Plutôt solide"
    elif score >= 55:
        verdict = "Intéressant mais à surveiller"
    elif score >= 35:
        verdict = "Spéculatif"
    else:
        verdict = "Très risqué"

    return score, verdict, flags


def score_momentum(metrics: Dict[str, Optional[float]]) -> Tuple[int, str, List[str]]:
    score = 0
    signals = []
    price = metrics.get("price")
    r1 = metrics.get("return_1d")
    r5 = metrics.get("return_5d")
    r20 = metrics.get("return_20d")
    vol = metrics.get("volume_ratio")
    ma20 = metrics.get("ma20")
    ma50 = metrics.get("ma50")
    ma200 = metrics.get("ma200")
    rsi = metrics.get("rsi")
    dist_high = metrics.get("distance_high")

    if r1 is not None:
        if r1 >= 0.15:
            score += 10
            signals.append("Rush journalier très fort")
        elif r1 >= 0.05:
            score += 6
            signals.append("Bonne hausse aujourd'hui")
        elif r1 <= -0.08:
            signals.append("Forte baisse aujourd'hui")

    if r5 is not None:
        if r5 >= 0.30:
            score += 15
            signals.append("Momentum 5 jours très fort")
        elif r5 >= 0.12:
            score += 10
            signals.append("Momentum 5 jours positif")
        elif r5 <= -0.15:
            signals.append("Momentum 5 jours négatif")

    if r20 is not None:
        if r20 >= 0.50:
            score += 15
            signals.append("Hausse mensuelle explosive")
        elif r20 >= 0.20:
            score += 10
            signals.append("Tendance mensuelle forte")
        elif r20 <= -0.20:
            signals.append("Tendance mensuelle négative")

    if vol is not None:
        if vol >= 4:
            score += 20
            signals.append("Volume anormal énorme")
        elif vol >= 2:
            score += 12
            signals.append("Volume supérieur à la moyenne")
        elif vol < 0.6:
            signals.append("Volume faible")

    if price is not None and ma20 is not None and ma50 is not None:
        if price > ma20 > ma50:
            score += 12
            signals.append("Prix au-dessus MA20 et MA50")
        elif price > ma20:
            score += 6
            signals.append("Prix au-dessus MA20")
        elif price < ma20 < ma50:
            signals.append("Prix sous MA20 et MA50")

    if price is not None and ma200 is not None:
        if price > ma200:
            score += 8
            signals.append("Prix au-dessus MA200")
        else:
            signals.append("Prix sous MA200")

    if dist_high is not None:
        if dist_high >= -0.03:
            score += 12
            signals.append("Proche du plus haut 52 semaines / possible breakout")
        elif dist_high >= -0.10:
            score += 7
            signals.append("À moins de 10% du plus haut 52 semaines")

    if rsi is not None:
        if 55 <= rsi <= 70:
            score += 8
            signals.append("RSI fort mais pas extrême")
        elif rsi > 80:
            score += 2
            signals.append("RSI très élevé : surchauffe possible")
        elif rsi > 70:
            score += 4
            signals.append("RSI élevé : momentum fort mais risque de correction")
        elif rsi < 35:
            signals.append("RSI faible")

    # Anti-FOMO
    if (r1 is not None and r1 >= 0.25) or (r20 is not None and r20 >= 1.00):
        score -= 8
        signals.append("ANTI-FOMO : hausse très verticale")

    score = max(0, min(score, 100))
    if score >= 75:
        verdict = "Rush très fort"
    elif score >= 55:
        verdict = "Momentum intéressant"
    elif score >= 35:
        verdict = "Momentum moyen"
    else:
        verdict = "Pas de vrai rush"
    return score, verdict, signals


def risk_score(info: Dict, fund_flags: List[str], momo_signals: List[str]) -> Tuple[int, List[str]]:
    risk = 2
    reasons = []
    market_cap = safe_float(info.get("marketCap"))
    profit_margin = safe_float(info.get("profitMargins"))
    beta = safe_float(info.get("beta"))
    ps = safe_float(info.get("priceToSalesTrailing12Months"))

    if market_cap is not None and market_cap < 500_000_000:
        risk += 3
        reasons.append("Micro/small cap")
    elif market_cap is not None and market_cap < 2_000_000_000:
        risk += 2
        reasons.append("Petite capitalisation")

    if profit_margin is not None and profit_margin < 0:
        risk += 2
        reasons.append("Non rentable")

    if beta is not None and beta > 2:
        risk += 1
        reasons.append("Très volatile")

    if ps is not None and ps > 15:
        risk += 1
        reasons.append("Valorisation très élevée vs ventes")

    joined = " ".join(fund_flags + momo_signals).lower()
    if "anti-fomo" in joined or "surchauffe" in joined:
        risk += 2
        reasons.append("Surchauffe / FOMO")
    if "volume anormal" in joined:
        risk += 1
        reasons.append("Volume anormal : catalyseur à vérifier")

    risk = min(risk, 10)
    return risk, reasons


def final_verdict(fund_score: int, momo_score: int, risk: int) -> str:
    if fund_score >= 70 and momo_score >= 55 and risk <= 6:
        return "Bon dossier + momentum intéressant. À étudier sérieusement."
    if fund_score >= 70 and momo_score < 35:
        return "Dossier solide, mais pas de signal rush immédiat. Plutôt long terme / attendre point d'entrée."
    if fund_score < 45 and momo_score >= 65:
        return "Momentum fort mais fondamentaux faibles. Trade spéculatif seulement."
    if risk >= 8:
        return "Très risqué : possible hype/pump. Micro-position ou éviter."
    if momo_score >= 70:
        return "Rush détecté. Vérifie la news exacte avant d'acheter."
    if fund_score >= 55:
        return "Intéressant, mais à comparer avec d'autres actions du secteur."
    return "Pas assez convaincant pour l'instant. Watchlist plutôt qu'achat immédiat."


def suggested_position_size(risk: int) -> str:
    if risk >= 8:
        return "0,5% à 1% max du portefeuille"
    if risk >= 6:
        return "1% à 3% max"
    if risk >= 4:
        return "3% à 5% max"
    return "5% à 10% max"


# =========================================================
# ANALYSE COMPLETE
# =========================================================

def analyze_ticker(ticker: str) -> Dict:
    data = fetch_stock(ticker)
    metrics = technical_metrics(data.hist)
    fund_score, fund_verdict, fund_flags = score_fundamental(data.info)
    momo_score, momo_verdict, momo_signals = score_momentum(metrics)
    risk, risk_reasons = risk_score(data.info, fund_flags, momo_signals)
    verdict = final_verdict(fund_score, momo_score, risk)

    info = data.info
    return {
        "ticker": data.ticker,
        "name": info.get("longName") or info.get("shortName") or data.ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "info": info,
        "hist": data.hist,
        "news": data.news,
        "metrics": metrics,
        "fund_score": fund_score,
        "fund_verdict": fund_verdict,
        "fund_flags": fund_flags,
        "momo_score": momo_score,
        "momo_verdict": momo_verdict,
        "momo_signals": momo_signals,
        "risk": risk,
        "risk_reasons": risk_reasons,
        "verdict": verdict,
        "position_size": suggested_position_size(risk),
    }


# =========================================================
# UI COMPONENTS
# =========================================================

def metric_card(label: str, value: str, help_text: Optional[str] = None):
    st.metric(label=label, value=value, help=help_text)


def plot_price_chart(hist: pd.DataFrame, ticker: str):
    if hist is None or hist.empty:
        st.warning("Pas assez de données pour afficher le graphique.")
        return

    df = hist.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Prix"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))
    fig.update_layout(
        title=f"Graphique {ticker}",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_news(news: List[Dict], max_items: int = 5):
    if not news:
        st.info("Pas de news disponibles via yfinance pour ce ticker.")
        return

    for item in news[:max_items]:
        title = item.get("title") or "Sans titre"
        publisher = item.get("publisher") or "Source inconnue"
        link = item.get("link") or ""
        ts = item.get("providerPublishTime")
        date_str = ""
        if ts:
            try:
                date_str = datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M")
            except Exception:
                date_str = ""
        if link:
            st.markdown(f"**[{title}]({link})**  ")
        else:
            st.markdown(f"**{title}**")
        st.caption(f"{publisher} {date_str}")
        st.divider()


def show_analysis(report: Dict):
    info = report["info"]
    metrics = report["metrics"]

    st.subheader(f"{report['ticker']} — {report['name']}")
    st.caption(f"{report.get('sector') or 'Secteur N/A'} · {report.get('industry') or 'Industrie N/A'} · {report.get('country') or 'Pays N/A'}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Prix", fmt_num(metrics.get("price") or info.get("currentPrice")))
    with c2:
        metric_card("Market cap", fmt_num(info.get("marketCap")))
    with c3:
        metric_card("Score fonda", f"{color_score(report['fund_score'])} {report['fund_score']}/100")
    with c4:
        metric_card("Score rush", f"{color_score(report['momo_score'])} {report['momo_score']}/100")
    with c5:
        metric_card("Risque", f"{risk_label(report['risk'])} ({report['risk']}/10)")

    st.info(f"**Verdict :** {report['verdict']}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Graphique", "🧾 Fondamental", "🚀 Momentum", "📰 News", "🧠 Plan"])

    with tab1:
        plot_price_chart(report["hist"], report["ticker"])

    with tab2:
        a, b, c, d = st.columns(4)
        a.metric("Chiffre d'affaires", fmt_num(info.get("totalRevenue")))
        b.metric("Croissance CA", fmt_pct(info.get("revenueGrowth")))
        c.metric("Marge brute", fmt_pct(info.get("grossMargins")))
        d.metric("Marge nette", fmt_pct(info.get("profitMargins")))

        e, f, g, h = st.columns(4)
        e.metric("P/E", fmt_num(info.get("trailingPE")))
        f.metric("Forward P/E", fmt_num(info.get("forwardPE")))
        g.metric("P/S", fmt_num(info.get("priceToSalesTrailing12Months")))
        h.metric("Beta", fmt_num(info.get("beta")))

        i, j, k = st.columns(3)
        i.metric("Cash", fmt_num(info.get("totalCash")))
        j.metric("Dette", fmt_num(info.get("totalDebt")))
        k.metric("Reco analystes", str(info.get("recommendationKey") or "N/A"))

        st.markdown(f"**Verdict fondamental :** {report['fund_verdict']}")
        if report["fund_flags"]:
            st.warning("Points d'attention : " + ", ".join(report["fund_flags"]))

    with tab3:
        a, b, c, d, e, f = st.columns(6)
        a.metric("1 jour", fmt_pct(metrics.get("return_1d")))
        b.metric("5 jours", fmt_pct(metrics.get("return_5d")))
        c.metric("1 mois", fmt_pct(metrics.get("return_20d")))
        d.metric("3 mois", fmt_pct(metrics.get("return_60d")))
        e.metric("Volume x20j", f"{fmt_num(metrics.get('volume_ratio'))}x")
        f.metric("RSI", fmt_num(metrics.get("rsi")))

        a2, b2, c2, d2 = st.columns(4)
        a2.metric("MA20", fmt_num(metrics.get("ma20")))
        b2.metric("MA50", fmt_num(metrics.get("ma50")))
        c2.metric("MA200", fmt_num(metrics.get("ma200")))
        d2.metric("Distance high 52w", fmt_pct(metrics.get("distance_high")))

        st.markdown(f"**Verdict momentum :** {report['momo_verdict']}")
        if report["momo_signals"]:
            for signal in report["momo_signals"]:
                if "ANTI-FOMO" in signal or "surchauffe" in signal.lower():
                    st.error(signal)
                else:
                    st.write("- " + signal)

    with tab4:
        show_news(report["news"])

    with tab5:
        price = safe_float(metrics.get("price") or info.get("currentPrice"))
        st.markdown("### Plan avant achat")
        st.write("**Taille de position indicative :**", report["position_size"])
        if report["risk_reasons"]:
            st.write("**Pourquoi ce risque :**", ", ".join(report["risk_reasons"]))

        if price:
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Entrée actuelle", fmt_num(price))
            col_b.metric("Stop -10%", fmt_num(price * 0.90))
            col_c.metric("Objectif +20%", fmt_num(price * 1.20))
            col_d.metric("Objectif +35%", fmt_num(price * 1.35))

        st.markdown(
            """
            Avant d'acheter, écris noir sur blanc :
            1. pourquoi ça peut monter ;
            2. quelle news/catalyseur explique le mouvement ;
            3. combien tu acceptes de perdre ;
            4. où tu coupes si tu as tort ;
            5. où tu prends une partie des gains.
            """
        )


def scan_watchlist(tickers: List[str]) -> pd.DataFrame:
    rows = []
    progress = st.progress(0)
    for idx, ticker in enumerate(tickers):
        progress.progress((idx + 1) / max(len(tickers), 1))
        try:
            r = analyze_ticker(ticker)
            info = r["info"]
            m = r["metrics"]
            total = r["fund_score"] * 0.55 + r["momo_score"] * 0.45 - r["risk"] * 1.5
            rows.append({
                "Ticker": r["ticker"],
                "Nom": r["name"],
                "Prix": m.get("price") or info.get("currentPrice"),
                "Market cap": info.get("marketCap"),
                "Secteur": r.get("sector"),
                "Croissance CA": info.get("revenueGrowth"),
                "Marge nette": info.get("profitMargins"),
                "P/S": info.get("priceToSalesTrailing12Months"),
                "Perf 1j": m.get("return_1d"),
                "Perf 5j": m.get("return_5d"),
                "Perf 1m": m.get("return_20d"),
                "Volume x20j": m.get("volume_ratio"),
                "RSI": m.get("rsi"),
                "Fonda": r["fund_score"],
                "Rush": r["momo_score"],
                "Risque": r["risk"],
                "Score total": total,
                "Verdict": r["verdict"],
            })
        except Exception as e:
            rows.append({"Ticker": ticker, "Erreur": str(e)})
    progress.empty()
    df = pd.DataFrame(rows)
    if not df.empty and "Score total" in df.columns:
        df = df.sort_values("Score total", ascending=False)
    return df


def format_watchlist_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Prix", "Market cap", "P/S", "Volume x20j", "RSI", "Score total"]:
        if col in out.columns:
            out[col] = out[col].apply(fmt_num)
    for col in ["Croissance CA", "Marge nette", "Perf 1j", "Perf 5j", "Perf 1m"]:
        if col in out.columns:
            out[col] = out[col].apply(fmt_pct)
    return out


# =========================================================
# PAPER TRADING
# =========================================================

if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []


def paper_trading_ui():
    st.subheader("🧪 Paper trading")
    st.caption("Simulation gratuite : ça ne passe aucun ordre réel. C'est juste pour t'entraîner.")

    with st.form("add_trade"):
        c1, c2, c3, c4 = st.columns(4)
        ticker = c1.text_input("Ticker", value="IREN").upper()
        qty = c2.number_input("Quantité", min_value=0.0, value=1.0, step=1.0)
        entry = c3.number_input("Prix d'entrée", min_value=0.0, value=0.0, step=0.01)
        note = c4.text_input("Note", value="test")
        submitted = st.form_submit_button("Ajouter le trade fictif")

    if submitted and ticker and qty > 0:
        if entry <= 0:
            try:
                entry = safe_float(analyze_ticker(ticker)["metrics"].get("price")) or 0
            except Exception:
                entry = 0
        st.session_state.paper_trades.append({
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Ticker": ticker,
            "Quantité": qty,
            "Prix entrée": entry,
            "Note": note,
        })
        st.success("Trade fictif ajouté.")

    if not st.session_state.paper_trades:
        st.info("Aucun trade fictif pour l'instant.")
        return

    rows = []
    for t in st.session_state.paper_trades:
        try:
            current = safe_float(analyze_ticker(t["Ticker"])["metrics"].get("price"))
        except Exception:
            current = None
        entry = safe_float(t["Prix entrée"])
        qty = safe_float(t["Quantité"])
        pnl = None
        pnl_pct = None
        if current is not None and entry is not None and qty is not None and entry > 0:
            pnl = (current - entry) * qty
            pnl_pct = (current / entry) - 1
        rows.append({**t, "Prix actuel": current, "P/L": pnl, "P/L %": pnl_pct})

    df = pd.DataFrame(rows)
    st.dataframe(format_watchlist_df(df), use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger le paper trading en CSV", data=csv, file_name="paper_trading.csv", mime="text/csv")

    if st.button("Effacer les trades fictifs"):
        st.session_state.paper_trades = []
        st.rerun()


# =========================================================
# MAIN UI
# =========================================================

st.title("📈 IA/Tech Stock Scanner")
st.caption("Analyse d'actions : fondamentaux, momentum, rushs, news, risque et paper trading.")

with st.sidebar:
    st.header("Réglages")
    st.markdown("**Mode d'emploi rapide**")
    st.write("1. Mets une watchlist")
    st.write("2. Clique sur Scanner")
    st.write("3. Ouvre une action intéressante")
    st.write("4. Vérifie la news avant achat")
    st.divider()
    raw_tickers = st.text_area("Watchlist", value=DEFAULT_TICKERS, height=160)
    tickers = parse_tickers(raw_tickers)
    selected_ticker = st.selectbox("Action à analyser", tickers if tickers else ["IREN"])
    st.divider()
    st.warning("Aucun score ne prédit le futur. Le but est d'éviter d'acheter au hasard.")

main_tab, detail_tab, paper_tab, learn_tab = st.tabs(["🔍 Scanner watchlist", "📌 Analyse détaillée", "🧪 Paper trading", "📚 Méthode"])

with main_tab:
    st.subheader("Scanner ta watchlist")
    st.write("Compare plusieurs actions pour repérer les dossiers solides, les rushs et les pièges FOMO.")

    if st.button("Scanner maintenant", type="primary"):
        if not tickers:
            st.error("Ajoute au moins un ticker.")
        else:
            df = scan_watchlist(tickers)
            st.session_state["last_scan"] = df

    if "last_scan" in st.session_state:
        df = st.session_state["last_scan"]
        st.dataframe(format_watchlist_df(df), use_container_width=True, height=520)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger le scan en CSV", data=csv, file_name="watchlist_scan.csv", mime="text/csv")

        st.markdown("### Comment lire le tableau")
        st.write("- **Fonda** : qualité fondamentale rapide.")
        st.write("- **Rush** : force du momentum court terme.")
        st.write("- **Risque** : volatilité / micro-cap / FOMO / non-rentabilité.")
        st.write("- **Score total** : tri global, pas une recommandation d'achat.")

with detail_tab:
    st.subheader("Analyse détaillée")
    ticker_input = st.text_input("Ticker", value=selected_ticker).upper()
    if st.button("Analyser cette action", type="primary"):
        with st.spinner("Analyse en cours..."):
            st.session_state["detail_report"] = analyze_ticker(ticker_input)

    if "detail_report" in st.session_state:
        show_analysis(st.session_state["detail_report"])
    else:
        st.info("Choisis un ticker puis clique sur Analyser.")

with paper_tab:
    paper_trading_ui()

with learn_tab:
    st.subheader("La méthode à suivre")
    st.markdown(
        """
        ### 1. Ne confonds pas long terme et rush
        - **Long terme** : entreprise solide, croissance durable, bilan acceptable.
        - **Rush/trade** : catalyseur récent, gros volume, cassure technique.

        ### 2. Le meilleur signal
        Le meilleur setup, c'est souvent :
        **fondamentaux corrects + catalyseur clair + volume anormal + pas trop de FOMO.**

        ### 3. Les pièges
        - Acheter après +50% sans comprendre la news.
        - Mettre trop sur une micro-cap.
        - Garder un trade raté en disant soudainement “c'est long terme”.
        - Acheter une boîte non rentable sans regarder cash/dette/dilution.

        ### 4. Règle avant achat
        Écris toujours :
        - pourquoi j'achète ;
        - quelle news explique le mouvement ;
        - mon prix maximum ;
        - mon stop mental ;
        - mon objectif ;
        - combien je peux perdre.
        """
    )


# =========================================================
# requirements.txt à créer séparément
# =========================================================
# streamlit
# yfinance
# pandas
# numpy
# plotly
