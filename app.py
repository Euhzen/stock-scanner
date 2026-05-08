"""
IA/Tech Stock Scanner — Streamlit Premium UI

App gratuite pour analyser des actions : fondamentaux + momentum + news récentes + watchlist + paper trading.

Fichiers nécessaires :
1) app.py
2) requirements.txt avec :
   streamlit
   yfinance
   pandas
   numpy
   plotly
   feedparser

Lancer :
streamlit run app.py

Disclaimer : outil d'analyse/éducation, pas un conseil financier.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import feedparser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Stock Radar",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_TICKERS = "IREN, OSS, AMD, NVDA, IONQ, SNDK, VECO, PLTR, SMCI, MU, AVGO, TSM, ASML, ARM, BBAI, SOUN, AEHR, POET"
PRESET_WATCHLISTS = {
    "IA / Semi-conducteurs": "NVDA, AMD, AVGO, TSM, ASML, ARM, MU, SMCI, VECO, AEHR, MRVL, QCOM, INTC",
    "IA spéculatif / small caps": "IREN, OSS, IONQ, BBAI, SOUN, POET, LAES, RGTI, QBTS, SERV, LUNR",
    "Software / Cloud / Cyber": "MSFT, GOOGL, AMZN, META, PLTR, SNOW, CRWD, PANW, NET, DDOG, NOW, MDB",
    "Belgique / Europe": "ABI.BR, UCB.BR, MELE.BR, EVS.BR, XIOR.BR, ASML.AS, MC.PA, AI.PA, SAP.DE",
    "Ma watchlist IA de base": DEFAULT_TICKERS,
}


# =========================================================
# STYLE
# =========================================================

def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --bg-card: rgba(255, 255, 255, 0.075);
            --bg-card-2: rgba(255, 255, 255, 0.045);
            --border: rgba(255, 255, 255, 0.13);
            --text-soft: rgba(255, 255, 255, 0.70);
            --green: #22c55e;
            --yellow: #eab308;
            --orange: #f97316;
            --red: #ef4444;
            --blue: #38bdf8;
            --purple: #a78bfa;
        }

        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 55%, #020617 100%);
        }

        .hero {
            padding: 1.4rem 1.55rem;
            border-radius: 24px;
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, .25), transparent 32%),
                radial-gradient(circle at bottom right, rgba(167, 139, 250, .20), transparent 35%),
                linear-gradient(135deg, rgba(15, 23, 42, .96), rgba(2, 6, 23, .96));
            border: 1px solid var(--border);
            box-shadow: 0 18px 50px rgba(0, 0, 0, .24);
            margin-bottom: 1.1rem;
        }

        .hero-title {
            font-size: 2.15rem;
            font-weight: 850;
            letter-spacing: -0.04em;
            margin: 0;
            color: white;
        }

        .hero-subtitle {
            color: rgba(255,255,255,.74);
            margin-top: .42rem;
            font-size: 1rem;
            max-width: 900px;
        }

        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: .5rem;
            margin-top: 1rem;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: .35rem;
            padding: .38rem .72rem;
            border-radius: 999px;
            background: rgba(255,255,255,.08);
            border: 1px solid rgba(255,255,255,.13);
            color: rgba(255,255,255,.82);
            font-size: .86rem;
        }

        .card {
            padding: 1.05rem 1.1rem;
            border-radius: 20px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            box-shadow: 0 10px 30px rgba(0, 0, 0, .14);
            min-height: 106px;
        }

        .card-small {
            padding: .85rem .95rem;
            border-radius: 18px;
            background: var(--bg-card-2);
            border: 1px solid var(--border);
        }

        .kpi-label {
            color: rgba(255,255,255,.62);
            font-size: .76rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin-bottom: .25rem;
        }

        .kpi-value {
            color: white;
            font-weight: 800;
            font-size: 1.55rem;
            letter-spacing: -0.03em;
            margin-bottom: .25rem;
        }

        .kpi-help {
            color: rgba(255,255,255,.58);
            font-size: .82rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            padding: .22rem .55rem;
            border-radius: 999px;
            font-size: .78rem;
            font-weight: 700;
            border: 1px solid rgba(255,255,255,.12);
        }
        .badge-green { background: rgba(34,197,94,.15); color: #86efac; }
        .badge-yellow { background: rgba(234,179,8,.15); color: #fde68a; }
        .badge-orange { background: rgba(249,115,22,.15); color: #fdba74; }
        .badge-red { background: rgba(239,68,68,.15); color: #fca5a5; }
        .badge-blue { background: rgba(56,189,248,.15); color: #7dd3fc; }
        .badge-purple { background: rgba(167,139,250,.15); color: #c4b5fd; }

        .verdict-box {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(56,189,248,.11), rgba(167,139,250,.09));
            border: 1px solid rgba(125,211,252,.22);
            margin: .5rem 0 1rem 0;
            color: rgba(255,255,255,.86);
        }

        .warning-box {
            padding: .9rem 1rem;
            border-radius: 16px;
            background: rgba(239, 68, 68, .10);
            border: 1px solid rgba(239, 68, 68, .25);
            color: rgba(255,255,255,.86);
        }

        .soft-box {
            padding: .9rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.10);
            color: rgba(255,255,255,.80);
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -.02em;
            margin: .4rem 0 .6rem 0;
        }

        .news-card {
            padding: .92rem 1rem;
            border-radius: 16px;
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.10);
            margin-bottom: .65rem;
        }

        .news-title {
            font-weight: 760;
            color: white;
            text-decoration: none;
        }

        .news-meta {
            margin-top: .35rem;
            color: rgba(255,255,255,.58);
            font-size: .8rem;
        }

        .stDataFrame {
            border-radius: 18px;
            overflow: hidden;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.10);
            padding: .75rem .9rem;
            border-radius: 16px;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.15rem;
            font-weight: 800;
        }

        .footer-note {
            color: rgba(255,255,255,.52);
            font-size: .8rem;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def fmt_num(value, decimals: int = 2) -> str:
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
    return f"{value:.{decimals}f}"


def fmt_pct(value, ratio: bool = True) -> str:
    value = safe_float(value)
    if value is None:
        return "N/A"
    if ratio:
        value *= 100
    return f"{value:+.2f}%"


def parse_tickers(raw: str) -> List[str]:
    return [t.strip().upper() for t in raw.replace(";", ",").replace("\n", ",").split(",") if t.strip()]


def score_class(score: int) -> str:
    if score >= 75:
        return "green"
    if score >= 55:
        return "yellow"
    if score >= 35:
        return "orange"
    return "red"


def badge(text: str, color: str = "blue") -> str:
    return f'<span class="badge badge-{color}">{text}</span>'


def kpi_card(label: str, value: str, helper: str = ""):
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📈 AI Stock Radar</div>
            <div class="hero-subtitle">
                Scanner d'actions IA/tech : fondamentaux, momentum, news récentes, risque, plan d'entrée et paper trading.
                Le but n'est pas de deviner le futur, mais de repérer les bons setups et d'éviter le FOMO.
            </div>
            <div class="pill-row">
                <span class="pill">🚀 Rush detector</span>
                <span class="pill">🧾 Fondamentaux</span>
                <span class="pill">📰 News 24h</span>
                <span class="pill">⚠️ Anti-FOMO</span>
                <span class="pill">🧪 Paper trading</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _parse_feed_time(entry) -> Optional[datetime]:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime.fromtimestamp(time.mktime(parsed), tz=timezone.utc)
        except Exception:
            return None
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_google_news_rss(ticker: str, company_name: str = "", max_age_hours: int = 24) -> List[Dict]:
    ticker = ticker.strip().upper()
    company_name = (company_name or "").strip()

    if max_age_hours <= 24:
        when_filter = "when:1d"
    elif max_age_hours <= 72:
        when_filter = "when:3d"
    else:
        when_filter = "when:7d"

    if company_name and company_name.upper() != ticker:
        query = f'("{company_name}" OR {ticker}) stock OR earnings OR revenue OR shares {when_filter}'
    else:
        query = f'{ticker} stock OR earnings OR revenue OR shares {when_filter}'

    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    parsed = feedparser.parse(url)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    results = []

    for entry in parsed.entries[:35]:
        published_dt = _parse_feed_time(entry)
        if published_dt is None or published_dt < cutoff:
            continue
        age_hours = (datetime.now(timezone.utc) - published_dt).total_seconds() / 3600
        results.append({
            "title": entry.get("title", "Sans titre"),
            "publisher": entry.get("source", {}).get("title", "Google News"),
            "link": entry.get("link", ""),
            "published": published_dt.strftime("%d/%m/%Y %H:%M UTC"),
            "published_dt": published_dt.isoformat(),
            "age_hours": age_hours,
            "source_type": "Google News RSS",
        })

    results.sort(key=lambda x: x.get("published_dt", ""), reverse=True)
    return results


def normalize_yfinance_news(news: List[Dict], max_age_hours: int = 24) -> List[Dict]:
    out = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

    for item in news or []:
        content = item.get("content") if isinstance(item.get("content"), dict) else {}
        title = item.get("title") or content.get("title") or "Sans titre"
        publisher = item.get("publisher") or content.get("provider", {}).get("displayName") or "Yahoo Finance"
        link = item.get("link") or content.get("canonicalUrl", {}).get("url") or ""
        ts = item.get("providerPublishTime") or content.get("pubDate")

        published_dt = None
        if isinstance(ts, (int, float)):
            try:
                published_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            except Exception:
                published_dt = None
        elif isinstance(ts, str):
            try:
                published_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                published_dt = None

        if published_dt is None or published_dt < cutoff:
            continue

        age_hours = (datetime.now(timezone.utc) - published_dt).total_seconds() / 3600
        out.append({
            "title": title,
            "publisher": publisher,
            "link": link,
            "published": published_dt.strftime("%d/%m/%Y %H:%M UTC"),
            "published_dt": published_dt.isoformat(),
            "age_hours": age_hours,
            "source_type": "Yahoo/yfinance",
        })
    return out


def get_combined_news(ticker: str, company_name: str, yfinance_news: List[Dict], max_age_hours: int = 24) -> List[Dict]:
    combined = []
    combined.extend(normalize_yfinance_news(yfinance_news, max_age_hours=max_age_hours))
    combined.extend(fetch_google_news_rss(ticker, company_name, max_age_hours=max_age_hours))

    seen = set()
    unique = []
    for item in combined:
        key = (item.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)

    unique.sort(key=lambda x: x.get("published_dt", ""), reverse=True)
    return unique[:12]


# =========================================================
# TECHNICALS
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

    if cash is not None and debt is not None:
        if debt == 0 or cash >= debt:
            score += 15
        elif cash >= debt * 0.5:
            score += 8
        else:
            flags.append("Dette élevée vs cash")
    else:
        flags.append("Cash/dette indisponibles")

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
        reasons.append("Valorisation élevée vs ventes")

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
        return "Bon dossier + momentum intéressant. À étudier sérieusement, surtout si la news est solide."
    if fund_score >= 70 and momo_score < 35:
        return "Dossier solide, mais pas de signal rush immédiat. Plutôt long terme / attendre un meilleur point d'entrée."
    if fund_score < 45 and momo_score >= 65:
        return "Momentum fort mais fondamentaux faibles. Trade spéculatif seulement."
    if risk >= 8:
        return "Très risqué : possible hype/pump. Micro-position ou éviter."
    if momo_score >= 70:
        return "Rush détecté. Vérifie la news exacte avant d'acheter."
    if fund_score >= 55:
        return "Intéressant, mais à comparer avec d'autres actions du même secteur."
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
# ANALYSE
# =========================================================

def analyze_ticker(ticker: str) -> Dict:
    data = fetch_stock(ticker)
    info = data.info
    metrics = technical_metrics(data.hist)
    fund_score, fund_verdict, fund_flags = score_fundamental(info)
    momo_score, momo_verdict, momo_signals = score_momentum(metrics)
    risk, risk_reasons = risk_score(info, fund_flags, momo_signals)
    verdict = final_verdict(fund_score, momo_score, risk)

    company_name = info.get("longName") or info.get("shortName") or data.ticker

    return {
        "ticker": data.ticker,
        "name": company_name,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "info": info,
        "hist": data.hist,
        "news": get_combined_news(data.ticker, company_name, data.news, max_age_hours=24),
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
# CHARTS
# =========================================================

def plot_price_chart(hist: pd.DataFrame, ticker: str):
    if hist is None or hist.empty:
        st.warning("Pas assez de données pour afficher le graphique.")
        return

    df = hist.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Prix",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20", line=dict(width=1.6, color="#38bdf8")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50", line=dict(width=1.6, color="#a78bfa")))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200", line=dict(width=1.7, color="#facc15")))
    fig.update_layout(
        title=f"{ticker} — prix + moyennes mobiles",
        height=520,
        margin=dict(l=20, r=20, t=55, b=20),
        hovermode="x unified",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_score_gauge(label: str, score: int, max_score: int = 100):
    if score >= 75:
        color = "#22c55e"
    elif score >= 55:
        color = "#eab308"
    elif score >= 35:
        color = "#f97316"
    else:
        color = "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": f"/{max_score}"},
        title={"text": label},
        gauge={
            "axis": {"range": [0, max_score]},
            "bar": {"color": color},
            "bgcolor": "rgba(255,255,255,.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(239,68,68,.18)"},
                {"range": [35, 55], "color": "rgba(249,115,22,.18)"},
                {"range": [55, 75], "color": "rgba(234,179,8,.18)"},
                {"range": [75, 100], "color": "rgba(34,197,94,.18)"},
            ],
        },
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=12, r=12, t=35, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# UI ANALYSE DETAILLEE
# =========================================================

def show_news(news: List[Dict], max_items: int = 8):
    if not news:
        st.markdown(
            """
            <div class="soft-box">
                Aucune news trouvée dans les dernières 24h via les sources gratuites. C'est plutôt bon signe si tu veux éviter les vieilles infos, mais ça veut aussi dire qu'il faut vérifier manuellement si tu vois un gros mouvement.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for item in news[:max_items]:
        title = item.get("title") or "Sans titre"
        publisher = item.get("publisher") or "Source inconnue"
        link = item.get("link") or ""
        published = item.get("published") or ""
        source_type = item.get("source_type") or "News"
        age = item.get("age_hours")
        age_txt = f" · il y a {age:.1f}h" if isinstance(age, (int, float)) else ""
        if link:
            title_html = f'<a class="news-title" href="{link}" target="_blank">{title}</a>'
        else:
            title_html = f'<span class="news-title">{title}</span>'
        st.markdown(
            f"""
            <div class="news-card">
                {title_html}
                <div class="news-meta">{publisher} · {source_type} · {published}{age_txt}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_analysis(report: Dict):
    info = report["info"]
    metrics = report["metrics"]
    price = metrics.get("price") or info.get("currentPrice")

    col_title, col_badges = st.columns([2.2, 1])
    with col_title:
        st.markdown(f"### {report['ticker']} — {report['name']}")
        st.caption(f"{report.get('sector') or 'Secteur N/A'} · {report.get('industry') or 'Industrie N/A'} · {report.get('country') or 'Pays N/A'}")
    with col_badges:
        st.markdown(
            f"{badge('Fonda ' + str(report['fund_score']) + '/100', score_class(report['fund_score']))} "
            f"{badge('Rush ' + str(report['momo_score']) + '/100', score_class(report['momo_score']))} "
            f"{badge('Risque ' + str(report['risk']) + '/10', 'red' if report['risk'] >= 8 else 'orange' if report['risk'] >= 6 else 'yellow' if report['risk'] >= 4 else 'green')}",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Prix", fmt_num(price), "Dernier cours disponible")
    with c2:
        kpi_card("Market cap", fmt_num(info.get("marketCap")), "Taille de l'entreprise")
    with c3:
        kpi_card("Perf 5j", fmt_pct(metrics.get("return_5d")), "Momentum court terme")
    with c4:
        kpi_card("Volume", f"{fmt_num(metrics.get('volume_ratio'))}x", "Volume vs moyenne 20j")
    with c5:
        kpi_card("RSI", fmt_num(metrics.get("rsi")), "Surchauffe si très haut")

    st.markdown(f"<div class='verdict-box'><b>Verdict :</b> {report['verdict']}</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Graphique", "🎯 Scores", "🧾 Fondamental", "🚀 Momentum", "📰 News 24h", "🧠 Plan"])

    with tab1:
        plot_price_chart(report["hist"], report["ticker"])

    with tab2:
        g1, g2, g3 = st.columns(3)
        with g1:
            plot_score_gauge("Fondamental", report["fund_score"])
        with g2:
            plot_score_gauge("Rush", report["momo_score"])
        with g3:
            plot_score_gauge("Risque", report["risk"], 10)

        st.markdown("<div class='section-title'>Lecture rapide</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="soft-box">
            <b>Score fondamental :</b> qualité business/valorisation/bilan.<br>
            <b>Score rush :</b> momentum, volume, proximité des plus hauts, tendance.<br>
            <b>Risque :</b> micro-cap, non-rentabilité, volatilité, surchauffe/FOMO.<br><br>
            <b>Taille indicative :</b> {report['position_size']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab3:
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

    with tab4:
        a, b, c, d, e, f = st.columns(6)
        a.metric("1 jour", fmt_pct(metrics.get("return_1d")))
        b.metric("5 jours", fmt_pct(metrics.get("return_5d")))
        c.metric("1 mois", fmt_pct(metrics.get("return_20d")))
        d.metric("3 mois", fmt_pct(metrics.get("return_60d")))
        e.metric("6 mois", fmt_pct(metrics.get("return_120d")))
        f.metric("1 an", fmt_pct(metrics.get("return_1y")))

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

    with tab5:
        show_news(report["news"])

    with tab6:
        st.markdown("<div class='section-title'>Plan avant achat</div>", unsafe_allow_html=True)
        if report["risk_reasons"]:
            st.markdown(f"<div class='warning-box'><b>Sources de risque :</b> {', '.join(report['risk_reasons'])}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='soft-box'>Pas de gros signal de risque détecté automatiquement, mais vérifie quand même les news et les résultats.</div>", unsafe_allow_html=True)

        if price:
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Prix actuel", fmt_num(price))
            col_b.metric("Repère stop -10%", fmt_num(price * 0.90))
            col_c.metric("Objectif +20%", fmt_num(price * 1.20))
            col_d.metric("Objectif +35%", fmt_num(price * 1.35))

        st.markdown(
            """
            <div class="soft-box">
            <b>Checklist obligatoire avant achat :</b><br>
            1. Je comprends ce que fait l'entreprise.<br>
            2. Je sais quelle news/catalyseur explique le mouvement.<br>
            3. Je sais combien je peux perdre.<br>
            4. Je sais à quel prix je coupe si j'ai tort.<br>
            5. Je n'achète pas uniquement parce que c'est vert.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# WATCHLIST
# =========================================================

def scan_watchlist(tickers: List[str]) -> pd.DataFrame:
    rows = []
    progress = st.progress(0)
    for idx, ticker in enumerate(tickers):
        progress.progress((idx + 1) / max(len(tickers), 1), text=f"Analyse de {ticker}...")
        try:
            r = analyze_ticker(ticker)
            info = r["info"]
            m = r["metrics"]
            total = r["fund_score"] * 0.55 + r["momo_score"] * 0.45 - r["risk"] * 1.5
            setup = "Long terme" if r["fund_score"] >= 70 and r["momo_score"] < 45 else "Rush" if r["momo_score"] >= 60 else "Spéculatif" if r["risk"] >= 7 else "À surveiller"
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
                "Setup": setup,
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


def show_rankings(df: pd.DataFrame):
    if df.empty or "Ticker" not in df.columns:
        return
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("#### 🚀 Top rush")
        cols = ["Ticker", "Rush", "Perf 5j", "Volume x20j", "Risque"]
        st.dataframe(format_watchlist_df(df.sort_values("Rush", ascending=False).head(5)[cols]), use_container_width=True, hide_index=True)
    with r2:
        st.markdown("#### 🧾 Top fondamental")
        cols = ["Ticker", "Fonda", "Croissance CA", "Marge nette", "Risque"]
        st.dataframe(format_watchlist_df(df.sort_values("Fonda", ascending=False).head(5)[cols]), use_container_width=True, hide_index=True)
    with r3:
        st.markdown("#### ⚠️ Plus risquées")
        cols = ["Ticker", "Risque", "Rush", "Perf 1j", "Verdict"]
        st.dataframe(format_watchlist_df(df.sort_values("Risque", ascending=False).head(5)[cols]), use_container_width=True, hide_index=True)


# =========================================================
# PAPER TRADING
# =========================================================

if "paper_trades" not in st.session_state:
    st.session_state.paper_trades = []


def paper_trading_ui():
    st.markdown("### 🧪 Paper trading")
    st.caption("Simulation gratuite : aucun ordre réel n'est passé. C'est fait pour t'entraîner.")

    with st.form("add_trade", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        ticker = c1.text_input("Ticker", value="IREN").upper()
        qty = c2.number_input("Quantité", min_value=0.0, value=1.0, step=1.0)
        entry = c3.number_input("Prix d'entrée", min_value=0.0, value=0.0, step=0.01)
        note = c4.text_input("Pourquoi ce trade fictif ?", value="test setup")
        submitted = st.form_submit_button("Ajouter le trade fictif", type="primary")

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
    total_entry = 0.0
    total_value = 0.0
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
            total_entry += entry * qty
            total_value += current * qty
        rows.append({**t, "Prix actuel": current, "Valeur actuelle": current * qty if current and qty else None, "P/L": pnl, "P/L %": pnl_pct})

    if total_entry > 0:
        p1, p2, p3 = st.columns(3)
        p1.metric("Montant fictif investi", fmt_num(total_entry))
        p2.metric("Valeur actuelle", fmt_num(total_value))
        p3.metric("Performance", fmt_pct((total_value / total_entry) - 1))

    df = pd.DataFrame(rows)
    st.dataframe(format_watchlist_df(df), use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger le paper trading en CSV", data=csv, file_name="paper_trading.csv", mime="text/csv")

    if st.button("Effacer les trades fictifs"):
        st.session_state.paper_trades = []
        st.rerun()


# =========================================================
# WATCHLISTS PERSONNALISÉES
# =========================================================

if "custom_watchlists" not in st.session_state:
    st.session_state.custom_watchlists = {}

if "builder_tickers" not in st.session_state:
    st.session_state.builder_tickers = []


def unique_tickers(tickers: List[str]) -> List[str]:
    """Nettoie une liste de tickers en gardant l'ordre et en supprimant les doublons."""
    seen = set()
    cleaned = []
    for t in tickers:
        t = str(t).strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        cleaned.append(t)
    return cleaned


def tickers_to_text(tickers: List[str]) -> str:
    return ", ".join(unique_tickers(tickers))


def get_all_watchlists() -> Dict[str, str]:
    """Combine les watchlists intégrées, la liste en cours et les watchlists créées dans l'app."""
    builder_text = tickers_to_text(st.session_state.builder_tickers)
    builder = {"🧺 Liste en cours": builder_text}
    custom = {f"⭐ {name}": tickers for name, tickers in st.session_state.custom_watchlists.items()}
    return {**PRESET_WATCHLISTS, **builder, **custom}


def watchlist_manager_ui(current_tickers: str = ""):
    """Interface pour créer une watchlist au fur et à mesure directement dans l'app."""
    with st.expander("⭐ Créer / gérer mes watchlists", expanded=True):
        st.caption(
            "Tu peux ajouter des tickers un par un dans une liste en cours, puis l'enregistrer. "
            "Sur Streamlit gratuit, télécharge le JSON pour la garder même après redémarrage de l'app."
        )

        st.markdown("#### 🧺 Liste en cours")

        add_col1, add_col2 = st.columns([2, 1])
        with add_col1:
            ticker_to_add = st.text_input("Ajouter une action", placeholder="Ex : IREN, AMD, BBAI", key="ticker_to_add")
        with add_col2:
            st.write("")
            if st.button("➕ Ajouter", use_container_width=True):
                new_items = parse_tickers(ticker_to_add)
                if not new_items:
                    st.error("Entre au moins un ticker.")
                else:
                    st.session_state.builder_tickers = unique_tickers(st.session_state.builder_tickers + new_items)
                    st.session_state.selected_watchlist = "🧺 Liste en cours"
                    st.session_state.tickers_text = tickers_to_text(st.session_state.builder_tickers)
                    st.session_state.ticker_to_add = ""
                    st.success("Ticker ajouté à ta liste en cours.")
                    st.rerun()

        if st.button("📌 Ajouter la liste affichée à ma liste en cours", use_container_width=True):
            st.session_state.builder_tickers = unique_tickers(st.session_state.builder_tickers + parse_tickers(current_tickers))
            st.session_state.selected_watchlist = "🧺 Liste en cours"
            st.session_state.tickers_text = tickers_to_text(st.session_state.builder_tickers)
            st.rerun()

        if st.session_state.builder_tickers:
            st.markdown(" ".join([badge(t, "purple") for t in st.session_state.builder_tickers]), unsafe_allow_html=True)

            remove_items = st.multiselect(
                "Retirer des actions de la liste en cours",
                st.session_state.builder_tickers,
                key="remove_builder_tickers",
            )
            rm_col1, rm_col2 = st.columns(2)
            with rm_col1:
                if st.button("➖ Retirer la sélection", use_container_width=True):
                    st.session_state.builder_tickers = [t for t in st.session_state.builder_tickers if t not in remove_items]
                    st.session_state.tickers_text = tickers_to_text(st.session_state.builder_tickers)
                    st.session_state.selected_watchlist = "🧺 Liste en cours"
                    st.rerun()
            with rm_col2:
                if st.button("🧹 Vider la liste en cours", use_container_width=True):
                    st.session_state.builder_tickers = []
                    st.session_state.tickers_text = ""
                    st.session_state.selected_watchlist = "🧺 Liste en cours"
                    st.rerun()
        else:
            st.info("Ta liste en cours est vide. Ajoute une action au-dessus.")

        st.markdown("#### 💾 Enregistrer la liste en cours")
        save_name = st.text_input("Nom de la watchlist à enregistrer", placeholder="Ex : IA data centers", key="save_builder_name")
        if st.button("💾 Enregistrer cette watchlist", use_container_width=True):
            name = save_name.strip()
            tickers_value = tickers_to_text(st.session_state.builder_tickers)
            if not name:
                st.error("Donne un nom à ta watchlist.")
            elif not tickers_value:
                st.error("Ta liste en cours est vide.")
            else:
                st.session_state.custom_watchlists[name] = tickers_value
                st.session_state.selected_watchlist = f"⭐ {name}"
                st.session_state.tickers_text = tickers_value
                st.success(f"Watchlist '{name}' enregistrée.")
                st.rerun()

        st.markdown("#### 📦 Sauvegarde / import")
        if st.session_state.custom_watchlists:
            selected_delete = st.selectbox(
                "Supprimer une watchlist enregistrée",
                list(st.session_state.custom_watchlists.keys()),
                key="delete_watchlist_select"
            )
            if st.button("🗑️ Supprimer cette watchlist", use_container_width=True):
                st.session_state.custom_watchlists.pop(selected_delete, None)
                st.success("Watchlist supprimée.")
                st.rerun()

            export_payload = {
                "custom_watchlists": st.session_state.custom_watchlists,
                "builder_tickers": st.session_state.builder_tickers,
            }
            export_data = json.dumps(export_payload, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                "⬇️ Télécharger mes watchlists",
                data=export_data,
                file_name="mes_watchlists_stock_scanner.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Aucune watchlist enregistrée pour l'instant. Ta liste en cours existe seulement dans cette session.")

        uploaded = st.file_uploader("Importer mes watchlists JSON", type=["json"], key="watchlist_import")
        if uploaded is not None:
            try:
                imported = json.loads(uploaded.getvalue().decode("utf-8"))
                if isinstance(imported, dict) and "custom_watchlists" in imported:
                    custom = imported.get("custom_watchlists", {})
                    builder = imported.get("builder_tickers", [])
                elif isinstance(imported, dict):
                    custom = imported
                    builder = []
                else:
                    custom = {}
                    builder = []

                cleaned = {str(k): str(v) for k, v in custom.items() if str(k).strip() and str(v).strip()}
                if st.button("📥 Importer ce fichier", use_container_width=True):
                    st.session_state.custom_watchlists.update(cleaned)
                    if builder:
                        st.session_state.builder_tickers = unique_tickers(builder)
                    st.success("Watchlists importées.")
                    st.rerun()
            except Exception as e:
                st.error(f"Impossible de lire le fichier : {e}")


# =========================================================
# APP
# =========================================================

inject_css()
render_hero()

with st.sidebar:
    st.markdown("### ⚙️ Contrôle")
    all_watchlists = get_all_watchlists()
    options = list(all_watchlists.keys())

    if "selected_watchlist" not in st.session_state or st.session_state.selected_watchlist not in options:
        st.session_state.selected_watchlist = options[0]

    preset = st.selectbox("Watchlist", options, key="selected_watchlist")

    # Quand tu changes de watchlist, on remplit automatiquement la zone Tickers.
    if st.session_state.get("_last_watchlist") != preset:
        st.session_state.tickers_text = all_watchlists[preset]
        st.session_state._last_watchlist = preset

    raw_tickers = st.text_area(
        "Tickers",
        height=180,
        key="tickers_text",
        help="Tu peux modifier cette zone à la main, ou construire une liste progressivement plus bas."
    )
    tickers = parse_tickers(raw_tickers)

    watchlist_manager_ui(raw_tickers)

    st.markdown("---")
    st.markdown("### 🎯 Filtres scan")
    min_fonda = st.slider("Score fondamental min", 0, 100, 0, 5)
    min_rush = st.slider("Score rush min", 0, 100, 0, 5)
    max_risk = st.slider("Risque max", 1, 10, 10, 1)

    st.markdown("---")
    st.markdown(
        """
        <div class="soft-box">
        <b>Règle :</b><br>
        News récente + volume + setup propre > action verte sans raison.
        </div>
        """,
        unsafe_allow_html=True,
    )

main_tab, detail_tab, paper_tab, learn_tab = st.tabs(["🔍 Radar", "📌 Analyse détaillée", "🧪 Paper trading", "📚 Méthode"])

with main_tab:
    st.markdown("<div class='section-title'>Radar watchlist</div>", unsafe_allow_html=True)
    st.write("Scanne plusieurs actions et compare rapidement lesquelles méritent une vraie analyse.")

    top_c1, top_c2, top_c3, top_c4 = st.columns(4)
    with top_c1:
        kpi_card("Tickers", str(len(tickers)), "Nombre d'actions dans la watchlist")
    with top_c2:
        kpi_card("News", "24h", "Anciennes news filtrées")
    with top_c3:
        kpi_card("Mode", "IA/Tech", "Momentum + fondamentaux")
    with top_c4:
        kpi_card("Usage", "Éducation", "Aucun achat automatique")

    if st.button("🚀 Scanner maintenant", type="primary", use_container_width=True):
        if not tickers:
            st.error("Ajoute au moins un ticker.")
        else:
            with st.spinner("Scan en cours..."):
                df = scan_watchlist(tickers)
                if not df.empty:
                    df_filtered = df.copy()
                    if "Fonda" in df_filtered.columns:
                        df_filtered = df_filtered[df_filtered["Fonda"] >= min_fonda]
                    if "Rush" in df_filtered.columns:
                        df_filtered = df_filtered[df_filtered["Rush"] >= min_rush]
                    if "Risque" in df_filtered.columns:
                        df_filtered = df_filtered[df_filtered["Risque"] <= max_risk]
                    st.session_state["last_scan"] = df
                    st.session_state["last_scan_filtered"] = df_filtered

    if "last_scan_filtered" in st.session_state:
        df_show = st.session_state["last_scan_filtered"]
        df_raw = st.session_state["last_scan"]
        show_rankings(df_raw)
        st.markdown("#### Tableau complet")
        st.dataframe(format_watchlist_df(df_show), use_container_width=True, height=520, hide_index=True)
        csv = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger le scan complet en CSV", data=csv, file_name="watchlist_scan.csv", mime="text/csv")

with detail_tab:
    st.markdown("<div class='section-title'>Analyse détaillée</div>", unsafe_allow_html=True)
    default_detail = tickers[0] if tickers else "IREN"
    d1, d2 = st.columns([2, 1])
    with d1:
        ticker_input = st.text_input("Ticker à analyser", value=default_detail).upper()
    with d2:
        st.write("")
        st.write("")
        analyze_btn = st.button("Analyser cette action", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner(f"Analyse de {ticker_input}..."):
            st.session_state["detail_report"] = analyze_ticker(ticker_input)

    if "detail_report" in st.session_state:
        show_analysis(st.session_state["detail_report"])
    else:
        st.info("Entre un ticker puis clique sur Analyser.")

with paper_tab:
    paper_trading_ui()

with learn_tab:
    st.markdown("### 📚 Méthode simple")
    st.markdown(
        """
        <div class="soft-box">
        <b>Le meilleur setup :</b><br>
        1. Une entreprise que tu comprends.<br>
        2. Une vraie news ou un catalyseur récent.<br>
        3. Un volume supérieur à la moyenne.<br>
        4. Un RSI fort mais pas complètement en surchauffe.<br>
        5. Une taille de position raisonnable.<br><br>
        <b>À éviter :</b><br>
        Acheter après +50% sans comprendre pourquoi, mettre trop sur une micro-cap, ou transformer un trade raté en “investissement long terme”.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 Mini-dictionnaire")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            - **MA20** : moyenne mobile 20 jours, court terme.
            - **MA50** : moyenne mobile 50 jours, moyen terme.
            - **MA200** : moyenne mobile 200 jours, long terme.
            - **RSI** : mesure si l'action est forte ou en surchauffe.
            """
        )
    with col_b:
        st.markdown(
            """
            - **Volume x20j** : volume actuel comparé à la moyenne 20 jours.
            - **P/S** : valorisation par rapport au chiffre d'affaires.
            - **Market cap** : taille totale de l'entreprise en bourse.
            - **Beta** : volatilité par rapport au marché.
            """
        )

st.markdown("<div class='footer-note'>AI Stock Radar · Données gratuites via yfinance et Google News RSS · Pas de conseil financier · Vérifie toujours les sources officielles.</div>", unsafe_allow_html=True)
