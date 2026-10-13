
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import timedelta
from dotenv import load_dotenv, find_dotenv


from litellm import completion


_ = load_dotenv(find_dotenv(usecwd=True))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "groq/openai/gpt-oss-20b")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merged_msft_with_signals.csv")

DISCLAIMER = (
    "Disclaimer: This project is an educational tool. It describes technical indicators "
    "for one stock using historical data and does not provide investment advice, "
    "recommendations, or trading strategies. Nothing here should be used to make "
    "investment decisions."
)

st.set_page_config(page_title="LLM-Powered Investment Insight", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # light safety
    for c in ["PX","ret_1d","rsi_14","macd","bb_pos","avg_sentiment"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

df = load_data()
min_d, max_d = df["date"].min(), df["date"].max()

st.sidebar.warning(DISCLAIMER)

st.sidebar.header("View")
lookback_days = st.sidebar.slider("Lookback (days)", 30, 365, 120)
end_date = st.sidebar.date_input("End date", value=max_d.date(), min_value=min_d.date(), max_value=max_d.date())
start_date = pd.to_datetime(end_date) - timedelta(days=lookback_days)
mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
view = df.loc[mask].copy()

st.sidebar.header("LLM")
st.sidebar.write("Model:", DEFAULT_MODEL)
if not GROQ_API_KEY:
    st.sidebar.error("GROQ_API_KEY not found. Locally, add it to .env. On Streamlit Cloud, put it in Secrets.")
else:
    st.sidebar.success("API key loaded")


FIGSIZE = (9, 4.5)
CHART_WIDTH = 760  # px; both charts share this size


def plot_price_signals(data: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=FIGSIZE)


    ax1.plot(data["date"], data["PX"], linewidth=2, label="Price")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Price")

    # the data only contains long (1) and flat (0) signals
    if "signal" in data.columns:
        long_idx  = data["signal"].fillna(0) == 1
        ax1.scatter(data.loc[long_idx, "date"],  data.loc[long_idx, "PX"],  s=40, marker="^", label="Long signal")

    ax1.grid(alpha=.3)
    ax1.set_title("Price & Signals")
    ax1.legend(loc="upper left")
    fig.tight_layout()
    return fig

def plot_equity_curves(data: pd.DataFrame):
    sig = data.get("signal", pd.Series(0, index=data.index)).fillna(0).shift(1).fillna(0)
    strat_ret = sig * data["ret_1d"].fillna(0)
    bh_ret    = data["ret_1d"].fillna(0)
    strat_curve = (1 + strat_ret).cumprod()
    bh_curve    = (1 + bh_ret).cumprod()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # buy & hold first and thinner, so the signal line stays visible where they overlap
    ax.plot(data["date"], bh_curve, label="Buy & Hold", color="tab:orange", linewidth=1.5, alpha=.8)
    ax.plot(data["date"], strat_curve, label="Signal (long/flat)", color="tab:blue", linewidth=2.5, zorder=3)
    ax.set_title("Signal vs Buy & Hold (illustrative backtest, not validated)")
    ax.set_ylabel("Growth of $1"); ax.set_xlabel("Date")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    return fig


SYS_PROMPT = (
    "You are part of an educational tool that describes technical indicators for one stock "
    "using historical data. Describe what the indicators show in neutral, factual terms. "
    "Do not give investment advice, recommendations, price targets, or buy/sell/hold suggestions. "
    "Output 3-5 short bullet points and a one-sentence summary."
)

def row_context(row):

    return (
        f"Date: {row['date']:%Y-%m-%d}\n"
        f"Price PX: {row['PX']:.2f}\n"
        f"1D return: {row.get('ret_1d', np.nan):+.3%}\n"
        f"RSI(14): {row.get('rsi_14', np.nan):.1f}\n"
        f"MACD: {row.get('macd', np.nan):.2f}\n"
        f"Bollinger position: {row.get('bb_pos', np.nan):.2f}\n"
        f"Rule-based signal (1=long, 0=flat): {int(row.get('signal', 0))}\n"
    )

def llm_insight_from_row(row):
    if not GROQ_API_KEY:
        return "⚠️ No API key found. Add GROQ_API_KEY to your .env or Streamlit secrets."
    resp = completion(
        model=DEFAULT_MODEL,
        api_key=GROQ_API_KEY,
        messages=[{"role":"system","content":SYS_PROMPT},
                  {"role":"user","content":f"Context:\n{row_context(row)}"}],
        temperature=0.2, max_tokens=600
    )
    return resp["choices"][0]["message"]["content"].strip()


st.title("LLM-Powered Investment Insight Generator")

col1, col2 = st.columns([2,1])
with col1:
    st.pyplot(plot_price_signals(view), clear_figure=True, width=CHART_WIDTH)
with col2:
    if len(view):
        latest = view.iloc[-1]
        st.subheader("Indicator Summary (LLM)")
        if st.button("Generate summary for latest date"):
            st.write(llm_insight_from_row(latest))

st.divider()
st.pyplot(plot_equity_curves(view), clear_figure=True, width=CHART_WIDTH)
st.caption("Illustrative backtest, not validated. No transaction costs, no out-of-sample test; not a performance claim.")


INDICATOR_GUIDE = """
| Indicator | What it measures |
|---|---|
| **Daily return** | The percentage change in closing price from the previous trading day. |
| **RSI (14)** | Relative Strength Index: compares the size of recent gains and losses over 14 days on a 0–100 scale. Readings above 70 or below 30 are conventionally called "overbought" or "oversold". |
| **MACD** | Moving Average Convergence Divergence: the 12-day exponential moving average minus the 26-day one. Positive values mean the shorter-term average is above the longer-term one. |
| **SMA cross** | Whether the 5-day simple moving average is above (+1) or below (−1) the 20-day one, a simple description of short-term versus longer-term trend. |
| **Bollinger position** | Where the price sits within its Bollinger Bands (20-day average ± 2 standard deviations): about 0 at the middle band, about +1 at the upper band and −1 at the lower band. |
"""

with st.expander("Indicator guide"):
    st.markdown(INDICATOR_GUIDE)
    st.caption("Educational descriptions only, not trading advice.")
