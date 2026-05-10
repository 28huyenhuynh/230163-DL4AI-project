
import os, sys, pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

st.set_page_config(page_title="VN Stock Predictor", layout="wide")
st.title("Vietnam Stock Price Predictor")
st.caption("LSTM-based next-day close price prediction")

VN_PRICE_DIR = r"c:\Users\Administrator\230163-DL4AI-project\data-vn-20230228\stock-historical-data"

def _add_indicators(df):
    df = df.copy()
    c = df["Close"]
    df["sma5"]  = c.rolling(5).mean()
    df["sma20"] = c.rolling(20).mean()
    d    = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / loss)
    e12  = c.ewm(span=12, adjust=False).mean()
    e26  = c.ewm(span=26, adjust=False).mean()
    df["macd"]   = e12 - e26
    df["macd_s"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_h"] = df["macd"] - df["macd_s"]
    s20  = c.rolling(20).std()
    df["bb_u"] = df["sma20"] + 2*s20
    df["bb_l"] = df["sma20"] - 2*s20
    df["bb_w"] = df["bb_u"] - df["bb_l"]
    return df.dropna()

@st.cache_resource
def load_artifacts():
    model  = tf.keras.models.load_model("saved_models/vn_price_model")
    scaler = pickle.load(open("saved_models/scaler_price.pkl", "rb"))
    meta   = pickle.load(open("saved_models/meta.pkl", "rb"))
    return model, scaler, meta["WINDOW"], meta["VN_FEATS"]

@st.cache_data
def load_ticker(ticker):
    for suffix in ["VNINDEX", "HNXIndex", "UpcomIndex"]:
        path = os.path.join(VN_PRICE_DIR, f"{ticker}-{suffix}-History.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["TradingDate"])
            return df.rename(columns={"TradingDate": "Date"}).sort_values("Date")
    return None

# ── sidebar ──────────────────────────────────────────────────────────────
ticker = st.sidebar.selectbox("Ticker", ["HPG", "VNM", "FPT", "VCB", "MBB", "MSN", "VHM", "SSI"])
n_days = st.sidebar.slider("Chart history (days)", 60, 500, 200)

df_raw = load_ticker(ticker)
if df_raw is None:
    st.error(f"No CSV found for {ticker}")
    st.stop()

df = _add_indicators(df_raw)
recent = df.tail(n_days)

# ── price chart ───────────────────────────────────────────────────────────
fig = go.Figure()
fig.add_candlestick(x=recent["Date"], open=recent["Open"], high=recent["High"],
                    low=recent["Low"], close=recent["Close"], name="OHLC")
fig.add_scatter(x=recent["Date"], y=recent["sma20"], mode="lines",
                name="SMA20", line=dict(color="orange", dash="dash"))
fig.update_layout(title=f"{ticker} — last {n_days} trading days",
                  xaxis_title="Date", yaxis_title="Price (VND)",
                  xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)

# ── metrics row ───────────────────────────────────────────────────────────
last = df.iloc[-1]
col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Close",  f"{last['Close']:,.0f}")
col2.metric("SMA20",       f"{last['sma20']:,.0f}")
col3.metric("RSI14",       f"{last['rsi14']:.1f}")
col4.metric("MACD",        f"{last['macd']:.2f}")

# ── prediction ────────────────────────────────────────────────────────────
st.divider()
if st.button("Predict next-day close price", type="primary"):
    try:
        model, scaler, WINDOW, VN_FEATS = load_artifacts()
        feat_data = df[VN_FEATS].dropna()
        if len(feat_data) < WINDOW:
            st.error(f"Need at least {WINDOW} rows, only have {len(feat_data)}")
        else:
            window_data = feat_data.values[-WINDOW:]
            scaled = scaler.transform(window_data)[np.newaxis]
            pred_lr    = float(model.predict(scaled, verbose=0)[0][0])
            last_close = float(feat_data.iloc[-1]["Close"])
            pred_close = last_close * np.exp(pred_lr)
            direction  = "UP" if pred_lr > 0 else "DOWN"
            color      = "normal" if pred_lr > 0 else "inverse"
            c1, c2, c3 = st.columns(3)
            c1.metric("Last Close",    f"{last_close:,.0f} VND")
            c2.metric("Predicted",     f"{pred_close:,.0f} VND",
                      f"{pred_lr*100:+.2f}%", delta_color=color)
            c3.metric("Direction",     direction)
    except FileNotFoundError:
        st.error("Models not found — run cell t51-save in the notebook first.")
    except Exception as e:
        st.error(f"Error: {e}")
