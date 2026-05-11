
import os, pickle, numpy as np, pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app    = FastAPI(title="VN Stock Prediction API", version="1.0")
MODEL  = tf.keras.models.load_model("saved_models/vn_price_model.keras")
SCALER = pickle.load(open("saved_models/scaler_price.pkl", "rb"))
META   = pickle.load(open("saved_models/meta.pkl", "rb"))
WINDOW    = META["WINDOW"]
VN_FEATS  = META["VN_FEATS"]

def _add_indicators(df):
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

class OHLCVRequest(BaseModel):
    # rows: list of [Open, High, Low, Close, Volume] — need at least WINDOW+26 rows
    rows: List[List[float]]

@app.get("/health")
def health():
    return {"status": "ok", "window": WINDOW, "features": len(VN_FEATS)}

@app.post("/predict/price")
def predict_price(req: OHLCVRequest):
    if len(req.rows) < WINDOW + 26:
        raise HTTPException(400, f"Need >= {WINDOW + 26} rows, got {len(req.rows)}")
    df = pd.DataFrame(req.rows, columns=["Open", "High", "Low", "Close", "Volume"])
    df = _add_indicators(df)[VN_FEATS].dropna()
    if len(df) < WINDOW:
        raise HTTPException(400, "Not enough rows after indicator computation")
    window_data = df.values[-WINDOW:]
    scaled = SCALER.transform(window_data)[np.newaxis]   # (1, WINDOW, n_feats)
    pred_lr    = float(MODEL.predict(scaled, verbose=0)[0][0])
    last_close = float(df.iloc[-1]["Close"])
    pred_close = last_close * float(np.exp(pred_lr))
    return {
        "last_close":           round(last_close, 2),
        "predicted_log_return": round(pred_lr, 6),
        "predicted_next_close": round(pred_close, 2),
        "direction":            "UP" if pred_lr > 0 else "DOWN"
    }
