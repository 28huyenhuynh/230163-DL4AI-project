"""
Standalone test for Task 1.3 fix — runs only the cells needed for t13
and saves the plot to test_t13_output.png
"""
import os, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # no GUI needed
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

NASDAQ_DIR = r"c:\Users\Administrator\230163-DL4AI-project\data_nasdaq_csv\csv"
MIN_DAYS   = 500
WINDOW     = 60
TICKER_NQ  = "AAPL"
NQ_FEATS   = ["Open","High","Low","Close","Volume","Adjusted Close",
              "sma5","sma20","rsi14","macd","macd_h","bb_w"]

# ── load AAPL ─────────────────────────────────────────────────────────────
frames = []
for f in os.listdir(NASDAQ_DIR):
    if not f.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(NASDAQ_DIR, f), parse_dates=["Date"], on_bad_lines="skip")
    df["Ticker"] = f.replace(".csv", "")
    frames.append(df)
nasdaq = pd.concat(frames, ignore_index=True).sort_values(["Ticker","Date"])
aapl   = nasdaq[nasdaq["Ticker"] == TICKER_NQ].sort_values("Date").copy()
print(f"AAPL rows: {len(aapl)}  price range: {aapl['Close'].min():.2f}–{aapl['Close'].max():.2f}")

# ── helpers ───────────────────────────────────────────────────────────────
def clean_ohlcv(df):
    df = df.copy()
    for col in ("Open","High","Low","Close"):
        if col in df.columns:
            df = df[df[col] > 0]
    if "Close" in df.columns and len(df) > 30:
        lr = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        q1, q3 = lr.quantile(0.01), lr.quantile(0.99)
        fence = 3 * (q3 - q1)
        bad = lr[(lr < q1 - fence) | (lr > q3 + fence)].index
        df = df.drop(index=bad, errors="ignore")
    return df.reset_index(drop=True)

def add_indicators(df):
    df = df.copy(); c = df["Close"]
    df["sma5"]  = c.rolling(5).mean()
    df["sma20"] = c.rolling(20).mean()
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / loss)
    e12 = c.ewm(span=12,adjust=False).mean()
    e26 = c.ewm(span=26,adjust=False).mean()
    df["macd"]   = e12 - e26
    df["macd_s"] = df["macd"].ewm(span=9,adjust=False).mean()
    df["macd_h"] = df["macd"] - df["macd_s"]
    std20 = c.rolling(20).std()
    df["bb_u"]  = df["sma20"] + 2*std20
    df["bb_l"]  = df["sma20"] - 2*std20
    df["bb_w"]  = df["bb_u"] - df["bb_l"]
    return df.dropna()

def chrono_split(X, y, val=0.15, test=0.15):
    n  = len(X); t2 = int(n*(1-test)); t1 = int(t2*(1-val/(1-test)))
    return X[:t1],y[:t1],X[t1:t2],y[t1:t2],X[t2:],y[t2:]

def build_lstm(input_shape):
    inp = keras.Input(shape=input_shape)
    x   = layers.LSTM(128, return_sequences=True)(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(64, return_sequences=True)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(32)(x)
    x   = layers.Dropout(0.1)(x)
    out = layers.Dense(1)(x)
    m   = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m

def make_seq_ahead(feat_scaled, targets, window, d):
    X, y = [], []
    for i in range(window, len(feat_scaled) - d + 1):
        X.append(feat_scaled[i-window:i])
        y.append(targets[i+d-1])
    return np.array(X), np.array(y)

# ── prepare data ──────────────────────────────────────────────────────────
_nq = clean_ohlcv(aapl)
_nq = add_indicators(_nq)
_nq = _nq[NQ_FEATS].dropna().reset_index(drop=True)
close_arr = _nq.iloc[:,3].values
lr_arr    = np.concatenate([[0.0], np.log(close_arr[1:]/(close_arr[:-1]+1e-9))])
sc = MinMaxScaler()
s  = sc.fit_transform(_nq)
print(f"Data rows after cleaning: {len(_nq)}  close range: {close_arr.min():.2f}–{close_arr.max():.2f}")

# ── train K=7 separate models ─────────────────────────────────────────────
K = 7
yte_days, ypred_days = [], []

for d in range(1, K+1):
    X, y = make_seq_ahead(s, lr_arr, WINDOW, d)
    Xtr,ytr,Xval,yval,Xte,yte_d = chrono_split(X, y)
    es  = keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
    md  = build_lstm((WINDOW, len(NQ_FEATS)))
    md.fit(Xtr,ytr,validation_data=(Xval,yval),
           epochs=50,batch_size=32,callbacks=[es,rlr],verbose=0)
    yp_d = md.predict(Xte, verbose=0).flatten()
    yte_days.append(yte_d)
    ypred_days.append(yp_d)
    mae = mean_absolute_error(yte_d, yp_d)
    pred_std = yp_d.std()
    print(f"  day+{d}: MAE={mae:.4f}  pred_std={pred_std:.5f}  "
          f"pred_mean={yp_d.mean():.5f}  actual_std={yte_d.std():.5f}")

# ── reconstruct prices ────────────────────────────────────────────────────
X1, _ = make_seq_ahead(s, lr_arr, WINDOW, 1)
Xtr1,_,Xval1,_,_,_ = chrono_split(X1, np.zeros(len(X1)))
test_start_idx = len(Xtr1) + len(Xval1) + WINDOW - 1   # last day of input window
lc13 = close_arr[test_start_idx]
print(f"\nlc13 (anchor price) = {lc13:.4f}")

act_lr   = [float(yte_days[d][0])   for d in range(K)]
pred_lr  = [float(ypred_days[d][0]) for d in range(K)]
print(f"actual log-returns [0]: {[f'{v:.4f}' for v in act_lr]}")
print(f"predicted log-returns[0]: {[f'{v:.4f}' for v in pred_lr]}")

act_p  = [lc13 * np.exp(sum(act_lr[:d+1]))  for d in range(K)]
pred_p = [lc13 * np.exp(sum(pred_lr[:d+1])) for d in range(K)]
print(f"actual prices    : {[f'{v:.2f}' for v in act_p]}")
print(f"predicted prices : {[f'{v:.2f}' for v in pred_p]}")

# ── plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(range(1,K+1), act_p,  "o-",  label="Actual")
ax.plot(range(1,K+1), pred_p, "s--", label="Predicted")
ax.set_xlabel("Day ahead"); ax.set_ylabel("Reconstructed price ($)")
ax.set_title("Task 1.3 — 7-day consecutive forecast (separate model per horizon)")
ax.legend(); plt.tight_layout()
plt.savefig("test_t13_output.png", dpi=120)
print("\nPlot saved to test_t13_output.png")
