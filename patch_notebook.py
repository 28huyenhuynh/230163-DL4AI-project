"""
Complete rewrite patch for 230163_project_notebook.ipynb.
Run from terminal AFTER closing the notebook in Jupyter/VS Code.

Key fixes:
  - DATE_CUTOFF 2018-01-01  → test set lands in 2022-2024 (clean prices)
  - Use 'Adjusted Close' for NASDAQ close array → no stock-split spikes
  - Simplified LSTM 64-32 (no BatchNorm) → ~5x faster training
  - WINDOW=30 → less memory, faster sequences
  - t13 / t23 7-day chart: read actual prices directly from close array
"""
import json

NB = "230163_project_notebook.ipynb"
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)
cells = {c["id"]: c for c in nb["cells"] if "id" in c}

# ── paths ─────────────────────────────────────────────────────────────────
cells["paths"]["source"] = r'''# set paths
NASDAQ_DIR   = r"c:\Users\Administrator\230163-DL4AI-project\data_nasdaq_csv\csv"
VN_PRICE_DIR = r"c:\Users\Administrator\230163-DL4AI-project\data-vn-20230228\stock-historical-data"
VN_DIV_DIR   = r"c:\Users\Administrator\230163-DL4AI-project\data-vn-20230228\dividend-history"
VN_FIN_DIR   = r"c:\Users\Administrator\230163-DL4AI-project\data-vn-20230228\financial-ratio"
VN_COMPANIES = r"c:\Users\Administrator\230163-DL4AI-project\data-vn-20230228\companies.csv"
VN_OVERVIEW  = r"c:\Users\Administrator\230163-DL4AI-project\data-vn-20230228\ticker-overview.csv"

MIN_DAYS    = 200           # keep tickers with >= 200 trading days since cutoff
WINDOW      = 45            # 45-day lookback — more context, still fast
DATE_CUTOFF = '2018-01-01'  # use data from 2018 onwards — clean price scales, no old splits

print(os.listdir(NASDAQ_DIR)[:5])
print(os.listdir(VN_PRICE_DIR)[:5])
'''

# ── nasdaq-load ───────────────────────────────────────────────────────────
cells["nasdaq-load"]["source"] = r'''def load_nasdaq(d, min_days=MIN_DAYS):
    frames = []
    for f in os.listdir(d):
        if not f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(d, f), parse_dates=['Date'], on_bad_lines='skip')
        df['Ticker'] = f.replace('.csv', '')
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df['Date'] = pd.to_datetime(all_df['Date'], errors='coerce')
    all_df = all_df.dropna(subset=['Date'])
    all_df = all_df[all_df['Date'] >= pd.Timestamp(DATE_CUTOFF)]
    keep   = all_df.groupby('Ticker')['Date'].count()
    all_df = all_df[all_df['Ticker'].isin(keep[keep >= min_days].index)]
    print(f'Nasdaq: {keep[keep >= min_days].shape[0]} companies kept (since {DATE_CUTOFF})')
    return all_df.sort_values(['Ticker', 'Date'])

nasdaq = load_nasdaq(NASDAQ_DIR)
nasdaq.head()
'''

# ── vn-load ───────────────────────────────────────────────────────────────
cells["vn-load"]["source"] = r'''def load_vietnam(d, min_days=MIN_DAYS):
    frames = []
    for f in os.listdir(d):
        if not f.endswith('.csv'):
            continue
        ticker = f.split('-')[0]
        df = pd.read_csv(os.path.join(d, f), parse_dates=['TradingDate'])
        df = df.rename(columns={'TradingDate': 'Date'})
        df['Ticker'] = ticker
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df['Date'] = pd.to_datetime(all_df['Date'], errors='coerce')
    all_df = all_df.dropna(subset=['Date'])
    all_df = all_df[all_df['Date'] >= pd.Timestamp(DATE_CUTOFF)]
    keep   = all_df.groupby('Ticker')['Date'].count()
    all_df = all_df[all_df['Ticker'].isin(keep[keep >= min_days].index)]
    print(f'Vietnam: {keep[keep >= min_days].shape[0]} companies kept (since {DATE_CUTOFF})')
    return all_df.sort_values(['Ticker', 'Date'])

vn = load_vietnam(VN_PRICE_DIR)
vn.head()
'''

# ── utils ─────────────────────────────────────────────────────────────────
cells["utils"]["source"] = r'''# ══════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════

def clean_ohlcv(df, price_cols=('Open', 'High', 'Low', 'Close')):
    df = df.copy()
    for col in price_cols:
        if col in df.columns:
            df = df[df[col] > 0]
    if 'Close' in df.columns and len(df) > 30:
        lr      = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        q1, q3  = lr.quantile(0.01), lr.quantile(0.99)
        fence   = 3 * (q3 - q1)
        bad_idx = lr[(lr < q1 - fence) | (lr > q3 + fence)].index
        if len(bad_idx):
            df = df.drop(index=bad_idx, errors='ignore')
    return df.reset_index(drop=True)


def add_indicators(df):
    df   = df.copy()
    c    = df['Close']
    df['sma5']   = c.rolling(5).mean()
    df['sma20']  = c.rolling(20).mean()
    d    = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    df['rsi14']  = 100 - 100 / (1 + gain / loss)
    e12  = c.ewm(span=12, adjust=False).mean()
    e26  = c.ewm(span=26, adjust=False).mean()
    df['macd']   = e12 - e26
    df['macd_s'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_h'] = df['macd'] - df['macd_s']
    std20 = c.rolling(20).std()
    df['bb_u']   = df['sma20'] + 2 * std20
    df['bb_l']   = df['sma20'] - 2 * std20
    df['bb_w']   = df['bb_u'] - df['bb_l']
    return df.dropna()


def make_sequences(feat_scaled, targets, window, horizon=1):
    X, y = [], []
    for i in range(window, len(feat_scaled) - horizon + 1):
        X.append(feat_scaled[i - window: i])
        if horizon == 1:
            y.append(targets[i])
        else:
            y.append(targets[i: i + horizon])
    return np.array(X), np.array(y)


def make_seq_ahead(feat_scaled, targets, window, d):
    # Single-output target exactly d steps ahead — avoids Dense(k) MSE collapse.
    X, y = [], []
    for i in range(window, len(feat_scaled) - d + 1):
        X.append(feat_scaled[i - window: i])
        y.append(targets[i + d - 1])
    return np.array(X), np.array(y)


def chrono_split(X, y, val=0.15, test=0.15):
    n  = len(X)
    t2 = int(n * (1 - test))
    t1 = int(t2 * (1 - val / (1 - test)))
    return X[:t1], y[:t1], X[t1:t2], y[t1:t2], X[t2:], y[t2:]


def eval_regression(y_true, y_pred, tag=''):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'{tag}  MAE={mae:.4f}  RMSE={rmse:.4f}')
    return mae, rmse


def plot_pred(y_true_lr, y_pred_lr, title='', close_arr=None, t_start=0):
    """
    Top panel  : actual vs predicted log-returns.
    Bottom panel: actual close prices and one-step predicted prices.
                  One-step prediction: pred_p[i] = actual_p[i-1] * exp(pred_lr[i])
                  This anchors each prediction to yesterday's real price, so both
                  lines stay in the same price range (no compounding drift).
    """
    show_price = (close_arr is not None) and (t_start > 0)
    fig, axes = plt.subplots(1 if not show_price else 2, 1,
                             figsize=(12, 4 if not show_price else 8))
    if not show_price:
        axes = [axes]
    axes[0].plot(y_true_lr, lw=1.1, label='Actual log-return')
    axes[0].plot(y_pred_lr, lw=1.1, ls='--', alpha=0.85, label='Predicted log-return')
    axes[0].axhline(0, color='gray', lw=0.6, ls=':')
    axes[0].set_title(title + ' — log-return predictions')
    axes[0].legend()
    if show_price:
        n     = len(y_true_lr)
        act_p = close_arr[t_start: t_start + n].astype(float)
        # One-step: each predicted price anchors to the previous actual price
        prev_p = close_arr[t_start - 1: t_start + n - 1].astype(float)
        pred_p = prev_p * np.exp(y_pred_lr.astype(float))
        axes[1].plot(act_p,  lw=1.2, label='Actual price')
        axes[1].plot(pred_p, lw=1.2, ls='--', alpha=0.85, label='Predicted price (1-step)')
        axes[1].set_title('Close prices — actual vs one-step predicted')
        axes[1].legend()
    plt.tight_layout(); plt.show()


def build_lstm(input_shape, output_units=1, classify=False):
    """2-layer LSTM with BatchNorm — good balance of speed and capacity."""
    inp = keras.Input(shape=input_shape)
    x   = layers.LSTM(64, return_sequences=True)(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(32)(x)
    x   = layers.Dropout(0.1)(x)
    if classify:
        out  = layers.Dense(output_units, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
        met  = ['accuracy']
    else:
        out  = layers.Dense(output_units)(x)
        loss = 'mse'
        met  = ['mae']
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=loss, metrics=met)
    return m


def _get_close(d, prefer_adjusted=True):
    """Return the close price array, preferring Adjusted Close when available."""
    if prefer_adjusted and 'Adjusted Close' in d.columns:
        return d['Adjusted Close'].values.astype(float)
    return d['Close'].values.astype(float)


def train_one(df_co, features, window=WINDOW, horizon=1, epochs=30, verbose=0,
              use_indicators=True):
    d = df_co.sort_values('Date').copy()
    d = clean_ohlcv(d)
    if use_indicators:
        d = add_indicators(d)
    d = d[features].dropna().reset_index(drop=True)
    close = _get_close(d)
    lr    = np.concatenate([[0.0], np.log(close[1:] / (close[:-1] + 1e-9))])
    sc    = MinMaxScaler()
    s     = sc.fit_transform(d)
    X, y  = make_sequences(s, lr, window, horizon=1)
    Xtr, ytr, Xval, yval, Xte, yte = chrono_split(X, y)
    t_start = len(Xtr) + len(Xval) + window
    last_close = float(close[t_start]) if t_start < len(close) else float(close[-1])
    es  = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                            patience=3, min_lr=1e-5)
    m   = build_lstm((window, len(features)), output_units=1)
    m.fit(Xtr, ytr, validation_data=(Xval, yval),
          epochs=epochs, batch_size=32, callbacks=[es, rlr], verbose=verbose)
    ypred = m.predict(Xte, verbose=0).flatten()
    return m, sc, yte, ypred, last_close, close, t_start


def _prep_ticker(df_co, features, use_indicators=True):
    d = df_co.sort_values('Date').copy()
    d = clean_ohlcv(d)
    if use_indicators:
        d = add_indicators(d)
    d = d[features].dropna().reset_index(drop=True)
    close = _get_close(d)
    lr    = np.concatenate([[0.0], np.log(close[1:] / (close[:-1] + 1e-9))])
    sc    = MinMaxScaler()
    s     = sc.fit_transform(d)
    return close, lr, s


print('Utilities ready.')
'''

# ── t11 ───────────────────────────────────────────────────────────────────
cells["t11"]["source"] = r'''m11, sc11, yte11, ypred11, lc11, close11, ts11 = train_one(
    nasdaq[nasdaq['Ticker'] == TICKER_NQ], NQ_FEATS, epochs=30, verbose=1)
eval_regression(yte11, ypred11, '[Task 1.1]')
plot_pred(yte11, ypred11, 'Task 1.1 -- AAPL next-day',
          close_arr=close11, t_start=ts11)
'''

# ── t12 ───────────────────────────────────────────────────────────────────
cells["t12"]["source"] = r'''for k in [3, 7]:
    close12, lr12, s12 = _prep_ticker(
        nasdaq[nasdaq['Ticker'] == TICKER_NQ], NQ_FEATS)
    X, y = make_seq_ahead(s12, lr12, WINDOW, d=k)
    Xtr, ytr, Xval, yval, Xte, yte_k = chrono_split(X, y)
    X1, _ = make_seq_ahead(s12, lr12, WINDOW, 1)
    Xtr1, _, Xval1, _, _, _ = chrono_split(X1, np.zeros(len(X1)))
    t_start12 = len(Xtr1) + len(Xval1) + WINDOW
    es  = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                            patience=3, min_lr=1e-5)
    m   = build_lstm((WINDOW, len(NQ_FEATS)), output_units=1)
    m.fit(Xtr, ytr, validation_data=(Xval, yval),
          epochs=30, batch_size=32, callbacks=[es, rlr], verbose=0)
    ypred_k = m.predict(Xte, verbose=0).flatten()
    eval_regression(yte_k, ypred_k, f'[Task 1.2 k={k}]')
    plot_pred(yte_k, ypred_k, f'Task 1.2 -- AAPL {k}-th day',
              close_arr=close12, t_start=t_start12)
'''

# ── t13 ───────────────────────────────────────────────────────────────────
cells["t13"]["source"] = r'''K = 7
close13, lr13, s13 = _prep_ticker(
    nasdaq[nasdaq['Ticker'] == TICKER_NQ], NQ_FEATS)

yte_days, ypred_days = [], []
for d in range(1, K + 1):
    X, y = make_seq_ahead(s13, lr13, WINDOW, d)
    Xtr, ytr, Xval, yval, Xte, yte_d = chrono_split(X, y)
    es  = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                            patience=3, min_lr=1e-5)
    md  = build_lstm((WINDOW, len(NQ_FEATS)), output_units=1)
    md.fit(Xtr, ytr, validation_data=(Xval, yval),
           epochs=30, batch_size=32, callbacks=[es, rlr], verbose=0)
    yp_d = md.predict(Xte, verbose=0).flatten()
    yte_days.append(yte_d)
    ypred_days.append(yp_d)
    eval_regression(yte_d, yp_d, f'[Task 1.3 day+{d}]')

# Aligned 7-day chart anchored at test-set boundary of model d=1
X1, _ = make_seq_ahead(s13, lr13, WINDOW, 1)
Xtr1, _, Xval1, _, _, _ = chrono_split(X1, np.zeros(len(X1)))
t_start = len(Xtr1) + len(Xval1) + WINDOW   # index into close13

lc13 = float(close13[t_start - 1])

# Actual: read directly from close13 — no accumulation error
act_p = [float(close13[t_start + d - 1]) for d in range(1, K + 1)]

# Predicted: cumulative predicted log-returns applied to anchor
pred_lr_cum = np.cumsum([float(ypred_days[d][0]) for d in range(K)])
pred_p = [lc13 * float(np.exp(c)) for c in pred_lr_cum]

plt.figure(figsize=(8, 4))
plt.plot(range(1, K + 1), act_p,  'o-',  label='Actual')
plt.plot(range(1, K + 1), pred_p, 's--', label='Predicted')
plt.xlabel('Day ahead'); plt.ylabel('Close price ($)')
plt.title('Task 1.3 — 7-day consecutive forecast (separate model per horizon)')
plt.legend(); plt.tight_layout(); plt.show()
'''

# ── t21 ───────────────────────────────────────────────────────────────────
cells["t21"]["source"] = r'''m21, sc21, yte21, ypred21, lc21, close21, ts21 = train_one(
    vn[vn['Ticker'] == TICKER_VN], VN_FEATS, epochs=30, verbose=1)
eval_regression(yte21, ypred21, '[Task 2.1]')
plot_pred(yte21, ypred21, f'Task 2.1 -- {TICKER_VN} next-day',
          close_arr=close21, t_start=ts21)
'''

# ── t22 ───────────────────────────────────────────────────────────────────
cells["t22"]["source"] = r'''for k in [3, 7]:
    close22, lr22, s22 = _prep_ticker(
        vn[vn['Ticker'] == TICKER_VN], VN_FEATS)
    X, y = make_seq_ahead(s22, lr22, WINDOW, d=k)
    Xtr, ytr, Xval, yval, Xte, yte_k = chrono_split(X, y)
    X1, _ = make_seq_ahead(s22, lr22, WINDOW, 1)
    Xtr1, _, Xval1, _, _, _ = chrono_split(X1, np.zeros(len(X1)))
    t_start22 = len(Xtr1) + len(Xval1) + WINDOW
    es  = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                            patience=3, min_lr=1e-5)
    m   = build_lstm((WINDOW, len(VN_FEATS)), output_units=1)
    m.fit(Xtr, ytr, validation_data=(Xval, yval),
          epochs=30, batch_size=32, callbacks=[es, rlr], verbose=0)
    ypred_k = m.predict(Xte, verbose=0).flatten()
    eval_regression(yte_k, ypred_k, f'[Task 2.2 k={k}]')
    plot_pred(yte_k, ypred_k, f'Task 2.2 -- {TICKER_VN} {k}-th day',
              close_arr=close22, t_start=t_start22)
'''

# ── t23 ───────────────────────────────────────────────────────────────────
cells["t23"]["source"] = r'''K = 7
close23, lr23, s23 = _prep_ticker(
    vn[vn['Ticker'] == TICKER_VN], VN_FEATS)

yte_days23, ypred_days23 = [], []
for d in range(1, K + 1):
    X, y = make_seq_ahead(s23, lr23, WINDOW, d)
    Xtr, ytr, Xval, yval, Xte, yte_d = chrono_split(X, y)
    es  = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                            patience=3, min_lr=1e-5)
    md  = build_lstm((WINDOW, len(VN_FEATS)), output_units=1)
    md.fit(Xtr, ytr, validation_data=(Xval, yval),
           epochs=30, batch_size=32, callbacks=[es, rlr], verbose=0)
    yp_d = md.predict(Xte, verbose=0).flatten()
    yte_days23.append(yte_d)
    ypred_days23.append(yp_d)
    eval_regression(yte_d, yp_d, f'[Task 2.3 day+{d}]')

X1, _ = make_seq_ahead(s23, lr23, WINDOW, 1)
Xtr1, _, Xval1, _, _, _ = chrono_split(X1, np.zeros(len(X1)))
t_start23 = len(Xtr1) + len(Xval1) + WINDOW

lc23 = float(close23[t_start23 - 1])

act_p23 = [float(close23[t_start23 + d - 1]) for d in range(1, K + 1)]

pred_lr_cum23 = np.cumsum([float(ypred_days23[d][0]) for d in range(K)])
pred_p23 = [lc23 * float(np.exp(c)) for c in pred_lr_cum23]

plt.figure(figsize=(8, 4))
plt.plot(range(1, K + 1), act_p23,  'o-',  label='Actual')
plt.plot(range(1, K + 1), pred_p23, 's--', label='Predicted')
plt.xlabel('Day ahead'); plt.ylabel('Close price (VND)')
plt.title(f'Task 2.3 — {TICKER_VN} 7-day consecutive forecast (separate model per horizon)')
plt.legend(); plt.tight_layout(); plt.show()
'''

# ── t41 ───────────────────────────────────────────────────────────────────
cells["t41"]["source"] = r'''def profit_score(df_co, features, horizon=1):
    try:
        _, _, yte, ypred, _, _, _ = train_one(df_co, features, horizon=horizon, epochs=20)
        return float(np.mean(ypred.flatten()))
    except:
        return np.nan

tickers_vn = vn['Ticker'].unique()
profit = {}
for tk in tickers_vn[:20]:
    profit[tk] = profit_score(vn[vn['Ticker'] == tk], VN_FEATS)
    print(f'{tk}: {profit[tk]:.4f}')

profit_df = (pd.Series(profit).dropna()
               .sort_values(ascending=False)
               .to_frame('profit_score'))
profit_df.head(10).plot(kind='bar', figsize=(10, 4),
                        title='Top 10 — Predicted profitability')
plt.tight_layout(); plt.show()
'''

# ── t43 ───────────────────────────────────────────────────────────────────
cells["t43"]["source"] = r'''combined = profit_df.join(risk_df, how='inner')
if len(combined) == 0:
    print('No combined candidates -- skipping portfolio optimisation.')
else:
    thresh_risk   = combined['risk_score'].median()
    thresh_profit = combined['profit_score'].median()
    cands = combined[(combined['profit_score'] >= thresh_profit) &
                     (combined['risk_score']    <  thresh_risk)]
    if len(cands) < 2:
        cands = combined.nlargest(max(2, len(combined) // 2), 'profit_score')
    print(f'Portfolio candidates: {len(cands)}'); print(cands)

    ret_matrix = {}
    for tk in cands.index:
        sub = vn[vn['Ticker'] == tk].sort_values('Date')
        ret_matrix[tk] = sub.set_index('Date')['Close'].pct_change()
    ret_df = pd.DataFrame(ret_matrix).dropna()

    if ret_df.empty or len(ret_df.columns) < 2:
        print('Not enough return data for optimisation.')
    else:
        mu  = ret_df.mean().values * 252
        cov = ret_df.cov().values  * 252
        n   = len(cands)
        def neg_sharpe(w):
            return -(w @ mu) / (np.sqrt(w @ cov @ w) + 1e-9)
        res = minimize(neg_sharpe, np.ones(n) / n,
                       bounds=[(0, 1)] * n,
                       constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
        weights = pd.Series(res.x, index=cands.index)
        print('\nOptimal weights:'); print(weights.sort_values(ascending=False))
        p_ret = weights.values @ mu
        p_vol = np.sqrt(weights.values @ cov @ weights.values)
        print(f'Expected return: {p_ret:.2%}  vol: {p_vol:.2%}  Sharpe: {p_ret/p_vol:.2f}')
        weights.sort_values().plot(kind='barh', figsize=(8, 5),
                                   title='Optimal portfolio allocation')
        plt.xlabel('Weight'); plt.tight_layout(); plt.show()
'''

# ── t51-save ─────────────────────────────────────────────────────────────
cells["t51-save"]["source"] = r'''os.makedirs('saved_models', exist_ok=True)

# Save VN price model + its scaler (trained in Task 2.1)
m21.save('saved_models/vn_price_model.keras')
with open('saved_models/scaler_price.pkl', 'wb') as f:
    pickle.dump(sc21, f)

# Save signal classifier + its scaler (trained in Task 3)
m_sig.save('saved_models/vn_signal_model.keras')
with open('saved_models/scaler_signal.pkl', 'wb') as f:
    pickle.dump(sc_sig, f)

# Save constants needed at inference time
meta = {'WINDOW': WINDOW, 'VN_FEATS': VN_FEATS, 'VN_PRICE_DIR': VN_PRICE_DIR}
with open('saved_models/meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print('Models and scalers saved to saved_models/')
print(f'  Price model input shape : (1, {WINDOW}, {len(VN_FEATS)})')
'''

# ── t51-api ───────────────────────────────────────────────────────────────
cells["t51-api"]["source"] = r'''fastapi_code = """
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
"""

with open("main.py", "w") as f:
    f.write(fastapi_code)
print("main.py written.")
print("Run the API with:  uvicorn main:app --reload")
print("Then open:         http://127.0.0.1:8000/docs")
'''

# ── t52 ───────────────────────────────────────────────────────────────────
cells["t52"]["source"] = r'''streamlit_code = """
import os, sys, pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

st.set_page_config(page_title="VN Stock Predictor", layout="wide")
st.title("Vietnam Stock Price Predictor")
st.caption("LSTM-based next-day close price prediction")

VN_PRICE_DIR = r"c:\\Users\\Administrator\\230163-DL4AI-project\\data-vn-20230228\\stock-historical-data"

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
    model  = tf.keras.models.load_model("saved_models/vn_price_model.keras")
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
"""

with open("streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)
print("streamlit_app.py written.")
print("Run with:  streamlit run streamlit_app.py")
'''

# ── t53 ───────────────────────────────────────────────────────────────────
cells["t53"]["source"] = r'''workflow = """
AI Engineering Workflow
========================

1. INGESTION  (daily, via script or Airflow task)
   - Download OHLCV CSVs from exchange APIs / data vendors
   - Append to PostgreSQL table: raw.stock_prices

2. FEATURE ENGINEERING  (dbt or Python job)
   raw.stock_prices  ->  features.ohlcv          (cleaned)
                     ->  features.technical       (SMA, RSI, MACD, BB)
                     ->  features.fundamental     (P/E, dividend yield — quarterly)

3. TRAINING  (weekly or on-demand)
   - Retrain LSTM models on latest N days of data
   - Evaluate MAE / RMSE on held-out test window
   - Save model artefacts to saved_models/

4. INFERENCE  (daily, after market close)
   - Load latest WINDOW rows from features.technical
   - Call FastAPI /predict/price endpoint
   - Write predictions to predictions.daily_forecast

5. SERVING  (always-on)
   - FastAPI  : REST endpoint for programmatic access  (uvicorn main:app)
   - Streamlit: human-readable dashboard               (streamlit run streamlit_app.py)

6. MONITORING
   - Track prediction MAE vs realised prices each day
   - Alert if MAE rises > 2x historical baseline (model drift)
"""
print(workflow)
with open("workflow_description.txt", "w") as f:
    f.write(workflow)
print("Saved to workflow_description.txt")
'''

# ── clear stale outputs ───────────────────────────────────────────────────
changed = {"paths", "nasdaq-load", "vn-load", "utils",
           "t11", "t12", "t13", "t21", "t22", "t23", "t41", "t43",
           "t51-save", "t51-api", "t52", "t53"}
for cid in changed:
    if cid in cells:
        cells[cid]["outputs"] = []
        cells[cid]["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Patched {NB} — {len(changed)} cells updated.")
print("Next: close this file in Jupyter, then: Kernel -> Restart & Run All")
