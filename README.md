# CS313 Deep Learning for AI — Final Project
## LSTM-Based Stock Price Prediction & Portfolio Optimization

**Student:** Huynh Nhat Huyen — ID 230163

---

## Project Overview

This project applies Long Short-Term Memory (LSTM) neural networks to forecast stock prices in two markets (NASDAQ/AAPL and Vietnam/HPG), generate buy/sell trading signals, score and rank stocks for portfolio construction, and expose predictions through a FastAPI REST server and a Streamlit web dashboard.

---

## Folder Structure

```
230163-DL4AI-project/
├── 230163_project_notebook.ipynb   # Main Jupyter notebook (all tasks)
├── main.py                         # FastAPI REST server (Task 5.1)
├── streamlit_app.py                # Streamlit web app (Task 5.2)
├── workflow_description.txt        # AI engineering workflow notes (Task 5.3)
├── patch_notebook.py               # Utility: patch notebook cell outputs
├── run_notebook.py                 # Utility: execute notebook non-interactively
├── test_t13.py                     # Unit test for Task 1.3 multi-step forecasting
├── test_t13_output.png             # Test output screenshot
├── data_nasdaq_csv/
│   └── csv/                        # Per-ticker OHLCV CSVs (~1 523 NASDAQ companies)
├── data-vn-20230228/
│   ├── stock-historical-data/      # Per-ticker OHLCV CSVs (Vietnam exchange)
│   ├── dividend-history/           # Dividend payment records (Task 2.4)
│   ├── financial-ratio/            # Quarterly financial ratios (Task 2.4)
│   ├── industry-analysis/          # Sector/industry metadata
│   ├── companies.csv               # Vietnam company master list
│   └── ticker-overview.csv         # Market-cap & sector overview
└── saved_models/
    ├── vn_price_model.keras        # Trained VN price-prediction LSTM
    ├── vn_signal_model.keras       # Trained VN buy/sell/hold signal LSTM
    ├── scaler_price.pkl            # MinMaxScaler for price features
    ├── scaler_signal.pkl           # MinMaxScaler for signal features
    └── meta.pkl                    # Window size, feature list metadata
```

---

## Setup

### Requirements

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib plotly \
            fastapi uvicorn pydantic streamlit scipy ta
```

> Python 3.10+ and TensorFlow 2.21 (Keras 3) are recommended. GPU optional.

---

## How to Run

### 1. Jupyter Notebook (all tasks)

```bash
jupyter notebook 230163_project_notebook.ipynb
```

Run cells sequentially. Tasks are labelled `t11`, `t12`, `t13`, `t14`, `t21`–`t24`, `t30`–`t32`, `t41`–`t43`, `t51`–`t53`. The notebook trains models and saves artefacts to `saved_models/` before the API and Streamlit app can be used.

### 2. FastAPI REST Server (Task 5.1)

```bash
uvicorn main:app --reload
```

API available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health, window size, feature count |
| `POST` | `/predict/price` | Next-day close price & log-return prediction |

### 3. Streamlit Web App (Task 5.2)

```bash
streamlit run streamlit_app.py
```

App available at `http://localhost:8501`. Select a Vietnam ticker from the sidebar, view the candlestick chart with SMA20 overlay, and click **Predict next-day close price** to run the LSTM model in real time.

---

## Task Summary

| Task | Description | Target Variable | Model |
|------|-------------|-----------------|-------|
| 1.1 | AAPL next-day price prediction | Log-return (day+1) | 2-layer LSTM |
| 1.2 | AAPL k-th day prediction (k=3, 7) | Log-return (day+k) | 2-layer LSTM per k |
| 1.3 | AAPL 7 consecutive days | Log-return (day+1…+7) | One LSTM per horizon |
| 1.4 | Time-series cross-validation | Log-return | Rolling-window CV |
| 2.1 | HPG (Vietnam) next-day prediction | Log-return (day+1) | 2-layer LSTM |
| 2.2 | HPG k-th day prediction (k=3, 7) | Log-return (day+k) | 2-layer LSTM per k |
| 2.3 | HPG 7 consecutive days | Log-return (day+1…+7) | One LSTM per horizon |
| 2.4 | Dividend & financial data analysis | — | EDA / feature study |
| 3.0 | Label generation (buy/sell/hold) | 3-class signal | Symmetric window |
| 3.1 | Trading signal model training | 3-class | LSTM + class weights |
| 3.2 | Signal evaluation & backtesting | — | Accuracy / F1 |
| 4.1 | Profit scoring of 20 VN stocks | Cumulative return rank | Signal model |
| 4.2 | Risk metrics | Volatility, Sharpe, max drawdown | Statistical |
| 4.3 | Portfolio optimization | Max Sharpe weights | Mean-variance (scipy) |
| 5.1 | FastAPI REST server | — | Serving layer |
| 5.2 | Streamlit web dashboard | — | Interactive UI |
| 5.3 | AI engineering workflow design | — | Architecture diagram |

---

## Model Architecture (Summary)

```
Input (batch, 45, n_features)
  → LSTM(64, return_sequences=True) → BatchNorm → Dropout(0.2)
  → LSTM(32, return_sequences=False) → BatchNorm → Dropout(0.2)
  → Dense(1)   # log-return regression
```

- Features: 12 for NASDAQ (OHLCV + Adj Close + SMA5 + SMA20 + RSI14 + MACD + MACD_H + BB_W), 11 for Vietnam (no Adj Close)
- Target: `log(Close_t+k / Close_t)` — stationary, zero-mean
- Scaler: `MinMaxScaler` fitted on training split only

---

## Known Results

| Task | Metric | Value |
|------|--------|-------|
| 1.3 day+1 | MAE (log-return) | 0.38 |
| 1.3 day+7 | MAE (log-return) | ~0.52 |
| 3.1 VN signals | 3-class accuracy | see notebook |

---

## Notes

- The notebook must be executed before running the API or Streamlit app (models must be saved).
- Data cutoff: 2018-01-01. Chronological split 70 / 15 / 15 %.
- No data leakage: scalers are fitted on the training split only.
