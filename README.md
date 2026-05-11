# CS313 Deep Learning for AI — Final Project
## LSTM-Based Stock Price Prediction & Portfolio Optimization

**Student:** Huynh Nhat Huyen — ID 230163

---

## Project Overview

This project applies Long Short-Term Memory (LSTM) neural networks to forecast stock prices in two markets (NASDAQ/AAPL and Vietnam/HPG), generate buy/sell/hold trading signals, rank stocks for portfolio construction using mean-variance optimization, and expose predictions through a FastAPI REST server and a Streamlit web dashboard.

All prediction tasks use **log-returns** `r_t = log(P_t / P_{t-1})` as the target variable rather than raw prices, ensuring stationarity across different price regimes.

---

## Folder Structure

```
230163-DL4AI-project/
├── 230163_project_notebook.ipynb   # Main Jupyter notebook (all tasks)
├── main.py                         # FastAPI REST server (Task 5.1)
├── streamlit_app.py                # Streamlit web dashboard (Task 5.2)
├── pipeline.html                   # AI engineering workflow (visual, Task 5.3)
├── workflow_description.txt        # Plain-text workflow summary (Task 5.3)
├── .gitignore
├── data_nasdaq_csv/
│   └── csv/                        # Per-ticker OHLCV CSVs (~1 523 NASDAQ companies)
├── data-vn-20230228/
│   ├── stock-historical-data/      # Per-ticker OHLCV CSVs (Vietnam exchange)
│   ├── dividend-history/           # Dividend payment records
│   ├── financial-ratio/            # Quarterly financial ratios
│   ├── industry-analysis/          # Sector/industry metadata
│   ├── companies.csv               # Vietnam company master list
│   └── ticker-overview.csv         # Market-cap & sector overview
└── saved_models/                   # Generated after running notebook (git-ignored)
    ├── vn_price_model.keras
    ├── vn_signal_model.keras
    ├── scaler_price.pkl
    ├── scaler_signal.pkl
    └── meta.pkl
```

---

## Setup

### Requirements

- **Python 3.12** (TensorFlow does not support Python 3.13+)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn \
            plotly fastapi uvicorn pydantic streamlit scipy
```

---

## How to Run

### 1. Jupyter Notebook (all tasks)

```bash
jupyter notebook 230163_project_notebook.ipynb
```

Run cells sequentially from top to bottom. The notebook trains and saves models to `saved_models/` — this must be done before running the API or Streamlit app.

Key cell IDs: `imports` → `paths` → `utils` → `nasdaq-eda` → `t11`–`t14` → `vn-eda` → `t21`–`t24` → `t3-*` → `t41`–`t43` → `t51-save` → `t51-api` → `t52` → `t53` → `key-findings`

### 2. FastAPI REST Server (Task 5.1)

```bash
.venv\Scripts\Activate.ps1
uvicorn main:app --reload
```

Available at `http://127.0.0.1:8000`. Interactive Swagger docs at `http://127.0.0.1:8000/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server status, window size, feature count |
| `POST` | `/predict/price` | Next-day close price & log-return prediction |

### 3. Streamlit Web Dashboard (Task 5.2)

```bash
.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Available at `http://localhost:8501`. Select a Vietnam ticker, view the candlestick chart with SMA20 overlay, and click **Predict next-day close price** to run the LSTM in real time.

---

## Task Summary

| Task | Description | Target | Model |
|------|-------------|--------|-------|
| 1.1 | AAPL next-day prediction | Log-return day+1 | 2-layer LSTM |
| 1.2 | AAPL k-th day prediction (k=3,7) | Log-return day+k | 2-layer LSTM per k |
| 1.3 | AAPL 7 consecutive days | Log-return day+1…+7 | One LSTM per horizon |
| 1.4 | Time-series cross-validation (5-fold) | Log-return | Rolling-window CV |
| 2.1 | HPG (Vietnam) next-day prediction | Log-return day+1 | 2-layer LSTM |
| 2.2 | HPG k-th day prediction (k=3,7) | Log-return day+k | 2-layer LSTM per k |
| 2.3 | HPG 7 consecutive days | Log-return day+1…+7 | One LSTM per horizon |
| 2.4 | Dividend & financial ratio analysis | — | EDA |
| 3.0 | Buy/sell/hold label generation | 3-class signal | Symmetric window |
| 3.1 | Trading signal classifier | 3-class | LSTM + class weights |
| 3.2 | Signal evaluation | — | Accuracy / F1 |
| 4.1 | Profit scoring (20 VN stocks) | Predicted return rank | LSTM |
| 4.2 | Risk management | Volatility, Sharpe, max drawdown | Statistical |
| 4.3 | Portfolio optimization | Max-Sharpe weights | Mean-variance (scipy) |
| 5.1 | FastAPI REST server | — | Serving |
| 5.2 | Streamlit web dashboard | — | Interactive UI |
| 5.3 | AI engineering workflow (XML) | — | Architecture design |

---

## Model Architecture

```
Input  (batch, 45, n_features)
  → LSTM(64, return_sequences=True, recurrent_dropout=0.1)
  → BatchNormalization → Dropout(0.2)
  → LSTM(32, recurrent_dropout=0.1)
  → Dropout(0.1)
  → Dense(1)                          # log-return regression
```

- **Loss:** Huber (delta=0.02) — robust to large return spikes
- **Optimizer:** Adam (lr=5e-4)
- **Callbacks:** EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5, patience=3)
- **Window:** 45 trading days (~2 months of context)
- **Split:** 70% train / 15% validation / 15% test (chronological, no shuffling)
- **Features (NASDAQ):** Open, High, Low, Close, Volume, Adjusted Close, SMA-5, SMA-20, RSI-14, MACD, MACD-H, BB-Width (12 total)
- **Features (Vietnam):** Same minus Adjusted Close (11 total)
- **Target:** `log(Close_t / Close_{t-1})` — stationary across price regimes

---

## Design Notes

- **Why log-returns?** Raw scaled prices are non-stationary — a model trained on historical low prices fails to generalize to a bull-run or crash regime. Log-returns are approximately zero-mean and constant-variance regardless of price level.
- **Why Huber loss?** MSE forces the model to chase rare extreme spikes (±15–20% crashes). Huber switches to MAE for large errors, so the model focuses on typical days instead.
- **No data leakage:** `MinMaxScaler` is fitted on the training split only and applied to validation/test.
- **Direction accuracy** (`Dir`) is reported alongside MAE/RMSE — for trading, sign prediction above 52% is more useful than a low MAE on near-zero returns.
