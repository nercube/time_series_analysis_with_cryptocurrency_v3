import datetime as dt
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from tensorflow.keras.models import load_model

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "log_return",
    "ma7",
    "ma21",
    "volatility_7",
    "rsi14",
]

CRYPTO_CONFIG = {
    "Bitcoin (BTC)": {
        "ticker": "BTC-USD",
        "coingecko": "bitcoin",
        "arima": "arima_model.pkl",
        "prophet": "prophet_model.pkl",
        "lstm": "btc_lstm_logreturn_multifeature.h5",
        "scaler": "btc_feature_scaler.pkl",
    },
    "Ethereum (ETH)": {
        "ticker": "ETH-USD",
        "coingecko": "ethereum",
        "arima": "eth_arima.pkl",
        "prophet": "eth_prophet (1).pkl",
        "lstm": "eth_lstm (1).h5",
        "scaler": "eth_scaler (1).pkl",
    },
    "Tether (USDT)": {
        "ticker": "USDT-USD",
        "coingecko": "tether",
        "arima": "usdt_arima.pkl",
        "prophet": "usdt_prophet (1).pkl",
        "lstm": "usdt_lstm (1).h5",
        "scaler": "usdt_scaler.pkl",
    },
}

# =========================================================
# STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="Crypto Forecast Dashboard",
    page_icon="📈",
    layout="wide",
)

# =========================================================
# CSS
# =========================================================

st.markdown(
    """
<style>
body {
    background-color: #020617;
}

.block-container {
    padding-top: 2rem;
}

.glass-card {
    background: rgba(15,23,42,0.92);
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.18);
    padding: 1rem;
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] {
    background: #020617;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================


def compute_rsi(close: pd.Series, period: int = 14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)


def create_features(df: pd.DataFrame):
    feat = df.copy()

    feat["log_return"] = np.log(
        feat["Close"] / feat["Close"].shift(1)
    )

    feat["ma7"] = feat["Close"].rolling(7).mean()

    feat["ma21"] = feat["Close"].rolling(21).mean()

    feat["volatility_7"] = (
        feat["log_return"].rolling(7).std()
    )

    feat["rsi14"] = compute_rsi(feat["Close"])

    feat = feat.dropna()

    return feat


def make_lstm_sequence(
    feat_df,
    feature_cols,
    scaler,
    window_size=60,
):
    X_raw = feat_df[feature_cols].values

    X_scaled = scaler.transform(X_raw)

    last_window = X_scaled[-window_size:]

    X = np.expand_dims(last_window, axis=0)

    return X


def _load_pickle(path: Path):
    return joblib.load(path)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data(ttl=600)
def load_price_history(ticker: str):
    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)

    return df


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_models(crypto_key: str):
    cfg = CRYPTO_CONFIG[crypto_key]

    arima_model = _load_pickle(
        MODELS_DIR / cfg["arima"]
    )

    prophet_model = _load_pickle(
        MODELS_DIR / cfg["prophet"]
    )

    lstm_model = load_model(
        MODELS_DIR / cfg["lstm"],
        compile=False
    )

    scaler_path = MODELS_DIR / cfg["scaler"]

    st.write("Loading scaler:", scaler_path)

    scaler = _load_pickle(scaler_path)

    return (
        arima_model,
        prophet_model,
        lstm_model,
        scaler,
    )


# =========================================================
# PREDICTIONS
# =========================================================


def predict_arima(model, horizon):
    preds = model.predict(n_periods=horizon)

    return np.asarray(preds, dtype=float)


def predict_prophet(model, hist_df, horizon):
    prophet_df = pd.DataFrame({
        "ds": hist_df.index,
        "y": hist_df["Close"].values,
    })

    future = model.make_future_dataframe(
        periods=horizon,
        freq="D",
    )

    forecast = model.predict(future)

    preds = forecast["yhat"].tail(horizon)

    return preds.to_numpy(dtype=float)


def predict_lstm(
    model,
    scaler,
    hist_df,
    horizon,
    window_size=60,
):
    future_df = hist_df.copy()

    predictions = []

    current_close = float(
        hist_df["Close"].iloc[-1]
    )

    for _ in range(horizon):

        feat_df = create_features(future_df)

        X_last = make_lstm_sequence(
            feat_df,
            FEATURE_COLUMNS,
            scaler,
            window_size,
        )

        pred_log_return = float(
            model.predict(X_last, verbose=0)[0][0]
        )

        next_close = current_close * np.exp(pred_log_return)

        predictions.append(next_close)

        next_date = (
            future_df.index[-1]
            + dt.timedelta(days=1)
        )

        new_row = future_df.iloc[-1].copy()

        new_row["Close"] = next_close
        new_row["Open"] = next_close
        new_row["High"] = next_close
        new_row["Low"] = next_close

        future_df.loc[next_date] = new_row

        current_close = next_close

    return np.array(predictions)


# =========================================================
# MARKET SNAPSHOT
# =========================================================

@st.cache_data(ttl=60)
def fetch_market_snapshot(coin_id: str):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"

    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }

    try:
        r = requests.get(
            url,
            params=params,
            timeout=10,
        )

        r.raise_for_status()

        data = r.json()["market_data"]

        return {
            "price": data["current_price"]["usd"],
            "change_24h": data.get(
                "price_change_percentage_24h",
                0.0,
            ),
            "high_24h": data["high_24h"]["usd"],
            "low_24h": data["low_24h"]["usd"],
            "volume_24h": data["total_volume"]["usd"],
            "market_cap": data["market_cap"]["usd"],
        }

    except Exception:
        return None


# =========================================================
# MAIN PAGE
# =========================================================


def render_dashboard(crypto_key: str):

    cfg = CRYPTO_CONFIG[crypto_key]

    ticker = cfg["ticker"]

    st.title(f"{crypto_key} Forecast Dashboard")

    # =====================================================
    # SIDEBAR
    # =====================================================

    with st.sidebar:

        st.header("Controls")

        horizon = st.slider(
            "Forecast horizon (days)",
            min_value=1,
            max_value=90,
            value=30,
        )

        use_arima = st.checkbox(
            "ARIMA",
            value=True,
        )

        use_prophet = st.checkbox(
            "Prophet",
            value=True,
        )

        use_lstm = st.checkbox(
            "LSTM",
            value=True,
        )

    # =====================================================
    # LOAD DATA + MODELS
    # =====================================================

    price_df = load_price_history(ticker)

    (
        arima_model,
        prophet_model,
        lstm_model,
        scaler,
    ) = load_models(crypto_key)

    # =====================================================
    # PREDICTIONS
    # =====================================================

    preds = {}

    if use_arima:
        preds["ARIMA"] = predict_arima(
            arima_model,
            horizon,
        )

    if use_prophet:
        preds["Prophet"] = predict_prophet(
            prophet_model,
            price_df,
            horizon,
        )

    if use_lstm:
        preds["LSTM"] = predict_lstm(
            lstm_model,
            scaler,
            price_df,
            horizon,
        )

    # =====================================================
    # LAYOUT
    # =====================================================

    left_col, right_col = st.columns(
        [3.2, 1.2],
        gap="large",
    )

    # =====================================================
    # LEFT SIDE
    # =====================================================

    with left_col:

        st.subheader("Price + Forecast")

        recent = price_df.tail(120)

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=recent.index,
                open=recent["Open"],
                high=recent["High"],
                low=recent["Low"],
                close=recent["Close"],
                name="Historical",
            )
        )

        last_date = price_df.index[-1]

        last_close = float(
            price_df["Close"].iloc[-1]
        )

        future_dates = [
            last_date + dt.timedelta(days=i)
            for i in range(1, horizon + 1)
        ]

        for model_name, values in preds.items():

            fig.add_trace(
                go.Scatter(
                    x=[last_date] + future_dates,
                    y=[last_close] + values.tolist(),
                    mode="lines",
                    name=model_name,
                )
            )

        fig.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            plot_bgcolor="rgba(15,23,42,0.95)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb"),
        )

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Predicted Prices")

        if preds:

            table = pd.DataFrame({
                "Model": list(preds.keys()),
                "Predicted Close (USD)": [
                    float(v[-1])
                    for v in preds.values()
                ]
            })

            st.dataframe(
                table.style.format({
                    "Predicted Close (USD)": "${:,.2f}"
                }),
                use_container_width=True,
            )

    # =====================================================
    # RIGHT SIDE
    # =====================================================

    with right_col:

        st.subheader("Market Snapshot")

        snapshot = fetch_market_snapshot(
            cfg["coingecko"]
        )

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True,
        )

        if snapshot:

            st.metric(
                "Price",
                f"${snapshot['price']:,.2f}",
                f"{snapshot['change_24h']:.2f}%",
            )

            st.metric(
                "24h High",
                f"${snapshot['high_24h']:,.2f}",
            )

            st.metric(
                "24h Low",
                f"${snapshot['low_24h']:,.2f}",
            )

            st.metric(
                "24h Volume",
                f"${snapshot['volume_24h']:,.0f}",
            )

            st.metric(
                "Market Cap",
                f"${snapshot['market_cap']:,.0f}",
            )

        else:
            st.warning(
                "Could not load market snapshot."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# =========================================================
# APP ENTRY
# =========================================================

selected_crypto = st.sidebar.radio(
    "Choose Cryptocurrency",
    list(CRYPTO_CONFIG.keys()),
)

render_dashboard(selected_crypto)
