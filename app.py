import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Page setup
st.set_page_config(page_title="Mutual Fund Risk Dashboard", layout="wide")
st.title("📊 Mutual Fund Risk & Performance Dashboard")

# Sidebar Controls
st.sidebar.header("Portfolio Parameters")
fund_ticker = st.sidebar.text_input("Fund Ticker", value="VFIAX")  # Vanguard 500
benchmark_ticker = st.sidebar.text_input(
    "Benchmark Ticker", value="^GSPC"
)  # S&P 500
start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2021-01-01")
)
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
risk_free_rate = (
    st.sidebar.number_input("Risk-Free Rate (%)", value=4.0) / 100
)


# Data Retrieval Function
@st.cache_data
def get_price_series(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"][ticker]
    return data["Close"]


try:
    # Fetch Data
    fund_data = get_price_series(fund_ticker, start_date, end_date)
    bench_data = get_price_series(benchmark_ticker, start_date, end_date)

    # Combine into DataFrame & calculate returns
    df = pd.DataFrame({"Fund": fund_data, "Benchmark": bench_data}).dropna()
    returns = df.pct_change().dropna()

    # Risk Metrics Calculation
    trading_days = 252
    fund_cagr = (
        (1 + returns["Fund"]).prod() ** (trading_days / len(returns))
    ) - 1
    fund_vol = returns["Fund"].std() * np.sqrt(trading_days)
    sharpe_ratio = (fund_cagr - risk_free_rate) / fund_vol

    # Beta Calculation
    cov_matrix = np.cov(returns["Fund"], returns["Benchmark"])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    # Max Drawdown
    cumulative = (1 + returns["Fund"]).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Top KPI Metrics Display
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Return (CAGR)", f"{fund_cagr:.2%}")
    col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col3.metric("Beta (vs. Benchmark)", f"{beta:.2f}")
    col4.metric("Max Drawdown", f"{max_drawdown:.2%}")

    # Charts
    st.subheader("Cumulative Return Performance")
    cum_returns = (1 + returns).cumprod() - 1
    fig_perf = px.line(
        cum_returns, labels={"value": "Return", "index": "Date"}
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Historical Drawdown Analysis")
    fig_dd = px.area(
        drawdown,
        labels={"value": "Drawdown", "index": "Date"},
        title="Underwater Plot",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

except Exception as e:
    st.error(
        f"Error fetching data. Please verify tickers and date range. Details: {str(e)}"
    )
