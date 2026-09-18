import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Client Wealth Dashboard", layout="wide")
st.title("💼 Client Investment Performance Summary")
st.caption(
    "A clean overview of fund performance, returns, and risk exposure."
)

# Sidebar Controls
st.sidebar.header("Select Investments")
fund_ticker = st.sidebar.text_input("Mutual Fund Ticker", value="VFIAX")
benchmark_ticker = st.sidebar.text_input("Benchmark Index", value="^GSPC")
start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2021-01-01")
)
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))


@st.cache_data
def get_price_series(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"][ticker]
    return data["Close"]


try:
    fund_data = get_price_series(fund_ticker, start_date, end_date)
    bench_data = get_price_series(benchmark_ticker, start_date, end_date)

    df = pd.DataFrame({"Fund": fund_data, "Benchmark": bench_data}).dropna()
    returns = df.pct_change().dropna()

    # Calculations
    trading_days = 252
    fund_cagr = (
        (1 + returns["Fund"]).prod() ** (trading_days / len(returns))
    ) - 1
    fund_vol = returns["Fund"].std() * np.sqrt(trading_days)
    risk_free_rate = 0.04
    sharpe_ratio = (fund_cagr - risk_free_rate) / fund_vol

    cov_matrix = np.cov(returns["Fund"], returns["Benchmark"])
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]

    cumulative = (1 + returns["Fund"]).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # 1. Executive Summary Box
    st.info(
        f"📌 **Executive Takeaway:** Over this period, **{fund_ticker}** generated an average yearly return of **{fund_cagr:.1%}**. "
        f"During market downturns, the fund's largest drop from peak to trough was **{max_drawdown:.1%}**."
    )

    # 2. Client-Friendly Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Average Yearly Growth",
        f"{fund_cagr:.1%}",
        help="The overall annual return rate earned by the fund.",
    )
    col2.metric(
        "Risk Efficiency Score",
        f"{sharpe_ratio:.2f}",
        help="Measures excess return earned per unit of risk. Higher than 1.0 is considered good efficiency.",
    )
    col3.metric(
        "Market Volatility Factor",
        f"{beta:.2f}",
        help="Measures how closely the fund follows market swings. 1.0 means it moves in tandem with the index.",
    )
    col4.metric(
        "Worst Historical Drop",
        f"{max_drawdown:.1%}",
        help="The maximum peak-to-bottom loss experienced over the selected timeline.",
    )

    # 3. Main Chart: Growth of $10,000
    st.subheader("📈 Growth of a $10,000 Investment")
    portfolio_value = (1 + returns) * 10000
    cum_growth = (1 + returns).cumprod() * 10000
    fig_growth = px.line(
        cum_growth,
        labels={"value": "Portfolio Value ($)", "index": "Date"},
        title=f"Initial $10,000 in {fund_ticker} vs Benchmark",
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    # 4. Hidden Technical Expander
    with st.expander("🔍 Click to view deeper risk analytics & drawdown charts"):
        st.write(
            "**Historical Downside Risk (Underwater Chart)**: Displays how far the portfolio dipped below its all-time high during market pullbacks."
        )
        fig_dd = px.area(
            drawdown,
            labels={"value": "Decline from Peak", "index": "Date"},
            title="Portfolio Drawdown History",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

except Exception as e:
    st.error(f"Unable to load fund details. Please verify ticker: {str(e)}")
