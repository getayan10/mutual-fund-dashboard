import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Institutional Wealth Dashboard", layout="wide")
st.title("💼 Comprehensive Wealth & Risk Analytics Dashboard")
st.caption(
    "Institutional-grade portfolio analytics, crisis stress-testing, and forward wealth simulations."
)

# Pre-configured Ticker Dictionaries
FUND_OPTIONS = {
    "VFIAX - Vanguard 500 Index Fund": "VFIAX",
    "FXAIX - Fidelity 500 Index Fund": "FXAIX",
    "FBGRX - Fidelity Blue Chip Growth": "FBGRX",
    "AGTHX - American Funds Growth Fund": "AGTHX",
    "SGRAX - Allspring Large Cap Core (Wells Fargo)": "SGRAX",
    "EKJAX - Allspring Discovery Growth (Wells Fargo)": "EKJAX",
    "VBTLX - Vanguard Total Bond Market": "VBTLX",
    "Custom Ticker...": "CUSTOM",
}

BENCHMARK_OPTIONS = {
    "^GSPC - S&P 500 Index": "^GSPC",
    "^IXIC - NASDAQ Composite": "^IXIC",
    "^RUT - Russell 2000 (Small Cap)": "^RUT",
    "AGG - iShares Core U.S. Aggregate Bond": "AGG",
    "Custom Ticker...": "CUSTOM",
}

# Sidebar Controls
st.sidebar.header("1. Investment Selection")

# Fund Dropdown Selection
selected_fund_label = st.sidebar.selectbox(
    "Select Mutual Fund", list(FUND_OPTIONS.keys())
)
if FUND_OPTIONS[selected_fund_label] == "CUSTOM":
    fund_ticker = (
        st.sidebar.text_input("Enter Fund Ticker", value="VFIAX")
        .strip()
        .upper()
    )
else:
    fund_ticker = FUND_OPTIONS[selected_fund_label]

# Benchmark Dropdown Selection
selected_bench_label = st.sidebar.selectbox(
    "Select Benchmark Index", list(BENCHMARK_OPTIONS.keys())
)
if BENCHMARK_OPTIONS[selected_bench_label] == "CUSTOM":
    benchmark_ticker = (
        st.sidebar.text_input("Enter Benchmark Ticker", value="^GSPC")
        .strip()
        .upper()
    )
else:
    benchmark_ticker = BENCHMARK_OPTIONS[selected_bench_label]

start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2020-01-01")
)
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

st.sidebar.header("2. Simulation Settings")
initial_inv = st.sidebar.number_input(
    "Initial Portfolio Value ($)", value=100000, step=10000
)
sim_years = st.sidebar.slider("Projection Horizon (Years)", 1, 10, 5)


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

    # Core Metrics
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

    # KPI Overview Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Return", f"{fund_cagr:.2%}")
    col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col3.metric("Beta (vs Market)", f"{beta:.2f}")
    col4.metric("Worst Drawdown", f"{max_drawdown:.2%}")

    # Tabs Layout
    tab1, tab2, tab3 = st.tabs(
        [
            "📈 Performance & Growth",
            "🔮 Monte Carlo Simulation",
            "💥 Historical Crisis Stress-Test",
        ]
    )

    with tab1:
        st.subheader(f"Portfolio Growth (${initial_inv:,} Initial)")
        cum_growth = (1 + returns).cumprod() * initial_inv
        fig_growth = px.line(
            cum_growth, labels={"value": "Portfolio Value ($)", "index": "Date"}
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    with tab2:
        st.subheader(
            f"Monte Carlo Forward Projection ({sim_years} Years | 1,000 Scenarios)"
        )

        mean_daily = returns["Fund"].mean()
        std_daily = returns["Fund"].std()
        sim_days = sim_years * trading_days
        num_simulations = 1000

        sim_returns = np.random.normal(
            mean_daily, std_daily, (sim_days, num_simulations)
        )
        price_paths = initial_inv * np.cumprod(1 + sim_returns, axis=0)

        p10 = np.percentile(price_paths[-1, :], 10)
        p50 = np.percentile(price_paths[-1, :], 50)
        p90 = np.percentile(price_paths[-1, :], 90)

        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Conservative (10th Percentile)", f"${p10:,.0f}")
        scol2.metric("Expected Median (50th Percentile)", f"${p50:,.0f}")
        scol3.metric("Optimistic (90th Percentile)", f"${p90:,.0f}")

        fig_sim = go.Figure()
        for i in range(min(50, num_simulations)):
            fig_sim.add_trace(
                go.Scatter(
                    y=price_paths[:, i],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(100, 100, 250, 0.15)"),
                    showlegend=False,
                )
            )

        fig_sim.add_trace(
            go.Scatter(
                y=np.median(price_paths, axis=1),
                mode="lines",
                name="Median Path",
                line=dict(color="blue", width=3),
            )
        )
        fig_sim.update_layout(
            xaxis_title="Trading Days Forward",
            yaxis_title="Projected Wealth ($)",
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    with tab3:
        st.subheader("Historical Crisis Replay")
        st.write(
            "Evaluates how the selected fund performed during key historical stress periods:"
        )

        crises = {
            "2020 COVID Crash (Feb - Mar 2020)": ("2020-02-19", "2020-03-23"),
            "2022 Inflation / Rate Hike Selloff": ("2022-01-03", "2022-10-12"),
        }

        crisis_results = []
        for name, (c_start, c_end) in crises.items():
            try:
                c_df = df.loc[c_start:c_end]
                if not c_df.empty:
                    f_return = (
                        c_df["Fund"].iloc[-1] / c_df["Fund"].iloc[0]
                    ) - 1
                    b_return = (
                        c_df["Benchmark"].iloc[-1] / c_df["Benchmark"].iloc[0]
                    ) - 1
                    crisis_results.append(
                        {
                            "Crisis Period": name,
                            f"{fund_ticker} Return": f"{f_return:.2%}",
                            f"{benchmark_ticker} Return": f"{b_return:.2%}",
                            "Outperformance": f"{(f_return - b_return):.2%}",
                        }
                    )
            except Exception:
                pass

        if crisis_results:
            st.table(pd.DataFrame(crisis_results))
        else:
            st.warning(
                "Selected start date does not overlap with historical crisis events. Extend start date back to 2020."
            )

except Exception as e:
    st.error(f"Error executing analytics engine: {str(e)}")
