import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Institutional Wealth Dashboard", layout="wide"
)
st.title("💼 Comprehensive Wealth & Risk Analytics Dashboard")
st.caption(
    "Institutional-grade portfolio analytics, crisis stress-testing, and forward wealth simulations."
)

# Region-Categorized Fund & Benchmark Mapping
FUNDS_BY_REGION = {
    "US Market": {
        "VFIAX - Vanguard 500 Index Fund": "VFIAX",
        "FXAIX - Fidelity 500 Index Fund": "FXAIX",
        "SWPPX - Schwab S&P 500 Index Fund": "SWPPX",
        "FBGRX - Fidelity Blue Chip Growth": "FBGRX",
        "AGTHX - American Funds Growth Fund of America": "AGTHX",
        "DODGX - Dodge & Cox Stock Fund": "DODGX",
        "SGRAX - Allspring Large Cap Core (Wells Fargo)": "SGRAX",
        "EKJAX - Allspring Discovery Growth (Wells Fargo)": "EKJAX",
        "NVHAX - Allspring Special Small Cap Value (Wells Fargo)": "NVHAX",
        "VBTLX - Vanguard Total Bond Market Index": "VBTLX",
        "Custom Ticker...": "CUSTOM",
    },
    "EMEA Market (Europe, Middle East, Africa)": {
        "0P00000B2H.L - Fidelity European Growth Fund": "0P00000B2H.L",
        "0P0000XW01.L - Vanguard LifeStrategy 60% Equity": "0P0000XW01.L",
        "IWDA.L - iShares Core MSCI World UCITS ETF": "IWDA.L",
        "VEUR.L - Vanguard FTSE Developed Europe UCITS ETF": "VEUR.L",
        "IEAC.L - iShares Core EUR Corporate Bond UCITS ETF": "IEAC.L",
        "Custom Ticker...": "CUSTOM",
    },
    "Global & Emerging Markets": {
        "VTIAX - Vanguard Total International Stock Index": "VTIAX",
        "AEPGX - American Funds EuroPacific Growth": "AEPGX",
        "VTI - Vanguard Total Stock Market ETF": "VTI",
        "Custom Ticker...": "CUSTOM",
    },
}

BENCHMARKS_BY_REGION = {
    "US Market": {
        "^GSPC - S&P 500 Index (Broad US Large Cap)": "^GSPC",
        "^IXIC - NASDAQ Composite (Tech & Growth)": "^IXIC",
        "^DJI - Dow Jones Industrial Average": "^DJI",
        "^RUT - Russell 2000 Index (US Small Cap)": "^RUT",
        "AGG - iShares Core U.S. Aggregate Bond ETF": "AGG",
        "Custom Ticker...": "CUSTOM",
    },
    "EMEA Market (Europe, Middle East, Africa)": {
        "^GDAXI - DAX 40 Index (Germany Blue-Chip)": "^GDAXI",
        "^FTSE - FTSE 100 Index (UK)": "^FTSE",
        "^FCHI - CAC 40 Index (France)": "^FCHI",
        "FTSEMIB.MI - FTSE MIB Index (Italy)": "FTSEMIB.MI",
        "^STOXX50E - EURO STOXX 50 Index (Eurozone)": "^STOXX50E",
        "Custom Ticker...": "CUSTOM",
    },
    "Global & Emerging Markets": {
        "EFA - iShares MSCI EAFE ETF (Developed Ex-US)": "EFA",
        "EEM - iShares MSCI Emerging Markets ETF": "EEM",
        "ACWI - iShares MSCI ACWI ETF (All Country World)": "ACWI",
        "Custom Ticker...": "CUSTOM",
    },
}

# Sidebar Controls
st.sidebar.header("1. Geographic Focus")
selected_region = st.sidebar.selectbox(
    "Select Region Focus", list(FUNDS_BY_REGION.keys())
)

st.sidebar.header("2. Asset Selection")

current_funds = FUNDS_BY_REGION[selected_region]
current_benchmarks = BENCHMARKS_BY_REGION[selected_region]

selected_fund_label = st.sidebar.selectbox(
    "Select Mutual Fund", list(current_funds.keys())
)
if current_funds[selected_fund_label] == "CUSTOM":
    fund_ticker = (
        st.sidebar.text_input("Enter Custom Fund Ticker", value="VFIAX")
        .strip()
        .upper()
    )
else:
    fund_ticker = current_funds[selected_fund_label]

selected_bench_label = st.sidebar.selectbox(
    "Select Benchmark Index", list(current_benchmarks.keys())
)
if current_benchmarks[selected_bench_label] == "CUSTOM":
    benchmark_ticker = (
        st.sidebar.text_input("Enter Custom Benchmark Ticker", value="^GSPC")
        .strip()
        .upper()
    )
else:
    benchmark_ticker = current_benchmarks[selected_bench_label]

start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2020-01-01")
)
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

st.sidebar.header("3. Simulation Settings")
initial_inv = st.sidebar.number_input(
    "Initial Portfolio Value ($)", value=100000, step=10000
)
sim_years = st.sidebar.slider("Projection Horizon (Years)", 1, 10, 5)


@st.cache_data
def get_price_series(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    if data.empty:
        return pd.Series(dtype=float)
    if isinstance(data.columns, pd.MultiIndex):
        return data["Close"][ticker].dropna()
    return data["Close"].dropna()


try:
    fund_data = get_price_series(fund_ticker, start_date, end_date)
    bench_data = get_price_series(benchmark_ticker, start_date, end_date)

    if fund_data.empty or bench_data.empty:
        st.warning(
            "⚠️ No market data found for the selected ticker combination or date range. Please try expanding the date range or choosing different tickers."
        )
        st.stop()

    df = pd.DataFrame({"Fund": fund_data, "Benchmark": bench_data}).dropna()
    returns = df.pct_change().dropna()

    # Guard against insufficient overlapping date records
    if len(returns) < 5:
        st.warning(
            "⚠️ Insufficient overlapping historical trading days found between these two tickers. Please select a broader date range or matching market region."
        )
        st.stop()

    # Core Calculations with Safeguards against Division by Zero
    trading_days = 252
    num_records = len(returns)

    fund_cagr = (
        (1 + returns["Fund"]).prod() ** (trading_days / num_records)
    ) - 1
    fund_vol = returns["Fund"].std() * np.sqrt(trading_days)
    risk_free_rate = 0.04

    # Safeguard Sharpe Ratio
    sharpe_ratio = (
        (fund_cagr - risk_free_rate) / fund_vol if fund_vol > 0 else 0.0
    )

    # Safeguard Beta
    cov_matrix = np.cov(returns["Fund"], returns["Benchmark"])
    bench_variance = cov_matrix[1, 1]
    beta = (cov_matrix[0, 1] / bench_variance) if bench_variance > 0 else 1.0

    cumulative = (1 + returns["Fund"]).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    # Display KPI Metrics Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Return", f"{fund_cagr:.2%}")
    col2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col3.metric(f"Beta (vs {benchmark_ticker})", f"{beta:.2f}")
    col4.metric("Worst Drawdown", f"{max_drawdown:.2%}")

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
            "Evaluates fund performance during major historical volatility periods:"
        )

        crises = {
            "2020 COVID Crash (Feb - Mar 2020)": ("2020-02-19", "2020-03-23"),
            "2022 Inflation / Rate Hike Selloff": ("2022-01-03", "2022-10-12"),
        }

        crisis_results = []
        for name, (c_start, c_end) in crises.items():
            try:
                c_df = df.loc[c_start:c_end]
                if not c_df.empty and len(c_df) > 1:
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
