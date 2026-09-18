import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Institutional Wealth & Portfolio Analytics", layout="wide"
)
st.title("💼 Institutional Wealth & Portfolio Analytics Dashboard")
st.caption(
    "Wealth-management decision support platform: client profiling, downside risk, rolling metrics, and goal-based SIP modeling."
)

# ------------------------------------------------------------------------------
# 1. ASSET MAPPING & FUND-SPECIFIC BENCHMARKS
# ------------------------------------------------------------------------------
FUNDS_BY_REGION = {
    "US Market": {
        "VFIAX - Vanguard 500 Index Fund": "VFIAX",
        "FXAIX - Fidelity 500 Index Fund": "FXAIX",
        "FBGRX - Fidelity Blue Chip Growth": "FBGRX",
        "AGTHX - American Funds Growth Fund": "AGTHX",
        "DODGX - Dodge & Cox Stock Fund": "DODGX",
        "SGRAX - Allspring Large Cap Core (Wells Fargo)": "SGRAX",
        "EKJAX - Allspring Discovery Growth (Wells Fargo)": "EKJAX",
        "VBTLX - Vanguard Total Bond Market": "VBTLX",
        "Custom Ticker...": "CUSTOM",
    },
    "EMEA Market": {
        "IWDA.L - iShares Core MSCI World UCITS": "IWDA.L",
        "VEUR.L - Vanguard FTSE Developed Europe UCITS": "VEUR.L",
        "ISF.L - iShares Core FTSE 100 UCITS": "ISF.L",
        "EXW1.DE - iShares EURO STOXX 50 UCITS": "EXW1.DE",
        "IEAC.L - iShares Core EUR Corporate Bond UCITS": "IEAC.L",
        "Custom Ticker...": "CUSTOM",
    },
    "Global & Emerging": {
        "VTIAX - Vanguard Total International Stock Index": "VTIAX",
        "AEPGX - American Funds EuroPacific Growth": "AEPGX",
        "VTI - Vanguard Total Stock Market ETF": "VTI",
        "Custom Ticker...": "CUSTOM",
    },
}

# Fund-specific default benchmark mapping
DEFAULT_BENCHMARK_MAP = {
    "VFIAX": "^GSPC",
    "FXAIX": "^GSPC",
    "FBGRX": "^IXIC",
    "AGTHX": "^GSPC",
    "DODGX": "^GSPC",
    "SGRAX": "IWV",  # Russell 3000 Growth proxy
    "EKJAX": "^RUT",
    "VBTLX": "AGG",
    "IWDA.L": "ACWI",  # MSCI World
    "VEUR.L": "^STOXX50E",
    "ISF.L": "^FTSE",
    "EXW1.DE": "^STOXX50E",
    "IEAC.L": "AGG",
    "VTIAX": "EFA",
    "AEPGX": "EFA",
    "VTI": "^GSPC",
}

BENCHMARK_OPTIONS = {
    "^GSPC - S&P 500 Index": "^GSPC",
    "^IXIC - NASDAQ Composite": "^IXIC",
    "IWV - Russell 3000 Index": "IWV",
    "^RUT - Russell 2000 Index": "^RUT",
    "AGG - US Aggregate Bond ETF": "AGG",
    "^STOXX50E - EURO STOXX 50": "^STOXX50E",
    "^FTSE - FTSE 100 Index": "^FTSE",
    "^GDAXI - DAX 40 Index": "^GDAXI",
    "EFA - MSCI EAFE ETF": "EFA",
    "ACWI - MSCI ACWI ETF": "ACWI",
    "Custom Ticker...": "CUSTOM",
}

# ------------------------------------------------------------------------------
# 2. SIDEBAR INPUTS & CLIENT SUITABILITY
# ------------------------------------------------------------------------------
st.sidebar.header("1. Geographic & Asset Focus")
selected_region = st.sidebar.selectbox(
    "Select Region", list(FUNDS_BY_REGION.keys())
)

current_funds = FUNDS_BY_REGION[selected_region]
selected_fund_label = st.sidebar.selectbox(
    "Select Fund / Asset", list(current_funds.keys())
)

if current_funds[selected_fund_label] == "CUSTOM":
    fund_ticker = (
        st.sidebar.text_input("Enter Fund Ticker", value="VFIAX")
        .strip()
        .upper()
    )
    default_bench = "^GSPC"
else:
    fund_ticker = current_funds[selected_fund_label]
    default_bench = DEFAULT_BENCHMARK_MAP.get(fund_ticker, "^GSPC")

# Auto-map benchmark with manual override
st.sidebar.caption(f"Default Benchmark mapped: **{default_bench}**")
bench_override = st.sidebar.checkbox("Override Default Benchmark?")
if bench_override:
    selected_bench_label = st.sidebar.selectbox(
        "Select Custom Benchmark", list(BENCHMARK_OPTIONS.keys())
    )
    if BENCHMARK_OPTIONS[selected_bench_label] == "CUSTOM":
        benchmark_ticker = (
            st.sidebar.text_input("Enter Benchmark Ticker", value="^GSPC")
            .strip()
            .upper()
        )
    else:
        benchmark_ticker = BENCHMARK_OPTIONS[selected_bench_label]
else:
    benchmark_ticker = default_bench

st.sidebar.header("2. Financial & Dates Parameters")
start_date = st.sidebar.date_input(
    "Start Date", value=pd.to_datetime("2020-01-01")
)
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
risk_free_rate_pct = st.sidebar.number_input(
    "Dynamic Risk-Free Rate (%)",
    min_value=0.0,
    max_value=15.0,
    value=4.25,
    step=0.25,
)
risk_free_rate = risk_free_rate_pct / 100.0

st.sidebar.header("3. Wealth Planning & Goal Inputs")
initial_inv = st.sidebar.number_input(
    "Initial Lump Sum ($)", value=100000, step=10000
)
monthly_contrib = st.sidebar.number_input(
    "Monthly Contribution ($)", value=1000, step=250
)
step_up_pct = (
    st.sidebar.number_input(
        "Annual Contribution Step-Up (%)", value=5.0, step=1.0
    )
    / 100.0
)
horizon_years = st.sidebar.slider("Investment Horizon (Years)", 1, 30, 10)
target_wealth = st.sidebar.number_input(
    "Target Wealth Goal ($)", value=500000, step=25000
)

st.sidebar.header("4. Client Risk Profiler Questionnaire")
age_band = st.sidebar.selectbox(
    "Client Age Group", ["< 35", "35 - 50", "51 - 65", "65+"]
)
inv_horizon = st.sidebar.selectbox(
    "Liquidity Horizon",
    ["< 2 Years", "2 - 5 Years", "5 - 10 Years", "10+ Years"],
)
loss_tolerance = st.sidebar.selectbox(
    "Max Tolerable 1-Yr Drawdown", ["< 5%", "5% - 15%", "15% - 25%", "> 25%"]
)

# Calculate Client Risk Score (1 to 4)
risk_points = 0
risk_points += (
    4
    if loss_tolerance == "> 25%"
    else (
        3
        if loss_tolerance == "15% - 25%"
        else (2 if loss_tolerance == "5% - 15%" else 1)
    )
)
risk_points += (
    3
    if inv_horizon == "10+ Years"
    else (2 if inv_horizon == "5 - 10 Years" else 1)
)

if risk_points <= 3:
    client_risk_band = "Conservative"
elif risk_points <= 5:
    client_risk_band = "Balanced"
elif risk_points <= 7:
    client_risk_band = "Growth"
else:
    client_risk_band = "Aggressive Growth"


# ------------------------------------------------------------------------------
# 3. DATA FETCHING & COMPUTATION ENGINE
# ------------------------------------------------------------------------------
@st.cache_data
def get_price_series(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return pd.Series(dtype=float)
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.levels[0]:
                df_close = data["Close"]
                if ticker in df_close.columns:
                    return df_close[ticker].dropna()
                return df_close.iloc[:, 0].dropna()
        if "Close" in data.columns:
            return data["Close"].dropna()
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


try:
    fund_data = get_price_series(fund_ticker, start_date, end_date)
    bench_data = get_price_series(benchmark_ticker, start_date, end_date)

    if fund_data.empty or bench_data.empty:
        st.warning(
            "⚠️ Unable to load market data for selected assets/dates. Please adjust selections."
        )
        st.stop()

    df = pd.DataFrame({"Fund": fund_data, "Benchmark": bench_data}).dropna()
    returns = df.pct_change().dropna()

    if len(returns) < 10:
        st.warning("⚠️ Insufficient historical data for reliable analysis.")
        st.stop()

    trading_days = 252
    num_records = len(returns)

    # Core Metrics
    fund_cagr = (
        (1 + returns["Fund"]).prod() ** (trading_days / num_records)
    ) - 1
    bench_cagr = (
        (1 + returns["Benchmark"]).prod() ** (trading_days / num_records)
    ) - 1

    fund_vol = returns["Fund"].std() * np.sqrt(trading_days)
    bench_vol = returns["Benchmark"].std() * np.sqrt(trading_days)

    sharpe_ratio = (
        (fund_cagr - risk_free_rate) / fund_vol if fund_vol > 0 else 0.0
    )

    cov_matrix = np.cov(returns["Fund"], returns["Benchmark"])
    bench_var = cov_matrix[1, 1]
    beta = (cov_matrix[0, 1] / bench_var) if bench_var > 0 else 1.0

    cumulative = (1 + returns["Fund"]).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    # Downside Metrics
    downside_returns = returns["Fund"][returns["Fund"] < 0]
    downside_dev = (
        np.sqrt(np.mean(downside_returns**2)) * np.sqrt(trading_days)
        if len(downside_returns) > 0
        else 0.0001
    )
    sortino_ratio = (fund_cagr - risk_free_rate) / downside_dev
    calmar_ratio = (
        fund_cagr / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0
    )

    var_95_daily = np.percentile(returns["Fund"], 5)
    var_95_annual = var_95_daily * np.sqrt(trading_days)
    cvar_95_daily = returns["Fund"][returns["Fund"] <= var_95_daily].mean()

    # Recovery Time Calculation
    is_in_dd = drawdown < 0
    dd_lengths = []
    current_length = 0
    for in_dd in is_in_dd:
        if in_dd:
            current_length += 1
        else:
            if current_length > 0:
                dd_lengths.append(current_length)
                current_length = 0
    max_recovery_days = max(dd_lengths) if dd_lengths else 0

    # Benchmark-Relative Metrics
    alpha = fund_cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))
    tracking_diff = returns["Fund"] - returns["Benchmark"]
    tracking_error = tracking_diff.std() * np.sqrt(trading_days)
    information_ratio = (
        (fund_cagr - bench_cagr) / tracking_error if tracking_error > 0 else 0.0
    )

    up_mask = returns["Benchmark"] > 0
    down_mask = returns["Benchmark"] < 0
    up_capture = (
        returns["Fund"][up_mask].mean() / returns["Benchmark"][up_mask].mean()
        if returns["Benchmark"][up_mask].mean() != 0
        else 1.0
    )
    down_capture = (
        returns["Fund"][down_mask].mean()
        / returns["Benchmark"][down_mask].mean()
        if returns["Benchmark"][down_mask].mean() != 0
        else 1.0
    )

    # ------------------------------------------------------------------------------
    # 4. DASHBOARD TABS
    # ------------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🎯 Suitability & Insights",
            "📊 Downside & Relative Risk",
            "📈 Rolling Metrics",
            "🔮 Goal-Based Monte Carlo",
            "💥 Crisis Stress-Test",
        ]
    )

    # TAB 1: SUITABILITY & AUTOMATED INSIGHTS
    with tab1:
        st.subheader("Client Risk Band Mapping & Investment Insights")

        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Assigned Risk Profile", client_risk_band)
        mcol2.metric("Fund Annualized Volatility", f"{fund_vol:.2%}")
        mcol3.metric("Fund Beta (vs Benchmark)", f"{beta:.2f}")

        # Rule-based Insights Generation
        insights = []
        if fund_cagr > bench_cagr and max_drawdown < (
            (1 + returns["Benchmark"]).cumprod().cummax()
            - (1 + returns["Benchmark"]).cumprod()
        ).min():
            insights.append(
                "✅ **Superior Efficiency:** The fund outperformed its benchmark on an absolute basis while maintaining lower downside risk."
            )
        elif fund_cagr > bench_cagr:
            insights.append(
                "⚠️ **Growth at High Risk:** The fund delivered higher annualized returns than the benchmark, but experienced larger peak-to-trough drawdowns."
            )
        else:
            insights.append(
                "ℹ️ **Benchmark Lag:** The fund underperformed its benchmark over the selected historical period."
            )

        if client_risk_band in ["Conservative", "Balanced"] and fund_vol > 0.18:
            insights.append(
                "🚨 **Suitability Mismatch:** Fund volatility exceeds the threshold suitable for Conservative/Balanced profiles."
            )
        else:
            insights.append(
                "✅ **Suitability Aligned:** Fund risk metrics fit well within the client's declared risk capacity."
            )

        if down_capture < 0.90:
            insights.append(
                f"🛡️ **Downside Protection:** Excellent downside capital preservation, capturing only {down_capture:.0%} of market declines."
            )

        st.info("\n\n".join(insights))

        st.markdown("---")
        st.subheader("Executive Performance Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fund CAGR", f"{fund_cagr:.2%}")
        col2.metric("Benchmark CAGR", f"{bench_cagr:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col4.metric("Worst Drawdown", f"{max_drawdown:.2%}")

    # TAB 2: DOWNSIDE & BENCHMARK-RELATIVE RISK
    with tab2:
        st.subheader("Institutional Downside Risk Metrics")
        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        dcol1.metric(
            "Sortino Ratio",
            f"{sortino_ratio:.2f}",
            help="Measures return per unit of bad downside volatility.",
        )
        dcol2.metric(
            "Calmar Ratio",
            f"{calmar_ratio:.2f}",
            help="CAGR divided by Max Drawdown.",
        )
        dcol3.metric(
            "Value at Risk (95% Daily)",
            f"{var_95_daily:.2%}",
            help="Maximum daily expected loss at 95% confidence level.",
        )
        dcol4.metric(
            "Conditional VaR (CVaR)",
            f"{cvar_95_daily:.2%}",
            help="Average expected loss when VaR threshold is breached.",
        )

        st.write(
            f"**Longest Drawdown Recovery Duration:** {max_recovery_days} trading days."
        )

        st.markdown("---")
        st.subheader("Benchmark-Relative Metrics")
        bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
        bcol1.metric(
            "Alpha (Jensen's)",
            f"{alpha:.2%}",
            help="Risk-adjusted excess return vs CAPM expectation.",
        )
        bcol2.metric("Tracking Error", f"{tracking_error:.2%}")
        bcol3.metric("Information Ratio", f"{information_ratio:.2f}")
        bcol4.metric("Up Capture Ratio", f"{up_capture:.2%}")
        bcol5.metric("Down Capture Ratio", f"{down_capture:.2%}")

    # TAB 3: ROLLING METRICS ENGINE
    with tab3:
        st.subheader("Rolling 1-Year (252-Day) Analytics")
        rolling_window = 252

        rolling_cagr = returns["Fund"].rolling(rolling_window).apply(
            lambda x: (1 + x).prod() - 1
        ) * np.sqrt(252 / rolling_window)
        rolling_vol = returns["Fund"].rolling(rolling_window).std() * np.sqrt(
            252
        )
        rolling_sharpe = (rolling_cagr - risk_free_rate) / rolling_vol

        roll_df = pd.DataFrame(
            {
                "Rolling 1Y CAGR": rolling_cagr,
                "Rolling 1Y Volatility": rolling_vol,
                "Rolling 1Y Sharpe": rolling_sharpe,
            }
        ).dropna()

        fig_roll = px.line(
            roll_df,
            title="Rolling Performance & Risk Profile Over Time",
            labels={"value": "Metric Value", "index": "Date"},
        )
        st.plotly_chart(fig_roll, use_container_width=True)

    # TAB 4: GOAL-BASED MONTE CARLO & SIP ENGINE
    with tab4:
        st.subheader("SIP & Goal-Based Wealth Simulation")

        sim_mode = st.radio(
            "Simulation Method",
            ["Historical Bootstrap Resampling", "Parametric Random Walk"],
            horizontal=True,
        )

        months = horizon_years * 12
        num_sims = 1000

        # Convert daily returns to monthly returns for SIP compounding
        monthly_returns = (
            df["Fund"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
        )
        m_mean = monthly_returns.mean()
        m_std = monthly_returns.std()

        sim_paths = np.zeros((months, num_sims))

        for sim in range(num_sims):
            path_val = initial_inv
            curr_monthly_contrib = monthly_contrib

            for m in range(months):
                if m > 0 and m % 12 == 0:
                    curr_monthly_contrib *= 1 + step_up_pct

                if sim_mode == "Historical Bootstrap Resampling":
                    r = np.random.choice(monthly_returns)
                else:
                    r = np.random.normal(m_mean, m_std)

                path_val = (path_val + curr_monthly_contrib) * (1 + r)
                sim_paths[m, sim] = path_val

        final_wealths = sim_paths[-1, :]
        p10 = np.percentile(final_wealths, 10)
        p50 = np.percentile(final_wealths, 50)
        p90 = np.percentile(final_wealths, 90)

        success_rate = (final_wealths >= target_wealth).mean()

        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric(
            "Goal Success Probability",
            f"{success_rate:.1%}",
            f"Target: ${target_wealth:,.0f}",
        )
        scol2.metric("Conservative (10th %ile)", f"${p10:,.0f}")
        scol3.metric("Expected Median (50th %ile)", f"${p50:,.0f}")
        scol4.metric("Optimistic (90th %ile)", f"${p90:,.0f}")

        fig_mc = go.Figure()
        for i in range(min(50, num_sims)):
            fig_mc.add_trace(
                go.Scatter(
                    y=sim_paths[:, i],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(100, 100, 250, 0.15)"),
                    showlegend=False,
                )
            )

        fig_mc.add_trace(
            go.Scatter(
                y=np.median(sim_paths, axis=1),
                mode="lines",
                name="Median Path",
                line=dict(color="blue", width=3),
            )
        )
        fig_mc.add_hline(
            y=target_wealth,
            line_dash="dash",
            line_color="red",
            annotation_text="Target Goal",
        )
        fig_mc.update_layout(
            xaxis_title="Months Forward", yaxis_title="Projected Wealth ($)"
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # TAB 5: CRISIS STRESS-TEST
    with tab5:
        st.subheader("Historical Crisis Replay Engine")

        crises = {
            "2020 COVID Crash": ("2020-02-19", "2020-03-23"),
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
                            "Crisis Scenario": name,
                            f"{fund_ticker} Return": f"{f_return:.2%}",
                            f"{benchmark_ticker} Return": f"{b_return:.2%}",
                            "Relative Outperformance": f"{(f_return - b_return):.2%}",
                        }
                    )
            except Exception:
                pass

        if crisis_results:
            st.table(pd.DataFrame(crisis_results))
        else:
            st.warning(
                "Selected date range does not overlap with historical crisis events."
            )

except Exception as e:
    st.error(f"Error executing analytics engine: {str(e)}")
