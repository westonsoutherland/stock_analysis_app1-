# app.py — Stock Comparison and Analysis Application
# Run locally with:  uv run streamlit run app.py
# -------------------------------------------------------

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(page_title="Stock Comparison & Analysis", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

with st.sidebar.expander("ℹ️ About / Methodology", expanded=False):
    st.markdown(
        """
**What this app does**  
Compare and analyze 2–5 stocks over a custom date range using historical
adjusted closing prices from Yahoo Finance.

**Key assumptions**
- Returns: simple (arithmetic) — `pct_change()`
- Annualization: 252 trading days/year  
  - Mean return × 252  
  - Std × √252
- Cumulative wealth: `(1 + r).cumprod()`
- Equal-weight portfolio: average of daily returns across all stocks

**Data source:** Yahoo Finance via `yfinance`
        """
    )

# ── Inputs ───────────────────────────────────────────────────────────────────
raw_input = st.sidebar.text_input(
    "Ticker symbols (2–5, comma-separated)",
    value="AAPL, MSFT, GOOGL",
    help="Example: AAPL, MSFT, TSLA, AMZN",
)

default_start = date.today() - timedelta(days=365 * 3)
start_date = st.sidebar.date_input("Start date", value=default_start)
end_date = st.sidebar.date_input("End date", value=date.today())

# ── Input validation ─────────────────────────────────────────────────────────
tickers_raw = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
input_errors = []

if len(tickers_raw) < 2:
    input_errors.append("Please enter **at least 2** ticker symbols.")
if len(tickers_raw) > 5:
    input_errors.append("Please enter **no more than 5** ticker symbols.")
if start_date >= end_date:
    input_errors.append("Start date must be **before** end date.")
if (end_date - start_date).days < 365:
    input_errors.append("Date range must be **at least 1 year**.")

if input_errors:
    for e in input_errors:
        st.sidebar.error(e)
    st.warning("Please fix the inputs in the sidebar to continue.")
    st.stop()

# ── Data download ─────────────────────────────────────────────────────────────
BENCHMARK = "^GSPC"


@st.cache_data(show_spinner=False, ttl=3600)
def download_prices(tickers: tuple, start: date, end: date) -> dict:
    """Download adjusted close prices. Returns dict of {ticker: Series}."""
    all_tickers = list(tickers) + [BENCHMARK]
    result = {}
    failed = []
    for tk in all_tickers:
        try:
            raw = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if raw.empty or len(raw) < 30:
                failed.append(tk)
                continue
            # Handle MultiIndex columns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            result[tk] = raw["Close"].rename(tk)
        except Exception:
            failed.append(tk)
    return result, failed


with st.spinner("Downloading price data…"):
    price_dict, failed_tickers = download_prices(tuple(tickers_raw), start_date, end_date)

# Report failures
user_failed = [t for t in failed_tickers if t != BENCHMARK]
if user_failed:
    st.error(
        f"Could not download data for: **{', '.join(user_failed)}**. "
        "Check the ticker symbols and try again."
    )
if len(price_dict) - (1 if BENCHMARK in price_dict else 0) < 2:
    st.error("At least 2 valid tickers are required to continue.")
    st.stop()

# Keep only successful user tickers
tickers = [t for t in tickers_raw if t in price_dict]

# ── Align to common date range ────────────────────────────────────────────────
prices_df = pd.concat([price_dict[t] for t in tickers], axis=1).dropna()
benchmark_series = price_dict.get(BENCHMARK)
if benchmark_series is not None:
    benchmark_series = benchmark_series.reindex(prices_df.index).dropna()

if prices_df.empty:
    st.error("No overlapping trading days found for the selected tickers and date range.")
    st.stop()

# Warn if date range was truncated
actual_start = prices_df.index[0].date()
actual_end = prices_df.index[-1].date()
if actual_start > start_date or actual_end < end_date:
    st.info(
        f"Data aligned to overlapping range: **{actual_start}** → **{actual_end}**."
    )

# ── Compute returns ───────────────────────────────────────────────────────────
returns_df = prices_df.pct_change().dropna()

# Equal-weight portfolio
ew_returns = returns_df.mean(axis=1)

# Benchmark returns
bench_returns = benchmark_series.pct_change().dropna() if benchmark_series is not None else None

# Wealth index ($10,000 start)
wealth_df = (1 + returns_df).cumprod() * 10_000
ew_wealth = (1 + ew_returns).cumprod() * 10_000
if bench_returns is not None:
    bench_wealth = (1 + bench_returns.reindex(returns_df.index).fillna(0)).cumprod() * 10_000

# ── TABS ──────────────────────────────────────────────────────────────────────
st.title("📈 Stock Comparison & Analysis")
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Price & Returns", "📉 Risk & Distribution", "🔗 Correlation & Portfolio", "📋 Data"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Price & Returns
# ═══════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Stock selector ───────────────────────────────────────────────────
    st.subheader("Adjusted Closing Prices")
    visible = st.multiselect(
        "Show/hide stocks", options=tickers, default=tickers, key="price_vis"
    )

    fig_price = go.Figure()
    for tk in visible:
        fig_price.add_trace(
            go.Scatter(x=prices_df.index, y=prices_df[tk], mode="lines", name=tk)
        )
    fig_price.update_layout(
        title="Adjusted Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=420,
        legend_title="Ticker",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # ── Summary statistics table ─────────────────────────────────────────
    st.subheader("Summary Statistics")

    def summary_stats(ret_series: pd.Series, label: str) -> dict:
        ann_ret = ret_series.mean() * 252
        ann_vol = ret_series.std() * math.sqrt(252)
        return {
            "Ticker": label,
            "Ann. Return": f"{ann_ret:.2%}",
            "Ann. Volatility": f"{ann_vol:.2%}",
            "Skewness": f"{ret_series.skew():.3f}",
            "Kurtosis": f"{ret_series.kurtosis():.3f}",
            "Min Daily Return": f"{ret_series.min():.2%}",
            "Max Daily Return": f"{ret_series.max():.2%}",
        }

    rows = [summary_stats(returns_df[tk], tk) for tk in tickers]
    if bench_returns is not None:
        bench_aligned = bench_returns.reindex(returns_df.index).dropna()
        rows.append(summary_stats(bench_aligned, "S&P 500 (^GSPC)"))

    st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

    # ── Cumulative wealth index ───────────────────────────────────────────
    st.subheader("Cumulative Wealth Index (Starting $10,000)")

    fig_wealth = go.Figure()
    for tk in tickers:
        fig_wealth.add_trace(
            go.Scatter(x=wealth_df.index, y=wealth_df[tk], mode="lines", name=tk)
        )
    fig_wealth.add_trace(
        go.Scatter(
            x=ew_wealth.index,
            y=ew_wealth,
            mode="lines",
            name="Equal-Weight Portfolio",
            line=dict(dash="dash", width=2),
        )
    )
    if bench_returns is not None:
        fig_wealth.add_trace(
            go.Scatter(
                x=bench_wealth.index,
                y=bench_wealth,
                mode="lines",
                name="S&P 500",
                line=dict(dash="dot", width=2, color="gray"),
            )
        )
    fig_wealth.update_layout(
        title="Growth of $10,000 Investment",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (USD)",
        template="plotly_white",
        height=450,
        legend_title="Series",
    )
    st.plotly_chart(fig_wealth, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Risk & Distribution
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Rolling volatility ────────────────────────────────────────────────
    st.subheader("Rolling Annualized Volatility")
    vol_window = st.select_slider(
        "Window (trading days)", options=[30, 60, 90], value=30, key="vol_win"
    )

    fig_vol = go.Figure()
    for tk in tickers:
        rolling_vol = returns_df[tk].rolling(vol_window).std() * math.sqrt(252)
        fig_vol.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol, mode="lines", name=tk)
        )
    fig_vol.update_layout(
        title=f"{vol_window}-Day Rolling Annualized Volatility",
        xaxis_title="Date",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".0%",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── Distribution analysis ─────────────────────────────────────────────
    st.subheader("Return Distribution Analysis")
    dist_ticker = st.selectbox("Select stock for distribution analysis", tickers, key="dist_tk")
    dist_returns = returns_df[dist_ticker].dropna()

    # Histogram vs Q-Q toggle
    dist_view = st.radio(
        "View", ["Histogram + Normal Fit", "Q-Q Plot"], horizontal=True, key="dist_view"
    )

    if dist_view == "Histogram + Normal Fit":
        mu, sigma = stats.norm.fit(dist_returns)
        x_range = np.linspace(dist_returns.min(), dist_returns.max(), 300)
        pdf_vals = stats.norm.pdf(x_range, mu, sigma)

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=dist_returns,
                nbinsx=70,
                histnorm="probability density",
                name="Daily Returns",
                marker_color="steelblue",
                opacity=0.7,
            )
        )
        fig_hist.add_trace(
            go.Scatter(
                x=x_range,
                y=pdf_vals,
                mode="lines",
                name="Normal Fit",
                line=dict(color="red", width=2),
            )
        )
        fig_hist.update_layout(
            title=f"{dist_ticker} — Daily Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            xaxis_tickformat=".1%",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        qq_theoretical, qq_sample = stats.probplot(dist_returns, dist="norm")[:2][0], stats.probplot(dist_returns, dist="norm")[0][1]
        qq_theoretical = stats.probplot(dist_returns, dist="norm")[0][0]
        fit_line = np.polyval(np.polyfit(qq_theoretical, qq_sample, 1), qq_theoretical)

        fig_qq = go.Figure()
        fig_qq.add_trace(
            go.Scatter(
                x=qq_theoretical,
                y=qq_sample,
                mode="markers",
                name="Sample Quantiles",
                marker=dict(color="steelblue", size=4),
            )
        )
        fig_qq.add_trace(
            go.Scatter(
                x=qq_theoretical,
                y=fit_line,
                mode="lines",
                name="Normal Reference",
                line=dict(color="red", width=2),
            )
        )
        fig_qq.update_layout(
            title=f"{dist_ticker} — Q-Q Plot vs. Normal Distribution",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_qq, use_container_width=True)

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(dist_returns)
    reject = jb_p < 0.05
    st.info(
        f"**Jarque-Bera normality test** — Statistic: {jb_stat:.2f} | "
        f"p-value: {jb_p:.4f} → "
        + ("**Rejects normality (p < 0.05)**" if reject else "**Fails to reject normality (p ≥ 0.05)**")
    )

    # ── Box plot ──────────────────────────────────────────────────────────
    st.subheader("Daily Return Distributions — Box Plot")

    fig_box = go.Figure()
    for tk in tickers:
        fig_box.add_trace(
            go.Box(y=returns_df[tk], name=tk, boxpoints="outliers")
        )
    fig_box.update_layout(
        title="Daily Return Distributions (All Stocks)",
        yaxis_title="Daily Return",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Correlation & Portfolio
# ═══════════════════════════════════════════════════════════════════════════
with tab3:

    # ── Correlation heatmap ───────────────────────────────────────────────
    st.subheader("Correlation Heatmap")

    corr = returns_df.corr()
    fig_heat = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig_heat.update_layout(
        title="Pairwise Return Correlation Matrix",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Scatter plot ──────────────────────────────────────────────────────
    st.subheader("Return Scatter Plot")
    col_a, col_b = st.columns(2)
    sc_a = col_a.selectbox("Stock A", tickers, index=0, key="sc_a")
    sc_b = col_b.selectbox("Stock B", tickers, index=min(1, len(tickers) - 1), key="sc_b")

    fig_scatter = go.Figure(
        go.Scatter(
            x=returns_df[sc_a],
            y=returns_df[sc_b],
            mode="markers",
            marker=dict(size=4, opacity=0.6),
            name=f"{sc_a} vs {sc_b}",
        )
    )
    fig_scatter.update_layout(
        title=f"Daily Returns: {sc_a} vs {sc_b}",
        xaxis_title=f"{sc_a} Daily Return",
        yaxis_title=f"{sc_b} Daily Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Rolling correlation ───────────────────────────────────────────────
    st.subheader("Rolling Correlation")
    col_ra, col_rb, col_rw = st.columns(3)
    rc_a = col_ra.selectbox("Stock A", tickers, index=0, key="rc_a")
    rc_b = col_rb.selectbox("Stock B", tickers, index=min(1, len(tickers) - 1), key="rc_b")
    rc_win = col_rw.select_slider("Window", options=[30, 60, 90, 120], value=60, key="rc_win")

    rolling_corr = returns_df[rc_a].rolling(rc_win).corr(returns_df[rc_b])
    fig_rc = go.Figure(
        go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            mode="lines",
            name=f"Rolling {rc_win}-day corr",
        )
    )
    fig_rc.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_rc.update_layout(
        title=f"{rc_win}-Day Rolling Correlation: {rc_a} & {rc_b}",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1, 1]),
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_rc, use_container_width=True)

    # ── Two-asset portfolio explorer ──────────────────────────────────────
    st.subheader("Two-Asset Portfolio Explorer")

    st.markdown(
        """
**What this shows:** When you combine two stocks into a portfolio, the portfolio's volatility
depends on not just each stock's own volatility, but also the correlation between them.
When correlation is less than 1, combining stocks produces a portfolio with **lower volatility
than either stock individually** — this is the diversification effect. The lower the correlation,
the more dramatic the dip in the volatility curve.
        """
    )

    col_pa, col_pb = st.columns(2)
    port_a = col_pa.selectbox("Stock A", tickers, index=0, key="port_a")
    port_b = col_pb.selectbox("Stock B", tickers, index=min(1, len(tickers) - 1), key="port_b")

    weight_a = st.slider(
        f"Weight on {port_a} (%)", min_value=0, max_value=100, value=50, step=1, key="wt_a"
    )
    weight_b = 100 - weight_a

    # Annualized stats
    ret_a = returns_df[port_a]
    ret_b = returns_df[port_b]

    ann_ret_a = ret_a.mean() * 252
    ann_ret_b = ret_b.mean() * 252
    ann_vol_a = ret_a.std() * math.sqrt(252)
    ann_vol_b = ret_b.std() * math.sqrt(252)
    cov_matrix = returns_df[[port_a, port_b]].cov() * 252
    cov_ab = cov_matrix.loc[port_a, port_b]

    w = weight_a / 100
    port_ret = w * ann_ret_a + (1 - w) * ann_ret_b
    port_var = (
        w**2 * ann_vol_a**2
        + (1 - w)**2 * ann_vol_b**2
        + 2 * w * (1 - w) * cov_ab
    )
    port_vol = math.sqrt(max(port_var, 0))

    col1, col2 = st.columns(2)
    col1.metric("Portfolio Ann. Return", f"{port_ret:.2%}")
    col2.metric("Portfolio Ann. Volatility", f"{port_vol:.2%}")

    # Full volatility curve across all weights
    weights_range = np.linspace(0, 1, 201)
    vols_curve = []
    for ww in weights_range:
        var = (
            ww**2 * ann_vol_a**2
            + (1 - ww)**2 * ann_vol_b**2
            + 2 * ww * (1 - ww) * cov_ab
        )
        vols_curve.append(math.sqrt(max(var, 0)))

    fig_port = go.Figure()
    fig_port.add_trace(
        go.Scatter(
            x=weights_range * 100,
            y=vols_curve,
            mode="lines",
            name="Portfolio Volatility",
            line=dict(color="royalblue", width=2),
        )
    )
    # Mark current slider position
    fig_port.add_trace(
        go.Scatter(
            x=[weight_a],
            y=[port_vol],
            mode="markers",
            name="Current Weight",
            marker=dict(color="red", size=12, symbol="diamond"),
        )
    )
    # Horizontal reference lines for individual stocks
    fig_port.add_hline(
        y=ann_vol_a, line_dash="dot", line_color="green",
        annotation_text=f"{port_a} vol", annotation_position="right"
    )
    fig_port.add_hline(
        y=ann_vol_b, line_dash="dot", line_color="orange",
        annotation_text=f"{port_b} vol", annotation_position="right"
    )
    fig_port.update_layout(
        title=f"Portfolio Volatility Curve: {port_a} ({weight_a}%) + {port_b} ({weight_b}%)",
        xaxis_title=f"Weight on {port_a} (%)",
        yaxis_title="Annualized Volatility",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=430,
    )
    st.plotly_chart(fig_port, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Raw Data
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Adjusted Closing Prices")
    st.dataframe(prices_df.tail(120), use_container_width=True)

    st.subheader("Daily Returns")
    st.dataframe(returns_df.tail(120).style.format("{:.4%}"), use_container_width=True)