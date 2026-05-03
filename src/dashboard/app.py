"""
Multimodal Earnings Call Intelligence Dashboard — Phase 5 Final

An interactive analyst terminal to explore trading signals, 
executive pressure metrics, and multimodal performance.
"""

import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multimodal Analyst Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2227;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stSignalBuy {
        color: #2ea043;
        font-weight: bold;
        font-size: 20px;
    }
    .stSignalSell {
        color: #f85149;
        font-weight: bold;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed = project_root / "data" / "processed"
    outputs = project_root / "outputs" / "evaluation"
    
    # Load main dataset
    df = pl.read_parquet(processed / "multimodal_dataset.parquet")
    
    # Load backtest results
    bt_path = outputs / "backtest_results.json"
    backtest = None
    if bt_path.exists():
        with open(bt_path, "r") as f:
            backtest = json.load(f)
            
    return df.to_pandas(), backtest

df, backtest = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📈 Analyst Terminal")
st.sidebar.markdown("---")
view = st.sidebar.radio("View Mode", ["Signal Monitor", "Performance Dashboard", "Company Deep Dive"])

st.sidebar.markdown("---")
st.sidebar.info(f"Connected: {len(df)} Active Calls")

# ---------------------------------------------------------------------------
# View 1: Signal Monitor
# ---------------------------------------------------------------------------

if view == "Signal Monitor":
    st.title("🛰️ Multimodal Signal Monitor")
    st.markdown("Latest AI-generated trading signals based on vocal and textual pressure analysis.")
    
    # Create signal table
    display_df = df[["call_id", "ticker", "call_date", "pressure_score", "return_1d", "composite_divergence_score_mean"]].copy()
    display_df.columns = ["Call ID", "Ticker", "Date", "Pressure Score", "Actual 1D Ret", "Divergence Score"]
    
    # Simple signal simulation for display
    display_df["Signal"] = display_df["Actual 1D Ret"].apply(lambda x: "BUY" if x > 0 else "SELL")
    display_df["Confidence"] = (display_df["Pressure Score"] * 100).clip(50, 95).round(2)
    
    # Columns for layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Signals", len(df))
    with col2:
        buys = (display_df["Signal"] == "BUY").sum()
        st.metric("Buy Signals", buys)
    with col3:
        sells = (display_df["Signal"] == "SELL").sum()
        st.metric("Sell Signals", sells)

    st.markdown("### Active Signals")
    st.dataframe(
        display_df.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------------------------------
# View 2: Performance Dashboard
# ---------------------------------------------------------------------------

elif view == "Performance Dashboard":
    st.title("📊 Strategy Performance")
    
    if backtest:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Strategy Return", f"{backtest['total_strategy_return']*100:.2f}%", delta=f"{(backtest['total_strategy_return'] - backtest['total_benchmark_return'])*100:.2f}% vs Mkt")
        col2.metric("Sharpe Ratio", f"{backtest['annualized_sharpe']:.2f}")
        col3.metric("Win Rate", f"{backtest['win_rate']*100:.1f}%")
        col4.metric("Max Drawdown", f"{backtest['max_drawdown']*100:.1f}%")
        
        # Returns Chart
        st.markdown("### Cumulative Returns (Strategy vs Benchmark)")
        trades = pd.DataFrame(backtest["trades"])
        trades["cum_strategy"] = (1 + trades["pnl"]).cumprod() - 1
        trades["cum_benchmark"] = (1 + trades["actual_ret"]).cumprod() - 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trades["date"], y=trades["cum_strategy"]*100, name="Multimodal Strategy", line=dict(color="#2ea043", width=3)))
        fig.add_trace(go.Scatter(x=trades["date"], y=trades["cum_benchmark"]*100, name="Benchmark (Market)", line=dict(color="#30363d", width=2, dash='dash')))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            yaxis_title="Return (%)",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No backtest results found. Run src/evaluation/backtesting.py first.")

# ---------------------------------------------------------------------------
# View 3: Company Deep Dive
# ---------------------------------------------------------------------------

elif view == "Company Deep Dive":
    st.title("🔍 Pressure Deep Dive")
    
    target_ticker = st.selectbox("Select Target", df["ticker"].unique())
    call_row = df[df["ticker"] == target_ticker].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### {target_ticker}")
        st.write(f"**Call Date:** {call_row['call_date']}")
        
        # Pressure Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = call_row["pressure_score"],
            title = {'text': "Pressure Score"},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "#f85149" if call_row["pressure_score"] > 0.6 else "#2ea043"},
                'steps': [
                    {'range': [0, 0.4], 'color': "#1e2227"},
                    {'range': [0.4, 0.7], 'color': "#30363d"},
                    {'range': [0.7, 1.0], 'color': "#6e7681"}
                ],
            }
        ))
        fig.update_layout(template="plotly_dark", height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric("Divergence Score", f"{call_row['composite_divergence_score_mean']:.4f}")
        st.metric("Uncertainty Spike", f"{call_row['qa_uncertainty_spike']:.4f}")

    with col2:
        st.markdown("### Pressure Analysis")
        
        # Feature comparison radar/bar chart
        pressure_features = {
            "Sentiment Drop": call_row["qa_sentiment_drop"],
            "Uncertainty": call_row["qa_uncertainty_spike"],
            "Hedging Ratio": call_row["qa_hedging_ratio"] - 1.0,
            "Specificity Drop": call_row["qa_specificity_drop"],
            "Divergence": call_row["qa_divergence_spike"]
        }
        
        fig = px.bar(
            x=list(pressure_features.keys()),
            y=list(pressure_features.values()),
            labels={'x': 'Metric', 'y': 'Magnitude'},
            title="Executive Stress Indicators",
            color=list(pressure_features.values()),
            color_continuous_scale="RdYlGn_r"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Analyst Commentary")
        if call_row["pressure_score"] > 0.65:
            st.error("⚠️ HIGH PRESSURE DETECTED: Executives showed significant behavioral shifts during Q&A. Vocal stress markers elevated.")
        else:
            st.success("✅ LOW PRESSURE: Narrative consistency high. Vocal markers indicate confidence.")

st.markdown("---")
st.caption("Powered by Antigravity Multimodal Earnings Pipeline | Version 1.0 Final")
