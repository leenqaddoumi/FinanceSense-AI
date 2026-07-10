import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

# Imports from our custom modular package
from src.preprocessing import clean_transaction_data
from src.feature_engineering import engineer_advanced_features
from src.clustering import apply_auto_clustering
from src.forecasting import forecast_with_baseline
from src.anomaly_detection import detect_anomalies
from src.insights import generate_intelligent_insights

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="FinanceSense AI", layout="wide")
st.title("💰 FinanceSense AI")
st.markdown("A unified software pipeline executing automated data cleaning, Isolation Forest anomaly tracking, and SARIMA forecasting directly from your transaction logs.")

# --- FILE UPLOAD PIPELINE ---
st.sidebar.header("📁 Data Input Portal")
uploaded_file = st.sidebar.file_uploader("Upload your transaction dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success("🎉 Custom CSV loaded!")
else:
    # UPDATE THIS PATH HERE TO MATCH YOUR NEW LOGIC
    default_path = os.path.join("data", "Sample_Dataset.csv") 
    
    if os.path.exists(default_path):
        raw_df = pd.read_csv(default_path)
        st.sidebar.info("ℹ️ Running default sandbox profile data.")
    else:
        raw_df = pd.DataFrame()
        st.error(f"❌ Missing File: Please verify '{default_path}' location.")

if not raw_df.empty:
    # Run the modular backend data pipeline
    cleaned_df, dup_count = clean_transaction_data(raw_df)
    engineered_df = engineer_advanced_features(cleaned_df)
    clustered_df = apply_auto_clustering(engineered_df)
    final_df = detect_anomalies(clustered_df)
    
    # --- FILTERS ---
    st.sidebar.header("🕹️ Filter Controls")
    sel_year = st.sidebar.selectbox("Year Filter", ["All"] + sorted(list(final_df['Year'].unique()), reverse=True))
    sel_cat = st.sidebar.selectbox("Category Filter", ["All"] + sorted(list(final_df['Category'].unique())))
    
    view_df = final_df.copy()
    if sel_year != "All": view_df = view_df[view_df['Year'] == sel_year]
    if sel_cat != "All": view_df = view_df[view_df['Category'] == sel_cat]

    # --- SCORECARDS ---
    in_val = view_df[view_df['Type'] == 'Income']['Amount'].sum()
    ex_val = view_df[view_df['Type'] == 'Expense']['Amount'].sum()
    net_savings = in_val - ex_val
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Income Inflow", f"${in_val:,.2f}")
    c2.metric("Total Expense Outflow", f"${ex_val:,.2f}", delta=f"-${ex_val:,.2f}", delta_color="inverse")
    c3.metric("Net Savings Margin", f"${net_savings:,.2f}")
    if dup_count > 0: st.sidebar.warning(f"🧹 Cleaned out {dup_count} duplicate rows.")

    st.markdown("---")
    
    # --- SPENDING TRENDS CHARTS ---
    st.subheader("📉 Spending Trends")
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.markdown("**Running Liquidity Portfolio Profile Over Time**")
        f_bal = px.line(view_df, x='Date', y='Cumulative_Balance', template="plotly_dark")
        st.plotly_chart(f_bal, use_container_width=True)
    with vcol2:
        st.markdown("**Categorical Expense Allocation Distributions**")
        cat_agg = view_df[view_df['Type'] == 'Expense'].groupby('Category')['Amount'].sum().reset_index()
        f_pie = px.pie(cat_agg, values='Amount', names='Category', template="plotly_dark")
        st.plotly_chart(f_pie, use_container_width=True)

    st.markdown("---")

    # --- MACHINE LEARNING & FORECAST LAYER ---
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.subheader("🤖 Automated Behavioral Clustering")
        profiles = final_df.groupby(['Year', 'Month', 'Cluster_Label'], as_index=False)[['Amount']].sum()
        f_clus = px.scatter(profiles, x='Month', y='Amount', color='Cluster_Label', size='Amount', title="Optimal Silhouette K-Selection", template="plotly_dark")
        st.plotly_chart(f_clus, use_container_width=True)
        
    with m_col2:
        st.subheader("🔮 Time-Series Expense Forecasting")
        hist, f_res, model_desc = forecast_with_baseline(final_df)
        st.caption(f"Active Engine: {model_desc}")
        
        f_forecast = go.Figure()
        f_forecast.add_trace(go.Scatter(x=hist.index[-12:], y=hist.values[-12:], name="Historical"))
        f_forecast.add_trace(go.Scatter(x=f_res.index, y=f_res['mean'], name="Prediction", line=dict(dash='dash', color='orange')))
        f_forecast.add_trace(go.Scatter(x=f_res.index, y=f_res['upper'], mode='lines', line=dict(width=0), showlegend=False))
        f_forecast.add_trace(go.Scatter(x=f_res.index, y=f_res['lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255,165,0,0.12)', name="95% Confidence Bounds"))
        f_forecast.update_layout(template="plotly_dark")
        st.plotly_chart(f_forecast, use_container_width=True)

    st.markdown("---")

    # --- ANOMALIES VIEW ---
    st.subheader("🌲 Isolation Forest Outlier Exception Log")
    anomalies = final_df[final_df['Is_Anomaly'] == 1][['Date', 'Transaction Description', 'Category', 'Amount']].sort_values(by='Amount', ascending=False)
    if not anomalies.empty:
        st.warning(f"⚠️ Flagged {len(anomalies)} extreme transaction spikes outside normal baseline rules.")
        st.dataframe(anomalies, use_container_width=True)
    else:
        st.success("✅ No extreme anomalies identified across records.")

    st.markdown("---")

    # --- INSIGHTS VIEW ---
    st.subheader("📝 Smart Financial Insights")
    insights_list = generate_intelligent_insights(final_df)
    for line in insights_list:
        st.info(line)