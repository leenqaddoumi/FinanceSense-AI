📊 Personal Finance AI Dashboard

📌 GitHub Repository Details

Repository Description: AI-powered personal finance dashboard using machine learning for spending analysis, forecasting, anomaly detection, and interactive visualization.

Topics/Tags: python, machine-learning, streamlit, finance, pandas, scikit-learn, data-science, forecasting, plotly, isolation-forest, kmeans, sarima

📌 Overview

This is an interactive financial intelligence dashboard that helps users track, understand, and predict their personal expenses. Instead of manual categorization, the platform uses machine learning to automatically cluster spending habits, flag unusual transactions (anomalies), and forecast future monthly expenses.

Live Demo: [Insert Deployed App Link Here]

Tech Stack: Python, Streamlit, Pandas, Scikit-Learn, Statsmodels, Plotly

✨ Features

Spending Pattern Clustering: Groups transaction histories into behavioral profiles using K-Means clustering.

Expense Forecasting: Uses SARIMA modeling to forecast monthly expenses with a 95% confidence interval.

Anomaly Detection: Employs Isolation Forests to instantly flag unusual transaction spikes or double-charges.

Interactive Visualization: Dynamic multi-axis charts built with Plotly to drill down into monthly category limits.

🏗️ System Architecture

┌────────────────┐      ┌─────────────────┐      ┌───────────────────────────┐
│  CSV / Bank    │ ───> │  Data Cleaning  │ ───> │ Feature Engineering       │
│  Statement     │      │  & Imputation   │      │ (Temporal & Categorical)  │
└────────────────┘      └─────────────────┘      └───────────────────────────┘
                                                               │
                                                               ▼
┌────────────────┐      ┌─────────────────┐      ┌───────────────────────────┐
│ Interactive    │ <─── │ Streamlit       │ <─── │ ML Models                 │
│ Plotly Graphs  │      │ Frontend App    │      │ (K-Means, SARIMA, IF)     │
└────────────────┘      └─────────────────┘      └───────────────────────────┘


⚙️ Data Pipeline

[Raw CSV] 
   │
   ▼
[Data Cleaning] ────> Fills missing categories, parses dates, normalizes transaction amounts.
   │
   ▼
[Feature Extraction] ────> Extracts Day of Week, Day of Month, and rolling 30-day averages.
   │
   ├─> [Clustering Pipeline] ───> Fits K-Means to categorize spending behaviors.
   ├─> [Anomaly Pipeline] ────> Fits Isolation Forest to isolate extreme transaction values.
   └─> [Forecasting Pipeline] ──> Fits SARIMA on aggregate historical monthly sequences.


🤖 Machine Learning Models

1. Spending Pattern Clustering (K-Means)

Objective: Group transactions automatically to isolate core spending behaviors.

Features Used: Transaction amount, category frequency, time-of-day, and merchant type.

Evaluation: Optimized cluster count (K) using the Silhouette Score and the Elbow Method.

2. Expense Forecasting (SARIMA)

Objective: Forecast future monthly expenses based on historical trends and seasonality.

Configuration: SARIMA(p,d,q)(P,D,Q)s model fitted on historical monthly spending aggregates.

Visual Representation: Displays a rolling timeline with a shaded 95% confidence boundary.

3. Anomaly Detection (Isolation Forest)

Objective: Identify potential billing errors, fraud, or unexpected spending spikes.

Method: Evaluates multidimensional transaction features to flag instances with high outlier scores.

📊 Interactive Dashboard

The Streamlit frontend allows non-technical users to interact with their financial models:

Upload Area: Upload monthly bank statements in standard CSV format.

What-If Forecasts: Adjust budget goals to see how actual and forecasted spending patterns align.

Drill-Down Panels: Filter flagged transaction anomalies by date, merchant, or severity.

⚠️ Limitations & Assumptions

Data History: The SARIMA model requires at least 12–18 months of historical records to reliably model monthly seasonality.

Category Tagging: Transaction descriptions must contain recognizable keywords for optimal vector representation prior to clustering.

🚀 Quick Start & Installation

1. Clone the repository

git clone https://github.com/leenqaddoumi/finance-ai-dashboard.git
cd finance-ai-dashboard


2. Set up virtual environment & install packages

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


3. Run the application locally

streamlit run app.py


🔮 Future Enhancements

Multi-Account Aggregation: Connect to live banking APIs using secure ledger connectors.

Auto-Labeling Pipeline: Fine-tune a lightweight BERT classifier for semantic merchant-to-category mapping.