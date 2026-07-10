import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    """Uses an Isolation Forest to flag unusual transaction spikes."""
    expenses = df[df['Type'] == 'Expense'].copy()
    if expenses.empty:
        df['Is_Anomaly'] = 0
        return df
        
    iso = IsolationForest(contamination=0.03, random_state=42)
    expenses['Is_Anomaly'] = iso.fit_predict(expenses[['Amount', 'DaysSincePreviousTransaction']])
    expenses['Is_Anomaly'] = expenses['Is_Anomaly'].map({1: 0, -1: 1})
    
    df = pd.merge(df, expenses[['Is_Anomaly']], left_index=True, right_index=True, how='left')
    df['Is_Anomaly'] = df['Is_Anomaly'].fillna(0).astype(int)
    return df