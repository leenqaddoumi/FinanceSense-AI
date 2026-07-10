import pandas as pd

def engineer_advanced_features(df):
    """Creates rich temporal, delta, and operational accounting features."""
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.strftime('%b')
    df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Time delta features
    df['DaysSincePreviousTransaction'] = df['Date'].diff().dt.days.fillna(0)
    df['Cumulative_Balance'] = df['Signed_Amount'].cumsum()
    
    # Monthly financial ratios
    monthly_aggregates = df.groupby(['Year', 'Month']).agg(
        Monthly_Income=('Amount', lambda x: x[df.loc[x.index, 'Type'] == 'Income'].sum()),
        Monthly_Expense=('Amount', lambda x: x[df.loc[x.index, 'Type'] == 'Expense'].sum())
    ).reset_index()
    
    monthly_aggregates['Monthly_Savings'] = monthly_aggregates['Monthly_Income'] - monthly_aggregates['Monthly_Expense']
    monthly_aggregates['Monthly_Expense_Ratio'] = (monthly_aggregates['Monthly_Expense'] / monthly_aggregates['Monthly_Income']).fillna(0)
    
    df = pd.merge(df, monthly_aggregates, on=['Year', 'Month'], how='left')
    return df