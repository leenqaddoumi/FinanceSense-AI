import pandas as pd

def clean_transaction_data(df):
    """Handles missing values and cleans up exact duplicate entries."""
    # Ensure baseline data integrity
    df.dropna(subset=['Amount', 'Date'], inplace=True)
    df['Category'] = df['Category'].fillna('Other')
    df['Transaction Description'] = df['Transaction Description'].fillna('Unknown Transaction')
    
    # Structural duplicate controls
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)
    
    # Formatting adjustments
    df['Date'] = pd.to_datetime(df['Date'])
    df['Signed_Amount'] = df.apply(lambda row: -row['Amount'] if row['Type'] == 'Expense' else row['Amount'], axis=1)
    
    return df.sort_values(by='Date').reset_index(drop=True), duplicates_removed