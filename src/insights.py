import pandas as pd

def generate_intelligent_insights(df):
    """Generates dynamic financial alerts and actionable behavioral tracking suggestions."""
    insights = []
    expenses = df[df['Type'] == 'Expense']
    if expenses.empty:
        return ["No expense history identified yet."]
        
    # Day concentration evaluation
    day_mapping = {0:'Mondays', 1:'Tuesdays', 2:'Wednesdays', 3:'Thursdays', 4:'Fridays', 5:'Saturdays', 6:'Sundays'}
    peak_day_idx = expenses.groupby('DayOfWeek')['Amount'].sum().idxmax()
    insights.append(f"📅 **Spending Peak**: Outbound cash flow concentrates heaviest on **{day_mapping[peak_day_idx]}**.")
    
    # Dynamic month-over-month category swings
    recent_months = df.groupby(['Year', 'Month', 'Category'])['Amount'].sum().unstack(fill_value=0).tail(2)
    if len(recent_months) == 2:
        prev_mo, curr_mo = recent_months.index[0], recent_months.index[1]
        for cat in recent_months.columns:
            p_val = recent_months.loc[prev_mo, cat]
            c_val = recent_months.loc[curr_mo, cat]
            if p_val > 0:
                pct_change = ((c_val - p_val) / p_val) * 100
                if pct_change > 15:
                    insights.append(f"🚀 **Surge Alert**: Your spending on **{cat}** increased by **{pct_change:.1f}%** compared to last month.")
                elif pct_change < -15:
                    insights.append(f"📉 **Spending Drop**: Your spending on **{cat}** decreased by **{abs(pct_change):.1f}%** compared to last month.")
                    
    # Target allocation budget suggestions
    top_cat = expenses.groupby('Category')['Amount'].sum().idxmax()
    top_cat_monthly_avg = expenses[expenses['Category'] == top_cat].groupby(['Year', 'Month'])['Amount'].sum().mean()
    insights.append(f"💡 **Budget Suggestion**: You spend an average of **${top_cat_monthly_avg:,.2f}/mo** on **{top_cat}**. Consider capping next month's budget at **${top_cat_monthly_avg*0.85:,.2f}** to save 15%.")
    
    return insights