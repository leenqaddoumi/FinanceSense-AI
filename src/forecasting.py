import pandas as pd
import numpy as np
import statsmodels.api as sm

def forecast_with_baseline(df, steps=3):
    """Compares SARIMA projections against a Moving Average baseline with 95% CIs."""
    series = df[df['Type'] == 'Expense'].resample('ME', on='Date')['Amount'].sum()
    if len(series) < 5:
        idx = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='ME')
        return series, pd.DataFrame({'mean': [series.mean()]*steps, 'lower': [series.mean()]*steps, 'upper': [series.mean()]*steps}, index=idx), "Baseline MA"
        
    baseline_pred = series.rolling(window=3, min_periods=1).mean().iloc[-1]
    
    try:
        model = sm.tsa.statespace.SARIMAX(series, order=(1,1,0), seasonal_order=(0,1,0,12), enforce_stationarity=False)
        res = model.fit(method='powell', maxiter=100, disp=False)
        forecast = res.get_forecast(steps=steps)
        f_frame = forecast.summary_frame(alpha=0.05) # 95% Confidence Bounds
        
        forecast_df = pd.DataFrame({
            'mean': f_frame['mean'],
            'lower': f_frame['mean_ci_lower'],
            'upper': f_frame['mean_ci_upper']
        }, index=pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='ME'))
        
        return series, forecast_df, f"SARIMA optimized (Baseline MA was ${baseline_pred:,.2f})"
    except:
        idx = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='ME')
        return series, pd.DataFrame({'mean': [baseline_pred]*steps, 'lower': [baseline_pred*0.8]*steps, 'upper': [baseline_pred*1.2]*steps}, index=idx), "Baseline MA Fallback"