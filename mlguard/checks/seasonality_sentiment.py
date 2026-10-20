import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from mlguard.core.domain import Finding, Severity


def check_seasonality(
    df: pd.DataFrame, 
    time_col: str = None, 
    target_col: str = None, 
    threshold_ratio: float = 0.35
) -> list[Finding]:
    findings: list[Finding] = []

    if time_col is None:
        date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
        if len(date_cols) > 0:
            time_col = date_cols[0]
        else:
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    time_col = col
                    break

    if time_col is None or time_col not in df.columns:
        return findings

    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[time_col])
    if len(df_copy) < 30:
        return findings

    df_sorted = df_copy.sort_values(by=time_col).set_index(time_col)
    df_resampled = df_sorted.resample('D').mean(numeric_only=True).ffill().bfill()

    # Сканируем все числовые колонки (признаки), исключая сам целевой класс
    numeric_cols = [c for c in df_resampled.select_dtypes(include='number').columns if c != target_col]
    if not numeric_cols:
        numeric_cols = list(df_resampled.select_dtypes(include='number').columns)

    candidate_periods = [7, 14, 30]

    for col in numeric_cols:
        series = df_resampled[col].dropna()
        
        if len(series) < 30:
            continue

        best_ratio = 0.0
        var_total = float(series.var())
        if var_total == 0 or np.isnan(var_total):
            continue

        for p in candidate_periods:
            if len(series) < p * 6:
                continue
            try:
                stl = STL(series, period=p)
                res = stl.fit()
                seasonal_ratio = float(res.seasonal.var()) / var_total
                if seasonal_ratio > best_ratio:
                    best_ratio = seasonal_ratio
            except Exception:
                continue

        if best_ratio > threshold_ratio:
            findings.append(Finding(
                check='seasonality_sentiment',
                severity=Severity.WARNING,
                column=col,
                metric_value=round(float(best_ratio), 4),
                threshold=threshold_ratio,
                message=f"High seasonality in '{col}': seasonal component explains {best_ratio:.1%} of total variance."
            ))

    return findings
