import numpy as np
import pandas as pd
from mlguard.core.domain import Finding, Severity


def check_temporal_leakage(
    df: pd.DataFrame, 
    time_col: str = None, 
    target_col: str = None,
    threshold_corr: float = 0.90
) -> list[Finding]:
    """
    Детектирует временные утечки (Temporal Lookahead):
    1. Наличие временных меток признаков позже времени среза таргета (feature_time > target_time).
    2. Признаки, заглядывающие в будущее: корреляция фичи X[t] с будущим таргетом y[t+1] превышает порог.
    """
    findings: list[Finding] = []

    # Определяем time_col, если не указан явно
    if time_col is None:
        date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns
        if len(date_cols) > 0:
            time_col = date_cols[0]
        else:
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'timestamp' in col.lower():
                    time_col = col
                    break

    if time_col is None or time_col not in df.columns:
        return findings

    # Пытаемся привести time_col к datetime
    try:
        t_base = pd.to_datetime(df[time_col], errors='coerce')
    except Exception:
        return findings

    valid_mask = t_base.notna()
    if valid_mask.sum() < 20:
        return findings

    # 1. Проверка других временных колонок на события из будущего
    for col in df.columns:
        if col == time_col:
            continue
        col_lower = str(col).lower()
        if 'date' in col_lower or 'time' in col_lower or 'timestamp' in col_lower or pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                t_other = pd.to_datetime(df[col], errors='coerce')
                # Считаем количество записей, где t_other > t_base
                future_diff = t_other[valid_mask] - t_base[valid_mask]
                future_count = int((future_diff > pd.Timedelta(0)).sum())
                if future_count > 0:
                    findings.append(Finding(
                        check="temporal_leakage",
                        severity=Severity.CRITICAL,
                        column=col,
                        metric_value=float(future_count),
                        threshold=0.0,
                        message=f"Future timestamps detected in '{col}': {future_count} records occur after base time '{time_col}'. Direct lookahead leak."
                    ))
            except Exception:
                pass

    # 2. Проверка заглядывания в будущее по значению таргета (Lead target correlation)
    if target_col is not None and target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        df_sorted = df[valid_mask].sort_values(by=time_col)
        target_series = df_sorted[target_col]
        future_target = target_series.shift(-1)  # y[t+1]

        numeric_cols = [c for c in df_sorted.select_dtypes(include='number').columns if c != target_col]
        for col in numeric_cols:
            s = df_sorted[col]
            corr_curr = abs(s.corr(target_series))
            corr_future = abs(s.corr(future_target))

            if corr_future > threshold_corr and (corr_future > corr_curr + 0.05 or np.isnan(corr_curr)):
                findings.append(Finding(
                    check="temporal_leakage",
                    severity=Severity.CRITICAL,
                    column=col,
                    metric_value=round(float(corr_future), 4),
                    threshold=threshold_corr,
                    message=f"Feature '{col}' strongly correlates ({corr_future:.4f}) with next-step target '{target_col}[t+1]'. Lookahead feature leakage."
                ))

    return findings
