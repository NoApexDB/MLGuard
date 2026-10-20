import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from mlguard.core.domain import Finding, Severity


def check_metric_confusion(
    df: pd.DataFrame, 
    time_col: str = None, 
    vanity_col: str = None, 
    health_col: str = None, 
    alpha: float = 0.05,
    min_tau: float = 0.25,
    max_pairs: int = 5
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

    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)

    if vanity_col is None or health_col is None:
        numeric_cols = [c for c in df_sorted.select_dtypes(include='number').columns if c != time_col]
        
        increasing_cols = []
        decreasing_cols = []

        for col in numeric_cols:
            valid_s = df_sorted[[col]].dropna()
            if len(valid_s) < 10:
                continue
            tau, pval = kendalltau(valid_s.index, valid_s[col])
            if np.isnan(tau) or np.isnan(pval):
                continue
            if pval < alpha:
                # Фильтруем шум: требуем минимальную силу монотонного тренда
                if tau >= min_tau:
                    increasing_cols.append((col, tau, pval))
                elif tau <= -min_tau:
                    decreasing_cols.append((col, tau, pval))

        pairs = []
        for v_name, v_tau, v_p in increasing_cols:
            for h_name, h_tau, h_p in decreasing_cols:
                divergence = v_tau - h_tau
                pairs.append((divergence, v_name, v_tau, v_p, h_name, h_tau, h_p))

        # Сортируем пары по силе расхождения и ограничиваем top-N
        pairs.sort(key=lambda x: x[0], reverse=True)
        for divergence, v_name, v_tau, v_p, h_name, h_tau, h_p in pairs[:max_pairs]:
            findings.append(Finding(
                check='metric_confusion',
                severity=Severity.CRITICAL,
                column=f"{v_name} vs {h_name}",
                metric_value=round(float(v_p), 4),
                threshold=alpha,
                message=f"Metric divergence: '{v_name}' is increasing (tau={v_tau:.2f}, p={v_p:.4f}), while '{h_name}' is decreasing (tau={h_tau:.2f}, p={h_p:.4f})."
            ))
        return findings

    if vanity_col not in df.columns or health_col not in df.columns:
        return findings

    valid_v = df_sorted[[vanity_col]].dropna()
    valid_h = df_sorted[[health_col]].dropna()
    if len(valid_v) < 10 or len(valid_h) < 10:
        return findings

    tau_v, pval_v = kendalltau(valid_v.index, valid_v[vanity_col])
    tau_h, pval_h = kendalltau(valid_h.index, valid_h[health_col])

    if (tau_v >= min_tau and pval_v < alpha) and (tau_h <= -min_tau and pval_h < alpha):
        findings.append(Finding(
            check='metric_confusion',
            severity=Severity.CRITICAL,
            column=f"{vanity_col} vs {health_col}",
            metric_value=round(float(pval_v), 4),
            threshold=alpha,
            message=f"Metric divergence: '{vanity_col}' is increasing (tau={tau_v:.2f}, p={pval_v:.4f}), while '{health_col}' is decreasing (tau={tau_h:.2f}, p={pval_h:.4f})."
        ))

    return findings
