import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from mlguard.core.domain import Finding, Severity


def check_churn_blindness(
    df: pd.DataFrame, 
    status_col: str = None, 
    threshold_pvalue: float = 0.05,
    min_effect_size: float = 0.25
) -> list[Finding]:
    findings: list[Finding] = []

    if status_col is None:
        # Сначала ищем по характерным названиям колонок статуса/оттока
        status_candidates = [
            c for c in df.columns 
            if any(k in c.lower() for k in ["churn", "status", "target", "label", "exit", "attrit"])
            and df[c].nunique(dropna=True) == 2
        ]
        if status_candidates:
            status_col = status_candidates[0]
        else:
            possible_status_cols = [c for c in df.columns if set(df[c].dropna().unique()).issubset({0, 1})]
            if not possible_status_cols:
                return findings
            status_col = possible_status_cols[0]

    if status_col not in df.columns:
        return findings

    unique_vals = list(df[status_col].dropna().unique())
    if len(unique_vals) != 2:
        return findings

    # Разделение на две группы исходов (с поддержкой 0/1, True/False, текстовых классов)
    val_0, val_1 = unique_vals[0], unique_vals[1]
    if {0, 1}.issubset(set(unique_vals)):
        active_val, churn_val = 0, 1
    elif {True, False}.issubset(set(unique_vals)):
        active_val, churn_val = False, True
    else:
        active_val, churn_val = val_0, val_1

    active_df = df[df[status_col] == active_val]
    churned_df = df[df[status_col] == churn_val]

    if len(active_df) < 5 or len(churned_df) < 5:
        return findings

    # 1. Анализ числовых признаков через двухвыборочный KS-тест
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        if col == status_col:
            continue

        active_vals = active_df[col].dropna()
        churned_vals = churned_df[col].dropna()
        if len(active_vals) < 5 or len(churned_vals) < 5:
            continue

        stat_result = ks_2samp(active_vals, churned_vals)
        ks_stat = float(stat_result.statistic)
        p_val = float(stat_result.pvalue)

        if p_val < threshold_pvalue and ks_stat >= min_effect_size:
            findings.append(Finding(
                check='churn_blindness',
                severity=Severity.CRITICAL,
                column=col,
                metric_value=round(float(p_val), 4),
                threshold=threshold_pvalue,
                message=f"Distribution shift detected for '{col}' between groups of '{status_col}' (D={ks_stat:.2f}, p-value={p_val:.4f})."
            ))

    # 2. Анализ категориальных признаков через Chi-square тест сопряжённости
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns
    for col in cat_cols:
        if col == status_col:
            continue

        valid_df = df[[col, status_col]].dropna()
        if len(valid_df) < 20:
            continue

        contingency = pd.crosstab(valid_df[col], valid_df[status_col])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        try:
            chi2, p_val, _, _ = chi2_contingency(contingency)
            n_total = contingency.to_numpy().sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = float(np.sqrt(chi2 / (n_total * min_dim))) if n_total > 0 and min_dim > 0 else 0.0

            if p_val < threshold_pvalue and cramers_v >= min_effect_size:
                findings.append(Finding(
                    check='churn_blindness',
                    severity=Severity.CRITICAL,
                    column=col,
                    metric_value=round(float(p_val), 4),
                    threshold=threshold_pvalue,
                    message=f"Categorical shift detected for '{col}' across '{status_col}' groups (Cramer's V={cramers_v:.2f}, p-value={p_val:.4f})."
                ))
        except Exception:
            pass

    return findings
