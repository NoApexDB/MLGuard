import numpy as np
import pandas as pd
from mlguard.core.domain import Finding, Severity


def check_correlation_leakage(
    df: pd.DataFrame, 
    target_col: str = None, 
    threshold: float = 0.95
) -> list[Finding]:
    findings: list[Finding] = []
    numeric_df = df.select_dtypes(include='number')
    numeric_cols = list(numeric_df.columns)

    target_series = None
    target_is_binary_factorized = False

    if target_col is not None and target_col in df.columns:
        if target_col in numeric_df.columns:
            target_series = numeric_df[target_col]
        elif df[target_col].nunique(dropna=True) == 2:
            target_series = pd.Series(pd.factorize(df[target_col])[0], index=df.index, dtype=float)
            target_is_binary_factorized = True

    if target_series is not None:
        for col in numeric_cols:
            if col == target_col and not target_is_binary_factorized:
                continue

            s_col = numeric_df[col]
            p_corr = abs(s_col.corr(target_series, method='pearson'))
            s_corr = abs(s_col.corr(target_series, method='spearman'))
            corr_value = max(p_corr, s_corr)
            if not np.isnan(corr_value) and corr_value > threshold:
                corr_type = "Spearman" if s_corr > p_corr else "Pearson"
                findings.append(Finding(
                    check="leakage_detector",
                    severity=Severity.CRITICAL,
                    column=col,
                    metric_value=round(float(corr_value), 4),
                    threshold=threshold,
                    message=f"Feature '{col}' has {corr_type} correlation of {corr_value:.4f} with target '{target_col}'."
                ))
    elif len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                
                p_corr = abs(numeric_df[col1].corr(numeric_df[col2], method='pearson'))
                s_corr = abs(numeric_df[col1].corr(numeric_df[col2], method='spearman'))
                corr_value = max(p_corr, s_corr)
                if not np.isnan(corr_value) and corr_value > threshold:
                    findings.append(Finding(
                        check="leakage_detector",
                        severity=Severity.WARNING,
                        column=f"{col1} vs {col2}",
                        metric_value=round(float(corr_value), 4),
                        threshold=threshold,
                        message=f"High correlation ({corr_value:.4f}) detected between '{col1}' and '{col2}'."
                    ))

    # Проверка колонок-идентификаторов (ID/UUID), ведущих к переобучению/утечке
    if len(df) >= 30:
        for col in df.columns:
            if col == target_col:
                continue
            col_lower = str(col).lower()
            is_id_name = any(k in col_lower for k in ["id", "uuid", "guid", "token", "hash", "key", "index"])
            s = df[col].dropna()
            n_valid = len(s)
            if n_valid >= 30:
                id_ratio = s.nunique() / n_valid
                if id_ratio >= 0.99:
                    if is_id_name or (pd.api.types.is_integer_dtype(df[col]) and (s.to_numpy() == np.arange(len(s))).all()):
                        findings.append(Finding(
                            check="leakage_detector",
                            severity=Severity.WARNING,
                            column=col,
                            metric_value=round(float(id_ratio), 4),
                            threshold=0.99,
                            message=f"Unique identifier column '{col}' detected ({id_ratio:.1%} unique values). Trivial memorization risk."
                        ))

    return findings
