import numpy as np
import pandas as pd


def check_correlation_leakage(
    df: pd.DataFrame, 
    target_col: str = None, 
    threshold: float = 0.95
) -> list[dict]:
    findings = []
    numeric_df = df.select_dtypes(include='number')
    numeric_cols = list(numeric_df.columns)

    if len(numeric_cols) >= 2:
        # 1. Утечка целевой переменной в признаки (target leakage)
        if target_col is not None and target_col in numeric_df.columns:
            for col in numeric_cols:
                if col == target_col:
                    continue
                    
                p_corr = abs(numeric_df[col].corr(numeric_df[target_col], method='pearson'))
                s_corr = abs(numeric_df[col].corr(numeric_df[target_col], method='spearman'))
                corr_value = max(p_corr, s_corr)
                if corr_value > threshold:
                    corr_type = "Spearman" if s_corr > p_corr else "Pearson"
                    findings.append({
                        "check": "leakage_detector",
                        "severity": "CRITICAL",
                        "column": col,
                        "metric_value": round(float(corr_value), 4),
                        "threshold": threshold,
                        "message": f"Feature '{col}' has {corr_type} correlation of {corr_value:.4f} with target '{target_col}'."
                    })

        # 2. Мультиколлинеарность между признаками (feature vs feature)
        feat_cols = [c for c in numeric_cols if c != target_col]
        for i in range(len(feat_cols)):
            for j in range(i + 1, len(feat_cols)):
                col1 = feat_cols[i]
                col2 = feat_cols[j]
                
                p_corr = abs(numeric_df[col1].corr(numeric_df[col2], method='pearson'))
                s_corr = abs(numeric_df[col1].corr(numeric_df[col2], method='spearman'))
                corr_value = max(p_corr, s_corr)
                if corr_value > threshold:
                    findings.append({
                        "check": "leakage_detector",
                        "severity": "WARNING",
                        "column": f"{col1} vs {col2}",
                        "metric_value": round(float(corr_value), 4),
                        "threshold": threshold,
                        "message": f"High correlation ({corr_value:.4f}) detected between '{col1}' and '{col2}'."
                    })


    # Проверка колонок-идентификаторов (ID/UUID), ведущих к переобучению/утечке
    if len(df) >= 30:
        for col in df.columns:
            if col == target_col:
                continue
            col_lower = col.lower()
            is_id_name = any(k in col_lower for k in ["id", "uuid", "guid", "token", "hash", "key", "index"])
            s = df[col].dropna()
            if len(s) == len(df) and s.nunique() == len(df):
                if is_id_name or (pd.api.types.is_integer_dtype(df[col]) and (s.to_numpy() == np.arange(len(df))).all()):
                    findings.append({
                        "check": "leakage_detector",
                        "severity": "WARNING",
                        "column": col,
                        "metric_value": 1.0,
                        "threshold": 1.0,
                        "message": f"Unique identifier column '{col}' detected (100% unique values). Trivial memorization risk."
                    })

    return findings