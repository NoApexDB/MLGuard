import pandas as pd
import numpy as np


def check_sample_size(
    df: pd.DataFrame, 
    target_col: str = None, 
    min_samples: int = 200
) -> list[dict]:
    findings = []
    bootstrap_means = []

    if len(df) < min_samples:
        findings.append({
            "check": "small_sample_guard",
            "severity": "WARNING",
            "column": "all",
            "metric_value": len(df),
            "threshold": min_samples,
            "message": f"Sample size ({len(df)}) is below the minimum threshold ({min_samples})."
        })

    target_cols_to_check = []
    if target_col is not None and target_col in df.columns:
        if df[target_col].nunique() <= 10 or pd.api.types.is_bool_dtype(df[target_col]) or pd.api.types.is_object_dtype(df[target_col]):
            target_cols_to_check.append(target_col)
    else:
        target_cols_to_check = [c for c in df.columns if df[c].nunique() <= 10]

    for col in target_cols_to_check:
        proportions = df[col].value_counts(normalize=True)
        if len(proportions) > 1 and proportions.min() < 0.15:
            findings.append({
                "check": "small_sample_guard",
                "severity": "WARNING",
                "column": col,
                "metric_value": round(float(proportions.min()), 4),
                "threshold": 0.15,
                "message": f"Class imbalance detected in '{col}': minority class represents {proportions.min():.1%}."
            })

    num_cols = df.select_dtypes(include='number').columns
    eval_col = target_col if (target_col in num_cols) else (num_cols[0] if len(num_cols) > 0 else None)

    if eval_col is not None:
        col_series = df[eval_col].dropna()
        if len(col_series) > 1:
            col_std = float(col_series.std())
            col_data = col_series.to_numpy()
            n = len(col_data)

            for _ in range(1000):
                sample = np.random.choice(col_data, size=n, replace=True)
                bootstrap_means.append(sample.mean())

            p_97 = np.percentile(bootstrap_means, 97.5)
            p_2 = np.percentile(bootstrap_means, 2.5)
            ci_width = float(p_97 - p_2)

            # Относительная ширина ДИ к стандартному отклонению (инвариантна к масштабу единиц)
            norm_width = (ci_width / col_std) if col_std > 1e-9 else 0.0

            if norm_width > 0.20:
                findings.append({
                    "check": "small_sample_guard",
                    "severity": "CRITICAL",
                    "column": eval_col,
                    "metric_value": round(float(norm_width), 4),
                    "threshold": 0.20,
                    "message": f"High uncertainty for '{eval_col}': 95% CI [{p_2:.3f}, {p_97:.3f}], relative width {norm_width:.3f}."
                })

    return findings