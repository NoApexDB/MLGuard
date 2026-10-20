import pandas as pd
import numpy as np
from mlguard.core.domain import Finding, Severity


def check_sample_size(
    df: pd.DataFrame, 
    target_col: str = None, 
    min_samples: int = 200,
    threshold_ci_norm: float = 0.20
) -> list[Finding]:
    findings: list[Finding] = []

    if len(df) < min_samples:
        findings.append(Finding(
            check="small_sample_guard",
            severity=Severity.WARNING,
            column="all",
            metric_value=float(len(df)),
            threshold=float(min_samples),
            message=f"Sample size ({len(df)}) is below the minimum threshold ({min_samples})."
        ))

    target_cols_to_check = []
    if target_col is not None and target_col in df.columns:
        if df[target_col].nunique() <= 10 or pd.api.types.is_bool_dtype(df[target_col]) or pd.api.types.is_object_dtype(df[target_col]):
            target_cols_to_check.append(target_col)
    else:
        target_cols_to_check = [c for c in df.columns if df[c].nunique() <= 10]

    for col in target_cols_to_check:
        proportions = df[col].value_counts(normalize=True)
        if len(proportions) > 1 and proportions.min() < 0.15:
            findings.append(Finding(
                check="small_sample_guard",
                severity=Severity.WARNING,
                column=col,
                metric_value=round(float(proportions.min()), 4),
                threshold=0.15,
                message=f"Class imbalance detected in '{col}': minority class represents {proportions.min():.1%}."
            ))

    num_cols = df.select_dtypes(include='number').columns
    eval_col = target_col if (target_col in num_cols) else (num_cols[0] if len(num_cols) > 0 else None)

    if eval_col is not None:
        col_series = df[eval_col].dropna()
        if len(col_series) > 1:
            col_std = float(col_series.std())
            col_data = col_series.to_numpy()
            n = len(col_data)

            # Векторизованный Bootstrap (1000 ресэмплов в одной NumPy операции без циклов)
            samples = np.random.choice(col_data, size=(1000, n), replace=True)
            bootstrap_means = samples.mean(axis=1)

            p_97 = float(np.percentile(bootstrap_means, 97.5))
            p_2 = float(np.percentile(bootstrap_means, 2.5))
            ci_width = float(p_97 - p_2)

            # Относительная ширина ДИ к стандартному отклонению (инвариантна к масштабу единиц)
            norm_width = (ci_width / col_std) if col_std > 1e-9 else 0.0

            if norm_width > threshold_ci_norm:
                findings.append(Finding(
                    check="small_sample_guard",
                    severity=Severity.CRITICAL,
                    column=eval_col,
                    metric_value=round(float(norm_width), 4),
                    threshold=threshold_ci_norm,
                    message=f"High uncertainty for '{eval_col}': 95% CI [{p_2:.3f}, {p_97:.3f}], relative width {norm_width:.3f}."
                ))

    return findings
