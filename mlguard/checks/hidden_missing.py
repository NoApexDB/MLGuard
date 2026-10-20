import numpy as np
import pandas as pd
from mlguard.core.domain import Finding, Severity


def check_hidden_missing(
    df: pd.DataFrame, 
    threshold_ratio: float = 0.05
) -> list[Finding]:
    """
    Детектирует маскированные пропуски данных:
    1. Строковые псевдо-пропуски: '?', 'na', 'n/a', 'null', 'none', 'nan', 'missing', '-', ''
    2. Числовые sentinel-значения: -999, -9999, 999, 9999
    3. Бесконечные значения: np.inf, -np.inf
    """
    findings: list[Finding] = []
    numeric_sentinels = {-999, -9999, 999, 9999}
    string_sentinels = {"?", "na", "n/a", "null", "none", "nan", "missing", "-", "--", "unknown"}

    for col in df.columns:
        s = df[col].dropna()
        n = len(s)
        if n < 10:
            continue

        # 1. Проверка строковых псевдо-пропусков
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            str_clean = s.astype(str).str.strip().str.lower()
            mask = str_clean.isin(string_sentinels) | (str_clean == "")
            count = int(mask.sum())
            ratio = count / n
            if ratio >= threshold_ratio and count >= 5:
                findings.append(Finding(
                    check="hidden_missing",
                    severity=Severity.WARNING,
                    column=col,
                    metric_value=round(float(ratio), 4),
                    threshold=threshold_ratio,
                    message=f"Masked missing values detected in '{col}': {count} entries ({ratio:.1%}) match sentinels like '?', 'null', 'None'."
                ))

        # 2. Проверка числовых колонок (inf и sentinels)
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Проверка inf/-inf
            inf_count = int(np.isinf(s).sum())
            if inf_count > 0:
                findings.append(Finding(
                    check="hidden_missing",
                    severity=Severity.CRITICAL,
                    column=col,
                    metric_value=float(inf_count),
                    threshold=0.0,
                    message=f"Infinite values (inf/-inf) detected in '{col}': {inf_count} occurrences."
                ))

            # Числовые экстремальные sentinels (-999, 9999)
            val_counts = s.value_counts()
            for sentinel in numeric_sentinels:
                if sentinel in val_counts:
                    cnt = int(val_counts[sentinel])
                    ratio = cnt / n
                    if ratio >= threshold_ratio and cnt >= 5:
                        findings.append(Finding(
                            check="hidden_missing",
                            severity=Severity.WARNING,
                            column=col,
                            metric_value=round(float(ratio), 4),
                            threshold=threshold_ratio,
                            message=f"Suspicious numeric sentinel ({sentinel}) detected in '{col}': {cnt} entries ({ratio:.1%}). Likely masked missing data."
                        ))

    return findings
