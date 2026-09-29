import pandas as pd
from mlguard.checks.leakage_detector import check_correlation_leakage
from mlguard.checks.check_sample_size import check_sample_size
from mlguard.checks.churn_blindness import check_churn_blindness
from mlguard.checks.metric_confusion import check_metric_confusion
from mlguard.checks.seasonality_sentiment import check_seasonality
from mlguard.checks.hidden_missing import check_hidden_missing
from mlguard.checks.temporal_leakage import check_temporal_leakage


def run_audit(
    df: pd.DataFrame, 
    target_col: str = None, 
    time_col: str = None,
    checks: list[str] = None
) -> list[dict]:
    all_findings = []

    all_findings.extend(check_correlation_leakage(df, target_col=target_col))
    all_findings.extend(check_sample_size(df, target_col=target_col))
    all_findings.extend(check_churn_blindness(df, status_col=target_col))
    all_findings.extend(check_metric_confusion(df, time_col=time_col))
    all_findings.extend(check_seasonality(df, time_col=time_col, target_col=target_col))
    all_findings.extend(check_hidden_missing(df))
    all_findings.extend(check_temporal_leakage(df, time_col=time_col, target_col=target_col))

    if checks:
        checks_set = set(checks)
        all_findings = [f for f in all_findings if f["check"] in checks_set]

    return all_findings

