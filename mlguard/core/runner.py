import pandas as pd
from mlguard.core.domain import Finding
from mlguard.checks.leakage_detector import check_correlation_leakage
from mlguard.checks.check_sample_size import check_sample_size
from mlguard.checks.churn_blindness import check_churn_blindness
from mlguard.checks.metric_confusion import check_metric_confusion
from mlguard.checks.seasonality_sentiment import check_seasonality
from mlguard.checks.hidden_missing import check_hidden_missing
from mlguard.checks.temporal_leakage import check_temporal_leakage

CHECK_HANDLERS = {
    "leakage_detector": lambda df, target, time: check_correlation_leakage(df, target_col=target),
    "small_sample_guard": lambda df, target, time: check_sample_size(df, target_col=target),
    "churn_blindness": lambda df, target, time: check_churn_blindness(df, status_col=target),
    "metric_confusion": lambda df, target, time: check_metric_confusion(df, time_col=time),
    "seasonality_sentiment": lambda df, target, time: check_seasonality(df, time_col=time, target_col=target),
    "hidden_missing": lambda df, target, time: check_hidden_missing(df),
    "temporal_leakage": lambda df, target, time: check_temporal_leakage(df, time_col=time, target_col=target),
}


def run_audit(
    df: pd.DataFrame, 
    target_col: str = None, 
    time_col: str = None,
    checks: list[str] = None
) -> list[Finding]:
    """
    Запускает аудит данных с ленивым выполнением только активных чеков.
    """
    all_findings: list[Finding] = []

    active_checks = list(CHECK_HANDLERS.keys())
    if checks:
        checks_set = set(checks)
        active_checks = [c for c in active_checks if c in checks_set]

    for check_name in active_checks:
        handler = CHECK_HANDLERS[check_name]
        all_findings.extend(handler(df, target_col, time_col))

    return all_findings
