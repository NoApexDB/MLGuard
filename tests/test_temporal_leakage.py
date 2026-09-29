import numpy as np
import pandas as pd
from mlguard.checks.temporal_leakage import check_temporal_leakage


def test_temporal_leakage_catches_future_timestamps():
    n = 50
    base_dates = pd.date_range("2026-01-01", periods=n, freq="D")
    # Другая временная колонка содержит даты на 10 дней позже
    future_dates = base_dates + pd.Timedelta(days=10)

    df = pd.DataFrame({
        "event_time": base_dates,
        "completion_time": future_dates,
        "feature": np.random.normal(0, 1, n)
    })

    findings = check_temporal_leakage(df, time_col="event_time")
    assert len(findings) == 1
    assert findings[0]["column"] == "completion_time"
    assert "Future timestamps detected" in findings[0]["message"]


def test_temporal_leakage_catches_lead_feature_correlation():
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    target = np.cumsum(np.random.normal(0, 1, n))
    # Фича заглядывает на 1 шаг вперед и повторяет y[t+1]
    lookahead_feat = np.roll(target, -1)
    lookahead_feat[-1] = 0

    df = pd.DataFrame({
        "date": dates,
        "target": target,
        "lookahead_feat": lookahead_feat
    })

    findings = check_temporal_leakage(df, time_col="date", target_col="target")
    lead_findings = [f for f in findings if f["column"] == "lookahead_feat"]
    assert len(lead_findings) == 1
    assert "next-step target" in lead_findings[0]["message"]


def test_temporal_leakage_clean_data_no_findings():
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "feat": np.random.normal(0, 1, n),
        "target": np.random.normal(0, 1, n)
    })
    findings = check_temporal_leakage(df, time_col="date", target_col="target")
    assert len(findings) == 0
