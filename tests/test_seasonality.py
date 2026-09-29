import numpy as np
import pandas as pd
from mlguard.checks.seasonality_sentiment import check_seasonality


def test_seasonality_catches_periodic_signal():
    n = 100
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    t = np.arange(n)
    # Сильная недельная сезонность (период 7 дней)
    seasonal = np.sin(2 * np.pi * t / 7.0) * 30.0
    series = np.linspace(10, 20, n) + seasonal

    df = pd.DataFrame({
        "date": dates,
        "sales": series
    })

    findings = check_seasonality(df, time_col="date")
    assert len(findings) == 1
    assert findings[0]["column"] == "sales"
    assert "High seasonality" in findings[0]["message"]


def test_seasonality_handles_string_columns_without_typeerror():
    n = 80
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    # Датасет со строковыми колонками (раньше падал в resample('D').mean())
    df = pd.DataFrame({
        "date": dates,
        "category": ["A", "B", "C", "D"] * 20,
        "sales": np.random.normal(100, 5, n)
    })

    findings = check_seasonality(df, time_col="date")
    assert isinstance(findings, list)


def test_seasonality_clean_data_no_seasonality():
    np.random.seed(42)
    n = 400
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "pure_noise": np.random.normal(50, 5, n)
    })

    findings = check_seasonality(df, time_col="date", threshold_ratio=0.35)
    assert len(findings) == 0

