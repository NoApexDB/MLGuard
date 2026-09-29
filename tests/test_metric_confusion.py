import numpy as np
import pandas as pd
from mlguard.checks.metric_confusion import check_metric_confusion


def test_metric_confusion_catches_diverging_trends():
    n = 100
    dates = pd.date_range("2026-01-01", periods=n)
    # vanity metric растёт, health metric падает
    order_volume = np.linspace(10, 200, n) + np.random.normal(0, 2, n)
    profit_margin = np.linspace(0.40, 0.05, n) + np.random.normal(0, 0.01, n)

    df = pd.DataFrame({
        "date": dates,
        "order_volume": order_volume,
        "profit_margin": profit_margin
    })

    findings = check_metric_confusion(df, time_col="date")
    assert len(findings) == 1
    assert "order_volume vs profit_margin" in findings[0]["column"]
    assert findings[0]["severity"] == "CRITICAL"


def test_metric_confusion_filters_noise():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2026-01-01", periods=n)
    # Случайный шум без выраженного тренда
    df = pd.DataFrame({
        "date": dates,
        "metric1": np.random.normal(50, 5, n),
        "metric2": np.random.normal(100, 10, n)
    })

    findings = check_metric_confusion(df, time_col="date")
    assert len(findings) == 0


def test_metric_confusion_limits_max_pairs():
    n = 60
    dates = pd.date_range("2026-01-01", periods=n)
    data = {"date": dates}
    # 4 растущих и 4 падающих метрики (без лимита было бы 16 пар)
    for i in range(4):
        data[f"inc_{i}"] = np.linspace(1, 50, n)
        data[f"dec_{i}"] = np.linspace(50, 1, n)

    df = pd.DataFrame(data)
    findings = check_metric_confusion(df, time_col="date", max_pairs=3)
    assert len(findings) == 3
