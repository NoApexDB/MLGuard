import numpy as np
import pandas as pd
from mlguard.checks.check_sample_size import check_sample_size


def test_small_sample_size_flagged():
    df = pd.DataFrame({
        "feat": np.random.normal(0, 1, 50),
        "target": np.random.choice([0, 1], 50)
    })
    findings = check_sample_size(df, min_samples=200)
    sample_findings = [f for f in findings if f["column"] == "all"]
    assert len(sample_findings) == 1
    assert sample_findings[0]["severity"] == "WARNING"


def test_class_imbalance_flagged_for_discrete_target():
    # 5% класса 1 (сильный дисбаланс)
    target = np.array([0] * 190 + [1] * 10)
    df = pd.DataFrame({
        "feat": np.random.normal(0, 1, len(target)),
        "target": target
    })
    findings = check_sample_size(df, target_col="target")
    imbalance_findings = [f for f in findings if "Class imbalance" in f["message"]]
    assert len(imbalance_findings) == 1
    assert "Class imbalance detected" in imbalance_findings[0]["message"]



def test_no_false_imbalance_on_continuous_regression_target():
    # Регрессионный таргет с 300 уникальными значениями
    df = pd.DataFrame({
        "revenue": np.random.uniform(100, 10000, 300),
        "x": np.random.normal(0, 1, 300)
    })
    findings = check_sample_size(df, target_col="revenue")
    imbalance_findings = [f for f in findings if "Class imbalance" in f["message"]]
    assert len(imbalance_findings) == 0


def test_bootstrap_relative_scale_invariance_on_large_samples():
    np.random.seed(42)
    # Зарплата со средним 100,000 и большим разбросом на 500 строк
    salary = np.random.normal(100000, 15000, 500)
    df = pd.DataFrame({
        "salary": salary,
        "feat": np.random.normal(0, 1, 500)
    })
    findings = check_sample_size(df, target_col="salary")
    uncertainty_findings = [f for f in findings if "High uncertainty" in f["message"]]
    # При 500 строках погрешность среднего относительно std < 0.20, не должно быть CRITICAL
    assert len(uncertainty_findings) == 0


def test_sample_size_handles_nans():
    salary = [100.0, np.nan, 200.0, np.nan, 150.0] * 50
    df = pd.DataFrame({"salary": salary})
    # Не должно упасть с исключением из-за NaN
    findings = check_sample_size(df, target_col="salary")
    assert isinstance(findings, list)
