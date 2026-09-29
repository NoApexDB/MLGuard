import numpy as np
import pandas as pd
from mlguard.checks.churn_blindness import check_churn_blindness


def test_churn_blindness_catches_numerical_shift():
    np.random.seed(42)
    n = 200
    status = np.random.choice([0, 1], size=n)
    # Зарплата сильно отличается для ушедших и оставшихся
    salary = np.where(status == 1, np.random.normal(30000, 5000, n), np.random.normal(80000, 10000, n))

    df = pd.DataFrame({
        "salary": salary,
        "is_churn": status
    })

    findings = check_churn_blindness(df, status_col="is_churn")
    assert len(findings) == 1
    assert findings[0]["column"] == "salary"
    assert findings[0]["severity"] == "CRITICAL"


def test_churn_blindness_supports_string_targets():
    n = 150
    status = np.random.choice(["Yes", "No"], size=n)
    tenure = np.where(status == "Yes", np.random.exponential(5, n), np.random.exponential(40, n))

    df = pd.DataFrame({
        "tenure": tenure,
        "Churn": status
    })

    findings = check_churn_blindness(df, status_col="Churn")
    assert len(findings) == 1
    assert findings[0]["column"] == "tenure"


def test_churn_blindness_supports_boolean_targets():
    n = 150
    status = np.random.choice([True, False], size=n)
    score = np.where(status, np.random.normal(10, 2, n), np.random.normal(50, 5, n))

    df = pd.DataFrame({
        "score": score,
        "churned": status
    })

    findings = check_churn_blindness(df, status_col="churned")
    assert len(findings) == 1
    assert findings[0]["column"] == "score"


def test_churn_blindness_catches_categorical_shift_via_chi2():
    # Сильный сдвиг категории тарифа между активными и ушедшими
    status = [0] * 100 + [1] * 100
    plan = ["basic"] * 80 + ["premium"] * 20 + ["basic"] * 10 + ["premium"] * 90

    df = pd.DataFrame({
        "plan": plan,
        "status": status
    })

    findings = check_churn_blindness(df, status_col="status")
    cat_findings = [f for f in findings if f["column"] == "plan"]
    assert len(cat_findings) == 1
    assert "Categorical shift detected" in cat_findings[0]["message"]
    assert "Cramer's V" in cat_findings[0]["message"]


def test_churn_blindness_clean_data_no_findings():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "val": np.random.normal(50, 10, n),
        "status": np.random.choice([0, 1], n)
    })
    findings = check_churn_blindness(df, status_col="status")
    assert len(findings) == 0
