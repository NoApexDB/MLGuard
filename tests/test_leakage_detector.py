import numpy as np
import pandas as pd
from mlguard.checks.leakage_detector import check_correlation_leakage


def test_leakage_detector_catches_pearson_leak():
    n = 200
    target = np.random.choice([0, 1], size=n)
    leaky_feature = target * 10.0 + np.random.normal(0, 0.01, size=n)
    clean_feature = np.random.normal(50, 10, size=n)

    df = pd.DataFrame({
        "target": target,
        "leaky": leaky_feature,
        "clean": clean_feature
    })

    findings = check_correlation_leakage(df, target_col="target", threshold=0.95)
    assert len(findings) == 1
    assert findings[0]["column"] == "leaky"
    assert findings[0]["severity"] == "CRITICAL"


def test_leakage_detector_catches_spearman_nonlinear_leak():
    n = 200
    target = np.linspace(1, 100, n)
    # Экспоненциальная монотонная связь (ранговая корреляция 1.0)
    leaky_exp = np.exp(target / 20.0)

    df = pd.DataFrame({
        "target": target,
        "leaky_exp": leaky_exp,
        "noise": np.random.normal(0, 1, n)
    })

    findings = check_correlation_leakage(df, target_col="target", threshold=0.95)
    leaky_findings = [f for f in findings if f["column"] == "leaky_exp"]
    assert len(leaky_findings) == 1
    assert "Spearman" in leaky_findings[0]["message"]


def test_leakage_detector_catches_feature_multicollinearity():
    n = 150
    feat1 = np.random.normal(10, 2, n)
    feat2 = feat1 * 2.5 + np.random.normal(0, 0.01, n)
    target = np.random.normal(100, 15, n)

    df = pd.DataFrame({
        "feat1": feat1,
        "feat2": feat2,
        "target": target
    })

    findings = check_correlation_leakage(df, threshold=0.95)
    multicol_findings = [f for f in findings if f["severity"] == "WARNING"]
    assert len(multicol_findings) == 1
    assert "feat1 vs feat2" in multicol_findings[0]["column"]



def test_leakage_detector_catches_unique_id():
    n = 100
    df = pd.DataFrame({
        "customer_id": [f"user_{i:04d}" for i in range(n)],
        "age": np.random.randint(18, 70, size=n),
        "target": np.random.choice([0, 1], size=n)
    })

    findings = check_correlation_leakage(df, target_col="target")
    id_findings = [f for f in findings if f["column"] == "customer_id"]
    assert len(id_findings) == 1
    assert id_findings[0]["severity"] == "WARNING"


def test_leakage_detector_clean_data_no_findings():
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "feat1": np.random.normal(50, 10, size=n),
        "feat2": np.random.normal(100, 15, size=n),
        "target": np.random.choice([0, 1], size=n)
    })

    findings = check_correlation_leakage(df, target_col="target", threshold=0.95)
    assert len(findings) == 0
