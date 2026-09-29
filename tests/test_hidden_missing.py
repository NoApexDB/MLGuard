import numpy as np
import pandas as pd
from mlguard.checks.hidden_missing import check_hidden_missing


def test_hidden_missing_catches_string_sentinels():
    n = 100
    # 20% вопросительных знаков вместо реальных пропусков
    vals = ["valid_val"] * 80 + ["?"] * 20
    df = pd.DataFrame({"category": vals})

    findings = check_hidden_missing(df)
    assert len(findings) == 1
    assert findings[0]["column"] == "category"
    assert "Masked missing values" in findings[0]["message"]


def test_hidden_missing_catches_numeric_sentinels():
    n = 100
    # Числовой код пропуска -999 в 15% строк
    vals = np.array([50.0] * 85 + [-999.0] * 15)
    df = pd.DataFrame({"measurement": vals})

    findings = check_hidden_missing(df)
    assert len(findings) == 1
    assert findings[0]["column"] == "measurement"
    assert "Suspicious numeric sentinel (-999)" in findings[0]["message"]


def test_hidden_missing_catches_infinite_values():
    n = 50
    vals = [1.0] * 48 + [np.inf, -np.inf]
    df = pd.DataFrame({"score": vals})

    findings = check_hidden_missing(df)
    inf_findings = [f for f in findings if f["severity"] == "CRITICAL"]
    assert len(inf_findings) == 1
    assert "Infinite values" in inf_findings[0]["message"]


def test_hidden_missing_clean_data_no_findings():
    df = pd.DataFrame({
        "num": np.random.normal(50, 10, 100),
        "cat": ["A", "B", "C", "D"] * 25
    })
    findings = check_hidden_missing(df)
    assert len(findings) == 0
