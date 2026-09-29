import numpy as np
import pandas as pd
import mlguard
from mlguard import audit, AuditReport


def test_public_audit_with_dataframe():
    df = pd.DataFrame({
        "target": [0, 1] * 20,
        "leak": [0.0, 10.0] * 20,
        "clean": np.random.normal(0, 1, 40)
    })
    report = audit(df, target_col="target")
    assert isinstance(report, AuditReport)
    assert len(report) >= 1
    assert report.has_critical() is True
    assert isinstance(report.to_dataframe(), pd.DataFrame)
    assert "leakage_detector" in report.to_json()
    assert "MLGuard Audit Report" in report.to_html()
    assert report._repr_html_() == report.to_html()


def test_public_audit_with_file_path(tmp_path):
    p = tmp_path / "data.csv"
    df = pd.DataFrame({
        "target": [0, 1] * 30,
        "clean": list(range(60))
    })
    df.to_csv(p, index=False)
    report = mlguard.audit(str(p), target_col="target")
    assert isinstance(report, AuditReport)


def test_public_audit_checks_filter():
    df = pd.DataFrame({
        "target": [0, 1] * 20,
        "leak": [0.0, 10.0] * 20
    })
    # Запуск только с фильтром по small_sample_guard (не должен запускать leakage_detector)
    report = audit(df, target_col="target", checks=["small_sample_guard"])
    for f in report:
        assert f["check"] == "small_sample_guard"
