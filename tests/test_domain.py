import json
import pandas as pd
from mlguard.core.domain import Finding, Severity
from mlguard.core.runner import run_audit


def test_finding_domain_model():
    f = Finding(
        check="leakage_detector",
        severity=Severity.CRITICAL,
        column="feat_a",
        metric_value=0.992,
        threshold=0.95,
        message="Leakage test"
    )
    assert f.check == "leakage_detector"
    assert f.severity == "CRITICAL"
    assert f.column == "feat_a"
    assert f.metric_value == 0.992
    assert f.threshold == 0.95
    assert f.message == "Leakage test"
    
    # Dict-like compatibility
    assert f["check"] == "leakage_detector"
    assert f.get("column") == "feat_a"
    assert f.to_dict()["severity"] == "CRITICAL"

    # Native json serialization
    json_str = json.dumps([f])
    assert "leakage_detector" in json_str


def test_runner_lazy_execution():
    df = pd.DataFrame({
        "target": [0, 1] * 20,
        "feat": [1.0, 2.0] * 20
    })
    # Запрос только одной легкой проверки не должен вызывать остальные
    findings = run_audit(df, target_col="target", checks=["leakage_detector"])
    for item in findings:
        assert item["check"] == "leakage_detector"
