from click.testing import CliRunner
import pandas as pd
from mlguard.core.cli import cli


def test_cli_checks_list():
    runner = CliRunner()
    result = runner.invoke(cli, ["checks", "list"])
    assert result.exit_code == 0
    assert "leakage_detector" in result.output
    assert "churn_blindness" in result.output
    assert "small_sample_guard" in result.output


def test_cli_audit_table_output(tmp_path):
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "target": [0, 1] * 20,
        "feat_leak": [0.0, 10.0] * 20,
        "feat_clean": [5.0] * 40
    })
    df.to_csv(csv_file, index=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["audit", str(csv_file), "--target", "target"])
    # При наличии CRITICAL находок exit code = 1
    assert result.exit_code == 1
    assert "leakage_detector" in result.output


def test_cli_audit_json_output(tmp_path):
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "target": [0, 1] * 50,
        "feat_clean": list(range(100))
    })
    df.to_csv(csv_file, index=False)

    import json

    runner = CliRunner()
    result = runner.invoke(cli, ["audit", str(csv_file), "--target", "target", "-f", "json"])
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert any(item["check"] == "leakage_detector" for item in data)

