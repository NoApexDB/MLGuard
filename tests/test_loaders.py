import click
import pandas as pd
import pytest
from mlguard.core.loaders import load_data


def test_loader_csv(tmp_path):
    p = tmp_path / "test.csv"
    p.write_text("a,b\n1,2\n3,4", encoding="utf-8")
    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_loader_json(tmp_path):
    p = tmp_path / "test.json"
    p.write_text('[{"a": 1, "b": 2}, {"a": 3, "b": 4}]', encoding="utf-8")
    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_loader_jsonl(tmp_path):
    p = tmp_path / "test.jsonl"
    p.write_text('{"a": 1, "b": 2}\n{"a": 3, "b": 4}', encoding="utf-8")
    df = load_data(str(p))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_loader_unsupported_extension_raises_click_exception(tmp_path):
    p = tmp_path / "test.xyz"
    p.write_text("junk", encoding="utf-8")
    with pytest.raises(click.ClickException):
        load_data(str(p))
