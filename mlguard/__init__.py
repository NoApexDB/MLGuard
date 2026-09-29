"""
MLGuard — линтер для предиктивной аналитики бизнеса.
Находит методологические и статистические ловушки в данных без вреда продакшену.
"""
from pathlib import Path
import pandas as pd

from mlguard.core.loaders import load_data, register_loader
from mlguard.core.report import format_html, format_json, format_markdown, format_table
from mlguard.core.runner import run_audit

__version__ = "1.0.0"


class AuditReport:
    """Результат аудита данных MLGuard с поддержкой отображения в Jupyter и CLI."""

    def __init__(self, findings: list[dict]):
        self.findings = findings

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def __getitem__(self, idx):
        return self.findings[idx]

    def __bool__(self) -> bool:
        return len(self.findings) > 0

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if str(f.get("severity", "")).upper() == "CRITICAL")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if str(f.get("severity", "")).upper() == "WARNING")

    def has_critical(self) -> bool:
        return self.critical_count > 0

    def to_dict(self) -> list[dict]:
        return list(self.findings)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.findings)

    def to_json(self) -> str:
        return format_json(self.findings)

    def to_markdown(self) -> str:
        return format_markdown(self.findings)

    def to_html(self) -> str:
        return format_html(self.findings)

    def _repr_html_(self) -> str:
        """Интерактивное отображение отчёта внутри Jupyter Notebook."""
        return self.to_html()

    def __repr__(self) -> str:
        return f"<AuditReport findings={len(self.findings)} critical={self.critical_count} warning={self.warning_count}>"

    def __str__(self) -> str:
        return format_table(self.findings)


def audit(
    data: str | Path | pd.DataFrame,
    target_col: str = None,
    time_col: str = None,
    checks: list[str] = None
) -> AuditReport:
    """
    Запускает полный аудит датасета.

    Параметры:
        data: Путь к файлу (csv, parquet, json, jsonl, xlsx) или pd.DataFrame.
        target_col: Название колонки целевой переменной (таргета).
        time_col: Название колонки даты/времени.
        checks: Список проверяемых чеков (по умолчанию все зарегистрированные).
    """
    if isinstance(data, (str, Path)):
        df = load_data(str(data))
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Ожидался путь к файлу или pd.DataFrame, получено: {type(data)}")

    findings = run_audit(df, target_col=target_col, time_col=time_col, checks=checks)
    return AuditReport(findings)


__all__ = [
    "audit",
    "AuditReport",
    "load_data",
    "register_loader",
    "run_audit",
    "__version__",
]
