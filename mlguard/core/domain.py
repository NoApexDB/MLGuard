from enum import Enum
from typing import Any, Protocol
import pandas as pd


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.value


class Finding(dict):
    """
    Доменная сущность находки аудита.
    Наследуется от dict для нативной обратной совместимости с json.dumps,
    pd.DataFrame, click и старыми тестами, при этом предоставляя
    строгую типизацию атрибутов и свойств.
    """

    def __init__(
        self,
        check: str,
        severity: Severity | str,
        column: str,
        metric_value: float,
        threshold: float,
        message: str,
        **kwargs: Any
    ):
        sev_str = severity.value if isinstance(severity, Severity) else str(severity).upper()
        super().__init__(
            check=str(check),
            severity=sev_str,
            column=str(column),
            metric_value=float(metric_value),
            threshold=float(threshold),
            message=str(message),
            **kwargs
        )

    @property
    def check(self) -> str:
        return self["check"]

    @property
    def severity(self) -> str:
        return self["severity"]

    @property
    def column(self) -> str:
        return self["column"]

    @property
    def metric_value(self) -> float:
        return self["metric_value"]

    @property
    def threshold(self) -> float:
        return self["threshold"]

    @property
    def message(self) -> str:
        return self["message"]

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


class CheckProtocol(Protocol):
    name: str
    description: str

    def __call__(
        self,
        df: pd.DataFrame,
        target_col: str | None = None,
        time_col: str | None = None,
        **kwargs: Any
    ) -> list[Finding]:
        ...
