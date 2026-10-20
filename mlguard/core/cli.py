"""
CLI mlguard.

Использование:
    mlguard audit data.csv --target churn --time-col created_at
    mlguard audit data.csv --format json -o report.json
    mlguard audit data.csv --fail-on WARNING   # для CI/CD gate
    mlguard checks list
"""
import sys
from pathlib import Path

import click

from mlguard.core.config import load_config, merge_cli_overrides
from mlguard.core.loaders import load_data
from mlguard.core.report import FORMATTERS
from mlguard.core.runner import run_audit

SEVERITY_ORDER = ["INFO", "WARNING", "CRITICAL"]

# Имя чека -> человекочитаемое описание. Используется в `mlguard checks list`
# и как единственный источник правды для --only (сверяется с полем "check"
# в findings, а не с именами файлов — они, как видно на small_sample_guard,
# могут расходиться).
CHECK_REGISTRY = {
    "leakage_detector": "Data Leakage Guard — корреляционные и мультиколлинеарные утечки",
    "small_sample_guard": "Small Sample & Statistical Power — размер выборки, bootstrap CI",
    "churn_blindness": "Survivorship Bias Detector — сдвиг распределений между группами исходов",
    "metric_confusion": "Vanity Metrics Detector — расхождение трендов метрик (Mann-Kendall)",
    "seasonality_sentiment": "Seasonality & Trend Leak — STL-декомпозиция, сезонность",
    "hidden_missing": "Hidden Missing Values — маскированные пропуски ('?', '-999', inf)",
    "temporal_leakage": "Temporal Lookahead Guard — заглядывание в будущее и опережающие события",
}



def _severity_rank(severity: str) -> int:
    severity = str(severity).upper()
    return SEVERITY_ORDER.index(severity) if severity in SEVERITY_ORDER else 0


@click.group()
@click.version_option(package_name="mlguard")
def cli():
    """MLGuard — линтер для предиктивной аналитики бизнеса."""


@cli.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False))
@click.option("-t", "--target", "target_col", default=None, help="Колонка целевой переменной.")
@click.option("--time-col", default=None, help="Колонка даты/времени.")
@click.option(
    "--only", default=None,
    help="Список чеков через запятую, напр.: leakage_detector,small_sample_guard",
)
@click.option(
    "-f", "--format", "output_format",
    type=click.Choice(list(FORMATTERS), case_sensitive=False), default=None,
    help="Формат отчёта (по умолчанию table).",
)
@click.option("-o", "--output", "output_path", type=click.Path(), default=None,
              help="Сохранить отчёт в файл вместо вывода в stdout.")
@click.option(
    "--fail-on", type=click.Choice(SEVERITY_ORDER, case_sensitive=False), default=None,
    help="Минимальная severity, при которой команда вернёт exit code 1 (для CI/CD).",
)
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), default=None,
              help="Путь к mlguard.toml (по умолчанию ищется в текущей директории).")
def audit(source, target_col, time_col, only, output_format, output_path, fail_on, config_path):
    """Прогнать аудит по файлу данных (.csv / .parquet)."""
    config = load_config(config_path)
    config = merge_cli_overrides(
        config,
        target_col=target_col,
        time_col=time_col,
        format=output_format,
        fail_on=fail_on,
    )

    checks_filter = None
    if only:
        checks_filter = {c.strip() for c in only.split(",") if c.strip()}
    elif config.get("checks"):
        checks_filter = set(config["checks"])

    if checks_filter:
        unknown = checks_filter - set(CHECK_REGISTRY)
        if unknown:
            raise click.ClickException(
                f"Неизвестные чеки: {', '.join(sorted(unknown))}. "
                f"Смотри `mlguard checks list`."
            )

    df = load_data(source)
    findings = run_audit(
        df,
        target_col=config["target_col"],
        time_col=config["time_col"],
        checks=list(checks_filter) if checks_filter else None
    )

    fmt = (config.get("format") or "table").lower()
    rendered = FORMATTERS[fmt](findings)

    if output_path:
        Path(output_path).write_text(rendered, encoding="utf-8")
        click.echo(f"Отчёт сохранён: {output_path}")
    else:
        click.echo(rendered)

    fail_on_value = (config.get("fail_on") or "CRITICAL").upper()
    threshold_rank = _severity_rank(fail_on_value)
    is_blocking = any(_severity_rank(f.get("severity")) >= threshold_rank for f in findings)
    if is_blocking:
        sys.exit(1)


@cli.group()
def checks():
    """Информация о доступных проверках."""


@checks.command("list")
def checks_list():
    """Показать список чеков, которые запускает `mlguard audit`."""
    for name, description in CHECK_REGISTRY.items():
        click.echo(f"{click.style(name, bold=True):<28} {description}")


if __name__ == "__main__":
    cli()
