"""
Рендеринг findings, которые возвращает run_audit(), в разные форматы.

Ожидаемая схема каждого finding (см. mlguard/checks/*.py):
    {
        "check": str,          # имя чека, напр. "leakage_detector"
        "severity": str,       # "WARNING" | "CRITICAL" (регистр не важен)
        "column": str,
        "metric_value": float,
        "threshold": float,
        "message": str,
    }
"""
import html
import json

import click

_SEVERITY_COLOR = {
    "CRITICAL": "bright_red",
    "WARNING": "yellow",
    "INFO": "white",
}


def format_table(findings: list[dict]) -> str:
    if not findings:
        return click.style("✓ Проблем не найдено", fg="green", bold=True)

    lines = []
    for f in findings:
        sev = str(f.get("severity", "")).upper()
        badge = click.style(f"{sev:8}", fg=_SEVERITY_COLOR.get(sev, "white"), bold=True)
        check = f.get("check", "?")
        column = f.get("column", "-")
        message = f.get("message", "")
        lines.append(f"[{badge}] {check:<20} {column:<25} {message}")

    summary = _summary_line(findings)
    return "\n".join(lines) + "\n\n" + summary


def format_json(findings: list[dict]) -> str:
    return json.dumps(findings, ensure_ascii=False, indent=2)


def format_markdown(findings: list[dict]) -> str:
    if not findings:
        return "✅ **Проблем не найдено**"

    lines = [
        "| Severity | Check | Column | Value | Threshold | Message |",
        "|---|---|---|---|---|---|",
    ]
    for f in findings:
        lines.append(
            f"| {f.get('severity', '')} | {f.get('check', '')} | {f.get('column', '')} "
            f"| {f.get('metric_value', '')} | {f.get('threshold', '')} | {f.get('message', '')} |"
        )
    lines.append("")
    lines.append(_summary_line(findings, markdown=True))
    return "\n".join(lines)


def format_html(findings: list[dict]) -> str:
    critical = sum(1 for f in findings if str(f.get("severity", "")).upper() == "CRITICAL")
    warning = sum(1 for f in findings if str(f.get("severity", "")).upper() == "WARNING")
    total = len(findings)

    rows = []
    for f in findings:
        sev = str(f.get("severity", "INFO")).upper()
        color = "#e53e3e" if sev == "CRITICAL" else ("#dd6b20" if sev == "WARNING" else "#3182ce")
        bg_color = "#fff5f5" if sev == "CRITICAL" else ("#fffaf0" if sev == "WARNING" else "#ebf8ff")
        check_val = html.escape(str(f.get("check", "")))
        col_val = html.escape(str(f.get("column", "")))
        msg_val = html.escape(str(f.get("message", "")))
        m_val = html.escape(str(f.get("metric_value", "")))
        th_val = html.escape(str(f.get("threshold", "")))
        rows.append(f"""
        <tr>
            <td><span style="background:{bg_color}; color:{color}; padding:4px 8px; border-radius:4px; font-weight:600; font-size:12px;">{sev}</span></td>
            <td><code>{check_val}</code></td>
            <td><b>{col_val}</b></td>
            <td>{m_val}</td>
            <td>{th_val}</td>
            <td>{msg_val}</td>
        </tr>""")

    tbody = "\n".join(rows) if rows else "<tr><td colspan='6' style='text-align:center; padding:20px; color:#38a169;'><b>✓ Проблем не найдено</b></td></tr>"

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>MLGuard Audit Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f7fafc; color: #2d3748; margin: 0; padding: 30px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #ffffff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); padding: 30px; }}
        h1 {{ margin-top: 0; color: #1a202c; display: flex; align-items: center; gap: 10px; }}
        .stats {{ display: flex; gap: 20px; margin: 25px 0; }}
        .stat-card {{ flex: 1; padding: 15px 20px; border-radius: 6px; background: #edf2f7; }}
        .stat-card.critical {{ background: #fed7d7; color: #9b2c2c; }}
        .stat-card.warning {{ background: #feebc8; color: #9c4221; }}
        .stat-card.total {{ background: #e2e8f0; color: #2d3748; }}
        .stat-number {{ font-size: 28px; font-weight: 700; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
        th {{ background: #f8fafc; font-weight: 600; color: #4a5568; }}
        tr:hover {{ background: #f8fafc; }}
        code {{ background: #edf2f7; padding: 2px 5px; border-radius: 3px; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🛡️ MLGuard Audit Report</h1>
        <div class="stats">
            <div class="stat-card total"><div>Всего находок</div><div class="stat-number">{total}</div></div>
            <div class="stat-card critical"><div>Критические</div><div class="stat-number">{critical}</div></div>
            <div class="stat-card warning"><div>Предупреждения</div><div class="stat-number">{warning}</div></div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Уровень</th>
                    <th>Проверка</th>
                    <th>Колонка</th>
                    <th>Значение</th>
                    <th>Порог</th>
                    <th>Описание</th>
                </tr>
            </thead>
            <tbody>
                {tbody}
            </tbody>
        </table>
    </div>
</body>
</html>"""


def _summary_line(findings: list[dict], markdown: bool = False) -> str:
    critical = sum(1 for f in findings if str(f.get("severity", "")).upper() == "CRITICAL")
    warning = sum(1 for f in findings if str(f.get("severity", "")).upper() == "WARNING")
    text = f"Итого: {len(findings)} findings — {critical} critical, {warning} warning."
    return f"**{text}**" if markdown else click.style(text, bold=True)


FORMATTERS = {
    "table": format_table,
    "json": format_json,
    "markdown": format_markdown,
    "html": format_html,
}

