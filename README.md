# MLGuard

> Линтер для предиктивной аналитики бизнеса — находит методологические и статистические ловушки в данных без вреда продакшену.

[![CI](https://github.com/chernov-ramil-machinelearning/MLGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/chernov-ramil-machinelearning/MLGuard/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)

---

## Что делает проект

MLGuard — это CLI-инструмент и Python-библиотека для проверки табличных данных (`.csv`, `.parquet`, `.json`, `.jsonl`, `.xlsx`) на типовые математические и методологические ошибки, из-за которых предиктивная аналитика и ML-модели дают ложную уверенность в результатах.

В библиотеке реализовано 7 автономных проверок:

| Проверка | Что ищет | Математический аппарат |
|---|---|---|
| `leakage_detector` | Утечка таргета, мультиколлинеарность фичей и ID-колонки | Корреляции Пирсона и Спирмена, анализ уникальности идентификаторов |
| `small_sample_guard` | Малый объём данных, дисбаланс классов и нестабильность оценок | Scale-invariant Bootstrap ДИ среднего (1000 ресэмплов) |
| `churn_blindness` | Survivorship Bias — сдвиг распределения между группами | Тест Колмогорова-Смирнова (`ks_2samp`) для чисел, $\chi^2$ и $V$ Крамера для категорий |
| `metric_confusion` | Расхождение метрики привлечения и метрики здоровья бизнеса | Тест Манна-Кендалла на тренд (`kendalltau`) с порогом силы тренда и ранжированием |
| `seasonality_sentiment` | Сезонность, маскирующая реальный тренд в данных | STL-декомпозиция ряда на Trend, Seasonal, Residual |
| `hidden_missing` | Маскированные пропуски данных | Детекция псевдо-пропусков (`?`, `null`, `none`, `-999`, `inf`) |
| `temporal_leakage` | Временное заглядывание в будущее (Lookahead) | Сверка временных меток признаков и корреляция с будущим таргетом $y[t+1]$ |

---


## Валидация на синтетических данных

Оценка качества проведена на синтетических датасетах (клинические/чистые + датасеты с контролируемыми инъекциями дефектов), сгенерированных на независимых random seed. Ниже — результат по held-out прогону (100 датасетов):

| Проверка | Precision | Recall | F1 |
|---|---|---|---|
| `leakage_detector` | 1.00 | 1.00 | 1.00 |
| `small_sample_guard` | 1.00 | 1.00 | 1.00 |
| `churn_blindness` | 1.00 | 1.00 | 1.00 |
| `metric_confusion` | 1.00 | 1.00 | 1.00 |
| `seasonality_sentiment` | 0.83 | 1.00 | 0.91 |

---

## Валидация на реальных открытых датасетах (Kaggle / IBM / UCI)

Для проверки на реальных данных в проект включены 3 эталонных публичных датасета (`python -m mlguard.case_study.real_world_benchmark`):

1. **Kaggle Titanic (891 строк, 12 колонок)**:
   - `leakage_detector`: обнаружен первичный ключ `PassengerId` (100% уникальность).
   - `churn_blindness`: обнаружен сдвиг по выживаемости для категорий `Sex` ($V$ Крамера = 0.54, $p = 0.0000$) и билетов `Ticket`, а также сдвиг распределения числовых фичей `Pclass` ($D = 0.33$) и `Fare` ($D = 0.30$).
2. **IBM Telco Customer Churn (7043 строки, 21 колонка)**:
   - `leakage_detector`: обнаружен ID клиента `customerID`.
   - `churn_blindness`: подтвержден численный сдвиг времени жизни `tenure` ($D = 0.36$), а также 7 категориальных сдвигов: `Contract` ($V = 0.41$), `OnlineSecurity` ($V = 0.35$), `TechSupport` ($V = 0.34$), `InternetService` ($V = 0.32$).
3. **UCI Bike Sharing Daily (728 строк, 18 колонок)**:
   - `leakage_detector`: поймана мультиколлинеарность признаков `instant vs days_since_2011` ($r = 1.0000$) и `temp vs atemp` ($r = 0.9925$).
   - `seasonality_sentiment`: поймана выраженная недельная сезонность спроса `casual` (38.8% общей дисперсии).


---

## Быстрый старт

### Установка

```bash
git clone https://github.com/chernov-ramil-machinelearning/MLGuard.git
cd MLGuard
pip install -e .
```

> Требуется Python 3.11+.

### Запуск аудита из CLI

```bash
mlguard audit data.csv --target churn --time-col created_at
```

Пример вывода в консоль:

```text
[CRITICAL] leakage_detector     leaky_corr_feature        Feature 'leaky_corr_feature' has a correlation of 0.9961 with target 'target'.
[WARNING ] small_sample_guard   all                       Sample size (100) is below the minimum threshold (200).
[CRITICAL] churn_blindness      salary                    Distribution shift detected for 'salary' between active and churned clients (p-value = 0.0000).

Итого: 3 findings — 2 critical, 1 warning.
```

### Основные команды CLI

```bash
mlguard checks list                                  # список доступных проверок
mlguard audit data.csv --format html -o report.html  # интерактивный HTML-отчёт
mlguard audit data.csv --format json -o report.json  # отчёт в формате JSON
mlguard audit data.csv --format markdown             # отчёт в формате Markdown
mlguard audit data.csv --only leakage_detector,churn_blindness
mlguard audit data.csv --fail-on WARNING             # gate для CI/CD: exit code 1 при находках >= WARNING
```

### Использование как Python-библиотека (в скриптах и Jupyter)

```python
import mlguard
import pandas as pd

# 1. Аудит напрямую из файла (csv, parquet, json, jsonl, xlsx)
report = mlguard.audit("data.csv", target_col="churn", time_col="created_at")

# 2. Аудит из pandas DataFrame
df = pd.read_csv("train.csv")
report = mlguard.audit(df, target_col="Survived")

# Вывод в терминал или Jupyter (автоматический HTML-рендер)
print(report)

# Экспорт результатов
df_findings = report.to_dataframe()
html_code = report.to_html()
json_str = report.to_json()

if report.has_critical():
    print("Внимание! Обнаружены критические ошибки методологии.")
```

---

## Автор и поддержка

Проект разработан для предиктивной аналитики и валидации данных перед обучением ML-моделей: 7 статистических проверок, CLI, Python API, HTML-отчёты, интеграция с CI/CD.


* **Автор**: [Рамиль Чернов](https://github.com/chernov-ramil-machinelearning)
* **Telegram**: [@NoApexDB](https://t.me/NoApexDB)

## Лицензия

MIT — см. файл [LICENSE](LICENSE).
