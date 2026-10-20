"""
Экспериментальное исследование прикладного эффекта аудита MLGuard:
1. Замер деградации моделей машинного обучения в продакшене (Generalization Gap).
2. Бенчмарк вычислительной производительности и масштабируемости (Rows/sec).
3. Сравнение производительности векторизованного Bootstrap против итеративного.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from mlguard.core.runner import run_audit


def run_generalization_experiment():
    print("=" * 80)
    print(" ЭКСПЕРИМЕНТ 1: ПРЕДОТВРАЩЕНИЕ ДЕГРАДАЦИИ ML-МОДЕЛИ В ПРОДАКШЕНЕ")
    print("=" * 80)
    np.random.seed(42)
    n_train = 1000
    n_test = 500

    # 1. Генерация реального обучающего распределения (банковский скоринг/отток)
    age_tr = np.random.normal(42, 10, n_train)
    credit_tr = np.random.normal(640, 60, n_train)
    balance_tr = np.random.normal(55000, 20000, n_train)
    tenure_tr = np.random.exponential(24, n_train)

    logits_tr = (
        -2.0
        + 0.04 * (50 - age_tr)
        + 0.015 * (650 - credit_tr)
        + 0.00003 * (60000 - balance_tr)
    )
    prob_tr = 1.0 / (1.0 + np.exp(-logits_tr))
    y_train = (np.random.rand(n_train) < prob_tr).astype(int)

    # Дефект 1: Уникальный ID клиента (риск тривиального запоминания)
    id_tr = np.arange(10000, 10000 + n_train)

    # Дефект 2: Скрытая утечка таргета (статус закрытия счёта, выставленный биллингом после оттока)
    closure_leak_tr = y_train.copy()
    flip_tr = np.random.rand(n_train) < 0.005
    closure_leak_tr[flip_tr] = 1 - closure_leak_tr[flip_tr]

    train_df = pd.DataFrame({
        "customer_id": id_tr,
        "age": np.round(age_tr, 1),
        "credit_score": np.round(credit_tr, 0),
        "balance": np.round(balance_tr, 2),
        "tenure_months": np.round(tenure_tr, 1),
        "closure_event_flag": closure_leak_tr,
        "churn": y_train,
    })

    # 2. Продакшен / Held-out тест: Новые клиенты, утечка недоступна (белый шум/не заполнена)
    age_te = np.random.normal(42, 10, n_test)
    credit_te = np.random.normal(640, 60, n_test)
    balance_te = np.random.normal(55000, 20000, n_test)
    tenure_te = np.random.exponential(24, n_test)

    logits_te = (
        -2.0
        + 0.04 * (50 - age_te)
        + 0.015 * (650 - credit_te)
        + 0.00003 * (60000 - balance_te)
    )
    prob_te = 1.0 / (1.0 + np.exp(-logits_te))
    y_test = (np.random.rand(n_test) < prob_te).astype(int)

    id_te = np.arange(50000, 50000 + n_test)  # Новые ID в продакшене
    closure_leak_te = np.random.choice([0, 1], size=n_test)  # В проде признак не заполнен вовремя

    test_df = pd.DataFrame({
        "customer_id": id_te,
        "age": np.round(age_te, 1),
        "credit_score": np.round(credit_te, 0),
        "balance": np.round(balance_te, 2),
        "tenure_months": np.round(tenure_te, 1),
        "closure_event_flag": closure_leak_te,
        "churn": y_test,
    })

    # Сценарий А: Обучение БЕЗ аудита MLGuard (наивный Data Science)
    feature_cols_all = [c for c in train_df.columns if c != "churn"]
    clf_naive = GradientBoostingClassifier(max_depth=4, random_state=42)
    clf_naive.fit(train_df[feature_cols_all], y_train)

    train_auc_naive = roc_auc_score(y_train, clf_naive.predict_proba(train_df[feature_cols_all])[:, 1])
    test_auc_naive = roc_auc_score(y_test, clf_naive.predict_proba(test_df[feature_cols_all])[:, 1])
    gap_naive = train_auc_naive - test_auc_naive

    # Сценарий Б: Аудит через MLGuard
    t0 = time.perf_counter()
    findings = run_audit(train_df, target_col="churn")
    audit_time_ms = (time.perf_counter() - t0) * 1000

    print(f"\n[MLGuard] Время полного аудита ({len(train_df)} строк): {audit_time_ms:.1f} мс")
    print(f"[MLGuard] Выявлено находок: {len(findings)}")
    rejected_columns = set()
    for f in findings:
        print(f"  - [{f['severity']:<8}] {f['check']:<20} | {f['column']:<20} | {f['message']}")
        if f["check"] == "leakage_detector":
            rejected_columns.add(f["column"])

    # Санитизация данных: удаление скомпрометированных колонок
    clean_features = [c for c in feature_cols_all if c not in rejected_columns]
    clf_sanitized = GradientBoostingClassifier(max_depth=3, n_estimators=60, random_state=42)
    clf_sanitized.fit(train_df[clean_features], y_train)

    train_auc_clean = roc_auc_score(y_train, clf_sanitized.predict_proba(train_df[clean_features])[:, 1])
    test_auc_clean = roc_auc_score(y_test, clf_sanitized.predict_proba(test_df[clean_features])[:, 1])
    gap_clean = train_auc_clean - test_auc_clean

    print("\n" + "-" * 80)
    print(" СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК ML-МОДЕЛИ (TRAIN vs PROD TEST)")
    print("-" * 80)
    print(f"{'Конфигурация':<35} | {'Train ROC-AUC':<14} | {'Prod Test ROC-AUC':<18} | {'Generalization Gap':<18}")
    print("-" * 80)
    print(f"{'Без аудита (Naive Pipeline)':<35} | {train_auc_naive:<14.4f} | {test_auc_naive:<18.4f} | {gap_naive:<18.4f}")
    print(f"{'После аудита MLGuard (Sanitized)':<35} | {train_auc_clean:<14.4f} | {test_auc_clean:<18.4f} | {gap_clean:<18.4f}")
    print("-" * 80)
    print(f"Итог: Ликвидирована ложная уверенность модели, разрыв Train/Prod сокращен на {gap_naive - gap_clean:.4f}.")


def run_performance_benchmark():
    print("\n" + "=" * 80)
    print(" ЭКСПЕРИМЕНТ 2: ВЫЧИСЛИТЕЛЬНАЯ ПРОИЗВОДИТЕЛЬНОСТЬ И МАСШТАБИРУЕМОСТЬ")
    print("=" * 80)

    sizes = [1000, 10000, 50000]
    print(f"{'Объём строк':<15} | {'Колонок':<10} | {'Время аудита (мс)':<20} | {'Скорость (строк/сек)':<22}")
    print("-" * 75)

    for n in sizes:
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=n, freq="min")
        df = pd.DataFrame({
            "timestamp": dates,
            "feat_1": np.random.normal(0, 1, n),
            "feat_2": np.random.normal(50, 10, n),
            "feat_3": np.random.uniform(10, 100, n),
            "target": np.random.choice([0, 1], size=n),
        })

        t0 = time.perf_counter()
        _ = run_audit(df, target_col="target", time_col="timestamp")
        elapsed = time.perf_counter() - t0
        throughput = n / elapsed if elapsed > 0 else 0

        print(f"{n:<15} | {len(df.columns):<10} | {elapsed * 1000:<20.2f} | {throughput:<22.0f}")


def run_bootstrap_vectorization_benchmark():
    print("\n" + "=" * 80)
    print(" ЭКСПЕРИМЕНТ 3: БЕНЧМАРК ВЕКТОРИЗАЦИИ BOOTSTRAP (1000 РЕСЭМПЛОВ)")
    print("=" * 80)

    np.random.seed(42)
    data = np.random.normal(100, 15, 2000)
    n = len(data)

    # 1. Цикл Python
    t0 = time.perf_counter()
    means_loop = []
    for _ in range(1000):
        s = np.random.choice(data, size=n, replace=True)
        means_loop.append(s.mean())
    time_loop_ms = (time.perf_counter() - t0) * 1000

    # 2. Векторизованный NumPy
    t0 = time.perf_counter()
    samples = np.random.choice(data, size=(1000, n), replace=True)
    means_vec = samples.mean(axis=1)
    time_vec_ms = (time.perf_counter() - t0) * 1000

    speedup = time_loop_ms / time_vec_ms if time_vec_ms > 0 else 1.0

    print(f"Итеративный цикл (Python for-loop):  {time_loop_ms:.2f} мс")
    print(f"Векторизованный NumPy (Single Pass): {time_vec_ms:.2f} мс")
    print(f"Фактическое ускорение:                {speedup:.1f}x быстрее")
    print("=" * 80)


if __name__ == "__main__":
    run_generalization_experiment()
    run_performance_benchmark()
    run_bootstrap_vectorization_benchmark()
