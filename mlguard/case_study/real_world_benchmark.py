"""
Бенчмарк и аудит MLGuard на реальных открытых датасетах:
1. Kaggle Titanic (891 строк) — классическая бинарная классификация (выживание).
2. IBM Telco Customer Churn (7043 строки) — классический отток клиентов.
3. UCI Bike Sharing Daily (728 строк) — регрессия и временной ряд спроса на аренду.
"""
from pathlib import Path
from mlguard.core.loaders import load_data
from mlguard.core.runner import run_audit


def audit_real_world_datasets():
    datasets_dir = Path(__file__).resolve().parent.parent / "datasets" / "real_world"

    print("=" * 80)
    print(" АУДИТ MLGUARD НА РЕАЛЬНЫХ ОТКРЫТЫХ ДАТАСЕТАХ (KAGGLE / UCI / IBM)")
    print("=" * 80)

    # 1. Titanic
    titanic_path = datasets_dir / "titanic.csv"
    if titanic_path.exists():
        df_titanic = load_data(titanic_path)
        findings = run_audit(df_titanic, target_col="Survived")
        print(f"\n[1] Kaggle Titanic (строк: {len(df_titanic)}, колонок: {len(df_titanic.columns)})")
        print(f"    Целевая переменная: Survived")
        print(f"    Найдено проблем: {len(findings)}")
        for f in findings:
            print(f"    - [{f['severity']:<8}] {f['check']:<20} | {f['column']:<15} | {f['message']}")

    # 2. Telco Churn
    telco_path = datasets_dir / "telco_churn.csv"
    if telco_path.exists():
        df_telco = load_data(telco_path)
        findings = run_audit(df_telco, target_col="Churn")
        print(f"\n[2] IBM Telco Customer Churn (строк: {len(df_telco)}, колонок: {len(df_telco.columns)})")
        print(f"    Целевая переменная: Churn (Yes/No)")
        print(f"    Найдено проблем: {len(findings)}")
        for f in findings:
            print(f"    - [{f['severity']:<8}] {f['check']:<20} | {f['column']:<15} | {f['message']}")

    # 3. Bike Sharing Daily
    bike_path = datasets_dir / "bike_sharing.csv"
    if bike_path.exists():
        df_bike = load_data(bike_path)
        findings = run_audit(df_bike, target_col="cnt", time_col="dteday")
        print(f"\n[3] UCI Bike Sharing (строк: {len(df_bike)}, колонок: {len(df_bike.columns)})")
        print(f"    Целевая переменная: cnt (регрессия), время: dteday")
        print(f"    Найдено проблем: {len(findings)}")
        for f in findings:
            print(f"    - [{f['severity']:<8}] {f['check']:<20} | {f['column']:<15} | {f['message']}")

    print("\n" + "=" * 80)
    print(" АУДИТ ЗАВЕРШЁН")
    print("=" * 80)


if __name__ == "__main__":
    audit_real_world_datasets()
