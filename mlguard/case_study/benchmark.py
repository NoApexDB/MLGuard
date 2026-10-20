import json
from pathlib import Path
import numpy as np
import pandas as pd
from mlguard.core.runner import run_audit


def generate_clean_dataset(seed: int) -> tuple[pd.DataFrame, dict[str, bool]]:
    np.random.seed(seed)
    n = 500
    dates = pd.date_range(start="2026-01-01", periods=n, freq="D")
    
    target = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    feat1 = np.random.normal(50, 10, size=n)
    feat2 = np.random.normal(100, 15, size=n)
    feat3 = np.random.exponential(scale=5.0, size=n)
    
    df = pd.DataFrame({
        "date": dates,
        "feat1": np.round(feat1, 2),
        "feat2": np.round(feat2, 2),
        "feat3": np.round(feat3, 2),
        "target": target
    })
    
    ground_truth = {
        "leakage_detector": False,
        "small_sample_guard": False,
        "churn_blindness": False,
        "metric_confusion": False,
        "seasonality_sentiment": False,
        "hidden_missing": False,
        "temporal_leakage": False,
    }
    return df, ground_truth


def generate_dirty_dataset(seed: int, pitfall_type: str) -> tuple[pd.DataFrame, dict[str, bool]]:
    np.random.seed(seed)
    n = 500
    dates = pd.date_range(start="2026-01-01", periods=n, freq="D")
    target = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    
    feat1 = np.random.normal(50, 10, size=n)
    feat2 = np.random.normal(100, 15, size=n)
    
    ground_truth = {
        "leakage_detector": False,
        "small_sample_guard": False,
        "churn_blindness": False,
        "metric_confusion": False,
        "seasonality_sentiment": False,
        "hidden_missing": False,
        "temporal_leakage": False,
    }
    
    if pitfall_type == "leakage":
        ground_truth["leakage_detector"] = True
        ground_truth["churn_blindness"] = True
        feat_leak = target * 100.0 + np.random.normal(0, 0.1, size=n)
        df = pd.DataFrame({
            "date": dates,
            "feat1": feat1,
            "feat_leak": feat_leak,
            "target": target
        })
        
    elif pitfall_type == "small_sample":
        ground_truth["small_sample_guard"] = True
        n_small = 60
        df = pd.DataFrame({
            "date": dates[:n_small],
            "feat1": feat1[:n_small],
            "target": target[:n_small]
        })
        
    elif pitfall_type == "churn_blindness":
        ground_truth["churn_blindness"] = True
        ground_truth["leakage_detector"] = True
        salary = np.where(target == 1, np.random.normal(40000, 5000, n), np.random.normal(120000, 15000, n))
        df = pd.DataFrame({
            "date": dates,
            "salary": salary,
            "target": target
        })
        
    elif pitfall_type == "metric_confusion":
        ground_truth["metric_confusion"] = True
        vol = np.linspace(100, 500, n) + np.random.normal(0, 5, n)
        margin = np.linspace(0.40, 0.05, n) + np.random.normal(0, 0.01, n)
        df = pd.DataFrame({
            "date": dates,
            "order_volume": vol,
            "net_margin": margin,
            "target": target
        })
        
    elif pitfall_type == "seasonality":
        ground_truth["seasonality_sentiment"] = True
        t = np.arange(n)
        seasonal = np.sin(2 * np.pi * t / 7.0) * 40.0
        revenue = np.linspace(10, 20, n) + seasonal
        df = pd.DataFrame({
            "date": dates,
            "revenue": revenue,
            "target": target
        })

    elif pitfall_type == "hidden_missing":
        ground_truth["hidden_missing"] = True
        str_feat = np.array(["valid_value"] * n, dtype=object)
        mask = np.random.rand(n) < 0.15
        str_feat[mask] = "?"
        df = pd.DataFrame({
            "date": dates,
            "category": str_feat,
            "feat1": feat1,
            "target": target
        })

    elif pitfall_type == "temporal_leakage":
        ground_truth["temporal_leakage"] = True
        # Прямая временная утечка: колонка события из будущего
        future_dates = dates + pd.Timedelta(days=10)
        df = pd.DataFrame({
            "date": dates,
            "future_event_date": future_dates,
            "feat1": feat1,
            "target": target
        })

    else:
        df, ground_truth = generate_clean_dataset(seed)

    return df, ground_truth


def evaluate_dataset_split(start_seed: int, num_clean: int = 50, num_dirty_per_type: int = 10) -> dict:
    pitfall_types = [
        "leakage", 
        "small_sample", 
        "churn_blindness", 
        "metric_confusion", 
        "seasonality",
        "hidden_missing",
        "temporal_leakage"
    ]
    
    datasets = []
    seed_counter = start_seed
    
    for _ in range(num_clean):
        df, gt = generate_clean_dataset(seed_counter)
        datasets.append((df, gt))
        seed_counter += 1
        
    for ptype in pitfall_types:
        for _ in range(num_dirty_per_type):
            df, gt = generate_dirty_dataset(seed_counter, ptype)
            datasets.append((df, gt))
            seed_counter += 1

    checks = [
        "leakage_detector", 
        "small_sample_guard", 
        "churn_blindness", 
        "metric_confusion", 
        "seasonality_sentiment",
        "hidden_missing",
        "temporal_leakage"
    ]
    metrics = {c: {"TP": 0, "FP": 0, "TN": 0, "FN": 0} for c in checks}
    
    for df, gt in datasets:
        findings = run_audit(df, target_col="target", time_col="date")
        detected_checks = {f["check"] for f in findings}
        
        for c in checks:
            actual = gt.get(c, False)
            predicted = (c in detected_checks)
            
            if actual and predicted:
                metrics[c]["TP"] += 1
            elif not actual and predicted:
                metrics[c]["FP"] += 1
            elif actual and not predicted:
                metrics[c]["FN"] += 1
            else:
                metrics[c]["TN"] += 1
                
    results = {}
    for c, counts in metrics.items():
        tp = counts["TP"]
        fp = counts["FP"]
        fn = counts["FN"]
        tn = counts["TN"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[c] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4)
        }
        
    return results


def main():
    # Held-Out Out-of-Sample Test Set evaluation (все 7 модулей проверок)
    held_out_results = evaluate_dataset_split(start_seed=5000, num_clean=50, num_dirty_per_type=10)
    
    out_dir = Path(__file__).parent
    res_path = out_dir / "benchmark_results.json"
    res_path.write_text(json.dumps(held_out_results, indent=2), encoding="utf-8")
    
    print("=== HELD-OUT TEST SET EVALUATION METRICS (Все 7 проверок) ===")
    print(f"{'Check Name':<25} | {'TP':<4} | {'FP':<4} | {'FN':<4} | {'TN':<4} | {'Precision':<9} | {'Recall':<8} | {'F1':<6}")
    print("-" * 85)
    for check_name, m in held_out_results.items():
        print(f"{check_name:<25} | {m['TP']:<4} | {m['FP']:<4} | {m['FN']:<4} | {m['TN']:<4} | {m['Precision']:<9.4f} | {m['Recall']:<8.4f} | {m['F1']:<6.4f}")


if __name__ == "__main__":
    main()
