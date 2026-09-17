#!/usr/bin/env python3
"""
HackerRank Orchestrate: Buy or Wait?
Evaluation and Accuracy Benchmark Script.

Evaluates the predictions against ground-truth labels in dataset/sample_requests.csv.
Computes:
- Classification Accuracy, Precision, Recall, F1 for affordability_status
- Classification Accuracy, Precision, Recall, F1 for recommended_payment_method
- Mean Absolute Error (MAE) and tolerance accuracy for amount_safe_to_pay
- Exact match rate for earliest_date_for_full_payment
- Exact match rate for spending_changes_needed
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

def run_evaluation(predictions_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict[str, Any]:
    # Match on request_id
    merged = ground_truth_df.merge(predictions_df, on="request_id", suffixes=("_true", "_pred"))
    total_samples = len(merged)
    if total_samples == 0:
        print("No matching request_ids found between predictions and ground truth.")
        return {}

    print(f"\n======================================================================")
    print(f"       BUY OR WAIT? MODEL EVALUATION & BENCHMARK REPORT               ")
    print(f"======================================================================")
    print(f"Total Evaluated Benchmark Samples: {total_samples}\n")

    # 1. Affordability Status Accuracy & Precision
    status_true = merged["affordability_status_true"].astype(str)
    status_pred = merged["affordability_status_pred"].astype(str)
    status_acc = (status_true == status_pred).mean()

    # Compute per-class precision and recall
    classes = sorted(list(set(status_true.unique()) | set(status_pred.unique())))
    print("--- 1. Affordability Status Metrics ---")
    print(f"Overall Accuracy: {status_acc * 100:.2f}%\n")
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'Support':<8}")
    print("-" * 57)
    
    precisions, recalls = [], []
    for cls in classes:
        tp = ((status_pred == cls) & (status_true == cls)).sum()
        fp = ((status_pred == cls) & (status_true != cls)).sum()
        fn = ((status_pred != cls) & (status_true == cls)).sum()
        support = (status_true == cls).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        print(f"{cls:<25} {prec * 100:>10.2f}% {rec * 100:>10.2f}% {support:>8}")

    macro_prec_status = np.mean(precisions)
    macro_rec_status = np.mean(recalls)
    f1_status = (2 * macro_prec_status * macro_rec_status / (macro_prec_status + macro_rec_status)) if (macro_prec_status + macro_rec_status) > 0 else 0.0
    print(f"\nMacro Precision: {macro_prec_status * 100:.2f}% | Macro Recall: {macro_rec_status * 100:.2f}% | Macro F1: {f1_status * 100:.2f}%\n")

    # 2. Recommended Payment Method Metrics
    method_true = merged["recommended_payment_method_true"].astype(str)
    method_pred = merged["recommended_payment_method_pred"].astype(str)
    method_acc = (method_true == method_pred).mean()

    m_classes = sorted(list(set(method_true.unique()) | set(method_pred.unique())))
    print("--- 2. Recommended Payment Method Metrics ---")
    print(f"Overall Accuracy: {method_acc * 100:.2f}%\n")
    print(f"{'Method':<25} {'Precision':<12} {'Recall':<12} {'Support':<8}")
    print("-" * 57)
    
    m_precs, m_recs = [], []
    for m in m_classes:
        tp = ((method_pred == m) & (method_true == m)).sum()
        fp = ((method_pred == m) & (method_true != m)).sum()
        fn = ((method_pred != m) & (method_true == m)).sum()
        support = (method_true == m).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        m_precs.append(prec)
        m_recs.append(rec)
        print(f"{m:<25} {prec * 100:>10.2f}% {rec * 100:>10.2f}% {support:>8}")

    macro_prec_method = np.mean(m_precs)
    macro_rec_method = np.mean(m_recs)
    f1_method = (2 * macro_prec_method * macro_rec_method / (macro_prec_method + macro_rec_method)) if (macro_prec_method + macro_rec_method) > 0 else 0.0
    print(f"\nMacro Precision: {macro_prec_method * 100:.2f}% | Macro Recall: {macro_rec_method * 100:.2f}% | Macro F1: {f1_method * 100:.2f}%\n")

    # 3. Numeric Amount Safe to Pay (MAE & Tolerance)
    amt_true = pd.to_numeric(merged["amount_safe_to_pay_true"], errors="coerce").fillna(0.0)
    amt_pred = pd.to_numeric(merged["amount_safe_to_pay_pred"], errors="coerce").fillna(0.0)
    abs_errors = (amt_true - amt_pred).abs()
    mae = abs_errors.mean()
    within_5_pct = ((abs_errors / amt_true.replace(0, 1)) <= 0.05).mean()
    within_exact = (abs_errors < 1.0).mean()

    print("--- 3. Numerical Precision: Amount Safe to Pay ---")
    print(f"Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"Exact Matches (< 1.0 diff): {within_exact * 100:.2f}%")
    print(f"Within 5% Tolerance:        {within_5_pct * 100:.2f}%\n")

    # 4. Dates and Spending Changes Match Rates
    date_true = merged["earliest_date_for_full_payment_true"].fillna("").astype(str).str.strip()
    date_pred = merged["earliest_date_for_full_payment_pred"].fillna("").astype(str).str.strip()
    date_match = (date_true == date_pred).mean()

    changes_true = merged["spending_changes_needed_true"].fillna("none").astype(str).str.strip()
    changes_pred = merged["spending_changes_needed_pred"].fillna("none").astype(str).str.strip()
    changes_match = (changes_true == changes_pred).mean()

    print("--- 4. Temporal & Constraint Adherence ---")
    print(f"Earliest Safe Full Date Accuracy: {date_match * 100:.2f}%")
    print(f"Spending Changes Needed Match:    {changes_match * 100:.2f}%\n")

    # Overall Combined Composite Score
    composite_score = np.mean([status_acc, method_acc, within_5_pct, date_match, changes_match]) * 100
    print(f"======================================================================")
    print(f"OVERALL COMPOSITE DECISION SCORE: {composite_score:.2f} / 100.0")
    print(f"======================================================================\n")

    return {
        "status_accuracy": status_acc,
        "status_macro_precision": macro_prec_status,
        "status_macro_recall": macro_rec_status,
        "status_f1": f1_status,
        "method_accuracy": method_acc,
        "method_macro_precision": macro_prec_method,
        "method_macro_recall": macro_rec_method,
        "method_f1": f1_method,
        "amount_safe_mae": mae,
        "amount_safe_exact_pct": within_exact,
        "earliest_date_accuracy": date_match,
        "spending_changes_accuracy": changes_match,
        "composite_score": composite_score,
    }

def main():
    repo_root = Path(__file__).resolve().parent
    if (repo_root / "dataset").exists():
        base_dir = repo_root
    else:
        base_dir = repo_root.parent

    # Ground truth samples
    samples_path = base_dir / "dataset" / "sample_requests.csv"
    if not samples_path.exists():
        print(f"Ground truth dataset not found at {samples_path}")
        sys.exit(1)

    gt_df = pd.read_csv(samples_path)

    # Predictions can be evaluated either from a benchmark run or by executing sample evaluations
    pred_path = base_dir / "dataset" / "sample_predictions.csv"
    if not pred_path.exists():
        pred_path = base_dir / "output.csv"

    if not pred_path.exists() or len(pd.read_csv(pred_path).merge(gt_df, on="request_id")) == 0:
        print("Running fast benchmark evaluation across all 25 sample requests...")
        # Import deterministic and pipeline evaluator
        sys.path.insert(0, str(base_dir / "code"))
        from main import normalize_foreign_events, prepare_request_context, deterministic_financial_check

        profiles_df = pd.read_csv(base_dir / "dataset" / "financial_profiles.csv")
        events_df = pd.read_csv(base_dir / "dataset" / "financial_events.csv")
        rates_df = pd.read_csv(base_dir / "dataset" / "exchange_rates.csv")
        options_df = pd.read_csv(base_dir / "dataset" / "request_payment_options.csv")
        messages_df = pd.read_csv(base_dir / "dataset" / "messages.csv")
        
        # Load image cache
        cache_file = base_dir / "code" / "image_amount_cache.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                img_cache = json.load(f)
            for idx, r in events_df[events_df["amount"].isna()].iterrows():
                eid = str(r["event_id"])
                if eid in img_cache:
                    events_df.loc[idx, "amount"] = img_cache[eid]["amount"]

        norm_events = normalize_foreign_events(events_df, profiles_df, rates_df)
        profile_lookup = {str(r["user_id"]): r for _, r in profiles_df.iterrows()}

        pred_rows = []
        for _, r_row in gt_df.iterrows():
            uid = str(r_row["user_id"])
            prof_row = profile_lookup.get(uid)
            ctx = prepare_request_context(r_row, prof_row, norm_events, options_df, messages_df)
            res = deterministic_financial_check(ctx)
            pred_rows.append(res)

        preds_df = pd.DataFrame(pred_rows)
        preds_df.to_csv(base_dir / "dataset" / "sample_predictions.csv", index=False)
        print("Sample benchmark predictions saved to dataset/sample_predictions.csv")
    else:
        preds_df = pd.read_csv(pred_path)

    run_evaluation(preds_df, gt_df)

if __name__ == "__main__":
    main()
