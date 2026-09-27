#!/usr/bin/env python3
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "block_score",
    "name_token_sort", "name_partial", "name_ratio",
    "name_jaccard", "name_len_ratio", "name_prefix_len",
    "addr_token_sort", "addr_partial", "addr_ratio",
    "addr_jaccard", "addr_len_ratio",
    "combined_token_sort",
    "s1_name_missing", "s1_addr_missing",
    "cand_name_missing", "cand_addr_missing", "any_missing",
    "source_flag", "country_match",
]

def f_beta(p, r, beta=0.5):
    if p + r == 0: return 0.0
    return (1 + beta**2) * p * r / (beta**2 * p + r)

def compute_macro_f05(true_groups, pred_groups):
    scores = []
    all_ids = set(true_groups.keys()) | set(pred_groups.keys())
    for sid in all_ids:
        ts = true_groups.get(sid, set())
        ps = pred_groups.get(sid, set())
        if not ts and not ps:
            scores.append(1.0)
        elif not ps or not ts:
            scores.append(0.0)
        else:
            tp = len(ts & ps)
            prec = tp / len(ps) if ps else 0
            rec = tp / len(ts) if ts else 0
            scores.append(f_beta(prec, rec))
    return np.mean(scores) if scores else 0.0

def run_training(requested_device="cpu"):
    print(f"\n{'='*70}\n  STAGE 3: TRAINING\n{'='*70}")
    
    feat_path = CHECKPOINT_DIR / "features_train.parquet"
    print(f"  Loading features from {feat_path}...")
    features_df = pd.read_parquet(str(feat_path))
    
    gt_path = DATASET_DIR / "train" / "train_ground_truth.tsv"
    gt_df = pd.read_csv(str(gt_path), sep="\t", dtype=str, keep_default_na=False)
    gt_df.columns = gt_df.columns.str.strip()

    true_pairs = set()
    true_groups = {}
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]
        if matched and str(matched).strip():
            mids = set(m.strip() for m in str(matched).split(","))
            true_groups[s1_id] = mids
            for mid in mids: true_pairs.add((s1_id, mid))
        else:
            true_groups[s1_id] = set()

    s1_ids = features_df["source1_entity_id"].values
    cand_ids = features_df["candidate_id"].values
    labels = np.array([1 if (s1_ids[i], cand_ids[i]) in true_pairs else 0
                       for i in range(len(features_df))], dtype=np.int32)
    features_df["label"] = labels

    population_path = CHECKPOINT_DIR / "train_source1_population.parquet"
    if population_path.exists():
        unique_s1 = pd.read_parquet(population_path)["source1_entity_id"].values
    else:
        # Backward-compatible fallback for checkpoints made by older blocker versions.
        unique_s1 = features_df["source1_entity_id"].unique()
    s1_train, s1_val = train_test_split(unique_s1, test_size=0.2, random_state=42)
    s1_val_set = set(s1_val)

    train_mask = ~features_df["source1_entity_id"].isin(s1_val_set)
    val_mask = features_df["source1_entity_id"].isin(s1_val_set)

    X_tr = features_df.loc[train_mask, FEATURE_COLS].values.astype(np.float32)
    y_tr = features_df.loc[train_mask, "label"].values
    X_va = features_df.loc[val_mask, FEATURE_COLS].values.astype(np.float32)
    y_va = features_df.loc[val_mask, "label"].values

    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    
    print("  Training XGBoost...")
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_COLS)
    dval = xgb.DMatrix(X_va, label=y_va, feature_names=FEATURE_COLS)

    device = "cpu"
    if requested_device in ("auto", "cuda"):
        try:
            cuda_value = xgb.build_info().get("USE_CUDA", False)
            cuda_build = str(cuda_value).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            cuda_build = False
        if requested_device == "cuda" or cuda_build:
            device = "cuda"
        elif requested_device == "auto":
            print("  CUDA support not found; using CPU.")
    print(f"  XGBoost training device: {device}")
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "max_depth": 8, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 5, "scale_pos_weight": spw,
        "tree_method": "hist", "device": device, "nthread": 8, "seed": 42,
    }

    try:
        model = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=30, verbose_eval=50,
        )
    except xgb.core.XGBoostError:
        if device != "cuda": raise
        print("  CUDA training failed; retrying with CPU histogram training.")
        device = "cpu"
        params["device"] = "cpu"
        model = xgb.train(
            params, dtrain, num_boost_round=500,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=30, verbose_eval=50,
        )

    y_val_proba = model.predict(dval)
    val_df = features_df.loc[val_mask]
    # Include validation entities with zero retrieved candidates: these are essential
    # singleton/empty-prediction cases in the challenge's macro score.
    val_s1_ids = list(s1_val)
    group_index = {sid: i for i, sid in enumerate(val_s1_ids)}
    row_groups = np.fromiter((group_index[sid] for sid in val_df["source1_entity_id"].values),
                             dtype=np.int32, count=len(val_df))
    true_counts = np.fromiter((len(true_groups.get(sid, set())) for sid in val_s1_ids),
                              dtype=np.int32, count=len(val_s1_ids))
    best_thresh, best_f05 = 0.5, 0.0
    for thresh in np.arange(0.30, 0.96, 0.02):
        above = y_val_proba >= thresh
        pred_counts = np.bincount(row_groups[above], minlength=len(val_s1_ids))
        tp_counts = np.bincount(row_groups[above & (y_va == 1)], minlength=len(val_s1_ids))
        denom = 0.25 * true_counts + pred_counts
        per_entity = np.divide(1.25 * tp_counts, denom, out=np.zeros(len(val_s1_ids), dtype=np.float64),
                               where=denom > 0)
        per_entity[(true_counts == 0) & (pred_counts == 0)] = 1.0
        f05 = float(per_entity.mean()) if len(per_entity) else 0.0
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
        print(f"    thresh={thresh:.2f} F0.5={f05:.4f}")

    print(f"\n  Best threshold: {best_thresh:.2f} (F0.5={best_f05:.4f})")
    
    print("  Retraining on full data...")
    X_all = features_df[FEATURE_COLS].values.astype(np.float32)
    y_all = features_df["label"].values
    dall = xgb.DMatrix(X_all, label=y_all, feature_names=FEATURE_COLS)

    final_model = xgb.train(
        params, dall, num_boost_round=model.best_iteration + 10,
        evals=[(dall, "train")], verbose_eval=100,
    )

    final_model.save_model(str(CHECKPOINT_DIR / "xgb_model_final.json"))
    with open(str(CHECKPOINT_DIR / "training_device.txt"), "w") as f:
        f.write(device)
    with open(str(CHECKPOINT_DIR / "best_threshold.txt"), "w") as f:
        f.write(str(best_thresh))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the entity matching classifier.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu",
                        help="Training device (default: cpu; cuda errors fall back to CPU).")
    run_training(parser.parse_args().device)
