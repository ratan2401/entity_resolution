#!/usr/bin/env python3
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

def run_inference():
    print(f"\n{'='*70}\n  STAGE 4: INFERENCE\n{'='*70}")
    
    model_path = CHECKPOINT_DIR / "xgb_model_final.json"
    print(f"  Loading model from {model_path}...")
    model = xgb.Booster()
    model.load_model(str(model_path))

    thresh_path = CHECKPOINT_DIR / "best_threshold.txt"
    with open(str(thresh_path), "r") as f:
        threshold = float(f.read().strip())
    print(f"  Decision threshold: {threshold:.2f}")

    feat_path = CHECKPOINT_DIR / "features_test.parquet"
    print(f"  Loading features from {feat_path}...")
    features_file = pq.ParquetFile(str(feat_path))
    match_dict = defaultdict(set)
    scored = predicted = 0
    for record_batch in features_file.iter_batches(batch_size=100000):
        features_df = record_batch.to_pandas()
        X = features_df[FEATURE_COLS].values.astype(np.float32)
        dtest = xgb.DMatrix(X, feature_names=FEATURE_COLS)
        probabilities = model.predict(dtest)
        mask = probabilities >= threshold
        scored += len(mask); predicted += int(mask.sum())
        for sid, cid in zip(features_df.loc[mask, "source1_entity_id"],
                            features_df.loc[mask, "candidate_id"]):
            match_dict[sid].add(cid)
        print(f"  Scored {scored:,}/{features_file.metadata.num_rows:,} candidates")
    print(f"  Above threshold: {predicted:,}/{scored:,}")

    test_s1_path = DATASET_DIR / "test" / "test_source1.tsv"
    if not test_s1_path.exists():
        test_s1_path = DATASET_DIR / "test" / "test_sourcel.tsv"
    
    out_path = OUTPUT_DIR / "matching_results.tsv"
    row_count = 0
    with open(out_path, "w", encoding="utf-8", newline="") as output_file:
        output_file.write("source1_entity_id\tmatched_entity_ids\n")
        for chunk in pd.read_csv(str(test_s1_path), sep="\t", dtype=str, usecols=["entity_id"],
                                 chunksize=200000, keep_default_na=False):
            for sid in chunk["entity_id"]:
                mids = match_dict.get(sid, ())
                output_file.write(sid + "\t" + ",".join(sorted(mids)) + "\n")
            row_count += len(chunk)
    print(f"  Saved matching_results.tsv ({row_count:,} rows)")

if __name__ == "__main__":
    run_inference()
