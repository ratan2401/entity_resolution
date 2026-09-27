#!/usr/bin/env python3
import os
import sys
import time
import re
import unicodedata
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_COLS = [
    "block_score", "name_token_sort", "name_partial", "name_ratio", "name_jaccard",
    "name_len_ratio", "name_prefix_len", "addr_token_sort", "addr_partial", "addr_ratio",
    "addr_jaccard", "addr_len_ratio", "combined_token_sort", "s1_name_missing", "s1_addr_missing",
    "cand_name_missing", "cand_addr_missing", "any_missing", "source_flag", "country_match",
]

_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

def normalize_text(text):
    if not isinstance(text, str) or not text.strip(): return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _RE_PUNCT.sub(" ", text)
    return _RE_MULTI_SPACE.sub(" ", text).strip()

def jaccard_tokens(a, b):
    sa = set(a.split()) if a else set()
    sb = set(b.split()) if b else set()
    if not sa and not sb: return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0

def length_ratio(a, b):
    la, lb = len(a), len(b)
    if la == 0 and lb == 0: return 1.0
    if la == 0 or lb == 0: return 0.0
    return min(la, lb) / max(la, lb)

def common_prefix_len(a, b):
    i = 0
    for ca, cb in zip(a, b):
        if ca != cb: break
        i += 1
    return i

def load_source(path, wanted_ids):
    """Read only records referenced by candidate pairs, in bounded chunks."""
    print(f"  Scanning {path} for {len(wanted_ids):,} referenced IDs...")
    pieces = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=200000, usecols=lambda c: c.strip() in
                             {"entity_id", "business_name", "business_address", "country"}):
        chunk.columns = chunk.columns.str.strip()
        found = chunk[chunk["entity_id"].isin(wanted_ids)]
        if not found.empty:
            pieces.append(found)
    df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["entity_id", "business_name", "business_address", "country"])
    df["name_norm"] = df["business_name"].apply(normalize_text)
    df["addr_norm"] = df["business_address"].apply(normalize_text)
    df["country_norm"] = df["country"].apply(normalize_text)
    return df[["entity_id", "name_norm", "addr_norm", "country_norm"]]

def build_lookup(df):
    lookup = {}
    eids = df["entity_id"].values
    names = df["name_norm"].values
    addrs = df["addr_norm"].values
    countries = df["country_norm"].values
    for i in range(len(eids)):
        lookup[eids[i]] = (names[i], addrs[i], countries[i])
    return lookup

def compute_features_vectorized(pairs_df, s1_lookup, s2s3_lookup):
    n = len(pairs_df)
    s1_ids = pairs_df["source1_entity_id"].values
    cand_ids = pairs_df["candidate_id"].values
    block_scores = pairs_df["block_score"].values

    empty = ("", "", "")
    s1_rows = [s1_lookup.get(x, empty) for x in s1_ids]
    cand_rows = [s2s3_lookup.get(x, empty) for x in cand_ids]
    s1n = [x[0] for x in s1_rows]; s1a = [x[1] for x in s1_rows]
    cn = [x[0] for x in cand_rows]; ca = [x[1] for x in cand_rows]

    def aligned(a, b, scorer):
        return process.cpdist(a, b, scorer=scorer, processor=None, dtype=np.float32,
                              workers=-1) / 100.0

    name_token_sort = aligned(s1n, cn, fuzz.token_sort_ratio)
    name_partial = aligned(s1n, cn, fuzz.partial_ratio)
    name_ratio = aligned(s1n, cn, fuzz.ratio)
    addr_token_sort = aligned(s1a, ca, fuzz.token_sort_ratio)
    addr_partial = aligned(s1a, ca, fuzz.partial_ratio)
    addr_ratio = aligned(s1a, ca, fuzz.ratio)
    combined_token_sort_arr = aligned([a + " " + b for a, b in zip(s1n, s1a)],
                                       [a + " " + b for a, b in zip(cn, ca)],
                                       fuzz.token_sort_ratio)
    # Lightweight set/length signals; expensive edit-distance work above runs in C++.
    name_jaccard_arr = np.fromiter((jaccard_tokens(a, b) for a, b in zip(s1n, cn)), np.float32, n)
    addr_jaccard_arr = np.fromiter((jaccard_tokens(a, b) for a, b in zip(s1a, ca)), np.float32, n)
    name_len_ratio_arr = np.fromiter((length_ratio(a, b) for a, b in zip(s1n, cn)), np.float32, n)
    addr_len_ratio_arr = np.fromiter((length_ratio(a, b) for a, b in zip(s1a, ca)), np.float32, n)
    name_prefix_arr = np.fromiter((common_prefix_len(a, b) for a, b in zip(s1n, cn)), np.float32, n)
    s1_name_miss = np.fromiter((not x for x in s1n), np.int8, n)
    s1_addr_miss = np.fromiter((not x for x in s1a), np.int8, n)
    cand_name_miss = np.fromiter((not x for x in cn), np.int8, n)
    cand_addr_miss = np.fromiter((not x for x in ca), np.int8, n)
    any_miss = ((s1_name_miss | s1_addr_miss | cand_name_miss | cand_addr_miss) != 0).astype(np.int8)
    source_flag = np.fromiter((x.startswith("S3") for x in cand_ids), np.int8, n)
    country_match_arr = np.fromiter((a[2] == b[2] for a, b in zip(s1_rows, cand_rows)), np.int8, n)

    return pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_id": cand_ids,
        "block_score": block_scores,
        "name_token_sort": name_token_sort,
        "name_partial": name_partial,
        "name_ratio": name_ratio,
        "name_jaccard": name_jaccard_arr,
        "name_len_ratio": name_len_ratio_arr,
        "name_prefix_len": name_prefix_arr,
        "addr_token_sort": addr_token_sort,
        "addr_partial": addr_partial,
        "addr_ratio": addr_ratio,
        "addr_jaccard": addr_jaccard_arr,
        "addr_len_ratio": addr_len_ratio_arr,
        "combined_token_sort": combined_token_sort_arr,
        "s1_name_missing": s1_name_miss,
        "s1_addr_missing": s1_addr_miss,
        "cand_name_missing": cand_name_miss,
        "cand_addr_missing": cand_addr_miss,
        "any_missing": any_miss,
        "source_flag": source_flag,
        "country_match": country_match_arr,
    })

def run_feature_engineering(mode):
    print(f"\n{'='*70}\n  STAGE 2: FEATURE ENGINEERING ({mode.upper()})\n{'='*70}")
    
    ckpt_path = CHECKPOINT_DIR / f"candidates_{mode}.parquet"
    print(f"  Loading candidates from {ckpt_path}...")
    import pyarrow.parquet as pq
    cand_file = pq.ParquetFile(str(ckpt_path))
    
    data_dir = DATASET_DIR / mode
    s1_path = data_dir / f"{mode}_source1.tsv"
    if not s1_path.exists():
        s1_path = data_dir / f"{mode}_sourcel.tsv"
        
    s1_needed, refs_needed = set(), set()
    for rb in cand_file.iter_batches(columns=["source1_entity_id", "candidate_id"], batch_size=500000):
        values = rb.to_pydict()
        s1_needed.update(values["source1_entity_id"])
        refs_needed.update(values["candidate_id"])
    s1_df = load_source(str(s1_path), s1_needed)
    s2_df = load_source(str(data_dir / f"{mode}_source2.tsv"), refs_needed)
    s3_df = load_source(str(data_dir / f"{mode}_source3.tsv"), refs_needed)
    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)

    t0 = time.time()
    s1_lookup = build_lookup(s1_df)
    s2s3_lookup = build_lookup(s2s3_df)
    
    feat_path = CHECKPOINT_DIR / f"features_{mode}.parquet"
    # Keep peak feature-matrix memory bounded; RapidFuzz computes aligned pairs in C++.
    import pyarrow as pa
    writer = None
    processed = 0
    batch_size = 100000
    for rb in cand_file.iter_batches(batch_size=batch_size):
        cand_batch = rb.to_pandas()
        batch = compute_features_vectorized(cand_batch, s1_lookup, s2s3_lookup)
        table = pa.Table.from_pandas(batch, preserve_index=False)
        if writer is None: writer = pq.ParquetWriter(str(feat_path), table.schema, compression="zstd")
        writer.write_table(table)
        processed += len(cand_batch)
        print(f"    Features: {processed:,}/{cand_file.metadata.num_rows:,}")
    if writer is not None: writer.close()
    else:
        empty_cols = ["source1_entity_id", "candidate_id", *FEATURE_COLS[1:]]
        pd.DataFrame(columns=empty_cols).to_parquet(feat_path, index=False)
    print(f"\n  Saved features to {feat_path} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    run_feature_engineering(mode)
