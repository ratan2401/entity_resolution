#!/usr/bin/env python3
"""Memory-bounded multilingual candidate generation using country/token blocks."""
import argparse
import gc
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_DIR, CHECKPOINT_DIR, OUTPUT_DIR = BASE_DIR / "dataset", BASE_DIR / "checkpoints", BASE_DIR / "output"
for directory in (CHECKPOINT_DIR, OUTPUT_DIR): directory.mkdir(parents=True, exist_ok=True)
TOP_K = 40
MAX_POSTING = 5000
MAX_RANK_POOL = 300
SPACE = re.compile(r"\s+")
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

def norm(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return SPACE.sub(" ", PUNCT.sub(" ", value)).strip()

def load(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    for col in ("entity_id", "business_name", "business_address", "country"):
        if col not in df: raise ValueError(f"{path} is missing required column {col}")
    for col in ("business_name", "business_address", "country"):
        df[col + "_n"] = df[col].map(norm)
    return df[["entity_id", "business_name_n", "business_address_n", "country_n"]]

def terms(row):
    # Name tokens are weighted separately; retaining short tokens helps non-Latin scripts.
    ns = set(row.business_name_n.split())
    ads = set(row.business_address_n.split())
    return ns, ads

def generate(s1, refs):
    """Retrieve bounded candidates from rare-token postings; never forms an N x M matrix."""
    for country, left in s1.groupby("country_n", sort=False):
        right = refs[refs.country_n == country]
        if left.empty: continue
        if right.empty:
            for row in left.itertuples(index=False): yield row.entity_id, []
            continue
        ids = right.entity_id.tolist()
        names = right.business_name_n.tolist()
        addrs = right.business_address_n.tolist()
        postings = defaultdict(list)
        ignored_tokens = set()
        for j, (name, addr) in enumerate(zip(names, addrs)):
            for key in ("n:" + token for token in set(name.split())):
                if key in ignored_tokens: continue
                postings[key].append(j)
                if len(postings[key]) > MAX_POSTING:
                    del postings[key]
                    ignored_tokens.add(key)
            for key in ("a:" + token for token in set(addr.split())):
                if key in ignored_tokens: continue
                postings[key].append(j)
                if len(postings[key]) > MAX_POSTING:
                    del postings[key]
                    ignored_tokens.add(key)
        for row in left.itertuples(index=False):
            nt, at = terms(row)
            hits = defaultdict(float)
            for token in nt:
                for j in postings.get("n:" + token, ()):
                    hits[j] += 2.0 / max(1, len(nt))
            for token in at:
                for j in postings.get("a:" + token, ()):
                    hits[j] += 1.0 / max(1, len(at))
            # A rare address/name token is sufficient to enter the pool. Rank only this
            # pool using fast string similarities, then retain a compact per-row set.
            # First bound the pool by cheap inverted-index overlap. RapidFuzz then
            # scores aligned candidates in native code, avoiding Python pair loops.
            pool = sorted(hits.items(), key=lambda item: item[1], reverse=True)[:MAX_RANK_POOL]
            if not pool:
                yield row.entity_id, []
                continue
            idx = [j for j, _ in pool]
            name_scores = process.cdist([row.business_name_n], [names[j] for j in idx],
                                        scorer=fuzz.token_sort_ratio, dtype=np.float32)[0] / 100.0
            addr_scores = process.cdist([row.business_address_n], [addrs[j] for j in idx],
                                        scorer=fuzz.token_sort_ratio, dtype=np.float32)[0] / 100.0
            overlap_scores = np.asarray([min(hits[j], 1.0) for j in idx], dtype=np.float32)
            scores = 0.55 * name_scores + 0.35 * addr_scores + 0.10 * overlap_scores
            order = np.argsort(scores)[-TOP_K:][::-1]
            row_candidates = []
            for pos in order:
                j, score = idx[pos], float(scores[pos])
                row_candidates.append((ids[j], score))
            yield row.entity_id, row_candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "test"))
    parser.add_argument("--sample", type=int, default=250000, help="Training Source 1 sample; 0 means all rows")
    args = parser.parse_args()
    folder = DATASET_DIR / args.mode
    s1 = load(folder / f"{args.mode}_source1.tsv")
    if args.mode == "train" and args.sample and args.sample < len(s1):
        # Fixed seed makes model-development candidate data reproducible.
        s1 = s1.sample(n=args.sample, random_state=42).reset_index(drop=True)
    if args.mode == "train":
        pd.DataFrame({"source1_entity_id": s1.entity_id}).to_parquet(
            CHECKPOINT_DIR / "train_source1_population.parquet", index=False)
    s2 = load(folder / f"{args.mode}_source2.tsv")
    s3 = load(folder / f"{args.mode}_source3.tsv")
    refs = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    out = CHECKPOINT_DIR / f"candidates_{args.mode}.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([("source1_entity_id", pa.string()), ("candidate_id", pa.string()), ("block_score", pa.float32())])
    writer = pq.ParquetWriter(str(out), schema, compression="zstd")
    batch = []
    pair_count = 0
    found_true = 0
    if args.mode == "train":
        truth = pd.read_csv(folder / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
        truth.columns = truth.columns.str.strip()
        selected = set(s1.entity_id)
        truth_pairs = set()
        for sid, value in zip(truth.source1_entity_id, truth.matched_entity_ids):
            if sid in selected and value.strip():
                truth_pairs.update((sid, mid.strip()) for mid in value.split(",") if mid.strip())
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    candidate_file = open(candidate_path, "w", encoding="utf-8", newline="") if args.mode == "test" else None
    if candidate_file: candidate_file.write("source1_entity_id\tcandidate_entity_ids\n")
    s1_count = 0
    for sid, candidates in generate(s1, refs):
        s1_count += 1
        if candidate_file:
            candidate_file.write(sid + "\t" + ",".join(sorted(x[0] for x in candidates)) + "\n")
        for cid, score in candidates:
            batch.append((sid, cid, score))
            if args.mode == "train" and (sid, cid) in truth_pairs: found_true += 1
        if len(batch) >= 100000:
            table = pa.Table.from_arrays([pa.array([x[0] for x in batch]), pa.array([x[1] for x in batch]),
                                          pa.array([x[2] for x in batch], type=pa.float32())], schema=schema)
            writer.write_table(table); pair_count += len(batch); batch.clear()
        if s1_count % 50000 == 0: print(f"  Blocked {s1_count:,}/{len(s1):,} Source 1 records")
    if batch:
        table = pa.Table.from_arrays([pa.array([x[0] for x in batch]), pa.array([x[1] for x in batch]),
                                      pa.array([x[2] for x in batch], type=pa.float32())], schema=schema)
        writer.write_table(table); pair_count += len(batch)
    writer.close()
    if candidate_file: candidate_file.close()
    print(f"Saved {pair_count:,} candidate pairs to {out}")
    if args.mode == "train":
        recall = found_true / len(truth_pairs) if truth_pairs else 0.0
        print(f"Training candidate pair recall on selected Source 1 rows: {recall:.4%} ({found_true:,}/{len(truth_pairs):,})")
        print(f"Mean candidates per selected Source 1: {pair_count / max(len(s1), 1):.2f}")
    if args.mode == "test":
        print(f"Saved candidate list for {s1_count:,} Source 1 records to {candidate_path}")

if __name__ == "__main__": main()
