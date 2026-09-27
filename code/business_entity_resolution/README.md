# Business Entity Resolution Pipeline

## Prerequisites
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`

## Dataset Layout
Place the dataset folder at the workspace root:
```
workspace_root/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── 01_blocker.py
│       │   ├── 02_feature_engine.py
│       │   ├── 03_trainer.py
│       │   └── 04_inference.py
│       ├── README.md
│       └── requirements.txt
└── output/
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

## Reproduce End-to-End

Run each stage sequentially from the **workspace root**:

### Stage 1: Blocking (Candidate Generation)
```bash
# Generate development training candidates (reproducible Source 1 sample)
python code/business_entity_resolution/src/01_blocker.py train --sample 250000

# Generate test candidates (for final inference + candidate_pairs.tsv)
python code/business_entity_resolution/src/01_blocker.py test
```

### Stage 2: Feature Engineering
```bash
# Compute features for training candidates
python code/business_entity_resolution/src/02_feature_engine.py train

# Compute features for test candidates
python code/business_entity_resolution/src/02_feature_engine.py test
```

### Stage 3: Model Training
```bash
# Train on CPU (portable default) + tune F_0.5 threshold
python code/business_entity_resolution/src/03_trainer.py --device cpu
```

### Stage 4: Inference
```bash
# Generate final matching_results.tsv
python code/business_entity_resolution/src/04_inference.py
```

### Validate Submission
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Checkpoints
Intermediate results are saved under `checkpoints/`:
- `candidates_train.parquet` / `candidates_test.parquet`
- `features_train.parquet` / `features_test.parquet`
- `xgb_model.json` / `xgb_model_final.json`
- `best_threshold.txt`

This allows re-running any stage independently without recomputing prior stages.

After validating the approach and measuring blocking recall, run `01_blocker.py train --sample 0` if full training candidates are required. The blocker uses country-partitioned token postings, ranks a bounded pool of at most 300 candidates, and retains at most 40 per Source 1 record. It writes candidates incrementally. Feature loading scans source TSVs in chunks, retains only referenced records, and computes RapidFuzz aligned similarities in 100,000-pair batches before appending compressed Parquet row groups. Validate candidate recall before changing caps; token-only blocking may miss records with no shared useful token.

CPU is the default, so this pipeline runs on machines without a GPU. To try GPU training on a CUDA-capable system, pass `--device cuda`; if CUDA initialization fails, the trainer falls back to CPU. `--device auto` uses CUDA only when the installed XGBoost build reports support. GPU training only affects the classifier; blocking and RapidFuzz feature generation remain CPU work.
