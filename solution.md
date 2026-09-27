# Business Entity Resolution: Implemented Solution

## Objective and constraints

The task is to return zero or more Source 2/3 records for every Source 1 record. Evaluation is macro-averaged F0.5, so false merges are costly; true singleton records must remain unmatched. `candidate_pairs.tsv` must list the final candidates actually scored, and every predicted match must be in that list. Processing uses only challenge data.

## Data scale and practical plan

The supplied files contain about 2.2 million training Source 1 records and 1.73 million test Source 1 records, with about five million records in each other source per split. The full test split is close to 10 million records. Development candidate generation defaults to a reproducible 250,000-row sample of training Source 1, paired against the full training reference sources. Set `--sample 0` to generate candidates for all training Source 1 rows.

Iterate on training candidates and entity-level validation before running test inference. Save the candidate and feature checkpoints; changing the classifier or threshold should not require running blocking again. The laptop run time is governed mainly by candidate generation and feature computation, not classifier fitting. Keep an eye on available memory and disk space when processing the full test set.

## Candidate generation

`src/01_blocker.py` uses country-partitioned inverted indexes rather than a dense all-pairs similarity matrix. It normalizes Unicode text, indexes name and address tokens from Sources 2 and 3, skips ubiquitous postings, selects at most 300 candidates from token-overlap hits, ranks this bounded pool with RapidFuzz's native batch scorer, then retains at most 40 candidates per Source 1 record. Candidate pairs are written incrementally to Parquet and, for test, `candidate_pairs.tsv` is written as each Source 1 entity is processed. This avoids quadratic comparison and dense similarity allocations while supporting unseen country labels such as France.

Blocking recall is the ceiling on achievable model recall. The training blocker reports pair recall and mean candidates per selected Source 1 row. This is a development-sample diagnostic, not proof of full-data recall. Before relying on a candidate cap, measure the proportion of ground-truth links present on held-out training entities. If recall is inadequate, increase `TOP_K` or `MAX_RANK_POOL`, or relax `MAX_POSTING`, then reevaluate candidate counts and macro F0.5. Token blocking may miss pairs with no shared useful token; add a bounded character n-gram retrieval channel if validation reveals meaningful misses. No score-preserving setting is established until that validation is run.

## Features and model

The feature stage scans source TSVs in chunks and retains only records referenced by candidates. It computes aligned name/address similarities with RapidFuzz's compiled `cpdist` scorer in 100,000-pair batches, plus token Jaccard and length signals, missing-field indicators, country agreement, source identity, and blocker score. Feature batches are appended to compressed Parquet row groups. Candidate-pair training data is produced by the same blocker used at inference, so the model learns to distinguish retrieved negatives rather than arbitrary pairs.

The trainer uses a compact XGBoost binary classifier. Split validation by Source 1 entity to prevent records for the same entity appearing on both sides. Tune the probability threshold against the competition's macro F0.5 metric, including entities with no true matches. Prefer conservative thresholds when validation shows false merges, especially for singleton-heavy groups. CPU histogram training is the default for portability. `--device cuda` optionally tries CUDA and falls back to CPU if initialization fails; `--device auto` uses CUDA only when the installed build reports support. Blocking and string feature generation remain CPU work, so GPU training cannot by itself guarantee a particular end-to-end runtime.

## Running the pipeline

From the workspace root:

```bash
python code/business_entity_resolution/src/01_blocker.py train --sample 250000
python code/business_entity_resolution/src/02_feature_engine.py train
python code/business_entity_resolution/src/03_trainer.py
python code/business_entity_resolution/src/01_blocker.py test
python code/business_entity_resolution/src/02_feature_engine.py test
python code/business_entity_resolution/src/04_inference.py
```

Use `--sample 0` for full training candidate generation after development. The test blocker writes `output/candidate_pairs.tsv`; inference writes `output/matching_results.tsv`. Verify output format and candidate containment using the provided submission validator before packaging.

## Limitations

The token blocker favors shared tokens and may miss transliterations or heavily corrupted records with no shared name/address token. The held-out blocking-recall check is essential; add a complementary character n-gram retrieval channel if validation shows material misses. Candidate caps and thresholds are tunable engineering choices, not guaranteed score-preserving settings. Report measured validation score, blocking recall, candidate count distribution, runtime, and peak memory in the final methodology rather than claiming unmeasured performance.
