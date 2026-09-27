# Business Entity Resolution: Solution Methodology

## 1. Executive Summary & Project Architecture
This document outlines our professional-grade, highly scalable methodology for the Business Entity Resolution challenge. Given the massive dataset scale (over 10 million combined records) and the strict prohibition on external data lookups, we employ a **decoupled, multi-stage machine learning pipeline**. 

To ensure reproducibility, prevent Out-of-Memory (OOM) errors, and comply with the final submission constraints, the local workspace is structured to strictly isolate the multi-gigabyte dataset from the final submission code:

```text
your_workspace_root/
|-- dataset/                               <-- Kept OUTSIDE the code/ folder to prevent zip bloat
|-- code/                                  <-- Only this folder gets submitted
|   `-- business_entity_resolution/
|       |-- src/
|       |   |-- 01_blocker.py
|       |   |-- 02_feature_engine.py
|       |   |-- 03_trainer.py
|       |   `-- 04_inference.py
|       |-- README.md
|       `-- requirements.txt
|-- output/                                <-- Target directory for final predictions
|   |-- matching_results.tsv
|   `-- candidate_pairs.tsv
|-- utils/
|   `-- validate_submission.py
`-- Documentation_template.md              <-- This methodology document
```

---

## 2. The Development Workflow
To rapidly iterate on a dataset of this magnitude, our development cycle utilizes the standard high-performance hackathon workflow, abandoning single monolithic scripts in favor of modular disk checkpointing:

1.  **Subsample for Speed (The 1-5% Rule):** We do not train on the full 2.2 million Source 1 entities during development. We create a stratified "micro-sample" to build the pipeline and tune hyperparameters.
2.  **Decouple the Pipeline (Disk Checkpointing):** 
    *   We run the heavy blocking script (`01_blocker.py`) *once* and save the candidate pairs to disk as a `.parquet` file.
    *   We run the vectorized feature engineering script (`02_feature_engine.py`) *once* on those candidates and save the resulting numeric matrix to disk.
    *   We load only the pre-computed numeric matrix into RAM to train the classifier (`03_trainer.py`), reducing iteration time.
3.  **Hard Negative Mining:** We avoid training on easy, random negatives. Our blocking algorithm intentionally retrieves "hard negatives"—distinct businesses that share very similar names or addresses.
4.  **The Full-Scale Final Run:** The fully optimized pipeline is run on the full dataset at the very end of development to generate the final model weights for inference (`04_inference.py`).

---

## 3. Handling Multilingual Data & Unseen Countries
The dataset exhibits extreme noise and spans multiple languages and scripts. The test set introduces an unseen country: France.

*   **Language-Agnostic Processing:** We explicitly avoid English-specific NLP tools.
*   **Open-Set Country Treatment:** The `country` column is treated purely as an open set of string labels. The pipeline dynamically partitions data by whatever normalized country string appears.
*   **Standardization:** Text is normalized globally using standard Unicode-aware lowercasing, aggressive punctuation stripping, and multiple-whitespace reduction to ensure characters from Hindi/Kannada scripts and French accents are preserved and matched mathematically.

---

## 4. Stage 1: Scalable Candidate Generation (Blocking)
To reduce the search space into a small, high-recall candidate set (`candidate_pairs.tsv`), we use a highly optimized approach. 

1.  **Country Pre-Partitioning:** Blocking is executed independently per normalized country string.
2.  **TF-IDF Sparse Matrix Blocking:** Instead of slow MinHash iteration, we utilize `scikit-learn`'s TfidfVectorizer on character n-grams and perform sparse matrix dot products to compute cosine similarity instantly across millions of records.
3.  **Pruning:** The union of these candidates is ranked by the cosine similarity score and pruned to a top-K limit per Source 1 entity. 

---

## 5. Stage 2: Vectorized Feature Engineering
We completely bypass slow Python `iterrows()` loops using array-based C-optimized vectorization (numpy and rapidfuzz).

*   **Vectorized String Distances:** We compute Token Sort Ratio, Partial Ratio, and Ratio using `rapidfuzz` efficiently across arrays.
*   **Feature Set:** For every (Source 1, Candidate) pair, we compute Name/Address Jaccard, Token Sort, Partial, Ratios, length comparisons, and common prefix lengths.
*   **Missing Value Handling:** Missing components are *not* imputed. Instead, structural absence is captured via explicit boolean features.
*   **Source Flags:** A categorical indicator identifying if the candidate originated from Source 2 or Source 3.

---

## 6. Stage 3: ML Classification & F_0.5 Optimization
The final matching decision is a binary classification task prioritizing precision.

*   **Model Selection:** We utilize **XGBoost**, satisfying the open-source and sub-8B parameter limits.
*   **Discarding the Default Cutoff:** The competition evaluates using the macro-averaged $F_{0.5}$ score. We sweep a threshold parameter across our validation fold to find the exact probability cutoff that maximizes $F_{0.5}$.
*   **Singleton Protection:** Because singletons cost a full 1.0 point on false merges, this tuned threshold strictly bounds precision and minimizes false positives.
*   **Final Output:** Pairs scoring above the tuned threshold are aggregated into comma-separated strings and exported to `output/matching_results.tsv`.
