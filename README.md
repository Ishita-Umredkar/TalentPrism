# TalentPrism — Redrob Candidate Ranking System

Three-stage candidate ranking engine for the Redrob Hackathon v4.  
Scores, ranks, and justifies candidates from a 100K pool — entirely on CPU, with optimized stage-wise processing.

---

## Architecture Blueprint

```
rank.py (orchestrator)
  │
  ├── Stage 1: Credibility Check (stage1/)
  │   └── 10 honeypot detectors — flags timeline fraud, company age
  │       mismatches, skill inflation, and profile inconsistencies.
  │       Outputs a credibility multiplier per candidate (0.0–1.0).
  │
  ├── Stage 2: Semantic Fit Scoring (stage2/)
  │   └── Scores each candidate against 17 JD constraints extracted
  │       from the job description, grouped into 8 categories.
  │       Uses pre-embedded BGE cosine similarities.
  │
  └── Stage 3: Ranking, Tie-Breaking & Explainability (stage3/)
      └── Combines Stage 1 × Stage 2 scores, breaks ties
          deterministically, and generates rank-aware natural
          language justifications per candidate.
```

* **Final Score** = `Fit Score × Credibility Score`  
* **Tie-break** = Hiring readiness score → Candidate ID (ascending)

---

## Setup & Installation

1. **Clone and enter the repository**:
    ```bash
    git clone https://github.com/Ishita-Umredkar/TalentPrism.git
    cd TalentPrism
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/macOS:
    source venv/bin/activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run (Step-by-Step)

For Stage 3 replication, the pipeline executes in two simple steps:

### Step 1: Pre-Computation (Generate Embeddings)
Before running the ranker, you must generate semantic embeddings for your candidate dataset. This step can run on GPU (if available) or CPU.

Run the following command:
```bash
python pipelines/final_pipeline/stage2/scripts/embeddings/generate_candidate_embeddings.py --candidates candidates.jsonl --output pipelines/final_pipeline/stage2/outputs/candidates_100k_embedded.pkl
```
* **Runtime**: **~40 minutes** (tested on local Intel Core i7 PC / CPU+GPU setup)
* **Output**: `pipelines/final_pipeline/stage2/outputs/candidates_100k_embedded.pkl`

*(Note: Pre-computed test embeddings `test_candidates_embedded.pkl` and `embedded_constraints.pkl` are already whitelisted and included in the repository for sandbox verification).*

### Step 2: Final Candidate Ranking
Once the candidate embeddings are generated, run the final candidate ranking orchestrator. This step runs entirely on **CPU only** without internet.

Run the following command:
```bash
python pipelines/final_pipeline/rank.py --candidates candidates.jsonl --out submission.csv
```
* **Runtime**: **~70 seconds** on CPU.
* **Output**: `submission.csv` at the repository root containing exactly the top 100 ranked candidates.

---

## Sandbox / Small-Sample Verification
To verify the system end-to-end on a custom small test sample (such as inside a Google Colab sandbox or offline test run):

1. **Pre-compute embeddings for the test set**:
   ```bash
   python pipelines/final_pipeline/stage2/scripts/embeddings/generate_candidate_embeddings.py --candidates ./data/test/100candidates.json --output ./pipelines/final_pipeline/stage2/outputs/test_candidates_embedded.pkl
   ```

2. **Run the ranking engine with the custom embeddings**:
   ```bash
   python pipelines/final_pipeline/rank.py --candidates ./data/test/100candidates.json --embeddings ./pipelines/final_pipeline/stage2/outputs/test_candidates_embedded.pkl --out test_submission.csv
   ```
   * **Runtime**: Completes in **under 1 second** on CPU.
   * **Note**: If no custom `--embeddings` file is specified, the ranker will automatically default to look for `test_candidates_embedded.pkl` if the input candidates file has 1,000 or fewer records.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | ≥ 2.2.0 | BGE embedding model (`BAAI/bge-base-en-v1.5`) |
| `torch` | ≥ 2.0.0 | Tensor backend for sentence-transformers |
| `numpy` | ≥ 1.20.0 | Array operations and scoring |
| `tqdm` | ≥ 4.60.0 | Progress bars during embedding generation |

---

## Submission Metadata
See [`submission_metadata.yaml`](submission_metadata.yaml) for full team details, compute environment, and AI tools declaration.
