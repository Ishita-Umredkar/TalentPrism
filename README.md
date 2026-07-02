# TalentPrism — Redrob Candidate Ranking System

Three-stage candidate ranking engine for the Redrob Hackathon v4.  
Scores, ranks, and justifies the top 100 candidates from a 100K pool — entirely on CPU, under 3 minutes.

---

## Quick Start — Reproduce the Submission

```bash
git clone https://github.com/Ishita-Umredkar/TalentPrism.git
cd TalentPrism
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt

python pipelines/final_pipeline/rank.py --candidates ./resources/candidates.jsonl --out submission.csv
```

> **Output**: `submission.csv` — 100 rows with `candidate_id`, `rank`, `score`, `reasoning`.

---

## Architecture

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
  │       from the job description, grouped into 8 categories
  │       (Retrieval, Production ML, LLMs, Domain, Career Quality,
  │       Tech Breadth, External Validation, Hiring Readiness).
  │       Uses pre-embedded BGE cosine similarities.
  │
  └── Stage 3: Ranking, Tie-Breaking & Explainability (stage3/)
      └── Combines Stage 1 × Stage 2 scores, breaks ties
          deterministically, and generates rank-aware natural
          language justifications per candidate.
```

**Final Score** = `Fit Score × Credibility Score`  
**Tie-break** = Hiring readiness score → Candidate ID (ascending)

---

## Directory Structure

```
TalentPrism/
├── pipelines/
│   ├── final_pipeline/              # Production pipeline
│   │   ├── rank.py                  # CLI entry point (single reproduce command)
│   │   ├── stage1/                  # Honeypot detection (10 detectors)
│   │   ├── stage2/                  # Constraint extraction, embeddings & scoring
│   │   │   ├── outputs/             # Pre-computed artifacts (*.pkl, *.json)
│   │   │   └── scripts/             # Embedding generation & ranking logic
│   │   └── stage3/                  # Scoring, explainability & CSV reporting
│   └── initial_pipeline/            # Baseline similarity pipeline (v1, superseded)
├── data/test/                       # Small candidate samples for sandbox testing
├── resources/                       # Official candidates.jsonl, JD, and specs
├── submission.csv                   # Main output — the submission file
├── submission_metadata.yaml         # Portal metadata (team, compute, AI tools)
└── requirements.txt                 # Python dependencies
```

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

## Running the Pipeline

### Full Ranking (100K candidates → submission CSV)

```bash
python pipelines/final_pipeline/rank.py --candidates ./resources/candidates.jsonl --out submission.csv
```

- **Runtime**: ~170s on Intel Core i7, 16GB RAM  
- **Output**: `submission.csv` at the repo root

### Sandbox / Small-Sample Test

```bash
python pipelines/final_pipeline/rank.py --candidates ./data/test/test_candidates.json --out test_submission.csv
```

Uses pre-computed test embeddings. Completes in seconds.

---

## Pre-Computation (runs once, outside the 5-minute window)

The ranking step (`rank.py`) depends on pre-computed embedding artifacts stored in `pipelines/final_pipeline/stage2/outputs/`. These are already included in the repo as `.pkl` files. To regenerate them from scratch:

| Step | Script | Output | Time |
|------|--------|--------|------|
| 1. Extract JD constraints | `stage2/scripts/extract_constraints.py` | `extracted_constraints_v2.json` | ~30s (requires Gemini API key) |
| 2. Embed constraints | `stage2/scripts/embeddings/generate_constraint_embeddings.py` | `embedded_constraints.pkl` | ~10s |
| 3. Embed 100K candidates | `stage2/scripts/embeddings/generate_candidate_embeddings.py` | `candidates_100k_embedded.pkl` | ~40min |

> Pre-computation is **not required** for reproduction — all artifacts are committed. The ranking step itself runs well within the 5-minute CPU budget.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `sentence-transformers` | ≥ 2.2.0 | BGE embedding model (`BAAI/bge-base-en-v1.5`) |
| `torch` | ≥ 2.0.0 | Tensor backend for sentence-transformers |
| `numpy` | ≥ 1.20.0 | Array operations and scoring |
| `tqdm` | ≥ 4.60.0 | Progress bars during embedding generation |

No GPU required for the ranking step. Pre-computation (embedding generation) uses GPU if available. No network calls during ranking.

---

## Submission Metadata

See [`submission_metadata.yaml`](submission_metadata.yaml) for full team details, compute environment, and AI tools declaration.
