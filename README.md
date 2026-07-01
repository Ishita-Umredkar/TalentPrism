# TalentPrism - Redrob Candidate Ranking System

This repository contains the candidate ranking engine for the Redrob Hackathon v4. 
It implements two distinct candidate evaluation pipelines.

---

## 1. Directory Structure

*   `pipelines/`
    *   `initial_pipeline/`: Represents the baseline semantic similarity pipeline. It calculates similarity score metrics across 6 distinct profile areas and ranks candidates based on simple sum aggregates.
    *   `final_pipeline/`: Represents the advanced production pipeline incorporating:
        *   `rank.py`: The orchestrator and CLI entry point.
        *   `stage3/`: Contains clean SOLID-compliant modules (`scoring.py`, `explainability.py`, `reporting.py`) and text database (`phrasing_templates.json`).
        *   **Stage 1 Credibility Check (in `stage1/`)**: An active honeypot detector (10 distinct detectors checking timeline discrepancies, company founding history, and profile inflation).
        *   **Stage 2 Constraint Evaluator (in `stage2/`)**: A hierarchical gatekeeper that scores candidates against 17 distinct extracted JD requirements, grouping them into 8 high-level conceptual categories.
*   `data/`: Contains raw taxonomy mappings and small-sample test candidate lists.
*   `resources/`: Holds the official `candidates.jsonl` database (100,000 candidates), job description requirements, and hackathon specs.
*   `pipelines/final_pipeline/ranking/`: Output folder for final submissions.
    *   `top100.csv`: The official submission CSV containing candidate IDs, ranks, scores, and natural text justifications.
    *   `top100.json`: Detailed JSON array listing the top 100 candidate profiles in ranked order with metadata.
    *   `explanation.txt`: Distribution metrics and ranking reports.

---

## 2. Setup & Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Ishita-Umredkar/TalentPrism.git
    cd TalentPrism
    ```

2.  **Initialize Virtual Environment**:
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On Linux/macOS:
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 3. How to Run the Pipelines

### A. Advanced Ranking Pipeline (Stage 1 + Stage 2 + Ranking)
The advanced scoring and credibility pipeline is executed using `pipelines/final_pipeline/rank.py`.

*   **Reproduce Command**:
    ```bash
    python pipelines/final_pipeline/rank.py --candidates ./resources/candidates.jsonl --out ./pipelines/final_pipeline/ranking/top100.csv --output-explanation ./pipelines/final_pipeline/ranking/explanation.txt
    ```
*   **Sandbox / Test Command**:
    For sandbox validation on a small subset (e.g., 10 candidates), the ranker dynamically detects candidate count and uses test embeddings:
    ```bash
    python pipelines/final_pipeline/rank.py --candidates ./data/test/test_candidates.json --out ./pipelines/final_pipeline/ranking/test_top10.csv --output-explanation ./pipelines/final_pipeline/ranking/test_explanation.txt
    ```

### B. Initial Semantic Similarity Pipeline
The baseline similarity calculation pipeline can be executed using:
```bash
python pipelines/initial_pipeline/scripts/similarity_calculation/compare_candidates.py
```

---

## 4. Submission Metadata

Refer to `submission_metadata.yaml` at the root of the repository for full metadata details, Contact info, compute platform specs, and AI usage summaries.
