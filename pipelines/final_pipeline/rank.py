"""
rank.py

SOLID Orchestrator for Candidate Discovery and Ranking System.
Deconstructs processes into scoring, explainability, and reporting modules.
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "stage3"))

# Import modular components
from stage3.scoring import CredibilityEvaluator, FitEvaluator
from stage3.explainability import ReasoningGenerator
from stage3.reporting import ReportWriter

# Import paths from stage2
from stage2.scripts.rank_100k import (
    CONSTRAINTS_FILE,
    EMBEDDED_CONSTRAINTS_FILE
)

def load_candidate_profiles(candidates_path):
    """Loads candidate profiles database file (supports JSONL and JSON lists)."""
    print(f"Loading candidate profiles from {candidates_path}...")
    candidates = {}
    is_jsonl = candidates_path.endswith(".jsonl")
    
    with open(candidates_path, "r", encoding="utf-8") as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cand = json.loads(line)
                candidates[cand["candidate_id"]] = cand
        else:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for cand in data:
                        candidates[cand["candidate_id"]] = cand
                else:
                    candidates[data["candidate_id"]] = data
            except json.JSONDecodeError:
                f.seek(0)
                content = f.read()
                decoder = json.JSONDecoder()
                pos = 0
                while pos < len(content):
                    while pos < len(content) and content[pos].isspace():
                        pos += 1
                    if pos >= len(content):
                        break
                    obj, next_pos = decoder.raw_decode(content, pos)
                    pos = next_pos
                    candidates[obj["candidate_id"]] = obj
    print(f"Loaded {len(candidates)} candidates.")
    return candidates

def main():
    default_csv = str(ROOT / "ranking" / "top100.csv")
    default_exp = str(ROOT / "ranking" / "explanation.txt")
    parser = argparse.ArgumentParser(description="Rank candidates using Stage 1 and Stage 2 scores.")
    parser.add_argument("--candidates", default="resources/candidates.jsonl", help="Path to candidates file")
    parser.add_argument("--output-csv", "--out", default=default_csv, help="Path to output CSV")
    parser.add_argument("--output-explanation", default=default_exp, help="Path to output explanation TXT")
    args = parser.parse_args()

    # 1. Load candidates
    candidates_path = args.candidates
    if not os.path.isabs(candidates_path):
        cwd_path = Path(candidates_path)
        if cwd_path.exists():
            candidates_path = str(cwd_path.resolve())
        else:
            candidates_path = str((PROJECT_ROOT / candidates_path).resolve())

    candidates = load_candidate_profiles(candidates_path)

    # 2. Stage 1: Credibility Evaluation
    credibility_evaluator = CredibilityEvaluator()
    credibility_scores = credibility_evaluator.evaluate_credibility(candidates)

    # 3. Load embeddings cache and constraints
    print(f"Loading pre-embedded constraints from: {EMBEDDED_CONSTRAINTS_FILE}")
    with open(EMBEDDED_CONSTRAINTS_FILE, "rb") as f:
        query_embeddings_cache = pickle.load(f)
        
    print(f"Loading constraints structure from: {CONSTRAINTS_FILE}")
    with open(CONSTRAINTS_FILE, "r", encoding="utf-8") as f:
        constraints_data = json.load(f)

    # Determine candidate embeddings file to load
    if len(candidates) <= 1000:
        emb_file = ROOT / "stage2" / "outputs" / "test_candidates_embedded.pkl"
    else:
        emb_file = ROOT / "stage2" / "outputs" / "candidates_100k_embedded.pkl"

    print(f"Loading candidate embeddings from: {emb_file}...")
    with open(emb_file, "rb") as f:
        embedded_candidates = pickle.load(f)

    embedded_candidates_map = {ec["candidate_id"]: ec for ec in embedded_candidates}

    # 4. Stage 2: Fit Evaluation & Sorting
    fit_evaluator = FitEvaluator(
        candidates, constraints_data, embedded_candidates, query_embeddings_cache, credibility_scores
    )
    ranked_results = fit_evaluator.evaluate_and_rank()

    # 5. Explainability Generator & Reporting
    templates_path = ROOT / "stage3" / "phrasing_templates.json"
    reasoning_generator = ReasoningGenerator(templates_path)
    
    # Resolve output paths to absolute paths to prevent directory creation errors
    output_csv = os.path.abspath(args.output_csv)
    output_explanation = os.path.abspath(args.output_explanation)
    
    report_writer = ReportWriter(output_csv, output_explanation)
    top_limit = min(len(ranked_results), 100)
    top_n = ranked_results[:top_limit]
    
    report_writer.write_reports(
        top_n, candidates, embedded_candidates_map, reasoning_generator, fit_evaluator
    )
    
    print("Ranking process completed successfully!")

if __name__ == "__main__":
    main()
