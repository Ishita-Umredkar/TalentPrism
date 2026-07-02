import streamlit as st
import pandas as pd
import json
import tempfile
import time
import os
import sys
import pickle
from pathlib import Path

# Add project directories to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "pipelines" / "final_pipeline"))
sys.path.append(str(PROJECT_ROOT / "pipelines" / "final_pipeline" / "stage3"))

from pipelines.final_pipeline.stage3.scoring import CredibilityEvaluator, FitEvaluator
from pipelines.final_pipeline.stage3.explainability import ReasoningGenerator
from pipelines.final_pipeline.stage3.reporting import ReportWriter
from pipelines.final_pipeline.rank import load_candidate_profiles

# Import stage2 paths
from pipelines.final_pipeline.stage2.scripts.rank_100k import (
    CONSTRAINTS_FILE,
    EMBEDDED_CONSTRAINTS_FILE
)

st.set_page_config(
    page_title="TalentPrism Candidate Ranker Sandbox",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 TalentPrism Candidate Ranker")
st.markdown("### Redrob Hackathon v4 — Sandbox Demo Link")

st.markdown("""
This hosted sandbox environment evaluates candidate profiles against Job Description requirements. 
Upload a candidate json/jsonl file containing up to 100 profiles to verify the ranking engine.
""")

uploaded_file = st.file_uploader("Upload Candidates JSON or JSONL file", type=["json", "jsonl"])

if uploaded_file is not None:
    # Save uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json" if uploaded_file.name.endswith(".json") else ".jsonl") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
        
    try:
        candidates = load_candidate_profiles(tmp_path)
        
        if len(candidates) > 1000:
            st.error("Please upload a file containing 1000 or fewer candidates for sandbox testing.")
        else:
            st.success(f"Successfully loaded {len(candidates)} candidates.")
            
            if st.button("Run Ranking Engine"):
                with st.spinner("Processing Stage 1 and Stage 2 pipeline..."):
                    start_time = time.time()
                    
                    # 1. Stage 1: Credibility
                    credibility_evaluator = CredibilityEvaluator()
                    credibility_scores = credibility_evaluator.evaluate_credibility(candidates)
                    
                    # 2. Stage 2 Constraints
                    # Convert to paths relative to PROJECT_ROOT
                    embedded_constraints_path = PROJECT_ROOT / "pipelines" / "final_pipeline" / "stage2" / "outputs" / "embedded_constraints.pkl"
                    constraints_path = PROJECT_ROOT / "pipelines" / "final_pipeline" / "stage2" / "outputs" / "extracted_constraints_v2.json"
                    
                    with open(embedded_constraints_path, "rb") as f:
                        query_embeddings_cache = pickle.load(f)
                    with open(constraints_path, "r", encoding="utf-8") as f:
                        constraints_data = json.load(f)
                        
                    # Load test candidates embeddings
                    emb_file = PROJECT_ROOT / "pipelines" / "final_pipeline" / "stage2" / "outputs" / "test_candidates_embedded.pkl"
                    with open(emb_file, "rb") as f:
                        embedded_candidates = pickle.load(f)
                    embedded_candidates_map = {ec["candidate_id"]: ec for ec in embedded_candidates}
                    
                    # Filter to uploaded candidates
                    embedded_subset = [ec for ec in embedded_candidates if ec["candidate_id"] in candidates]
                    
                    # 3. Score candidates
                    fit_evaluator = FitEvaluator(
                        candidates, constraints_data, embedded_subset, query_embeddings_cache, credibility_scores
                    )
                    ranked_results = fit_evaluator.evaluate_and_rank()
                    
                    # 4. Generate reasoning
                    templates_path = PROJECT_ROOT / "pipelines" / "final_pipeline" / "stage3" / "phrasing_templates.json"
                    reasoning_generator = ReasoningGenerator(templates_path)
                    
                    # Output path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as out_tmp:
                        output_path = out_tmp.name
                        
                    report_writer = ReportWriter(output_path)
                    report_writer.write_reports(
                        ranked_results, candidates, embedded_candidates_map, reasoning_generator, fit_evaluator
                    )
                    
                    elapsed = time.time() - start_time
                    st.success(f"Ranking completed in {elapsed:.2f} seconds!")
                    
                    # Read the CSV output to show in table
                    df = pd.read_csv(output_path)
                    st.dataframe(df)
                    
                    # Download button
                    with open(output_path, "r", encoding="utf-8") as f:
                        csv_data = f.read()
                        
                    st.download_button(
                        label="📥 Download Ranked CSV",
                        data=csv_data,
                        file_name="sandbox_submission.csv",
                        mime="text/csv"
                    )
                    
                    # Clean up
                    os.remove(output_path)
                    
    except Exception as e:
        st.error(f"Error parsing candidates file: {e}")
        
    # Clean up temp upload file
    os.remove(tmp_path)
