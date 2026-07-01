import json
import pickle
import numpy as np
from typing import List, Dict


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return cosine similarity between two vectors.
    Handles zero‑vector edge case by returning 0.0.
    """
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def calculate_edu_similarity(
    jd_json_path: str,
    candidates_json_path: str,
    jd_pkl_path: str,
    candidates_pkl_path: str
) -> Dict[str, float]:
    """Calculates education credentials similarity scores for all candidates.

    Compares the candidate's education entries with the JD's preferred education entries.
    Finds the maximum cosine similarity between any JD education embedding and any
    candidate education embedding, and multiplies it by the overall education weight.

    Args:
        jd_json_path: Path to raw job description JSON.
        candidates_json_path: Path to raw candidates JSON.
        jd_pkl_path: Path to job description pickle with embeddings.
        candidates_pkl_path: Path to candidates pickle with embeddings.

    Returns:
        A dictionary mapping candidate_id to their education similarity score.
    """
    with open(jd_pkl_path, "rb") as f:
        jd_schema = pickle.load(f)

    with open(candidates_pkl_path, "rb") as f:
        candidates = pickle.load(f)

    # Get the JD education section and weight
    jd_edu_section = jd_schema.get("education_credentials", {})
    jd_weight = jd_edu_section.get("weight", 1.0)
    jd_educations = jd_edu_section.get("education", [])

    if not jd_educations:
        return {cand.get("candidate_id"): 0.0 for cand in candidates if cand.get("candidate_id")}

    results = {}
    for cand_schema in candidates:
        cand_id = cand_schema.get("candidate_id")
        if not cand_id:
            continue

        cand_edu_section = cand_schema.get("education_credentials", {})
        cand_educations = cand_edu_section.get("education", [])

        if not cand_educations:
            results[cand_id] = 0.0
            continue

        max_sim = 0.0
        for jd_edu in jd_educations:
            jd_emb = jd_edu.get("embedding")
            if jd_emb is None:
                continue

            for cand_edu in cand_educations:
                cand_emb = cand_edu.get("embedding")
                if cand_emb is None:
                    continue

                sim = cosine_similarity(jd_emb, cand_emb)
                if sim > max_sim:
                    max_sim = sim

        results[cand_id] = max_sim * jd_weight

    return results
