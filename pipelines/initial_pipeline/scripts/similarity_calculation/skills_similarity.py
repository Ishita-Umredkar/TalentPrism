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


def compute_skill_similarity(jd_schema: Dict, cand_schema: Dict) -> float:
    """Calculate similarity score for the *technical_capability / technical_skills* section.

    Steps:
    1. For every skill in the JD, compute the maximum cosine similarity with any
       skill in the candidate.
    2. Multiply the max similarity by the JD skill's weight.
    3. After summing over all JD skills, multiply the total by the ratio
       ``candidate_score / jd_score`` where each score is the sum of the
       ``score`` fields of the respective skill lists.
    """
    jd_skills = jd_schema.get("technical_capability", {}).get("technical_skills", [])
    cand_skills = cand_schema.get("technical_capability", {}).get("technical_skills", [])

    # Guard against missing skills
    if not jd_skills or not cand_skills:
        return 0.0

    # Pre‑extract embeddings for candidate skills for efficiency
    cand_embeddings = [skill.get("embedding") for skill in cand_skills if skill.get("embedding") is not None]
    if not cand_embeddings:
        return 0.0

    weighted_sum = 0.0
    for jd_skill in jd_skills:
        jd_emb = jd_skill.get("embedding")
        if jd_emb is None:
            continue
        # Compute max similarity with any candidate skill embedding
        max_sim = max(cosine_similarity(jd_emb, cand_emb) for cand_emb in cand_embeddings)
        weight = jd_skill.get("weight", 1.0)
        weighted_sum += max_sim * weight

    # Compute total raw scores for ratio scaling
    jd_total_score = sum(skill.get("score", 0.0) for skill in jd_skills)
    cand_total_score = sum(skill.get("score", 0.0) for skill in cand_skills)
    if jd_total_score == 0:
        ratio = 0.0
    else:
        ratio = cand_total_score / jd_total_score

    return weighted_sum * ratio


def calculate_skills_similarity(
    jd_json_path: str,
    candidates_json_path: str,
    jd_pkl_path: str,
    candidates_pkl_path: str
) -> Dict[str, float]:
    """Calculates technical skills similarity scores for all candidates.

    Args:
        jd_json_path: Path to raw job description JSON.
        candidates_json_path: Path to raw candidates JSON.
        jd_pkl_path: Path to job description pickle with embeddings.
        candidates_pkl_path: Path to candidates pickle with embeddings.

    Returns:
        A dictionary mapping candidate_id to their skills similarity score.
    """
    with open(jd_pkl_path, "rb") as f:
        jd_schema = pickle.load(f)

    with open(candidates_pkl_path, "rb") as f:
        candidates = pickle.load(f)

    results = {}
    for cand_schema in candidates:
        cand_id = cand_schema.get("candidate_id")
        if not cand_id:
            continue
        results[cand_id] = compute_skill_similarity(jd_schema, cand_schema)

    return results
