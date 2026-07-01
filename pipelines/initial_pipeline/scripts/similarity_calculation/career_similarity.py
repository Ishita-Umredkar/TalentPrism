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


def calculate_career_similarity(
    jd_json_path: str,
    candidates_json_path: str,
    jd_pkl_path: str,
    candidates_pkl_path: str,
    decay_base: float = 0.8
) -> Dict[str, float]:
    """Calculates career similarity scores for all candidates.

    The single JD role is compared against each candidate role. Each similarity
    score is multiplied by:
      - a combined factor: (cand_role_yoe / total_cand_yoe) * (decay_base ** cand_role_index)
      - the JD role's weight

    The results are summed across all candidate roles to produce the total similarity.

    Args:
        jd_json_path: Path to raw job description JSON.
        candidates_json_path: Path to raw candidates JSON.
        jd_pkl_path: Path to job description pickle with embeddings.
        candidates_pkl_path: Path to candidates pickle with embeddings.
        decay_base: The base multiplier for chronological decay (default 0.8).
                    Index 0 (most recent) gets 1.0, Index 1 gets 0.8, Index 2 gets 0.64, etc.

    Returns:
        A dictionary mapping candidate_id to their career similarity score.
    """
    with open(jd_pkl_path, "rb") as f:
        jd_schema = pickle.load(f)

    with open(candidates_pkl_path, "rb") as f:
        candidates = pickle.load(f)

    # Get the single JD role
    jd_roles = jd_schema.get("professional_experience", {}).get("roles", [])
    if not jd_roles:
        return {cand.get("candidate_id"): 0.0 for cand in candidates if cand.get("candidate_id")}

    jd_role = jd_roles[0]
    jd_emb = jd_role.get("embedding")
    jd_weight = jd_role.get("weight", 1.0)

    results = {}
    for cand_schema in candidates:
        cand_id = cand_schema.get("candidate_id")
        if not cand_id:
            continue

        cand_roles = cand_schema.get("professional_experience", {}).get("roles", [])
        if not cand_roles or jd_emb is None:
            results[cand_id] = 0.0
            continue

        # Total YOE of candidate for proportional weight factor
        total_yoe = sum(role.get("yoe", 0.0) for role in cand_roles)

        score_sum = 0.0
        # Compare JD role with each role in candidate's professional experience
        for idx, cand_role in enumerate(cand_roles):
            cand_emb = cand_role.get("embedding")
            if cand_emb is None:
                continue

            sim = cosine_similarity(jd_emb, cand_emb)

            # 1) YOE ratio factor: role_yoe / total_yoe
            yoe_factor = (cand_role.get("yoe", 0.0) / total_yoe) if total_yoe > 0 else 0.0

            # 2) Recency factor: decay based on reverse chronological order
            # Index 0 is the most recent, so decay_base ** 0 = 1.0
            recency_factor = decay_base ** idx

            combined_factor = yoe_factor * recency_factor

            # Multiply by combined factor and JD weight, then add to sum
            score_sum += sim * combined_factor * jd_weight

        results[cand_id] = score_sum

    return results
