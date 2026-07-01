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


def calculate_org_similarity(
    jd_json_path: str,
    candidates_json_path: str,
    jd_pkl_path: str,
    candidates_pkl_path: str,
    decay_base: float = 0.8
) -> Dict[str, float]:
    """Calculates organizational context similarity scores for all candidates.

    For each candidate organization context, a base similarity is computed by:
      - company similarity * company weight in JD (if company exists)
      - industry similarity * industry weight in JD (if industry exists)
      - company_size similarity * company_size weight in JD (if company_size exists)

    This base similarity is multiplied by the YOE factor and the recency factor.
    The final score is the sum of these weighted similarities across all candidate contexts.

    Args:
        jd_json_path: Path to raw job description JSON.
        candidates_json_path: Path to raw candidates JSON.
        jd_pkl_path: Path to job description pickle with embeddings.
        candidates_pkl_path: Path to candidates pickle with embeddings.
        decay_base: The base multiplier for chronological decay (default 0.8).
                    Index 0 (most recent) gets 1.0, Index 1 gets 0.8, etc.

    Returns:
        A dictionary mapping candidate_id to their organizational context similarity score.
    """
    with open(jd_pkl_path, "rb") as f:
        jd_schema = pickle.load(f)

    with open(candidates_pkl_path, "rb") as f:
        candidates = pickle.load(f)

    # Get the single JD organizational context
    jd_orgs = jd_schema.get("organizational_context", [])
    if isinstance(jd_orgs, dict):
        jd_org = jd_orgs
    elif isinstance(jd_orgs, list) and len(jd_orgs) > 0:
        jd_org = jd_orgs[0]
    else:
        return {cand.get("candidate_id"): 0.0 for cand in candidates if cand.get("candidate_id")}

    # Get JD weights
    jd_company = jd_org.get("company", {})
    jd_industry = jd_org.get("industry", {})
    jd_size = jd_org.get("company_size", {})

    jd_company_emb = jd_company.get("embedding")
    jd_company_weight = jd_company.get("weight", 0.0)

    jd_industry_emb = jd_industry.get("embedding")
    jd_industry_weight = jd_industry.get("weight", 0.0)

    jd_size_emb = jd_size.get("embedding")
    jd_size_weight = jd_size.get("weight", 0.0)

    results = {}
    for cand_schema in candidates:
        cand_id = cand_schema.get("candidate_id")
        if not cand_id:
            continue

        cand_orgs = cand_schema.get("organizational_context", [])
        if not cand_orgs:
            results[cand_id] = 0.0
            continue

        # Total YOE of candidate org context for proportional factor
        total_yoe = sum(org.get("years", {}).get("number_of_years", 0.0) for org in cand_orgs)

        score_sum = 0.0
        for idx, cand_org in enumerate(cand_orgs):
            # 1. Company Similarity
            cand_company = cand_org.get("company", {})
            cand_company_emb = cand_company.get("embedding")
            company_score = 0.0
            if cand_company_emb is not None and jd_company_emb is not None:
                company_score = cosine_similarity(jd_company_emb, cand_company_emb) * jd_company_weight

            # 2. Industry Similarity
            cand_industry = cand_org.get("industry", {})
            cand_industry_emb = cand_industry.get("embedding")
            industry_score = 0.0
            if cand_industry_emb is not None and jd_industry_emb is not None:
                industry_score = cosine_similarity(jd_industry_emb, cand_industry_emb) * jd_industry_weight

            # 3. Company Size Similarity
            cand_size = cand_org.get("company_size", {})
            cand_size_emb = cand_size.get("embedding")
            size_score = 0.0
            if cand_size_emb is not None and jd_size_emb is not None:
                size_score = cosine_similarity(jd_size_emb, cand_size_emb) * jd_size_weight

            # Base similarity for this org
            base_sim = company_score + industry_score + size_score

            # YOE Factor: org_yoe / total_yoe
            org_yoe = cand_org.get("years", {}).get("number_of_years", 0.0)
            yoe_factor = (org_yoe / total_yoe) if total_yoe > 0 else 0.0

            # Recency Factor
            recency_factor = decay_base ** idx

            combined_factor = yoe_factor * recency_factor

            # Multiply base similarity by factors and accumulate
            score_sum += base_sim * combined_factor

        results[cand_id] = score_sum

    return results
