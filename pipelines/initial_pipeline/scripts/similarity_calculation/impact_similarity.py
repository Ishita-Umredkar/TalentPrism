import json
import pickle
from typing import Dict


def calculate_impact_similarity(
    jd_json_path: str,
    candidates_json_path: str,
    jd_pkl_path: str,
    candidates_pkl_path: str
) -> Dict[str, float]:
    """Calculates execution impact similarity scores for all candidates.

    For each impact field (production_experience, ownership_leadership, impact):
      - compares candidate score with JD score: comparison_score = 1.0 - abs(cand_score - jd_score)
      - multiplies by the ratio: cand_score / jd_score
      - multiplies by the JD field's weight

    The results are summed across all three fields.

    Args:
        jd_json_path: Path to raw job description JSON.
        candidates_json_path: Path to raw candidates JSON.
        jd_pkl_path: Path to job description pickle (not directly used but kept for signature consistency).
        candidates_pkl_path: Path to candidates pickle (not directly used but kept for signature consistency).

    Returns:
        A dictionary mapping candidate_id to their impact similarity score.
    """
    # Load raw JSONs to read execution_impact scores and weights
    with open(jd_json_path, "r", encoding="utf-8") as f:
        jd_schema = json.load(f)

    with open(candidates_json_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    jd_impact = jd_schema.get("execution_impact", {})
    fields = ["production_experience", "ownership_leadership", "impact"]

    results = {}
    for cand_schema in candidates:
        cand_id = cand_schema.get("candidate_id")
        if not cand_id:
            continue

        cand_impact = cand_schema.get("execution_impact", {})
        total_score = 0.0

        for field in fields:
            # Get JD score and weight
            jd_field_data = jd_impact.get(field, {})
            jd_score = jd_field_data.get("score", 0.0)
            jd_weight = jd_field_data.get("weight", 0.0)

            # Candidate score is direct float in extracted candidate
            cand_score = cand_impact.get(field, 0.0)

            if jd_score == 0:
                continue

            # Compare candidate and JD score: 1.0 - absolute difference
            comparison_score = 1.0 - abs(cand_score - jd_score)

            # Ratio of candidate score to JD score
            ratio = cand_score / jd_score

            # Combined score for this field
            field_score = comparison_score * ratio * jd_weight
            total_score += field_score

        results[cand_id] = total_score

    return results
