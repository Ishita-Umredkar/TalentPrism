import argparse
import json
import pathlib
import sys
from typing import Dict

# Add project root to sys.path to allow running directly from any directory
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from scripts.similarity_calculation.skills_similarity import calculate_skills_similarity
from scripts.similarity_calculation.career_similarity import calculate_career_similarity
from scripts.similarity_calculation.impact_similarity import calculate_impact_similarity
from scripts.similarity_calculation.org_similarity import calculate_org_similarity
from scripts.similarity_calculation.edu_similarity import calculate_edu_similarity
from scripts.similarity_calculation.logistics_similarity import calculate_logistics_similarity


def get_max_scores(jd_json_path: str) -> Dict[str, float]:
    """Dynamically calculates the maximum possible scores for each section from the JD JSON."""
    with open(jd_json_path, "r", encoding="utf-8") as f:
        jd = json.load(f)

    skills = jd.get("technical_capability", {}).get("technical_skills", [])
    max_skills = sum(skill.get("weight", 0.0) for skill in skills)

    roles = jd.get("professional_experience", {}).get("roles", [])
    max_career = sum(role.get("weight", 0.0) for role in roles)

    impact = jd.get("execution_impact", {})
    max_impact = sum(impact.get(f, {}).get("weight", 0.0) for f in ["production_experience", "ownership_leadership", "impact"])

    org = jd.get("organizational_context", {})
    if isinstance(org, list) and len(org) > 0:
        org = org[0]
    max_org = (
        org.get("company", {}).get("weight", 0.0) +
        org.get("industry", {}).get("weight", 0.0) +
        org.get("company_size", {}).get("weight", 0.0)
    )

    edu = jd.get("education_credentials", {})
    max_edu = edu.get("weight", 0.0)

    max_logistics = jd.get("logistics", {}).get("weight", 0.0)

    total_max = max_skills + max_career + max_impact + max_org + max_edu + max_logistics

    return {
        "skills": max_skills,
        "career": max_career,
        "impact": max_impact,
        "org": max_org,
        "edu": max_edu,
        "logistics": max_logistics,
        "total": total_max
    }


def compare_candidates(
    jd_json_path: str,
    candidates_json_path: str,
    jd_pkl_path: str,
    candidates_pkl_path: str
) -> None:
    """Calculates similarity for all sections and prints a comparison table."""
    # 1. Run similarity calculations for each section
    skills = calculate_skills_similarity(jd_json_path, candidates_json_path, jd_pkl_path, candidates_pkl_path)
    career = calculate_career_similarity(jd_json_path, candidates_json_path, jd_pkl_path, candidates_pkl_path)
    impact = calculate_impact_similarity(jd_json_path, candidates_json_path, jd_pkl_path, candidates_pkl_path)
    org = calculate_org_similarity(jd_json_path, candidates_json_path, jd_pkl_path, candidates_pkl_path)
    edu = calculate_edu_similarity(jd_json_path, candidates_json_path, jd_pkl_path, candidates_pkl_path)
    logistics = calculate_logistics_similarity(jd_json_path, candidates_json_path, jd_pkl_path, candidates_pkl_path)

    # 2. Get maximum scores
    max_scores = get_max_scores(jd_json_path)

    # Get a list of all candidate IDs
    candidate_ids = sorted(list(skills.keys()))

    # Print table header
    header_format = "{:<15} | {:<13} | {:<13} | {:<13} | {:<13} | {:<13} | {:<13} | {:<13}"
    row_format = "{:<15} | {:<13.4f} | {:<13.4f} | {:<13.4f} | {:<13.4f} | {:<13.4f} | {:<13.4f} | {:<13.4f}"

    print("\n" + "=" * 119)
    print("CANDIDATE SIMILARITY COMPARISON TABLE")
    print("=" * 119)
    print(header_format.format("Candidate ID", "Skills", "Career", "Impact", "Org Context", "Education", "Logistics", "Total Score"))
    print(header_format.format(
        "",
        f"(Max: {max_scores['skills']:.4f})",
        f"(Max: {max_scores['career']:.4f})",
        f"(Max: {max_scores['impact']:.4f})",
        f"(Max: {max_scores['org']:.4f})",
        f"(Max: {max_scores['edu']:.4f})",
        f"(Max: {max_scores['logistics']:.4f})",
        f"(Max: {max_scores['total']:.4f})"
    ))
    print("-" * 119)

    # Print candidates rows
    for cand_id in candidate_ids:
        s_score = skills.get(cand_id, 0.0)
        c_score = career.get(cand_id, 0.0)
        i_score = impact.get(cand_id, 0.0)
        o_score = org.get(cand_id, 0.0)
        e_score = edu.get(cand_id, 0.0)
        l_score = logistics.get(cand_id, 0.0)
        tot_score = s_score + c_score + i_score + o_score + e_score + l_score

        print(row_format.format(
            cand_id,
            s_score,
            c_score,
            i_score,
            o_score,
            e_score,
            l_score,
            tot_score
        ))

    print("=" * 119 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate candidate similarity comparison table.")
    parser.add_argument("jd_json", type=str, help="Path to raw JD JSON.")
    parser.add_argument("candidates_json", type=str, help="Path to raw candidates JSON.")
    parser.add_argument("jd_pkl", type=str, help="Path to JD embeddings pickle.")
    parser.add_argument("candidates_pkl", type=str, help="Path to candidates embeddings pickle.")
    args = parser.parse_args()

    compare_candidates(args.jd_json, args.candidates_json, args.jd_pkl, args.candidates_pkl)
