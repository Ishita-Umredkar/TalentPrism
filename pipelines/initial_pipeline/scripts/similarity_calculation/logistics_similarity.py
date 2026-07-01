import json
import pickle
import re
from typing import Dict, Any


def compare_location(jd_loc: str, cand_loc: str, jd_willing: bool, cand_willing: bool) -> float:
    """Compare locations based on candidate's city/country and relocation willingness."""
    if not jd_loc:
        return 1.0

    cand_loc_clean = cand_loc.lower().strip()
    jd_loc_clean = jd_loc.lower().strip()

    # Split the JD locations by delimiters (e.g., slash, comma, semicolon)
    jd_cities = [c.strip() for c in re.split(r'[/,;]', jd_loc_clean) if c.strip()]
    
    # 1. Direct match: check if candidate's location matches any JD cities/countries
    for city in jd_cities:
        if city in cand_loc_clean or cand_loc_clean in city:
            return 1.0

    # 2. Relocation match: if candidate is willing to relocate, it's a match
    if cand_willing:
        return 1.0

    # Otherwise, no match
    return 0.0


def compare_work_mode(jd_mode: str, cand_mode: str) -> float:
    """Compare preferred work modes (onsite, hybrid, remote, flexible)."""
    if not jd_mode or not cand_mode:
        return 1.0

    jd = jd_mode.lower().strip()
    cand = cand_mode.lower().strip()

    if jd == cand or jd == "flexible" or cand == "flexible" or jd == "any" or cand == "any":
        return 1.0

    if jd == "hybrid":
        if cand in ["onsite", "remote"]:
            return 0.5
    if jd == "onsite" and cand == "hybrid":
        return 0.5
    if jd == "remote" and cand == "hybrid":
        return 0.5

    return 0.0


def compare_notice_period(jd_notice: int, cand_notice: int) -> float:
    """Compare notice periods. Lower or equal is perfect (1.0). Higher decays the score."""
    if jd_notice <= 0 or cand_notice <= 0:
        return 1.0

    if cand_notice <= jd_notice:
        return 1.0

    return float(jd_notice / cand_notice)


def compare_salary(jd_salary: Any, cand_salary: Any) -> float:
    """Compare salary expectations if specified. Overlapping ranges is perfect."""
    if not jd_salary or not cand_salary:
        return 1.0

    # If they are dictionaries with min/max values
    if isinstance(jd_salary, dict) and isinstance(cand_salary, dict):
        jd_min = jd_salary.get("min_lpa", 0.0)
        jd_max = jd_salary.get("max_lpa", 0.0)
        cand_min = cand_salary.get("min_lpa", 0.0)
        
        # If JD has no salary specified
        if jd_max == 0.0:
            return 1.0

        # If candidate min is below or equal to JD max
        if cand_min <= jd_max:
            return 1.0

        # Decay if candidate wants more than JD max
        return float(jd_max / cand_min)

    return 1.0


def calculate_logistics_similarity(
    jd_json_path: str,
    candidates_json_path: str,
    jd_pkl_path: str,
    candidates_pkl_path: str
) -> Dict[str, float]:
    """Calculates logistics similarity scores for all candidates.

    Compares location (and relocation willingness), work mode, notice period, and salary.
    Excludes unspecified criteria from the average, then multiplies by logistics weight.

    Args:
        jd_json_path: Path to raw job description JSON.
        candidates_json_path: Path to raw candidates JSON.
        jd_pkl_path: Path to job description pickle (not directly used but kept for consistency).
        candidates_pkl_path: Path to candidates pickle (not directly used but kept for consistency).

    Returns:
        A dictionary mapping candidate_id to their logistics similarity score.
    """
    with open(jd_json_path, "r", encoding="utf-8") as f:
        jd_schema = json.load(f)

    with open(candidates_json_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    jd_logistics = jd_schema.get("logistics", {})
    jd_weight = jd_logistics.get("weight", 1.0)

    # Extract JD logistics details
    jd_loc = jd_logistics.get("location", "")
    jd_mode = jd_logistics.get("preferred_work_mode", "")
    jd_willing = jd_logistics.get("willing_to_relocate", False)
    jd_notice = jd_logistics.get("notice_period_days", 0)
    jd_salary = jd_logistics.get("salary_expectation", "")

    results = {}
    for cand_schema in candidates:
        cand_id = cand_schema.get("candidate_id")
        if not cand_id:
            continue

        cand_logistics = cand_schema.get("logistics", {})
        cand_loc = cand_logistics.get("location", "")
        cand_mode = cand_logistics.get("preferred_work_mode", "")
        cand_willing = cand_logistics.get("willing_to_relocate", False)
        cand_notice = cand_logistics.get("notice_period_days", 0)
        cand_salary = cand_logistics.get("salary_expectation", {})

        scores = []

        # 1. Location / Relocation
        if jd_loc:
            scores.append(compare_location(jd_loc, cand_loc, jd_willing, cand_willing))

        # 2. Preferred Work Mode
        if jd_mode:
            scores.append(compare_work_mode(jd_mode, cand_mode))

        # 3. Notice Period
        if jd_notice > 0:
            scores.append(compare_notice_period(jd_notice, cand_notice))

        # 4. Salary Expectation
        if jd_salary:
            scores.append(compare_salary(jd_salary, cand_salary))

        # If no logistics fields were defined in JD, default to 1.0
        avg_score = sum(scores) / len(scores) if scores else 1.0

        results[cand_id] = avg_score * jd_weight

    return results
