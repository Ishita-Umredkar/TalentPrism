import os
import csv
import json
import pickle
import math
import time
import re
import sys
import random
from pathlib import Path
import numpy as np

# Reconfigure stdout/stderr to use UTF-8 on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
CONSTRAINTS_FILE = ROOT / "stage2" / "outputs" / "extracted_constraints_v2.json"
CANDIDATES_EMBEDDED_FILE = ROOT / "stage2" / "outputs" / "candidates_100k_embedded.pkl"
EMBEDDED_CONSTRAINTS_FILE = ROOT / "stage2" / "outputs" / "embedded_constraints.pkl"

OUTPUT_OFFICIAL = ROOT / "stage2" / "outputs" / "top_100_submission_official.csv"
OUTPUT_WITH_SCORES = ROOT / "stage2" / "outputs" / "top_100_submission_with_scores.csv"

# ============================================================
# SIMILARITY NORMALIZATION
# ============================================================
def normalize_similarity(sim: float, min_val: float = 0.52, max_val: float = 0.68) -> float:
    if sim <= min_val:
        return 0.0
    if sim >= max_val:
        return 1.0
    return (sim - min_val) / (max_val - min_val)

# ============================================================
# SCORING ENGINE FUNCTIONS FROM FIT SCORE ENGINE
# ============================================================
def get_skill_multiplier(skill: dict) -> float:
    cached = skill.get("_multiplier")
    if cached is not None:
        return cached
    prof = skill.get("proficiency", "").lower()
    if "adv" in prof or "expert" in prof or "proficient" in prof or "lead" in prof:
        prof_mult = 1.0
    elif "inter" in prof:
        prof_mult = 0.8
    elif "begin" in prof:
        prof_mult = 0.5
    else:
        prof_mult = 0.8

    months = skill.get("duration_months", 0)
    if not isinstance(months, (int, float)):
        months = 0
    months_factor = min(1.0, max(0.0, months / 18.0))
    val = prof_mult * months_factor
    skill["_multiplier"] = val
    return val

def compute_skill_match_score(query_emb: np.ndarray, cand_skills: list) -> float:
    if not cand_skills:
        return 0.0
    max_sim = 0.0
    for sk in cand_skills:
        sk_emb = sk.get("embedding")
        if sk_emb is None:
            continue
        sim = np.dot(query_emb, sk_emb)
        sim_norm = normalize_similarity(sim, 0.60, 0.80)
        adjusted_sim = sim_norm * get_skill_multiplier(sk)
        if adjusted_sim > max_sim:
            max_sim = adjusted_sim
    return min(1.0, max(0.0, max_sim))

def compute_skill_aggregate_score(query_emb: np.ndarray, cand_skills: list) -> float:
    if not cand_skills:
        return 0.0
    total = 0.0
    valid_count = 0
    for sk in cand_skills:
        sk_emb = sk.get("embedding")
        if sk_emb is None:
            continue
        sim = np.dot(query_emb, sk_emb)
        sim_norm = normalize_similarity(sim, 0.60, 0.80)
        total += sim_norm
        valid_count += 1
    return (total / valid_count) if valid_count > 0 else 0.0

def check_consulting_company(company_name: str, cand_history: list) -> float:
    target = company_name.lower()
    for role in cand_history:
        comp = role.get("company", "").lower()
        if target in comp:
            return 1.0
    return 0.0

def is_entirely_consulting(cand: dict) -> float:
    history = cand.get("career_history", [])
    if not history:
        return 0.0
    consulting_keywords = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tech mahindra", "mindtree", "tata consultancy", "services"]
    for role in history:
        company = role.get("company", "").lower()
        industry = role.get("industry", "").lower()
        is_consulting = (
            any(k in company for k in consulting_keywords) or
            "service" in industry or "consult" in industry
        )
        if not is_consulting:
            return 0.0
    return 1.0

def has_product_experience(cand: dict) -> float:
    history = cand.get("career_history", [])
    consulting_keywords = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tech mahindra", "mindtree", "tata consultancy", "services"]
    for role in history:
        company = role.get("company", "").lower()
        industry = role.get("industry", "").lower()
        is_service = (
            any(k in company for k in consulting_keywords) or
            "service" in industry or "consult" in industry
        )
        if not is_service:
            return 1.0
    return 0.0

def startup_ratio(cand: dict) -> float:
    history = cand.get("career_history", [])
    if not history:
        return 0.0
    startup_months = 0
    total_months = 0
    for role in history:
        dur = role.get("duration_months", 0)
        total_months += dur
        size = role.get("company_size", "").lower()
        is_startup = (
            "1-10" in size or "11-50" in size or "51-200" in size or
            "1-50" in size or "201-500" in size or "under 500" in size
        )
        if is_startup:
            startup_months += dur
    return startup_months / total_months if total_months > 0 else 0.0

def recent_ai_experience_score(cand: dict) -> float:
    history = cand.get("career_history", [])
    ai_keywords = ["ai", "ml", "machine learning", "deep learning", "nlp", "computer vision", "llm", "large language model", "neural", "ranking", "retrieval", "search", "recommendation"]
    ai_months = 0
    for role in history:
        title = role.get("title", "").lower()
        desc = role.get("description_sentences", [])
        desc_text = " ".join([s.get("text", "") for s in desc]).lower()
        is_ai = any(k in title for k in ai_keywords) or any(k in desc_text for k in ai_keywords)
        if is_ai:
            ai_months += role.get("duration_months", 0)

    if ai_months <= 12:
        return 1.0
    elif ai_months >= 24:
        return 0.0
    else:
        return (24 - ai_months) / 12.0

def notice_period_score(cand: dict) -> float:
    np_days = cand.get("redrob_signals", {}).get("notice_period_days", 90)
    if np_days <= 30:
        return 1.0
    elif np_days <= 60:
        return 0.5
    else:
        return 0.0

def notice_period_30plus_score(cand: dict) -> float:
    np_days = cand.get("redrob_signals", {}).get("notice_period_days", 90)
    if np_days > 30:
        return 1.0
    else:
        return 0.0

def switching_companies_score(cand: dict) -> float:
    history = cand.get("career_history", [])
    if not history:
        return 0.0
    companies = set(role.get("company", "").strip().lower() for role in history if role.get("company"))
    num_companies = len(companies)
    if num_companies == 0:
        return 0.0
    total_months = sum(role.get("duration_months", 0) for role in history)
    avg_duration_years = (total_months / num_companies) / 12.0
    if avg_duration_years <= 1.5:
        return 1.0
    elif avg_duration_years >= 3.0:
        return 0.0
    else:
        return (3.0 - avg_duration_years) / 1.5

def closed_source_score(query_emb: np.ndarray, cand: dict) -> float:
    history = cand.get("career_history", [])
    matching_months = 0
    for role in history:
        role_emb = role.get("role_embedding")
        if role_emb is None:
            continue
        sim = np.dot(query_emb, role_emb)
        sim_norm = normalize_similarity(sim, 0.52, 0.68)
        if sim_norm > 0.5:
            matching_months += role.get("duration_months", 0)
    return min(1.0, matching_months / 60.0)

def evaluate_behavior_signal(category_name: str, cand: dict) -> float:
    signals = cand.get("redrob_signals", {})
    key = category_name.split(".")[-1]
    val = signals.get(key)
    if val is None:
        return 0.0

    if key == "profile_completeness_score":
        try:
            f_val = float(val)
            return max(0.0, min(1.0, f_val / 100.0)) if f_val >= 0 else 0.0
        except ValueError:
            return 0.0

    elif key == "open_to_work_flag":
        return 1.0 if val else 0.0

    elif key in ["profile_views_received_30d", "search_appearance_30d"]:
        try:
            f_val = float(val)
            if f_val < 0:
                return 0.0
            if f_val >= 500:
                return 1.0
            elif f_val >= 200:
                return 0.75 + (f_val - 200) / 300.0 * 0.25
            else:
                return 0.3 + (f_val / 200.0) * 0.45
        except ValueError:
            return 0.0

    elif key == "recruiter_response_rate":
        try:
            f_val = float(val)
            return max(0.0, min(1.0, 1.0 - f_val)) if f_val >= 0 else 1.0
        except ValueError:
            return 1.0

    elif key == "avg_response_time_hours":
        try:
            f_val = float(val)
            if f_val < 0:
                return 0.0
            if f_val <= 24:
                return 1.0
            elif f_val <= 48:
                return 1.0 - (f_val - 24) / 24.0 * 0.1
            else:
                return max(0.6, 0.9 - (f_val - 48) / 168.0 * 0.3)
        except ValueError:
            return 0.0

    elif key == "connection_count":
        try:
            f_val = float(val)
            if f_val < 0:
                return 0.0
            if f_val >= 500:
                return 1.0
            else:
                return 0.8 + (f_val / 500.0) * 0.2
        except ValueError:
            return 0.0

    elif key == "github_activity_score":
        try:
            f_val = float(val)
            return max(0.0, min(1.0, f_val / 100.0)) if f_val >= 0 else 0.0
        except ValueError:
            return 0.0

    elif key == "saved_by_recruiters_30d":
        try:
            f_val = float(val)
            if f_val < 0:
                return 0.0
            if f_val >= 5:
                return 1.0
            else:
                return 0.8 + (f_val / 5.0) * 0.2
        except ValueError:
            return 0.0

    elif key in ["interview_completion_rate", "offer_acceptance_rate"]:
        try:
            f_val = float(val)
            if f_val <= 0:
                return 0.0
            return f_val / 100.0 if f_val > 1.0 else f_val
        except ValueError:
            return 0.0

    elif key in ["verified_email", "verified_phone", "linkedin_connected"]:
        return 1.0 if val else 0.0

    if isinstance(val, (int, float)):
        return max(0.0, min(1.0, val)) if val >= 0 else 0.0
    return 1.0 if val else 0.0

def evaluate_category_score(
    query_emb: np.ndarray,
    cand: dict,
    category_name: str,
    strategy: str,
    sub_constraint_text: str,
    field_val: str = ""
) -> float:
    if strategy == "direct":
        if category_name == "redrob_signals.notice_period_days":
            np_days = cand.get("redrob_signals", {}).get("notice_period_days", 90)
            target_days = 30
            if field_val:
                try:
                    clean_days = "".join(c for c in str(field_val) if c.isdigit())
                    if clean_days:
                        target_days = int(clean_days)
                except ValueError:
                    pass
            is_negative_check = any(w in sub_constraint_text.lower() for w in ["greater", "above", "longer", "more than", ">", "exceed"])
            if is_negative_check:
                return 1.0 if np_days > target_days else 0.0
            else:
                if np_days <= target_days:
                    return 1.0
                elif np_days <= target_days * 2:
                    return 0.5
                else:
                    return 0.0

        if category_name == "redrob_signals.willing_to_relocate":
            expected = True
            if field_val:
                expected = (field_val.lower() == "true")
            return 1.0 if cand.get("redrob_signals", {}).get("willing_to_relocate", False) == expected else 0.0

        if category_name == "redrob_signals.open_to_work_flag":
            expected = True
            if field_val:
                expected = (field_val.lower() == "true")
            return 1.0 if cand.get("redrob_signals", {}).get("open_to_work_flag", False) == expected else 0.0

        if category_name == "profile.location":
            loc = cand.get("profile", {}).get("location", "").lower()
            willing = cand.get("redrob_signals", {}).get("willing_to_relocate", False)
            tier1_cities = ["mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata", "delhi", "gurgaon", "gurugram", "ghaziabad", "noida", "faridabad"]
            is_tier1 = any(c in loc for c in tier1_cities)

            if field_val:
                targets = [t.strip().lower() for t in field_val.split(",") if t.strip()]
            else:
                targets = ["pune", "noida"]

            def matches_any_target(candidate_loc, target_list):
                for t in target_list:
                    if t == "delhi ncr":
                        if any(ncr in candidate_loc for ncr in ["delhi", "noida", "gurgaon", "gurugram", "ghaziabad", "faridabad", "ncr"]):
                            return True
                    elif t in candidate_loc:
                        return True
                return False

            if matches_any_target(loc, targets):
                return 1.0
            if willing and is_tier1:
                return 1.0
            return 0.0

        if category_name == "profile.years_of_experience":
            yoe = cand.get("profile", {}).get("years_of_experience", 0.0)
            min_yoe, max_yoe = 5.0, 9.0
            if field_val:
                field_val_clean = str(field_val).strip()
                if "-" in field_val_clean:
                    parts = field_val_clean.split("-")
                    try:
                        min_yoe = float(parts[0].strip())
                        max_yoe = float(parts[1].strip())
                    except ValueError:
                        pass
                elif field_val_clean.endswith("+"):
                    try:
                        min_yoe = float(field_val_clean[:-1].strip())
                        max_yoe = 999.0
                    except ValueError:
                        pass
                else:
                    try:
                        min_yoe = float(field_val_clean)
                        max_yoe = float(field_val_clean)
                    except ValueError:
                        pass

            if min_yoe <= yoe <= max_yoe:
                return 1.0
            elif yoe < min_yoe:
                return (yoe / min_yoe) if min_yoe > 0 else 0.0
            else:
                return max(0.0, 1.0 - (yoe - max_yoe) / 5.0)

        if category_name == "career_history.company":
            for company_name in ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"]:
                if company_name.lower() in sub_constraint_text.lower():
                    return check_consulting_company(company_name, cand.get("career_history", []))

        if category_name == "career_history.industry":
            industry_text = cand.get("profile", {}).get("current_industry", "").lower()
            if "hr" in sub_constraint_text.lower() or "recruit" in sub_constraint_text.lower():
                if "hr" in industry_text or "recruit" in industry_text or "talent" in industry_text:
                    return 1.0
            if "marketplace" in sub_constraint_text.lower():
                if "market" in industry_text or "e-commerce" in industry_text:
                    return 1.0
            for role in cand.get("career_history", []):
                ind = role.get("industry", "").lower()
                desc = role.get("description_sentences", [])
                desc_text = " ".join([s.get("text", "") for s in desc]).lower()
                if "hr" in sub_constraint_text.lower() or "recruit" in sub_constraint_text.lower():
                    if "hr" in ind or "recruit" in ind or "talent" in ind or "recruiting" in desc_text or "hr-tech" in desc_text:
                        return 1.0
                if "marketplace" in sub_constraint_text.lower():
                    if "market" in ind or "e-commerce" in ind or "marketplace" in desc_text:
                        return 1.0
            return 0.0

        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name in ["career_history.title", "profile.current_title"]:
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name in ["career_history.company", "career_history.industry", "career_history.company_size",
                              "profile.current_company", "profile.current_industry", "profile.current_company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        signals = cand.get("redrob_signals", {})
        key = category_name.split(".")[-1]
        if key in signals:
            val = signals[key]
            if isinstance(val, (int, float)):
                if key == "profile_completeness_score":
                    return val / 100.0
                if key == "recruiter_response_rate":
                    return val
                if key == "search_appearance_30d" or key == "saved_by_recruiters_30d":
                    return min(1.0, val / 100.0)
            elif isinstance(val, bool):
                return 1.0 if val else 0.0
        return 0.0

    if strategy == "behavioral":
        return evaluate_behavior_signal(category_name, cand)

    if strategy == "max" and (category_name == "skills" or category_name == "skills.name"):
        return compute_skill_match_score(query_emb, cand.get("skills", []))

    if strategy == "max":
        if category_name == "career_history.description":
            max_sim = 0.0
            for role in cand.get("career_history", []):
                emb = role.get("role_embedding")
                if emb is not None:
                    sim = np.dot(query_emb, emb)
                    sim_norm = normalize_similarity(sim, 0.52, 0.68)
                    if sim_norm > max_sim:
                        max_sim = sim_norm
            return max_sim

        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name in ["career_history.title", "profile.current_title"]:
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name in ["career_history.company", "career_history.industry", "career_history.company_size",
                              "profile.current_company", "profile.current_industry", "profile.current_company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

    if strategy == "aggregate":
        if category_name == "skills.name" or category_name == "skills":
            return compute_skill_aggregate_score(query_emb, cand.get("skills", []))

        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name in ["career_history.title", "profile.current_title"]:
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name in ["career_history.company", "career_history.industry", "career_history.company_size",
                              "profile.current_company", "profile.current_industry", "profile.current_company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name == "career_history.duration_months":
            if "closed-source" in sub_constraint_text.lower():
                return closed_source_score(query_emb, cand)
            if "switching" in sub_constraint_text.lower() or "trajectory" in sub_constraint_text.lower():
                return switching_companies_score(cand)
            if "bounce" in sub_constraint_text.lower() or "startups" in sub_constraint_text.lower():
                return startup_ratio(cand)
            if "recent" in sub_constraint_text.lower() or "12 months" in sub_constraint_text.lower():
                return recent_ai_experience_score(cand)
            return 0.0

        if category_name == "career_history.company_size":
            if "startups" in sub_constraint_text.lower() or "bounce" in sub_constraint_text.lower():
                return startup_ratio(cand)
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name == "education.field_of_study" or category_name == "education":
            max_sim = 0.0
            for edu in cand.get("education", []):
                emb = edu.get("education_embedding")
                if emb is not None:
                    sim = np.dot(query_emb, emb)
                    sim_norm = normalize_similarity(sim, 0.52, 0.68)
                    if sim_norm > max_sim:
                        max_sim = sim_norm
            return max_sim

        if category_name == "career_history.title":
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name == "career_history.company":
            if "consulting" in sub_constraint_text.lower() or "service" in sub_constraint_text.lower():
                return is_entirely_consulting(cand)
            if "google" in sub_constraint_text.lower():
                return check_consulting_company("Google", cand.get("career_history", []))
            if "meta" in sub_constraint_text.lower():
                return check_consulting_company("Meta", cand.get("career_history", []))
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name == "career_history.industry":
            if "consulting" in sub_constraint_text.lower() or "service" in sub_constraint_text.lower():
                return is_entirely_consulting(cand)
            if "product" in sub_constraint_text.lower():
                return has_product_experience(cand)
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                return normalize_similarity(sim, 0.52, 0.68)
            return 0.0

        if category_name == "career_history.description":
            history = cand.get("career_history", [])
            if not history:
                return 0.0
            total_months = sum(r.get("duration_months", 0) for r in history)
            score_sum = 0.0
            for idx, role in enumerate(history):
                target_emb = role.get("role_embedding")
                if target_emb is None:
                    continue
                sim = np.dot(query_emb, target_emb)
                sim_norm = normalize_similarity(sim, 0.52, 0.68)
                dur = role.get("duration_months", 0)
                yoe_factor = (dur / total_months) if total_months > 0 else 0.0
                recency_factor = 0.8 ** idx
                score_sum += sim_norm * yoe_factor * recency_factor
            return score_sum

    return 0.0

def evaluate_section_score(
    section_name: str,
    constraints_list: list,
    cand: dict,
    query_embeddings_cache: dict
) -> float:
    total_weighted_score = 0.0
    section_total_weight = sum(c.get("weight", 0.0) for c in constraints_list)
    is_negative_section = section_name in ("negative", "rejection")

    if section_total_weight == 0.0:
        return 0.0

    for c in constraints_list:
        c_weight = c.get("weight", 0.0)
        sub_constraints = c.get("sub_constraints", [])

        if is_negative_section and c.get("type") == "conflicting":
            good_score = 0.0
            bad_score = 0.0
            for sub in sub_constraints:
                sub_item = sub.get("item", "")
                sub_type = sub.get("type", "bad")
                categories = sub.get("categories", [])
                query_emb = query_embeddings_cache[sub_item]

                cat_score_sum = 0.0
                cat_weight_sum = sum(cat.get("weight", 0.0) for cat in categories)

                for cat in categories:
                    cat_item = cat.get("item", "")
                    cat_strategy = cat.get("matching_strategy", "")
                    cat_weight = cat.get("weight", 0.0)
                    cat_field = cat.get("field", "")
                    cat_score = evaluate_category_score(
                        query_emb, cand, cat_item, cat_strategy, sub_item, cat_field
                    )
                    cat_score_sum += cat_score * cat_weight

                sub_score = (cat_score_sum / cat_weight_sum) if cat_weight_sum > 0 else 0.0
                if sub_type == "good":
                    good_score = sub_score
                else:
                    bad_score = sub_score
            constraint_score = bad_score * (1.0 - good_score)
        else:
            sub_total_score = 0.0
            for sub in sub_constraints:
                sub_weight = sub.get("weight", 0.0)
                sub_item = sub.get("item", "")
                sub_type = sub.get("type", "bad")
                categories = sub.get("categories", [])
                query_emb = query_embeddings_cache[sub_item]

                cat_score_sum = 0.0
                cat_weight_sum = sum(cat.get("weight", 0.0) for cat in categories)

                for cat in categories:
                    cat_item = cat.get("item", "")
                    cat_strategy = cat.get("matching_strategy", "")
                    cat_weight = cat.get("weight", 0.0)
                    cat_field = cat.get("field", "")
                    cat_score = evaluate_category_score(
                        query_emb, cand, cat_item, cat_strategy, sub_item, cat_field
                    )
                    cat_score_sum += cat_score * cat_weight

                sub_score = (cat_score_sum / cat_weight_sum) if cat_weight_sum > 0 else 0.0
                if is_negative_section and sub_type == "good":
                    sub_score = 1.0 - sub_score

                sub_total_score += sub_score * sub_weight
            constraint_score = sub_total_score
        
        total_weighted_score += constraint_score * c_weight

    return float(total_weighted_score / section_total_weight) if section_total_weight > 0 else 0.0

# ============================================================
# REASONING GENERATION LOGIC
# ============================================================
def clean_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip()

def find_matched_skills(cand) -> list[str]:
    matched = []
    skills_in_profile = [sk.get("name", "").lower() for sk in cand.get("skills", [])]
    career_texts = []
    for r in cand.get("career_history", []):
        career_texts.append(r.get("title", "").lower())
        career_texts.append(r.get("company", "").lower())
        career_texts.append(r.get("description", "").lower())
        for sent in r.get("description_sentences", []):
            career_texts.append(sent.get("text", "").lower())
    
    targets = {
        "Pinecone": ["pinecone"],
        "Weaviate": ["weaviate"],
        "Qdrant": ["qdrant"],
        "Milvus": ["milvus"],
        "FAISS": ["faiss"],
        "Elasticsearch": ["elasticsearch"],
        "OpenSearch": ["opensearch"],
        "Vector Search": ["vector database", "vector db", "vector search", "hybrid search"],
        "sentence-transformers": ["sentence-transformers", "sentence transformer"],
        "BGE": ["bge"],
        "E5": ["e5"],
        "OpenAI Embeddings": ["openai embedding"],
        "Embeddings": ["embedding-based", "embeddings"],
        "NDCG": ["ndcg"],
        "MRR": ["mrr"],
        "MAP": ["mean average precision", "map metric", "map score"],
        "A/B Testing": ["a/b test", "ab test", "ab-test"],
        "Ranking Evaluation": ["ranking evaluation", "ranking metrics", "evaluation framework"],
        "XGBoost": ["xgboost"],
        "LightGBM": ["lightgbm"],
        "LoRA/QLoRA": ["lora", "qlora"],
        "PEFT": ["peft"],
        "Python": ["python"],
        "Spark": ["spark"],
        "SQL": ["sql"]
    }
    
    for disp_name, kw_list in targets.items():
        if any(any(kw in sk_name for kw in kw_list) for sk_name in skills_in_profile):
            matched.append(disp_name)
        elif any(any(kw in text for kw in kw_list) for text in career_texts):
            matched.append(disp_name)
            
    return matched

def is_consulting(cand) -> bool:
    consulting_keywords = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tech mahindra", "mindtree", "tata consultancy"]
    history = cand.get("career_history", [])
    if not history:
        return False
    for r in history:
        comp = r.get("company", "").lower()
        if any(k in comp for k in consulting_keywords):
            return True
    return False

def is_job_hopper(cand) -> bool:
    history = cand.get("career_history", [])
    if not history:
        return False
    companies = set(r.get("company", "").strip().lower() for r in history if r.get("company"))
    if not companies:
        return False
    total_months = sum(r.get("duration_months", 0) for r in history)
    avg_years = (total_months / len(companies)) / 12.0
    return avg_years <= 1.5

def generate_candidate_reasoning(cand, rank: int) -> str:
    profile = cand.get("profile", {})
    name = clean_str(profile.get("anonymized_name", "Candidate"))
    title = clean_str(profile.get("current_title", "Software Engineer"))
    if not title:
        title = clean_str(profile.get("headline", "Applied Engineer")).split("|")[0].strip()
    yoe = profile.get("years_of_experience", 0.0)
    loc = clean_str(profile.get("location", ""))
    if "," in loc:
        loc = loc.split(",")[0].strip()
    
    signals = cand.get("redrob_signals", {})
    np_days = signals.get("notice_period_days", 90)
    willing_relocate = signals.get("willing_to_relocate", False)
    
    matched_skills = find_matched_skills(cand)
    consulting_flag = is_consulting(cand)
    hopper_flag = is_job_hopper(cand)
    
    cand_id = cand.get("candidate_id", "CAND_0000000")
    cand_num = int(re.sub(r'\D', '', cand_id))
    rng = random.Random(cand_num)
    
    # --- Sentence 1: Introduction and Tone ---
    intro = ""
    if rank <= 10:
        options = [
            f"Exceptional alignment at Rank {rank}: {name} is a {title} with {yoe} YOE, offering premium expertise in retrieval and ranking systems.",
            f"Top-tier candidate at Rank {rank}: A seasoned {title} with {yoe} YOE, showing strong specialization in search architectures.",
            f"Excellent fit at Rank {rank}: {name} ({title}, {yoe} YOE) represents a high-caliber match for applied ML and ranking roles."
        ]
        intro = rng.choice(options)
    elif rank <= 40:
        options = [
            f"Rank {rank}: Strong candidate with {yoe} YOE as a {title}, showing solid software engineering and applied ML capabilities.",
            f"Rank {rank}: {name} ({title}) brings {yoe} years of experience in technical data systems and ML applications.",
            f"Rank {rank}: High-potential {title} with {yoe} YOE, displaying good general alignment with the engineering stack."
        ]
        intro = rng.choice(options)
    elif rank <= 70:
        options = [
            f"Rank {rank}: Capable professional with {yoe} YOE as a {title}, displaying adjacent software development capabilities.",
            f"Rank {rank}: {name} ({title}, {yoe} YOE) presents general software engineering experience with some data/analytics exposure.",
            f"Rank {rank}: Technical profile ({title}, {yoe} YOE) with general engineering skills, acting as a moderate fit."
        ]
        intro = rng.choice(options)
    else:
        options = [
            f"Rank {rank}: Adjacent profile ({title}, {yoe} YOE) included as a lower-tier filler candidate based on general experience.",
            f"Rank {rank}: {name} ({title}) has {yoe} YOE in non-ML engineering/operations, showing minimal direct alignment.",
            f"Rank {rank}: Included at Rank {rank} ({title}, {yoe} YOE); technical skills are adjacent and lack specialized search experience."
        ]
        intro = rng.choice(options)

    # --- Sentence 2: Skills & JD Connection ---
    skills_sentence = ""
    if matched_skills:
        display_skills = matched_skills[:4]
        skills_str = ", ".join(display_skills)
        options = [
            f"Demonstrates production hands-on experience in {skills_str}, which matches critical JD requirements.",
            f"Technical profile highlights practical experience in {skills_str} for vector search and data processing.",
            f"Equipped with relevant skills like {skills_str}, supporting the search and retrieval requirements.",
            f"Proven competence with key technologies including {skills_str} in past positions."
        ]
        skills_sentence = rng.choice(options)
    else:
        options = [
            "Technical stack consists of general software development without direct ML retrieval or vector search skills.",
            "Lacks specific experience in vector databases, hybrid search, or ranking evaluation metrics.",
            "Background is focused on standard backend engineering, lacking direct relevance to the search/ranking stack."
        ]
        skills_sentence = rng.choice(options)

    # --- Sentence 3: Logistics & Concerns (Honest Concerns) ---
    logistics = []
    if np_days <= 30:
        logistics.append(rng.choice([f"available immediately ({np_days}-day notice)", f"has a short notice period of {np_days} days"]))
    elif np_days <= 60:
        logistics.append(f"available on a {np_days}-day notice")
    else:
        logistics.append(f"note: holds a longer notice period of {np_days} days")
        
    if loc:
        if willing_relocate:
            logistics.append(rng.choice([f"located in {loc} and open to relocate", f"willing to relocate from {loc}"]))
        else:
            logistics.append(f"based in {loc}")

    concerns = []
    if consulting_flag:
        concerns.append("primarily IT consulting/services background")
    if hopper_flag:
        concerns.append("potential tenure concern due to frequent job changes")
    if yoe < 5.0 and rank <= 40:
        concerns.append("experience is slightly under the preferred 5-year threshold")
        
    logistics_str = ", ".join(logistics)
    logistics_sentence = f"Candidate is {logistics_str}."
    
    if concerns:
        concern_str = " However, " + " and ".join(concerns) + "."
        logistics_sentence += concern_str

    sentence_1 = f"{intro} {skills_sentence}"
    sentence_2 = logistics_sentence
    full_reasoning = f"{sentence_1} {sentence_2}"
    full_reasoning = re.sub(r'\s+', ' ', full_reasoning)
    full_reasoning = full_reasoning.replace("..", ".").replace(" ,", ",").strip()
    
    return full_reasoning

# ============================================================
# MAIN PIPELINE USING PRE-EMBEDDED CANDIDATES
# ============================================================
def main():
    print("="*80)
    print("TALENTPRISM - 100K CANDIDATES SCORING & TOP 100 RANKING PIPELINE")
    print("="*80)
    
    start_time = time.time()
    
    print(f"Loading pre-embedded candidates from: {CANDIDATES_EMBEDDED_FILE}")
    if not CANDIDATES_EMBEDDED_FILE.exists():
        print(f"Error: Embedded file {CANDIDATES_EMBEDDED_FILE} not found. Run generate_100k_embeddings.py first.")
        return
        
    with open(CANDIDATES_EMBEDDED_FILE, "rb") as f:
        embedded_candidates = pickle.load(f)
    print(f"Loaded {len(embedded_candidates)} embedded candidates in {time.time() - start_time:.2f}s.")
    
    # 1. Load pre-embedded constraints from pickle file
    print(f"Loading pre-embedded constraints from: {EMBEDDED_CONSTRAINTS_FILE}")
    if not EMBEDDED_CONSTRAINTS_FILE.exists():
        raise FileNotFoundError(f"Embedded constraints file not found at {EMBEDDED_CONSTRAINTS_FILE}. Run generate_constraint_embeddings.py first.")
    
    with open(EMBEDDED_CONSTRAINTS_FILE, "rb") as f:
        query_embeddings_cache = pickle.load(f)
    print("Query cache loaded.")
    
    print("Loading constraints structure...")
    with open(CONSTRAINTS_FILE, "r", encoding="utf-8") as f:
        constraints_data = json.load(f)

    # 2. Score all candidates
    print("Evaluating fit scores for all 100k candidates on CPU...")
    candidate_scores = []
    
    last_print = time.time()
    for idx, ec in enumerate(embedded_candidates, 1):
        cand_id = ec["candidate_id"]
        
        # Periodic progress prints
        if idx % 10000 == 0 or idx == len(embedded_candidates):
            elapsed = time.time() - last_print
            print(f"Scored {idx}/{len(embedded_candidates)} candidates... (+{elapsed:.2f}s)")
            last_print = time.time()
        
        s_must = evaluate_section_score("must_have", constraints_data.get("must_have", []), ec, query_embeddings_cache)
        s_pref = evaluate_section_score("preferred", constraints_data.get("preferred", []), ec, query_embeddings_cache)
        s_rej = evaluate_section_score("rejection", constraints_data.get("rejection", []), ec, query_embeddings_cache)
        s_neg = evaluate_section_score("negative", constraints_data.get("negative", []), ec, query_embeddings_cache)

        # Combined raw positive and negative scores
        positive_score = 0.75 * s_must + 0.25 * s_pref
        negative_score = 0.75 * s_rej + 0.25 * s_neg
        final_fit_score = positive_score * (1.0 - negative_score)
        
        candidate_scores.append({
            "candidate_id": cand_id,
            "must": s_must,
            "pref": s_pref,
            "rej": s_rej,
            "neg": s_neg,
            "final": final_fit_score,
            "candidate_ref": ec
        })
        
    print(f"Finished scoring all candidates in {(time.time() - start_time)/60:.2f} mins.")
    
    # Sort candidates by final score descending, and candidate_id ascending for deterministic tie breaking
    print("Sorting rankings...")
    candidate_scores.sort(key=lambda x: (-x["final"], x["candidate_id"]))
    
    # 3. Process Top 100 and generate reasonings
    print("Processing top 100 and generating spec-compliant reasonings...")
    top_100_results = []
    for rank_idx, cs in enumerate(candidate_scores[:100], 1):
        cand = cs["candidate_ref"]
        reasoning = generate_candidate_reasoning(cand, rank_idx)
        top_100_results.append({
            "candidate_id": cs["candidate_id"],
            "rank": rank_idx,
            "score": cs["final"],
            "reasoning": reasoning,
            "must_have_raw": cs["must"],
            "preferred_raw": cs["pref"],
            "rejection_raw": cs["rej"],
            "negative_raw": cs["neg"],
            "final_score_raw": cs["final"]
        })

    # ============================================================
    # SAVE CSV 1: OFFICIAL SUBMISSION FORMAT
    # ============================================================
    print(f"Saving official submission format to: {OUTPUT_OFFICIAL}")
    os.makedirs(os.path.dirname(OUTPUT_OFFICIAL), exist_ok=True)
    with open(OUTPUT_OFFICIAL, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in top_100_results:
            writer.writerow([
                row["candidate_id"],
                row["rank"],
                f"{row['score']:.4f}",
                row["reasoning"]
            ])

    # ============================================================
    # SAVE CSV 2: WITH DETAILED RAW SCORES
    # ============================================================
    print(f"Saving detailed scores format to: {OUTPUT_WITH_SCORES}")
    with open(OUTPUT_WITH_SCORES, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "candidate_id", "rank", "score", "reasoning", 
            "must_have_raw", "preferred_raw", "rejection_raw", "negative_raw", "final_score_raw"
        ])
        for row in top_100_results:
            writer.writerow([
                row["candidate_id"],
                row["rank"],
                f"{row['score']:.4f}",
                row["reasoning"],
                f"{row['must_have_raw']:.4f}",
                f"{row['preferred_raw']:.4f}",
                f"{row['rejection_raw']:.4f}",
                f"{row['negative_raw']:.4f}",
                f"{row['final_score_raw']:.4f}"
            ])
            
    print("\n" + "="*80)
    print("TOP 10 CANDIDATES SCORED ON 100K DATASET")
    print("="*80)
    print(f"{'Rank':<5} | {'Candidate ID':<13} | {'Must':<6} | {'Pref':<6} | {'Rej':<6} | {'Neg':<6} | {'Final Score':<11}")
    print("-"*80)
    for row in top_100_results[:10]:
        print(f"{row['rank']:<5} | {row['candidate_id']:<13} | {row['must_have_raw']:.4f} | {row['preferred_raw']:.4f} | {row['rejection_raw']:.4f} | {row['negative_raw']:.4f} | {row['score']:.4f}")
    print("="*80)
    print(f"Pipeline completed successfully. Total duration: {(time.time() - start_time)/60:.2f} mins.")

if __name__ == "__main__":
    main()
