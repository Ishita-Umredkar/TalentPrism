import os
import sys
import json
import pickle
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
CANDIDATES_EMBEDDED_FILE = ROOT / "stage2" / "outputs" / "test_candidates_embedded.pkl"
EMBEDDED_CONSTRAINTS_FILE = ROOT / "stage2" / "outputs" / "embedded_constraints.pkl"

def clean_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip()

# ============================================================
# SIMILARITY NORMALIZATION
# ============================================================
def normalize_similarity(sim: float, min_val: float = 0.52, max_val: float = 0.68) -> float:
    if sim <= min_val:
        return 0.0
    if sim >= max_val:
        return 1.0
    return (sim - min_val) / (max_val - min_val)

def get_skill_multiplier(skill: dict) -> float:
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
    return prof_mult * months_factor

# ============================================================
# SCORING ENGINE HELPER FUNCTIONS
# ============================================================
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

# ============================================================
# EVALUATE DETAIL HELPER
# ============================================================
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
            # Invert the rate since the sub-constraint is "maintains a very low recruiter response rate"
            # (i.e. high response rate = 0 match with the bad behavior = 0 penalty).
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


def evaluate_details(query_emb, cand, category_name, strategy, sub_constraint_text, field_val="", section_name=""):
    """Returns (matched_item_text, raw_score_str_or_float, normalized_score_float)"""
    # 1. DIRECT / FIELD MATCH
    # Strategy "direct" covers both structured field matching (location, YOE, notice period)
    # and embedding similarity on profile/title/company fields.
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
                score = 1.0 if np_days > target_days else 0.0
                return f"{np_days} days (Target: >{target_days} days)", np_days, score
            else:
                if np_days <= target_days:
                    score = 1.0
                elif np_days <= target_days * 2:
                    score = 0.5
                else:
                    score = 0.0
                return f"{np_days} days (Target: <={target_days} days)", np_days, score

        if category_name == "redrob_signals.willing_to_relocate":
            expected = True
            if field_val:
                expected = (field_val.lower() == "true")
            val = cand.get("redrob_signals", {}).get("willing_to_relocate", False)
            score = 1.0 if val == expected else 0.0
            return f"Willing: {val} (Target: {expected})", 1.0 if val else 0.0, score

        if category_name == "redrob_signals.open_to_work_flag":
            expected = True
            if field_val:
                expected = (field_val.lower() == "true")
            val = cand.get("redrob_signals", {}).get("open_to_work_flag", False)
            score = 1.0 if val == expected else 0.0
            return f"Open to work: {val} (Target: {expected})", 1.0 if val else 0.0, score

        if category_name == "profile.location":
            loc = cand.get("profile", {}).get("location", "")
            loc_lower = loc.lower()
            willing = cand.get("redrob_signals", {}).get("willing_to_relocate", False)
            tier1_cities = ["mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata", "delhi", "gurgaon", "gurugram", "ghaziabad", "noida", "faridabad"]
            is_tier1 = any(c in loc_lower for c in tier1_cities)

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

            if matches_any_target(loc_lower, targets):
                score = 1.0
            elif willing and is_tier1:
                score = 1.0
            else:
                score = 0.0
            return f"'{loc}', relocate={willing} (Target: {field_val if field_val else 'Pune, Noida'})", score, score

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
                score = 1.0
            elif yoe < min_yoe:
                score = (yoe / min_yoe) if min_yoe > 0 else 0.0
            else:
                score = max(0.0, 1.0 - (yoe - max_yoe) / 5.0)
            return f"{yoe:.1f} YOE (Target: {field_val if field_val else '5-9'})", yoe, score

        if category_name == "career_history.company":
            history = cand.get("career_history", [])
            for company_name in ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"]:
                if company_name.lower() in sub_constraint_text.lower():
                    score = check_consulting_company(company_name, history)
                    return f"Consulting firm check ({company_name})", score, score

        if category_name == "career_history.industry":
            industry_text = cand.get("profile", {}).get("current_industry", "").lower()
            score = 0.0
            if "hr" in sub_constraint_text.lower() or "recruit" in sub_constraint_text.lower():
                if "hr" in industry_text or "recruit" in industry_text or "talent" in industry_text:
                    score = 1.0
            if "marketplace" in sub_constraint_text.lower():
                if "market" in industry_text or "e-commerce" in industry_text:
                    score = 1.0
            for role in cand.get("career_history", []):
                ind = role.get("industry", "").lower()
                desc = role.get("description_sentences", [])
                desc_text = " ".join([s.get("text", "") for s in desc]).lower()
                if "hr" in sub_constraint_text.lower() or "recruit" in sub_constraint_text.lower():
                    if "hr" in ind or "recruit" in ind or "talent" in ind or "recruiting" in desc_text:
                        score = 1.0
                if "marketplace" in sub_constraint_text.lower():
                    if "market" in ind or "e-commerce" in ind or "marketplace" in desc_text:
                        score = 1.0
            return f"Industry matches for '{sub_constraint_text}'", score, score

        # Embedding-based direct matching for profile/history fields
        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return "profile.summary embedding", sim, score

        if category_name in ["career_history.title", "profile.current_title"]:
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Titles chronological", sim, score

        if category_name in ["career_history.company", "career_history.industry", "career_history.company_size",
                              "profile.current_company", "profile.current_industry", "profile.current_company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Comp/Ind/Size chronological", sim, score

        signals = cand.get("redrob_signals", {})
        key = category_name.split(".")[-1]
        val = signals.get(key, 0.0)
        if key == "profile_completeness_score": score = val / 100.0
        elif key == "recruiter_response_rate": score = val
        elif key in ["search_appearance_30d", "saved_by_recruiters_30d"]: score = min(1.0, val / 100.0)
        else: score = 1.0 if val else 0.0
        return f"{key}: {val}", val, score

    # 2. BEHAVIORAL SIGNAL
    if strategy == "behavioral":
        score = evaluate_behavior_signal(category_name, cand)
        raw_val = cand.get("redrob_signals", {}).get(category_name.split(".")[-1], -1)
        return f"{category_name.split('.')[-1]}: {raw_val}", raw_val, score

    # Helper variables for chronological values
    history = cand.get("career_history", [])
    chronological_roles = history[::-1]

    def format_title_role_ec(r, is_current):
        title = clean_str(r.get("title", ""))
        if not title:
            return ""
        dur_m = r.get("duration_months", 0)
        if not isinstance(dur_m, (int, float)):
            dur_m = 0
        dur_y = dur_m / 12.0
        rounded = round(dur_y, 1)
        if rounded == 1.0:
            yoe_str = "1 year"
        elif rounded.is_integer():
            yoe_str = f"{int(rounded)} years"
        else:
            yoe_str = f"{rounded} years"
        if is_current:
            a_an = "an" if title[0].lower() in "aeiou" else "a"
            prefix = f"Currently {a_an}"
        else:
            prefix = "Worked as"
        return f"{prefix} {title} for {yoe_str}"

    def format_company_role_ec(r, is_current):
        company = clean_str(r.get("company", ""))
        industry = clean_str(r.get("industry", ""))
        size = clean_str(r.get("company_size", ""))
        if not company and not industry and not size:
            return ""
        prefix = "Currently at" if is_current else "Worked at"
        desc_parts = []
        if industry:
            a_an = "an" if industry[0].lower() in "aeiou" else "a"
            desc_parts.append(f"{a_an} {industry} company")
        if size:
            desc_parts.append(f"with {size} employees")
        if company:
            comp_str = f"{prefix} {company}"
            if desc_parts:
                return f"{comp_str}, {', '.join(desc_parts)}"
            return comp_str
        else:
            if desc_parts:
                return f"{prefix.lower()} {', '.join(desc_parts)}"
            return ""

    titles_with_dur = []
    for r in chronological_roles:
        is_curr = r.get("is_current", False)
        t = format_title_role_ec(r, is_curr)
        if t:
            titles_with_dur.append(t)
    titles_text = "->".join(titles_with_dur) if titles_with_dur else ""

    comp_ind_size_blocks = []
    for r in chronological_roles:
        is_curr = r.get("is_current", False)
        comp_str = format_company_role_ec(r, is_curr)
        if comp_str:
            comp_ind_size_blocks.append(comp_str)
    comp_ind_size_text = "->".join(comp_ind_size_blocks)

    # These strategy names are preserved for any legacy callers but "direct" is the
    # canonical name in v2. The blocks above already handle "direct".
    if strategy in ("direct_match",):
        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return "profile.summary embedding", sim, score
        if category_name in ["career_history.title", "profile.current_title"]:
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Titles chronological: {titles_text}", sim, score
        if category_name in ["career_history.company", "career_history.industry", "career_history.company_size",
                              "profile.current_company", "profile.current_industry", "profile.current_company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Comp/Ind/Size chronological: {comp_ind_size_text}", sim, score
        return "No direct match embedding", 0.0, 0.0

    if strategy == "max" and (category_name == "skills" or category_name == "skills.name"):
        cand_skills = cand.get("skills", [])
        if not cand_skills:
            return "No skills listed", 0.0, 0.0
        max_sim = 0.0
        best_skill = None
        best_raw_sim = 0.0
        best_adjusted = 0.0
        for sk in cand_skills:
            sk_emb = sk.get("embedding")
            if sk_emb is None:
                continue
            sim = np.dot(query_emb, sk_emb)
            # Skill Match uses 0.60 to 0.80 similarity bounds
            sim_norm = normalize_similarity(sim, 0.60, 0.80)
            mult = get_skill_multiplier(sk)
            adjusted = sim_norm * mult
            if adjusted > max_sim or best_skill is None:
                max_sim = adjusted
                best_skill = sk.get("name")
                best_raw_sim = sim
                best_adjusted = adjusted
        if best_skill:
            return f"Skill: '{best_skill}'", best_raw_sim, best_adjusted
        return "No matching skill embeddings", 0.0, 0.0

    if strategy == "max":
        if category_name == "career_history.description":
            max_sim = 0.0
            best_role = ""
            best_raw_sim = 0.0
            for role in cand.get("career_history", []):
                emb = role.get("role_embedding")
                if emb is not None:
                    sim = np.dot(query_emb, emb)
                    sim_norm = normalize_similarity(sim, 0.52, 0.68)
                    if sim_norm > max_sim or best_role == "":
                        max_sim = sim_norm
                        best_raw_sim = sim
                        best_role = f"{role.get('title')} at {role.get('company')}"
            return f"Role block {best_role}", best_raw_sim, max_sim

        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return "profile.summary embedding", sim, score
            return "No summary embedding", 0.0, 0.0

        if category_name in ["career_history.title", "profile.current_title"]:
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Titles chronological: {titles_text}", sim, score
            return "No titles chronological embedding", 0.0, 0.0

        if category_name in ["career_history.company", "career_history.industry", "career_history.company_size",
                              "profile.current_company", "profile.current_industry", "profile.current_company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Comp/Ind/Size chronological: {comp_ind_size_text}", sim, score
            return "No chronological embedding", 0.0, 0.0

    if strategy == "aggregate":
        if category_name == "skills.name" or category_name == "skills":
            cand_skills = cand.get("skills", [])
            if not cand_skills:
                return "No skills listed", 0.0, 0.0
            matching_details = []
            total_score = 0.0
            raw_sum = 0.0
            valid_count = 0
            for sk in cand_skills:
                sk_emb = sk.get("embedding")
                if sk_emb is None:
                    continue
                sim = np.dot(query_emb, sk_emb)
                sim_norm = normalize_similarity(sim, 0.60, 0.80)
                total_score += sim_norm
                raw_sum += sim
                valid_count += 1
                if sim_norm > 0.0:
                    matching_details.append(f"'{sk.get('name')}' ({sim_norm:.4f})")
            if valid_count > 0:
                avg_raw = raw_sum / valid_count
                avg_norm = total_score / valid_count
                if matching_details:
                    matched_str = "Skills: " + ", ".join(matching_details)
                else:
                    matched_str = "No matching skills"
                return matched_str, avg_raw, avg_norm
            return "No matching skill embeddings", 0.0, 0.0

        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return "profile.summary embedding", sim, score
            return "No summary embedding", 0.0, 0.0

        if category_name == "profile.current_title":
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Titles chronological: {titles_text}", sim, score
            return "No titles chronological embedding", 0.0, 0.0

        if category_name in ["profile.current_company", "profile.current_industry", "profile.current_company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Comp/Ind/Size chronological: {comp_ind_size_text}", sim, score
            return "No chronological embedding", 0.0, 0.0

        if category_name == "career_history.duration_months":
            if "closed-source" in sub_constraint_text.lower():
                score = closed_source_score(query_emb, cand)
                return "Closed-source roles aggregate", score, score
            if "switching" in sub_constraint_text.lower() or "trajectory" in sub_constraint_text.lower():
                score = switching_companies_score(cand)
                return "Switching frequency check", score, score
            if "bounce" in sub_constraint_text.lower() or "startups" in sub_constraint_text.lower():
                score = startup_ratio(cand)
                return "Startup ratio check", score, score
            if "recent" in sub_constraint_text.lower() or "12 months" in sub_constraint_text.lower():
                score = recent_ai_experience_score(cand)
                return "Recent AI/ML experience check", score, score
            return "Duration check", 0.0, 0.0

        if category_name == "career_history.company_size":
            if "startups" in sub_constraint_text.lower() or "bounce" in sub_constraint_text.lower():
                score = startup_ratio(cand)
                return "Startup size ratio check", score, score
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Comp/Ind/Size chronological: {comp_ind_size_text}", sim, score
            return "Company size check", 0.0, 0.0

        if category_name == "education.field_of_study" or category_name == "education":
            max_sim = 0.0
            best_edu = ""
            best_raw_sim = 0.0
            for edu in cand.get("education", []):
                emb = edu.get("education_embedding")
                if emb is not None:
                    sim = np.dot(query_emb, emb)
                    sim_norm = normalize_similarity(sim, 0.52, 0.68)
                    if sim_norm > max_sim or best_edu == "":
                        max_sim = sim_norm
                        best_raw_sim = sim
                        best_edu = f"{edu.get('degree')} at {edu.get('school', edu.get('institution'))}"
            return f"Edu: {best_edu}", best_raw_sim, max_sim

        if category_name == "career_history.title":
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Titles chronological: {titles_text}", sim, score
            return "No titles chronological embedding", 0.0, 0.0

        if category_name == "career_history.company":
            if "consulting" in sub_constraint_text.lower() or "service" in sub_constraint_text.lower():
                score = is_entirely_consulting(cand)
                return "IT consulting firm check across history", score, score
            if "google" in sub_constraint_text.lower():
                score = check_consulting_company("Google", history)
                return "Worked at Google across history", score, score
            if "meta" in sub_constraint_text.lower():
                score = check_consulting_company("Meta", history)
                return "Worked at Meta across history", score, score
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Comp/Ind/Size chronological: {comp_ind_size_text}", sim, score
            return "No chronological embedding", 0.0, 0.0

        if category_name == "career_history.industry":
            if "consulting" in sub_constraint_text.lower() or "service" in sub_constraint_text.lower():
                score = is_entirely_consulting(cand)
                return "IT services industry check across history", score, score
            if "product" in sub_constraint_text.lower():
                score = has_product_experience(cand)
                return "Product experience check across history", score, score
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Comp/Ind/Size chronological: {comp_ind_size_text}", sim, score
            return "No chronological embedding", 0.0, 0.0

        if category_name == "career_history.description":
            history = cand.get("career_history", [])
            if not history:
                return "No career history", 0.0, 0.0

            total_months = sum(r.get("duration_months", 0) for r in history)
            score_sum = 0.0
            details = []
            best_raw_sim = 0.0

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
                details.append(role.get("company", ""))
                if sim > best_raw_sim:
                    best_raw_sim = sim

            role_names = ", ".join(details)
            return f"Roles: {role_names}", best_raw_sim, score_sum

    return "Unknown strategy", 0.0, 0.0


# ============================================================
# MAIN EVALUATION BREAKDOWN
# ============================================================
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

                    _, _, cat_score = evaluate_details(
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
                sub_type = sub.get("type", "bad")  # v2 field: "good" or "bad"
                categories = sub.get("categories", [])

                query_emb = query_embeddings_cache[sub_item]

                cat_score_sum = 0.0
                cat_weight_sum = sum(cat.get("weight", 0.0) for cat in categories)

                for cat in categories:
                    cat_item = cat.get("item", "")
                    cat_strategy = cat.get("matching_strategy", "")
                    cat_weight = cat.get("weight", 0.0)
                    cat_field = cat.get("field", "")

                    # Fetch matching score using hybrid constraints logic
                    _, _, cat_score = evaluate_details(
                        query_emb, cand, cat_item, cat_strategy, sub_item, cat_field
                    )
                    cat_score_sum += cat_score * cat_weight

                sub_score = (cat_score_sum / cat_weight_sum) if cat_weight_sum > 0 else 0.0

                # In negative/rejection sections, a "good" sub-constraint means we want
                # this quality in the candidate — having it reduces the negative score.
                # So we flip: effective_score = 1 - sub_score.
                if is_negative_section and sub_type == "good":
                    sub_score = 1.0 - sub_score

                sub_total_score += sub_score * sub_weight
            constraint_score = sub_total_score
        
        total_weighted_score += constraint_score * c_weight

    return float(total_weighted_score / section_total_weight) if section_total_weight > 0 else 0.0

def main():
    if len(sys.argv) < 2:
        print("Usage: python stage2/scripts/explain_candidate.py <CANDIDATE_ID>")
        print("Example: python stage2/scripts/explain_candidate.py CAND_0000165")
        sys.exit(1)

    target_id = sys.argv[1].strip()

    if not CONSTRAINTS_FILE.exists():
        print(f"Error: {CONSTRAINTS_FILE} not found.")
        sys.exit(1)
    if not CANDIDATES_EMBEDDED_FILE.exists():
        print(f"Error: {CANDIDATES_EMBEDDED_FILE} not found.")
        sys.exit(1)
    if not EMBEDDED_CONSTRAINTS_FILE.exists():
        print(f"Error: {EMBEDDED_CONSTRAINTS_FILE} not found.")
        sys.exit(1)

    # Load resources
    with open(CONSTRAINTS_FILE, "r", encoding="utf-8") as f:
        constraints_data = json.load(f)
    with open(CANDIDATES_EMBEDDED_FILE, "rb") as f:
        candidates = pickle.load(f)
    with open(EMBEDDED_CONSTRAINTS_FILE, "rb") as f:
        query_embeddings_cache = pickle.load(f)

    # Find the target candidate
    cand = next((c for c in candidates if c["candidate_id"] == target_id), None)
    if not cand:
        print(f"Error: Candidate with ID '{target_id}' not found.")
        sys.exit(1)

    print(f"\n# Candidate Fit Analysis: {cand['profile']['anonymized_name']} ({target_id})")
    print(f"**Headline:** {cand['profile']['headline']}\n")

    # 1. Print Detailed Tabular Breakdown
    print("## 📊 Detailed Sub-Constraint Category Matching Breakdown\n")
    print("| Section | Constraint | Sub-Constraint | Category Item | Strategy | Matched Content | Raw Score | Normalized Score |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |")

    sections = ["must_have", "preferred", "rejection", "negative"]
    section_scores = {}

    for sec in sections:
        section_scores[sec] = evaluate_section_score(sec, constraints_data.get(sec, []), cand, query_embeddings_cache)
        
        for c in constraints_data.get(sec, []):
            constraint_text = c.get("constraint", "")
            for sub in c.get("sub_constraints", []):
                sub_item = sub.get("item", "")
                query_emb = query_embeddings_cache[sub_item]
                for cat in sub.get("categories", []):
                    cat_item = cat.get("item", "")
                    cat_strategy = cat.get("matching_strategy", "")
                    cat_field = cat.get("field", "")
                    
                    matched_item, raw, norm = evaluate_details(
                        query_emb, cand, cat_item, cat_strategy, sub_item, cat_field, sec
                    )
                    
                    # Formatting values
                    raw_str = f"{raw:.4f}" if isinstance(raw, float) else str(raw)
                    norm_str = f"{norm:.4f}"
                    matched_item = matched_item.replace("|", "\\|")
                    
                    print(f"| {sec} | {constraint_text} | {sub_item} | {cat_item} | {cat_strategy} | {matched_item} | {raw_str} | {norm_str} |")

    print("\n## 🧮 Overall Section Scores & Final Fit Score\n")
    s_must = section_scores["must_have"]
    s_pref = section_scores["preferred"]
    s_rej = section_scores["rejection"]
    s_neg = section_scores["negative"]

    positive_score = 0.75 * s_must + 0.25 * s_pref
    negative_score = 0.75 * s_rej + 0.25 * s_neg
    final_score = positive_score * (1.0 - negative_score)

    print(f"* **Must-Have Score (`s_must`):** `{s_must:.4f}`")
    print(f"* **Preferred Score (`s_pref`):** `{s_pref:.4f}`")
    print(f"* **Rejection Score (`s_rej`):** `{s_rej:.4f}`")
    print(f"* **Negative Score (`s_neg`):** `{s_neg:.4f}`")
    print(f"* **Combined Positive Score:** `{positive_score:.4f}` (0.75 * Must + 0.25 * Pref)")
    print(f"* **Combined Negative Score:** `{negative_score:.4f}` (0.75 * Rej + 0.25 * Neg)")
    print(f"* **Final Fit Score:** **`{final_score:.4f}`** (Positive * (1.0 - Negative))")
    print()

if __name__ == "__main__":
    main()
