import json
import pickle
import math
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_EMBEDDED_FILE = ROOT / "stage2" / "outputs" / "test_candidates_embedded.pkl"
CONSTRAINTS_FILE = ROOT / "stage2" / "outputs" / "extracted_constraints_v2.json"

# ============================================================
# MODEL LOAD & QUERY CACHING
# ============================================================
print("Loading sentence-transformer model (BAAI/bge-base-en-v1.5) for query encoding...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
print("Model loaded successfully.")

def clean_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip()

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


def find_best_field_match(
    query_emb: np.ndarray,
    cand: dict,
    category_name: str,
    strategy: str,
    sub_constraint_text: str,
    field_val: str = "",
    section_name: str = ""
):
    """Returns (matched_field_description, score) representing what specifically matched."""
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
                return f"Notice period: {np_days} days (Target: >{target_days} days)", score
            else:
                if np_days <= target_days:
                    score = 1.0
                elif np_days <= target_days * 2:
                    score = 0.5
                else:
                    score = 0.0
                return f"Notice period: {np_days} days (Target: <={target_days} days)", score

        if category_name == "redrob_signals.willing_to_relocate":
            expected = True
            if field_val:
                expected = (field_val.lower() == "true")
            willing = cand.get("redrob_signals", {}).get("willing_to_relocate", False)
            score = 1.0 if willing == expected else 0.0
            return f"Willing to relocate: {willing} (Target: {expected})", score

        if category_name == "redrob_signals.open_to_work_flag":
            expected = True
            if field_val:
                expected = (field_val.lower() == "true")
            otw = cand.get("redrob_signals", {}).get("open_to_work_flag", False)
            score = 1.0 if otw == expected else 0.0
            return f"Open to work flag: {otw} (Target: {expected})", score

        if category_name == "profile.location":
            loc = cand.get("profile", {}).get("location", "")
            willing = cand.get("redrob_signals", {}).get("willing_to_relocate", False)
            loc_lower = loc.lower()
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
            return f"Location: '{loc}', Relocate: {willing} (Target: {field_val if field_val else 'Pune, Noida'})", score

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
            return f"Years of experience: {yoe:.1f} (Target: {field_val if field_val else '5-9'})", score

        if category_name == "career_history.company":
            history = cand.get("career_history", [])
            for company_name in ["TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini"]:
                if company_name.lower() in sub_constraint_text.lower():
                    score = check_consulting_company(company_name, history)
                    matched_roles = [r.get("company", "") for r in history if company_name.lower() in r.get("company", "").lower()]
                    return f"Worked at {company_name}: {matched_roles if matched_roles else 'No'}", score

        if category_name == "career_history.industry":
            industry_text = cand.get("profile", {}).get("current_industry", "").lower()
            score = 0.0
            details = []
            if "hr" in sub_constraint_text.lower() or "recruit" in sub_constraint_text.lower():
                if "hr" in industry_text or "recruit" in industry_text or "talent" in industry_text:
                    score = 1.0
                    details.append(f"Current industry: '{industry_text}'")
            if "marketplace" in sub_constraint_text.lower():
                if "market" in industry_text or "e-commerce" in industry_text:
                    score = 1.0
                    details.append(f"Current industry: '{industry_text}'")
            
            for role in cand.get("career_history", []):
                ind = role.get("industry", "").lower()
                desc = role.get("description_sentences", [])
                desc_text = " ".join([s.get("text", "") for s in desc]).lower()
                if "hr" in sub_constraint_text.lower() or "recruit" in sub_constraint_text.lower():
                    if "hr" in ind or "recruit" in ind or "talent" in ind or "recruiting" in desc_text or "hr-tech" in desc_text:
                        score = 1.0
                        details.append(f"Role '{role.get('title')}' at '{role.get('company')}' (industry: '{ind}')")
                if "marketplace" in sub_constraint_text.lower():
                    if "market" in ind or "e-commerce" in ind or "marketplace" in desc_text:
                        score = 1.0
                        details.append(f"Role '{role.get('title')}' at '{role.get('company')}' (industry: '{ind}')")
            return f"Industry matched: {details if details else 'No'}", score

        # Embedding-based direct matching for profile/history fields
        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Direct match on profile summary (raw sim: {sim:.4f})", score

        if category_name in ["career_history.title", "profile.current_title"]:
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Direct match on chronological titles (raw sim: {sim:.4f})", score

        if category_name in ["career_history.company", "career_history.industry", "career_history.company_size",
                              "profile.current_company", "profile.current_industry", "profile.current_company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Direct match on combined history (raw sim: {sim:.4f})", score

        signals = cand.get("redrob_signals", {})
        key = category_name.split(".")[-1]
        if key in signals:
            val = signals[key]
            if isinstance(val, (int, float)):
                if key == "profile_completeness_score":
                    return f"Profile completeness: {val}%", val / 100.0
                if key == "recruiter_response_rate":
                    return f"Recruiter response rate: {val:.2f}", val
                if key == "search_appearance_30d" or key == "saved_by_recruiters_30d":
                    return f"Signal {key}: {val}", min(1.0, val / 100.0)
            elif isinstance(val, bool):
                return f"Signal {key}: {val}", 1.0 if val else 0.0
        return f"Signal field '{category_name}' not matched", 0.0

    # 2. BEHAVIORAL SIGNAL
    if strategy == "behavioral":
        score = evaluate_behavior_signal(category_name, cand)
        raw_val = cand.get("redrob_signals", {}).get(category_name.split(".")[-1], -1)
        return f"{category_name.split('.')[-1]}: {raw_val}", score

    # Helper variables for chronological values
    history = cand.get("career_history", [])
    chronological_roles = history[::-1]

    def format_title_role_em(r, is_current):
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

    def format_company_role_em(r, is_current):
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
        t = format_title_role_em(r, is_curr)
        if t:
            titles_with_dur.append(t)
    titles_text = "->".join(titles_with_dur) if titles_with_dur else ""

    comp_ind_size_blocks = []
    for r in chronological_roles:
        is_curr = r.get("is_current", False)
        comp_str = format_company_role_em(r, is_curr)
        if comp_str:
            comp_ind_size_blocks.append(comp_str)
    comp_ind_size_text = "->".join(comp_ind_size_blocks)

    # 1.5. DIRECT MATCH (legacy name)
    if strategy == "direct_match":
        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Direct match on profile summary (raw sim: {sim:.4f})", score
        if category_name in ["career_history.title", "profile.current_title"]:
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Direct match on chronological titles: '{titles_text}' (raw sim: {sim:.4f})", score
        if category_name in ["career_history.company", "career_history.industry", "career_history.company_size",
                              "profile.current_company", "profile.current_industry", "profile.current_company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                score = normalize_similarity(sim, 0.52, 0.68)
                return f"Direct match on combined history: '{comp_ind_size_text}' (raw sim: {sim:.4f})", score
        return "No direct match embedding", 0.0

    # 3. SKILL MATCH
    if strategy == "max" and (category_name == "skills" or category_name == "skills.name"):
        cand_skills = cand.get("skills", [])
        if not cand_skills:
            return "No skills", 0.0
        max_sim = -1.0
        best_skill = None
        best_sim = 0.0
        best_adjusted_sim = 0.0
        for sk in cand_skills:
            sk_emb = sk.get("embedding")
            if sk_emb is None:
                continue
            sim = np.dot(query_emb, sk_emb)
            sim_norm = normalize_similarity(sim, 0.60, 0.80)
            adjusted_sim = sim_norm * get_skill_multiplier(sk)
            if adjusted_sim > max_sim:
                max_sim = adjusted_sim
                best_skill = sk.get("name", "")
                best_sim = sim
                best_adjusted_sim = adjusted_sim
        if best_skill is not None:
            return f"Skill match: '{best_skill}' (raw sim: {best_sim:.4f})", min(1.0, max(0.0, best_adjusted_sim))
        return "No matching skills", 0.0

    # 4. MAX EVIDENCE
    if strategy == "max":
        if category_name == "career_history.description":
            max_sim = -1.0
            best_desc = ""
            best_sim = 0.0
            best_sim_norm = 0.0
            for role in cand.get("career_history", []):
                emb = role.get("role_embedding")
                if emb is not None:
                    sim = np.dot(query_emb, emb)
                    sim_norm = normalize_similarity(sim, 0.52, 0.68)
                    if sim_norm > max_sim:
                        max_sim = sim_norm
                        best_desc = f"Role '{role.get('title')}' at '{role.get('company')}'"
                        best_sim = sim
                        best_sim_norm = sim_norm
            if best_desc:
                return f"Max description evidence: {best_desc} (raw sim: {best_sim:.4f})", best_sim_norm
            return "No career history roles", 0.0

        if category_name == "profile.summary":
            emb = cand.get("profile", {}).get("summary", {}).get("embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                sim_norm = normalize_similarity(sim, 0.52, 0.68)
                return f"Max summary evidence: profile summary embedding (raw sim: {sim:.4f})", sim_norm
            return "No summary embedding", 0.0

        if category_name in ["career_history.title"]:
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                sim_norm = normalize_similarity(sim, 0.52, 0.68)
                return f"Max title evidence: '{titles_text}' (raw sim: {sim:.4f})", sim_norm
            return "No title embedding", 0.0

        if category_name in ["career_history.company", "career_history.industry", "career_history.company_size"]:
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                sim_norm = normalize_similarity(sim, 0.52, 0.68)
                return f"Max comp/ind/size evidence: '{comp_ind_size_text}' (raw sim: {sim:.4f})", sim_norm
            return "No combined embedding", 0.0

    # 5. AGGREGATE EVIDENCE
    if strategy == "aggregate":
        if category_name == "skills.name" or category_name == "skills":
            cand_skills = cand.get("skills", [])
            if not cand_skills:
                return "No skills", 0.0
            matching_details = []
            total_score = 0.0
            valid_count = 0
            for sk in cand_skills:
                sk_emb = sk.get("embedding")
                if sk_emb is None:
                    continue
                sim = np.dot(query_emb, sk_emb)
                sim_norm = normalize_similarity(sim, 0.60, 0.80)
                total_score += sim_norm
                valid_count += 1
                if sim_norm > 0.0:
                    matching_details.append(f"'{sk.get('name')}' ({sim_norm:.4f})")
            if valid_count > 0:
                avg_norm = total_score / valid_count
                if matching_details:
                    matched_str = "Skills: " + ", ".join(matching_details)
                else:
                    matched_str = "No matching skills"
                return matched_str, avg_norm
            return "No matching skills", 0.0

        if category_name == "career_history.duration_months":
            if "closed-source" in sub_constraint_text.lower():
                score = closed_source_score(query_emb, cand)
                # Let's find which roles contributed to closed-source score
                matched_roles = []
                for role in cand.get("career_history", []):
                    role_emb = role.get("role_embedding")
                    if role_emb is not None:
                        sim = np.dot(query_emb, role_emb)
                        sim_norm = normalize_similarity(sim, 0.52, 0.68)
                        if sim_norm > 0.5:
                            matched_roles.append(f"'{role.get('title')}' at '{role.get('company')}' ({role.get('duration_months')}m, raw sim: {sim:.4f})")
                return f"Closed-source roles: {matched_roles if matched_roles else 'None'}", score

            if "switching" in sub_constraint_text.lower() or "trajectory" in sub_constraint_text.lower():
                score = switching_companies_score(cand)
                history = cand.get("career_history", [])
                companies = set(r.get("company", "").strip() for r in history if r.get("company"))
                total_months = sum(r.get("duration_months", 0) for r in history)
                avg_y = (total_months / len(companies)) / 12.0 if companies else 0.0
                return f"Switching: {len(companies)} companies in {total_months} months (avg duration {avg_y:.2f} yrs)", score

            if "bounce" in sub_constraint_text.lower() or "startups" in sub_constraint_text.lower():
                score = startup_ratio(cand)
                return f"Startup ratio: {score:.2%}", score

            if "recent" in sub_constraint_text.lower() or "12 months" in sub_constraint_text.lower():
                score = recent_ai_experience_score(cand)
                return f"Recent AI experience score: {score:.4f}", score
            return "Duration months aggregate match", 0.0

        if category_name == "career_history.company_size":
            if "startups" in sub_constraint_text.lower() or "bounce" in sub_constraint_text.lower():
                score = startup_ratio(cand)
                return f"Startup ratio: {score:.2%}", score
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                sim_norm = normalize_similarity(sim, 0.52, 0.68)
                return f"Aggregate match on company size: '{comp_ind_size_text}' (raw sim: {sim:.4f})", sim_norm
            return "Company size aggregate match", 0.0

        if category_name == "education.field_of_study" or category_name == "education":
            max_sim = -1.0
            best_edu = ""
            best_sim = 0.0
            best_sim_norm = 0.0
            for edu in cand.get("education", []):
                emb = edu.get("education_embedding")
                if emb is not None:
                    sim = np.dot(query_emb, emb)
                    sim_norm = normalize_similarity(sim, 0.52, 0.68)
                    if sim_norm > max_sim:
                        max_sim = sim_norm
                        best_edu = f"{edu.get('degree')} in '{edu.get('field_of_study')}' at '{edu.get('school', edu.get('institution'))}'"
                        best_sim = sim
                        best_sim_norm = sim_norm
            if best_edu:
                return f"Education: {best_edu} (raw sim: {best_sim:.4f})", best_sim_norm
            return "No education records", 0.0

        if category_name == "career_history.title":
            emb = cand.get("profile", {}).get("titles_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                sim_norm = normalize_similarity(sim, 0.52, 0.68)
                return f"Aggregate match on titles: '{titles_text}' (raw sim: {sim:.4f})", sim_norm
            return "No title embedding", 0.0

        if category_name == "career_history.company":
            if "consulting" in sub_constraint_text.lower() or "service" in sub_constraint_text.lower():
                score = is_entirely_consulting(cand)
                return f"Is entirely consulting: {score}", score
            if "google" in sub_constraint_text.lower():
                score = check_consulting_company("Google", history)
                return f"Worked at Google: {score}", score
            if "meta" in sub_constraint_text.lower():
                score = check_consulting_company("Meta", history)
                return f"Worked at Meta: {score}", score
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                sim_norm = normalize_similarity(sim, 0.52, 0.68)
                return f"Aggregate match on company: '{comp_ind_size_text}' (raw sim: {sim:.4f})", sim_norm
            return "No chronological embedding", 0.0

        if category_name == "career_history.industry":
            if "consulting" in sub_constraint_text.lower() or "service" in sub_constraint_text.lower():
                score = is_entirely_consulting(cand)
                return f"Is entirely consulting: {score}", score
            if "product" in sub_constraint_text.lower():
                score = has_product_experience(cand)
                return f"Has product experience: {score}", score
            emb = cand.get("profile", {}).get("company_industry_size_chronological_embedding")
            if emb is not None:
                sim = np.dot(query_emb, emb)
                sim_norm = normalize_similarity(sim, 0.52, 0.68)
                return f"Aggregate match on industry: '{comp_ind_size_text}' (raw sim: {sim:.4f})", sim_norm
            return "No chronological embedding", 0.0

        if category_name == "career_history.description":
            history = cand.get("career_history", [])
            if not history:
                return "No career history", 0.0

            # Weighted aggregate score across all roles
            total_months = sum(r.get("duration_months", 0) for r in history)
            score_sum = 0.0
            best_role_text = ""
            max_role_sim = -1.0
            
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
                
                if sim_norm > max_role_sim:
                    max_role_sim = sim_norm
                    best_role_text = f"Role '{role.get('title')}' at '{role.get('company')}' (raw sim: {sim:.4f})"

            return f"Aggregate match on description: Best Role: {best_role_text}", score_sum

    return f"Unknown matching strategy '{strategy}' for '{category_name}'", 0.0


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Loading raw constraints: {CONSTRAINTS_FILE}")
    with open(CONSTRAINTS_FILE, "r", encoding="utf-8") as f:
        constraints_data = json.load(f)

    print(f"Loading embedded candidates: {CANDIDATES_EMBEDDED_FILE}")
    with open(CANDIDATES_EMBEDDED_FILE, "rb") as f:
        candidates = pickle.load(f)

    # 1. Gather all sub-constraint items for batch embedding encoding
    print("Collecting and batch-encoding constraint texts...")
    unique_items = set()
    for sec in ["must_have", "preferred", "negative", "rejection"]:
        for c in constraints_data.get(sec, []):
            for sub in c.get("sub_constraints", []):
                unique_items.add(sub.get("item", ""))
    
    unique_items_list = list(unique_items)
    print(f"Total unique sub-constraint items to encode: {len(unique_items_list)}")
    encoded_vectors = model.encode(unique_items_list, normalize_embeddings=True)
    query_embeddings_cache = {text: vec for text, vec in zip(unique_items_list, encoded_vectors)}
    print("Query cache populated.")

    # Target Candidates: Top 5 IDs
    target_ids = ["CAND_0000031", "CAND_0000666", "CAND_0000422", "CAND_0000981", "CAND_0000273"]
    target_candidates = []
    for cid in target_ids:
        for c in candidates:
            if c.get("candidate_id") == cid:
                target_candidates.append(c)
                break

    # Analyze matches for each of the top 5 candidates
    for rank, cand in enumerate(target_candidates, 1):
        cid = cand.get("candidate_id")
        name = cand.get("profile", {}).get("anonymized_name")
        headline = cand.get("profile", {}).get("headline")
        print(f"\n==========================================================================================")
        print(f"RANK {rank}: CANDIDATE ID: {cid} | NAME: {name}")
        print(f"HEADLINE: {headline}")
        print(f"==========================================================================================")

        # We will collect matching results for all constraints in each type:
        # positive (must_have + preferred), rejection, negative
        
        # 1. Positive Constraints
        positive_results = []
        for sec in ["must_have", "preferred"]:
            for c in constraints_data.get(sec, []):
                constraint_text = c.get("constraint")
                c_weight = c.get("weight")
                
                # We want to find the best matching sub-constraint and category for this constraint
                sub_constraints = c.get("sub_constraints", [])
                best_sub_match_text = ""
                best_category_matched = ""
                best_evidence = ""
                max_score = -1.0
                
                for sub in sub_constraints:
                    sub_item = sub.get("item", "")
                    query_emb = query_embeddings_cache[sub_item]
                    categories = sub.get("categories", [])
                    
                    for cat in categories:
                        cat_item = cat.get("item", "")
                        cat_strategy = cat.get("matching_strategy", "")
                        cat_field = cat.get("field", "")
                        
                        evidence, score = find_best_field_match(
                            query_emb, cand, cat_item, cat_strategy, sub_item, cat_field, sec
                        )
                        if score > max_score:
                            max_score = score
                            best_sub_match_text = sub_item
                            best_category_matched = cat_item
                            best_evidence = evidence
                
                positive_results.append({
                    "constraint": constraint_text,
                    "section": sec,
                    "weight": c_weight,
                    "sub_item": best_sub_match_text,
                    "category": best_category_matched,
                    "evidence": best_evidence,
                    "score": max_score
                })
        
        # Sort positive results by score descending
        positive_results.sort(key=lambda x: x["score"], reverse=True)
        print(f"\n[TOP POSITIVE CONSTRAINTS MATCHED]")
        for idx, res in enumerate(positive_results[:5], 1):
            print(f"  {idx}. Constraint: {res['constraint']}")
            print(f"     Sub-constraint: {res['sub_item']} (weight: {res['weight']})")
            print(f"     Candidate Field Matched: {res['category']}")
            print(f"     Field Details: {res['evidence']}")
            print(f"     Match Score/Similarity: {res['score']:.4f}")

        # 2. Rejection Constraints
        rejection_results = []
        for c in constraints_data.get("rejection", []):
            constraint_text = c.get("constraint")
            c_weight = c.get("weight")
            
            sub_constraints = c.get("sub_constraints", [])
            best_sub_match_text = ""
            best_category_matched = ""
            best_evidence = ""
            max_score = -1.0
            
            for sub in sub_constraints:
                sub_item = sub.get("item", "")
                sub_type = sub.get("type", "bad")  # v2 field: "good" or "bad"
                query_emb = query_embeddings_cache[sub_item]
                categories = sub.get("categories", [])
                
                for cat in categories:
                    cat_item = cat.get("item", "")
                    cat_strategy = cat.get("matching_strategy", "")
                    cat_field = cat.get("field", "")
                    
                    evidence, score = find_best_field_match(
                        query_emb, cand, cat_item, cat_strategy, sub_item, cat_field, "rejection"
                    )
                    # Apply v2 type-aware inversion: "good" sub-constraints reduce negative score
                    if sub_type == "good":
                        score = 1.0 - score
                    if score > max_score:
                        max_score = score
                        best_sub_match_text = sub_item
                        best_category_matched = cat_item
                        best_evidence = evidence
            
            rejection_results.append({
                "constraint": constraint_text,
                "weight": c_weight,
                "sub_item": best_sub_match_text,
                "category": best_category_matched,
                "evidence": best_evidence,
                "score": max_score
            })
            
        # Sort rejection results by score descending
        rejection_results.sort(key=lambda x: x["score"], reverse=True)
        print(f"\n[TOP REJECTION CONSTRAINTS MATCHED]")
        for idx, res in enumerate(rejection_results[:5], 1):
            print(f"  {idx}. Constraint: {res['constraint']}")
            print(f"     Sub-constraint: {res['sub_item']} (weight: {res['weight']})")
            print(f"     Candidate Field Matched: {res['category']}")
            print(f"     Field Details: {res['evidence']}")
            print(f"     Match Score/Similarity: {res['score']:.4f}")

        # 3. Negative Constraints
        negative_results = []
        for c in constraints_data.get("negative", []):
            constraint_text = c.get("constraint")
            c_weight = c.get("weight")
            
            sub_constraints = c.get("sub_constraints", [])
            best_sub_match_text = ""
            best_category_matched = ""
            best_evidence = ""
            max_score = -1.0
            
            for sub in sub_constraints:
                sub_item = sub.get("item", "")
                sub_type = sub.get("type", "bad")  # v2 field: "good" or "bad"
                query_emb = query_embeddings_cache[sub_item]
                categories = sub.get("categories", [])
                
                for cat in categories:
                    cat_item = cat.get("item", "")
                    cat_strategy = cat.get("matching_strategy", "")
                    cat_field = cat.get("field", "")
                    
                    evidence, score = find_best_field_match(
                        query_emb, cand, cat_item, cat_strategy, sub_item, cat_field, "negative"
                    )
                    # Apply v2 type-aware inversion: "good" sub-constraints reduce negative score
                    if sub_type == "good":
                        score = 1.0 - score
                    if score > max_score:
                        max_score = score
                        best_sub_match_text = sub_item
                        best_category_matched = cat_item
                        best_evidence = evidence
            
            negative_results.append({
                "constraint": constraint_text,
                "weight": c_weight,
                "sub_item": best_sub_match_text,
                "category": best_category_matched,
                "evidence": best_evidence,
                "score": max_score
            })
            
        # Sort negative results by score descending
        negative_results.sort(key=lambda x: x["score"], reverse=True)
        print(f"\n[TOP NEGATIVE CONSTRAINTS MATCHED]")
        for idx, res in enumerate(negative_results[:5], 1):
            print(f"  {idx}. Constraint: {res['constraint']}")
            print(f"     Sub-constraint: {res['sub_item']} (weight: {res['weight']})")
            print(f"     Candidate Field Matched: {res['category']}")
            print(f"     Field Details: {res['evidence']}")
            print(f"     Match Score/Similarity: {res['score']:.4f}")

if __name__ == "__main__":
    main()
