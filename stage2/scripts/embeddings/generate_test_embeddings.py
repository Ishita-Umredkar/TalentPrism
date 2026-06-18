import json
import pickle
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[3]

CANDIDATES_FILE = ROOT / "data" / "test" / "test_candidates.json"
OUTPUT_DIR = ROOT / "stage2" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "test_candidates_embedded.pkl"

# Ensure outputs directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL LOAD
# ============================================================
print("Loading sentence-transformer model (BAAI/bge-base-en-v1.5)...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
print("Model loaded successfully.")


# ============================================================
# HELPERS
# ============================================================
def clean_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip()


def split_sentences(text: str) -> list[str]:
    text = clean_str(text)
    if not text:
        return []
    # Split on periods/exclamation/question marks followed by whitespace or string end.
    # Negatively look behind for a digit to avoid splitting decimals like 6.9, 12.5.
    raw_sentences = re.split(r'(?<!\d)\.(?=\s|$)|[!?](?=\s|$)', text)
    sentences = []
    for s in raw_sentences:
        s_clean = s.strip()
        if s_clean:
            sentences.append(s_clean)
    return sentences


def format_title_role(r, is_current):
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
    prefix = "Currently a" if is_current else "Worked as"
    if is_current:
        a_an = "an" if title[0].lower() in "aeiou" else "a"
        prefix = f"Currently {a_an}"
    return f"{prefix} {title} for {yoe_str}"


def format_company_role(r, is_current):
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


def embed_text(text: str):
    text_clean = clean_str(text)
    if not text_clean:
        # Return zero vector if text is empty (keeping same dimension 768)
        import numpy as np
        return np.zeros(768)
    return model.encode(text_clean, normalize_embeddings=True)


# ============================================================
def process_candidate(cand: dict) -> dict:
    cand_id = cand.get("candidate_id", "")
    print(f"Processing candidate: {cand_id}")

    # 1. Profile Summary
    profile_data = cand.get("profile", {})
    summary_text = clean_str(profile_data.get("summary", ""))
    summary_emb = embed_text(summary_text)

    # 2. Chronological Job Titles
    history = cand.get("career_history", [])
    chronological_roles = history[::-1]
    titles_with_dur = []
    for r in chronological_roles:
        is_curr = r.get("is_current", False)
        title_str = format_title_role(r, is_curr)
        if title_str:
            titles_with_dur.append(title_str)
    titles_text = "->".join(titles_with_dur) if titles_with_dur else ""
    titles_emb = embed_text(titles_text)

    # 3. Combined chronological company, industry, and size
    comp_ind_size_blocks = []
    for r in chronological_roles:
        is_curr = r.get("is_current", False)
        comp_str = format_company_role(r, is_curr)
        if comp_str:
            comp_ind_size_blocks.append(comp_str)
    comp_ind_size_text = "->".join(comp_ind_size_blocks)
    comp_ind_size_emb = embed_text(comp_ind_size_text)

    embedded_profile = {
        "anonymized_name": clean_str(profile_data.get("anonymized_name", "")),
        "headline": clean_str(profile_data.get("headline", "")),
        "summary": {
            "text": summary_text,
            "embedding": summary_emb
        },
        "location": clean_str(profile_data.get("location", "")),
        "country": clean_str(profile_data.get("country", "")),
        "years_of_experience": profile_data.get("years_of_experience", 0.0),
        "current_title": clean_str(profile_data.get("current_title", "")),
        "current_company": clean_str(profile_data.get("current_company", "")),
        "current_company_size": clean_str(profile_data.get("current_company_size", "")),
        "current_industry": clean_str(profile_data.get("current_industry", "")),
        "titles_chronological_embedding": titles_emb,
        "company_industry_size_chronological_embedding": comp_ind_size_emb
    }

    # 4. Career History
    embedded_history = []
    for idx, role in enumerate(history):
        title = clean_str(role.get("title", ""))
        company = clean_str(role.get("company", ""))
        duration = role.get("duration_months", 0)
        description = clean_str(role.get("description", ""))

        role_block_text = f"Title: {title}\nCompany: {company}\nDuration: {duration} months\nDescription: {description}"
        role_block_emb = embed_text(role_block_text)

        embedded_history.append({
            "role_id": idx,
            "title": title,
            "company": company,
            "industry": clean_str(role.get("industry", "")),
            "duration_months": duration,
            "role_embedding": role_block_emb,
            "description_sentences": [{"text": s, "embedding": None} for s in split_sentences(description)],
            "is_current": role.get("is_current", False),
            "company_size": clean_str(role.get("company_size", ""))
        })

    # 5. Technical Skills
    embedded_skills = []
    skills = cand.get("skills", [])
    if skills:
        skill_names = [clean_str(sk.get("name", "")) for sk in skills]
        encoded_skills = model.encode(skill_names, normalize_embeddings=True)
        for sk, emb in zip(skills, encoded_skills):
            embedded_skills.append({
                "name": clean_str(sk.get("name", "")),
                "proficiency": clean_str(sk.get("proficiency", "")),
                "endorsements": sk.get("endorsements", 0),
                "duration_months": sk.get("duration_months", 0),
                "embedding": emb
            })

    # 6. Education
    embedded_education = []
    education = cand.get("education", [])
    for edu in education:
        degree = clean_str(edu.get("degree", ""))
        field_of_study = clean_str(edu.get("field_of_study", ""))
        institution = clean_str(edu.get("institution", ""))
        tier = clean_str(edu.get("tier", ""))
        grade = clean_str(edu.get("grade", ""))

        edu_block_text = f"Degree: {degree}\nField: {field_of_study}\nInstitution: {institution}\nTier: {tier}\nGrade: {grade}"
        edu_emb = embed_text(edu_block_text)

        embedded_education.append({
            "degree": degree,
            "field_of_study": field_of_study,
            "institution": institution,
            "tier": tier,
            "grade": grade,
            "start_year": edu.get("start_year"),
            "end_year": edu.get("end_year"),
            "education_embedding": edu_emb
        })

    return {
        "candidate_id": cand_id,
        "profile": embedded_profile,
        "career_history": embedded_history,
        "skills": embedded_skills,
        "education": embedded_education,
        "certifications": cand.get("certifications", []),
        "languages": cand.get("languages", []),
        "redrob_signals": cand.get("redrob_signals", {})
    }


def main():

    print(f"Reading raw candidates from: {CANDIDATES_FILE}")
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    embedded_candidates = []
    for cand in candidates:
        embedded_cand = process_candidate(cand)
        embedded_candidates.append(embedded_cand)

    print(f"Saving embedded candidates to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(embedded_candidates, f)

    print("Embedding generation completed successfully.")


if __name__ == "__main__":
    main()
