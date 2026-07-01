import os
import re
import json
import pickle
import time
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

# Set PyTorch thread count to 8 for optimal CPU speed
torch.set_num_threads(8)

# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[3]
CANDIDATES_FILE = ROOT / "resources" / "candidates.jsonl"
OUTPUT_DIR = ROOT / "stage2" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "test_candidates_embedded.pkl"

# Ensure outputs directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLEANING HELPERS
# ============================================================
def clean_str(val) -> str:
    if val is None:
        return ""
    return str(val).strip()


def split_sentences(text: str) -> list[str]:
    text = clean_str(text)
    if not text:
        return []
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


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Reading first 1000 candidates from: {CANDIDATES_FILE}")
    candidates_list = []
    
    start_time = time.time()
    
    with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 1000:
                break
            if not line.strip():
                continue
            cand = json.loads(line)
            candidates_list.append(cand)
            
    print(f"Loaded {len(candidates_list)} candidates in {time.time() - start_time:.2f}s.")

    print("Loading SentenceTransformer model (BAAI/bge-base-en-v1.5)...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print("Model loaded successfully.")

    # 1. Collect all texts to embed
    print("Collecting candidate texts to batch-embed...")
    texts_to_embed = []
    text_mapping = []  # List of tuples: (candidate_idx, field_type, sub_idx, sub_sub_idx)
    
    for c_idx, cand in enumerate(candidates_list):
        # Profile summary (full)
        summary_text = clean_str(cand.get("profile", {}).get("summary", ""))
        if summary_text:
            texts_to_embed.append(summary_text)
            text_mapping.append((c_idx, "summary", 0, 0))
            
        history = cand.get("career_history", [])
        chronological_roles = history[::-1]  # Oldest to newest chronological order

        # Chronological job titles
        titles_with_dur = []
        for r in chronological_roles:
            is_curr = r.get("is_current", False)
            title_str = format_title_role(r, is_curr)
            if title_str:
                titles_with_dur.append(title_str)
        titles_text = "->".join(titles_with_dur) if titles_with_dur else ""
        texts_to_embed.append(titles_text)
        text_mapping.append((c_idx, "titles_chronological", 0, 0))

        # Combined chronological company, industry, and size
        comp_ind_size_blocks = []
        for r in chronological_roles:
            is_curr = r.get("is_current", False)
            comp_str = format_company_role(r, is_curr)
            if comp_str:
                comp_ind_size_blocks.append(comp_str)
        comp_ind_size_text = "->".join(comp_ind_size_blocks)
        if comp_ind_size_text:
            texts_to_embed.append(comp_ind_size_text)
            text_mapping.append((c_idx, "company_industry_size_chronological", 0, 0))
            
        # Career history role blocks
        for r_idx, role in enumerate(history):
            title = clean_str(role.get("title", ""))
            company = clean_str(role.get("company", ""))
            duration = role.get("duration_months", 0)
            description = clean_str(role.get("description", ""))
            
            role_block = f"Title: {title}\nCompany: {company}\nDuration: {duration} months\nDescription: {description}"
            texts_to_embed.append(role_block)
            text_mapping.append((c_idx, "role_block", r_idx, 0))
            
        # Skills
        for s_idx, sk in enumerate(cand.get("skills", [])):
            sk_name = clean_str(sk.get("name", ""))
            if sk_name:
                texts_to_embed.append(sk_name)
                text_mapping.append((c_idx, "skill", s_idx, 0))
                
        # Education
        for e_idx, edu in enumerate(cand.get("education", [])):
            degree = clean_str(edu.get("degree", ""))
            field = clean_str(edu.get("field_of_study", ""))
            inst = clean_str(edu.get("institution", ""))
            tier = clean_str(edu.get("tier", ""))
            grade = clean_str(edu.get("grade", ""))
            
            edu_block = f"Degree: {degree}\nField: {field}\nInstitution: {inst}\nTier: {tier}\nGrade: {grade}"
            texts_to_embed.append(edu_block)
            text_mapping.append((c_idx, "education_block", e_idx, 0))
            
    print(f"Total texts to encode: {len(texts_to_embed)}")
    
    # Run batch embedding
    embed_start = time.time()
    all_embeddings = model.encode(texts_to_embed, batch_size=512, show_progress_bar=True, normalize_embeddings=True)
    print(f"Embedding completed in {time.time() - embed_start:.2f}s.")
    
    # 2. Build embedded candidates structure matching generate_test_embeddings.py exactly
    embedded_candidates = []
    for cand in candidates_list:
        p = cand.get("profile", {})
        embedded_cand = {
            "candidate_id": cand.get("candidate_id", ""),
            "profile": {
                "anonymized_name": clean_str(p.get("anonymized_name", "")),
                "headline": clean_str(p.get("headline", "")),
                "summary": {"text": clean_str(p.get("summary", "")), "embedding": None},
                "location": clean_str(p.get("location", "")),
                "country": clean_str(p.get("country", "")),
                "years_of_experience": p.get("years_of_experience", 0.0),
                "current_title": clean_str(p.get("current_title", "")),
                "current_company": clean_str(p.get("current_company", "")),
                "current_company_size": clean_str(p.get("current_company_size", "")),
                "current_industry": clean_str(p.get("current_industry", "")),
                "titles_chronological_embedding": None,
                "company_industry_size_chronological_embedding": None
            },
            "career_history": [],
            "skills": [],
            "education": [],
            "certifications": cand.get("certifications", []),
            "languages": cand.get("languages", []),
            "redrob_signals": cand.get("redrob_signals", {})
        }
        
        for idx, role in enumerate(cand.get("career_history", [])):
            embedded_cand["career_history"].append({
                "role_id": idx,
                "title": clean_str(role.get("title", "")),
                "company": clean_str(role.get("company", "")),
                "industry": clean_str(role.get("industry", "")),
                "duration_months": role.get("duration_months", 0),
                "role_embedding": None,
                "description_sentences": [{"text": s, "embedding": None} for s in split_sentences(clean_str(role.get("description", "")))],
                "is_current": role.get("is_current", False),
                "company_size": clean_str(role.get("company_size", ""))
            })
            
        for sk in cand.get("skills", []):
            embedded_cand["skills"].append({
                "name": clean_str(sk.get("name", "")),
                "proficiency": clean_str(sk.get("proficiency", "")),
                "endorsements": sk.get("endorsements", 0),
                "duration_months": sk.get("duration_months", 0),
                "embedding": None
            })
            
        for edu in cand.get("education", []):
            embedded_cand["education"].append({
                "degree": clean_str(edu.get("degree", "")),
                "field_of_study": clean_str(edu.get("field_of_study", "")),
                "institution": clean_str(edu.get("institution", "")),
                "tier": clean_str(edu.get("tier", "")),
                "grade": clean_str(edu.get("grade", "")),
                "start_year": edu.get("start_year"),
                "end_year": edu.get("end_year"),
                "education_embedding": None
            })
            
        embedded_candidates.append(embedded_cand)
        
    # Assign embeddings back using mapping
    import numpy as np
    for emb, (c_idx, f_type, sub_idx, sub_sub_idx) in zip(all_embeddings, text_mapping):
        ec = embedded_candidates[c_idx]
        if f_type == "summary":
            ec["profile"]["summary"]["embedding"] = emb
        elif f_type == "titles_chronological":
            ec["profile"]["titles_chronological_embedding"] = emb
        elif f_type == "company_industry_size_chronological":
            ec["profile"]["company_industry_size_chronological_embedding"] = emb
        elif f_type == "role_block":
            ec["career_history"][sub_idx]["role_embedding"] = emb
        elif f_type == "skill":
            ec["skills"][sub_idx]["embedding"] = emb
        elif f_type == "education_block":
            ec["education"][sub_idx]["education_embedding"] = emb
            
    # Fallback to zero vectors for any missing embeddings to ensure shape validity
    for ec in embedded_candidates:
        if ec["profile"]["summary"]["embedding"] is None:
            ec["profile"]["summary"]["embedding"] = np.zeros(768)
        if ec["profile"].get("titles_chronological_embedding") is None:
            ec["profile"]["titles_chronological_embedding"] = np.zeros(768)
        if ec["profile"].get("company_industry_size_chronological_embedding") is None:
            ec["profile"]["company_industry_size_chronological_embedding"] = np.zeros(768)
        for r in ec["career_history"]:
            if r["role_embedding"] is None:
                r["role_embedding"] = np.zeros(768)
        for s in ec["skills"]:
            if s["embedding"] is None:
                s["embedding"] = np.zeros(768)
        for edu in ec["education"]:
            if edu["education_embedding"] is None:
                edu["education_embedding"] = np.zeros(768)

    print(f"Saving embedded candidates to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(embedded_candidates, f)

    print("Embedding generation completed successfully.")
    print(f"Total time taken: {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    main()
