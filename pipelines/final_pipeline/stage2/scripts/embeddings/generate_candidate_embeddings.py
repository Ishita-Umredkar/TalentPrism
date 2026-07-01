import os
import re
import sys
import json
import pickle
import time
import gc
import argparse
from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

# Reconfigure stdout/stderr to use UTF-8 on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Set PyTorch thread count for optimal CPU speed
torch.set_num_threads(8)

# Setup paths
ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[5]

# ============================================================
# CLEANING & FORMATTING HELPERS
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

def stream_candidates(file_path: Path):
    """Yields candidate profiles one by one from JSONL or JSON list."""
    is_jsonl = str(file_path).endswith(".jsonl")
    if is_jsonl:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        # Try loading as a JSON array/list
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for cand in data:
                        yield cand
                else:
                    yield data
        except json.JSONDecodeError:
            # Fallback for concatenated json objects
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(content):
                while pos < len(content) and content[pos].isspace():
                    pos += 1
                if pos >= len(content):
                    break
                obj, next_pos = decoder.raw_decode(content, pos)
                pos = next_pos
                yield obj

def main():
    start_total = time.time()
    parser = argparse.ArgumentParser(description="Generate candidate profile embeddings.")
    parser.add_argument(
        "--candidates", 
        type=str, 
        default="resources/candidates.jsonl",
        help="Path to candidates JSON/JSONL file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path to output pickle file"
    )
    args = parser.parse_args()

    # Resolve candidates path relative to project root if it is relative
    candidates_path = args.candidates
    if not os.path.isabs(candidates_path):
        cwd_path = Path(candidates_path)
        if cwd_path.exists():
            candidates_path = cwd_path.resolve()
        else:
            candidates_path = (PROJECT_ROOT / candidates_path).resolve()
    else:
        candidates_path = Path(candidates_path)

    if not candidates_path.exists():
        print(f"ERROR: Candidates file not found at {candidates_path}")
        sys.exit(1)

    # Determine default output file name and create directory
    output_dir = ROOT / "stage2" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        output_file = Path(args.output)
    else:
        # Determine based on candidates filename
        in_basename = candidates_path.stem
        # If 'candidates' -> 'candidates_100k_embedded.pkl' or if 'test_candidates' -> 'test_candidates_embedded.pkl'
        if in_basename == "candidates":
            output_file = output_dir / "candidates_100k_embedded.pkl"
        else:
            output_file = output_dir / f"{in_basename}_embedded.pkl"

    print("="*80)
    print("TALENTPRISM - CANDIDATE EMBEDDING GENERATION")
    print("="*80)
    print(f"Candidates file: {candidates_path}")
    print(f"Output embeddings file: {output_file}")
    print(f"PyTorch CPU threads: {torch.get_num_threads()}")

    print("\n[1/3] Loading SentenceTransformer model (BAAI/bge-base-en-v1.5)...")
    start_model = time.time()
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print(f"Model loaded successfully in {time.time() - start_model:.2f}s.")

    # Batch configuration
    BATCH_SIZE = 1000
    temp_dir = output_dir / "temp_embeds"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[2/3] Processing candidates...")
    start_pipeline = time.time()

    # Global cache to prevent re-embedding repetitive texts (e.g. skills, titles, degrees)
    embedding_cache = {}

    candidates_gen = stream_candidates(candidates_path)
    batch_idx = 0

    while True:
        # Read batch
        candidates_list = []
        for _ in range(BATCH_SIZE):
            try:
                candidates_list.append(next(candidates_gen))
            except StopIteration:
                break

        if not candidates_list:
            break

        batch_idx += 1
        batch_start = time.time()
        cand_start_num = (batch_idx - 1) * BATCH_SIZE + 1
        cand_end_num = cand_start_num + len(candidates_list) - 1

        print(f"\n--- Batch {batch_idx}: Processing Candidates {cand_start_num} to {cand_end_num} ---")

        # 1. Collect all texts to embed in this batch
        texts_to_embed = []
        text_mapping = []  # Tuples: (cand_batch_idx, field_type, sub_idx)

        for c_idx, cand in enumerate(candidates_list):
            # Summary text
            summary_text = clean_str(cand.get("profile", {}).get("summary", ""))
            if summary_text:
                texts_to_embed.append(summary_text)
                text_mapping.append((c_idx, "summary", 0))

            history = cand.get("career_history", [])
            chronological_roles = history[::-1]

            # Chronological titles
            titles_with_dur = []
            for r in chronological_roles:
                is_curr = r.get("is_current", False)
                title_str = format_title_role(r, is_curr)
                if title_str:
                    titles_with_dur.append(title_str)
            titles_text = "->".join(titles_with_dur) if titles_with_dur else ""
            texts_to_embed.append(titles_text)
            text_mapping.append((c_idx, "titles_chronological", 0))

            # Combined company/size/industry
            comp_ind_size_blocks = []
            for r in chronological_roles:
                is_curr = r.get("is_current", False)
                comp_str = format_company_role(r, is_curr)
                if comp_str:
                    comp_ind_size_blocks.append(comp_str)
            comp_ind_size_text = "->".join(comp_ind_size_blocks)
            if comp_ind_size_text:
                texts_to_embed.append(comp_ind_size_text)
                text_mapping.append((c_idx, "company_industry_size_chronological", 0))

            # Role blocks
            for r_idx, role in enumerate(history):
                title = clean_str(role.get("title", ""))
                company = clean_str(role.get("company", ""))
                duration = role.get("duration_months", 0)
                description = clean_str(role.get("description", ""))
                role_block = f"Title: {title}\nCompany: {company}\nDuration: {duration} months\nDescription: {description}"
                texts_to_embed.append(role_block)
                text_mapping.append((c_idx, "role_block", r_idx))

            # Skills
            for s_idx, sk in enumerate(cand.get("skills", [])):
                sk_name = clean_str(sk.get("name", ""))
                if sk_name:
                    texts_to_embed.append(sk_name)
                    text_mapping.append((c_idx, "skill", s_idx))

            # Education
            for e_idx, edu in enumerate(cand.get("education", [])):
                degree = clean_str(edu.get("degree", ""))
                field = clean_str(edu.get("field_of_study", ""))
                inst = clean_str(edu.get("institution", ""))
                tier = clean_str(edu.get("tier", ""))
                grade = clean_str(edu.get("grade", ""))
                edu_block = f"Degree: {degree}\nField: {field}\nInstitution: {inst}\nTier: {tier}\nGrade: {grade}"
                texts_to_embed.append(edu_block)
                text_mapping.append((c_idx, "education_block", e_idx))

        print(f"Total texts in batch: {len(texts_to_embed)}")
        if texts_to_embed:
            # Find unique texts in the batch that are not already in the cache
            unique_texts_needed = list(set(text for text in texts_to_embed if text not in embedding_cache))
            
            if unique_texts_needed:
                print(f"Encoding {len(unique_texts_needed)} new unique texts (out of {len(texts_to_embed)})...")
                new_embeddings = model.encode(
                    unique_texts_needed,
                    batch_size=512,
                    show_progress_bar=True,
                    normalize_embeddings=True
                )
                # Save to global cache
                for text, emb in zip(unique_texts_needed, new_embeddings):
                    embedding_cache[text] = emb
            else:
                print("All texts in this batch resolved from cache!")
            
            # Map batch texts to their embeddings using the cache
            all_embeddings = [embedding_cache[text] for text in texts_to_embed]
        else:
            all_embeddings = []

        # Reconstruct candidate objects with embeddings
        embedded_candidates_batch = []
        for cand in candidates_list:
            p = cand.get("profile", {})
            embedded_cand = {
                "candidate_id": cand.get("candidate_id", "UNKNOWN"),
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
                
            embedded_candidates_batch.append(embedded_cand)

        # Map embeddings back (and cast to float16)
        for emb, (c_idx, f_type, sub_idx) in zip(all_embeddings, text_mapping):
            ec = embedded_candidates_batch[c_idx]
            f16_emb = emb.astype(np.float16)
            if f_type == "summary":
                ec["profile"]["summary"]["embedding"] = f16_emb
            elif f_type == "titles_chronological":
                ec["profile"]["titles_chronological_embedding"] = f16_emb
            elif f_type == "company_industry_size_chronological":
                ec["profile"]["company_industry_size_chronological_embedding"] = f16_emb
            elif f_type == "role_block":
                ec["career_history"][sub_idx]["role_embedding"] = f16_emb
            elif f_type == "skill":
                ec["skills"][sub_idx]["embedding"] = f16_emb
            elif f_type == "education_block":
                ec["education"][sub_idx]["education_embedding"] = f16_emb

        # Fallback to zero vectors for any missing embeddings
        for ec in embedded_candidates_batch:
            if ec["profile"]["summary"]["embedding"] is None:
                ec["profile"]["summary"]["embedding"] = np.zeros(768, dtype=np.float16)
            if ec["profile"]["titles_chronological_embedding"] is None:
                ec["profile"]["titles_chronological_embedding"] = np.zeros(768, dtype=np.float16)
            if ec["profile"]["company_industry_size_chronological_embedding"] is None:
                ec["profile"]["company_industry_size_chronological_embedding"] = np.zeros(768, dtype=np.float16)
            for r in ec["career_history"]:
                if r["role_embedding"] is None:
                    r["role_embedding"] = np.zeros(768, dtype=np.float16)
            for s in ec["skills"]:
                if s["embedding"] is None:
                    s["embedding"] = np.zeros(768, dtype=np.float16)
            for edu in ec["education"]:
                if edu["education_embedding"] is None:
                    edu["education_embedding"] = np.zeros(768, dtype=np.float16)

        # Save batch to temporary pickle file on disk to prevent RAM bloat
        temp_file = temp_dir / f"batch_{batch_idx}.pkl"
        with open(temp_file, "wb") as temp_f:
            pickle.dump(embedded_candidates_batch, temp_f, protocol=5)

        print(f"Batch {batch_idx} completed in {time.time() - batch_start:.2f}s.")

        # Clean memory
        del candidates_list
        del texts_to_embed
        del text_mapping
        del all_embeddings
        del embedded_candidates_batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDeleting model and freeing system RAM/VRAM before merging...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[3/3] Merging and saving embedded candidates to {output_file}...")
    start_merge = time.time()

    embedded_candidates_all = []
    for idx in range(1, batch_idx + 1):
        temp_file = temp_dir / f"batch_{idx}.pkl"
        if temp_file.exists():
            with open(temp_file, "rb") as temp_f:
                batch_cands = pickle.load(temp_f)
                embedded_candidates_all.extend(batch_cands)
            temp_file.unlink()

    with open(output_file, "wb") as f:
        pickle.dump(embedded_candidates_all, f, protocol=5)

    print(f"File saved successfully in {time.time() - start_merge:.2f}s.")

    try:
        temp_dir.rmdir()
    except Exception:
        pass

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"Final output file size: {file_size_mb:.2f} MB")
    print(f"Total processing time: {(time.time() - start_pipeline)/60:.2f} mins.")
    print(f"Total execution time: {time.time() - start_total:.2f} seconds ({(time.time() - start_total)/60:.2f} minutes).")
    print("="*80)

if __name__ == "__main__":
    main()
