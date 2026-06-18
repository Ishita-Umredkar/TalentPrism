import pickle
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EMBEDDINGS_FILE = ROOT / "stage2" / "outputs" / "test_candidates_embedded.pkl"

def verify_embeddings():
    print(f"Reading embeddings from: {EMBEDDINGS_FILE}")
    if not EMBEDDINGS_FILE.exists():
        print(f"ERROR: File not found at {EMBEDDINGS_FILE}")
        return False

    with open(EMBEDDINGS_FILE, "rb") as f:
        candidates = pickle.load(f)

    print(f"Total candidates found: {len(candidates)}")
    if len(candidates) != 10:
        print(f"WARNING: Expected 10 candidates, but got {len(candidates)}")

    success = True
    for idx, cand in enumerate(candidates):
        cand_id = cand.get("candidate_id", "UNKNOWN")
        print(f"\nVerifying Candidate {idx+1}/{len(candidates)}: {cand_id}")
        
        # 1. Profile Checks
        profile = cand.get("profile", {})
        summary_data = profile.get("summary", {})
        summary_text = summary_data.get("text", "")
        summary_emb = summary_data.get("embedding")
        
        if summary_emb is None:
            print("  - ERROR: profile.summary.embedding is missing!")
            success = False
        else:
            emb_shape = np.shape(summary_emb)
            if emb_shape != (768,):
                print(f"  - ERROR: profile.summary.embedding shape is {emb_shape}, expected (768,)!")
                success = False
        
        summary_sents = profile.get("summary_sentences", [])
        print(f"  - Profile Summary: {len(summary_sents)} sentences split.")
        for s_idx, s in enumerate(summary_sents):
            s_text = s.get("text", "")
            s_emb = s.get("embedding")
            if s_emb is None or np.shape(s_emb) != (768,):
                print(f"    - ERROR: Summary sentence {s_idx} embedding issue!")
                success = False

        # 2. Career History Checks
        career_history = cand.get("career_history", [])
        print(f"  - Career History: {len(career_history)} roles found.")
        for r_idx, role in enumerate(career_history):
            title = role.get("title", "")
            t_emb = role.get("title_embedding")
            c_emb = role.get("company_embedding")
            i_emb = role.get("industry_embedding")
            role_emb = role.get("role_embedding")
            desc_sents = role.get("description_sentences", [])
            
            # check shapes
            for field, emb in [("title", t_emb), ("company", c_emb), ("industry", i_emb), ("role block", role_emb)]:
                if emb is None or np.shape(emb) != (768,):
                    print(f"    - ERROR: Role {r_idx} ({title}) {field} embedding issue!")
                    success = False
            
            print(f"    - Role {r_idx} ({title}): {len(desc_sents)} sentences split from description.")
            for s_idx, s in enumerate(desc_sents):
                s_text = s.get("text", "")
                s_emb = s.get("embedding")
                if s_emb is None or np.shape(s_emb) != (768,):
                    print(f"      - ERROR: Role {r_idx} desc sentence {s_idx} embedding issue!")
                    success = False

        # 3. Technical Skills Checks
        skills = cand.get("skills", [])
        print(f"  - Technical Skills: {len(skills)} skills found.")
        for sk_idx, sk in enumerate(skills):
            sk_name = sk.get("name", "")
            sk_emb = sk.get("embedding")
            if sk_emb is None or np.shape(sk_emb) != (768,):
                print(f"    - ERROR: Skill {sk_name} embedding issue!")
                success = False

        # 4. Education Checks
        education = cand.get("education", [])
        print(f"  - Education: {len(education)} education entries found.")
        for e_idx, edu in enumerate(education):
            degree = edu.get("degree", "")
            edu_emb = edu.get("education_embedding")
            if edu_emb is None or np.shape(edu_emb) != (768,):
                print(f"    - ERROR: Education {e_idx} ({degree}) embedding issue!")
                success = False

    if success:
        print("\nAll embeddings verified successfully! Dimensions and keys are correct.")
    else:
        print("\nVerification failed. Please review errors above.")
    
    return success

if __name__ == "__main__":
    verify_embeddings()
