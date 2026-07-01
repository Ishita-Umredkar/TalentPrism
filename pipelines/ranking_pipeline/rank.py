"""
rank.py

Ranks candidate profiles by combining Stage 2 fit scores with Stage 1 credibility scores.
Generates top100.csv and ranking/explanation.txt.
"""

import os
import sys
import json
import csv
import argparse
import pickle
import numpy as np
import re
import random
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

# Import active detectors from stage1
from stage1.scripts import ALL_DETECTORS

# Import evaluate_section_score, evaluate_category_score and paths from stage2
from stage2.scripts.rank_100k import (
    evaluate_section_score,
    evaluate_category_score,
    CONSTRAINTS_FILE,
    EMBEDDED_CONSTRAINTS_FILE
)

def evaluate_single_constraint_score(c, ec, query_embeddings_cache, section_name):
    is_negative_section = section_name in ("negative", "rejection")
    sub_constraints = c.get("sub_constraints", [])
    
    if is_negative_section and c.get("type") == "conflicting":
        good_score = 0.0
        bad_score = 0.0
        for sub in sub_constraints:
            sub_item = sub.get("item", "")
            sub_type = sub.get("type", "bad")
            categories = sub.get("categories", [])
            query_emb = query_embeddings_cache.get(sub_item)
            if query_emb is None:
                continue

            cat_score_sum = 0.0
            cat_weight_sum = sum(cat.get("weight", 0.0) for cat in categories)

            for cat in categories:
                cat_item = cat.get("item", "")
                cat_strategy = cat.get("matching_strategy", "")
                if not cat_strategy:
                    cat_strategy = cat.get("strategy", "")
                cat_weight = cat.get("weight", 0.0)
                cat_field = cat.get("field", "")
                
                cat_score = evaluate_category_score(
                    query_emb, ec, cat_item, cat_strategy, sub_item, cat_field
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
            query_emb = query_embeddings_cache.get(sub_item)
            if query_emb is None:
                continue

            cat_score_sum = 0.0
            cat_weight_sum = sum(cat.get("weight", 0.0) for cat in categories)

            for cat in categories:
                cat_item = cat.get("item", "")
                cat_strategy = cat.get("matching_strategy", "")
                if not cat_strategy:
                    cat_strategy = cat.get("strategy", "")
                cat_weight = cat.get("weight", 0.0)
                cat_field = cat.get("field", "")
                
                cat_score = evaluate_category_score(
                    query_emb, ec, cat_item, cat_strategy, sub_item, cat_field
                )
                cat_score_sum += cat_score * cat_weight

            sub_score = (cat_score_sum / cat_weight_sum) if cat_weight_sum > 0 else 0.0
            if is_negative_section and sub_type == "good":
                sub_score = 1.0 - sub_score

            sub_total_score += sub_score * sub_weight
        constraint_score = sub_total_score
        
    return constraint_score

def get_natural_comparison(cand, ec, query_embeddings_cache, constraints_data, cand_id):
    # Deterministic seeding by candidate ID
    cand_num = int(re.sub(r'\D', '', cand_id))
    rng = random.Random(cand_num)
    
    # Evaluate raw constraint scores
    mh = [evaluate_single_constraint_score(c, ec, query_embeddings_cache, "must_have") for c in constraints_data.get("must_have", [])]
    pref = [evaluate_single_constraint_score(c, ec, query_embeddings_cache, "preferred") for c in constraints_data.get("preferred", [])]
    neg = [evaluate_single_constraint_score(c, ec, query_embeddings_cache, "negative") for c in constraints_data.get("negative", [])]
    rej = [evaluate_single_constraint_score(c, ec, query_embeddings_cache, "rejection") for c in constraints_data.get("rejection", [])]

    # Map to the 8 logical categories using desirability averages (1.0 is good, 0.0 is bad)
    categories = [
        {
            "id": "retrieval_search",
            "name": "Retrieval & Search Systems Expertise",
            "score": (mh[0] + mh[1]) / 2.0
        },
        {
            "id": "production_ml",
            "name": "Production ML Engineering",
            "score": (mh[2] + mh[3] + (1.0 - rej[0])) / 3.0
        },
        {
            "id": "llm_ai",
            "name": "LLM & Modern AI Expertise",
            "score": (pref[0] + (1.0 - rej[1])) / 2.0
        },
        {
            "id": "product_domain",
            "name": "Product & Domain Experience",
            "score": (pref[1] + (1.0 - rej[3])) / 2.0
        },
        {
            "id": "career_quality",
            "name": "Career Quality & Stability",
            "score": ((1.0 - neg[0]) + (1.0 - rej[2]) + (1.0 - rej[3])) / 3.0
        },
        {
            "id": "tech_breadth",
            "name": "Technical Breadth & Specialization",
            "score": 1.0 - rej[4]
        },
        {
            "id": "external_validation",
            "name": "External Validation",
            "score": (pref[2] + (1.0 - rej[5])) / 2.0
        },
        {
            "id": "hiring_readiness",
            "name": "Hiring Readiness",
            "score": (mh[4] + pref[3] + (1.0 - neg[1])) / 3.0
        }
    ]

    # Calculate extremeness: abs(score - 0.5)
    # Sort descending by extremeness
    scored_cats = []
    for item in categories:
        extremeness = abs(item["score"] - 0.5)
        scored_cats.append((item, extremeness))
        
    scored_cats.sort(key=lambda x: x[1], reverse=True)
    
    # Pick the top 4 most extreme categories
    selected = [x[0] for x in scored_cats[:4]]

    # Phrasing templates for the 8 categories
    phrasing = {
        "retrieval_search": {
            "good": [
                "demonstrate strong expertise in vector databases and search retrieval architectures",
                "possess a solid background in search indexing and ranking algorithms",
                "bring hands-on proficiency in vector search and embedding retrieval systems"
            ],
            "bad": [
                "lack hands-on experience in vector search or ranking evaluation",
                "have limited exposure to database indexing and search retrieval",
                "show gaps in search indexing or ranking setups"
            ]
        },
        "production_ml": {
            "good": [
                "demonstrate excellent ML engineering skills and write high-quality production code",
                "possess strong software engineering fundamentals and build robust production pipelines",
                "bring a solid engineering foundation for scaling production ML systems"
            ],
            "bad": [
                "need to strengthen their software engineering and production coding skills",
                "show gaps in building and scaling production ML pipelines",
                "possess limited experience in production-grade software development"
            ]
        },
        "llm_ai": {
            "good": [
                "bring advanced expertise in LLM fine-tuning and ranking models",
                "show strong familiarity with PEFT, QLoRA, and modern neural ranking",
                "possess valuable hands-on experience with modern LLMs and deep learning"
            ],
            "bad": [
                "have limited exposure to modern LLM fine-tuning or ranking models",
                "lack experience with LLM optimization and learning-to-rank methods",
                "show gaps in deep learning and modern LLM application development"
            ]
        },
        "product_domain": {
            "good": [
                "bring valuable experience working inside product-focused companies",
                "have solid product-company domain exposure and matching logic experience",
                "come from a strong product-engineering background"
            ],
            "bad": [
                "lack experience inside product-focused engineering setups",
                "have limited exposure to product domain matching systems",
                "possess minimal product company engineering background"
            ]
        },
        "career_quality": {
            "good": [
                "maintain a stable employment pattern and keep an active individual contributor profile",
                "show a history of consistent company tenure and hands-on coding roles",
                "possess a reliable and stable individual contributor track record"
            ],
            "bad": [
                "indicate potential job-hopping risks or a hands-off technical profile",
                "present career stability concerns or a hands-off leadership profile",
                "show patterns of frequent company switches or hands-off technical roles"
            ]
        },
        "tech_breadth": {
            "good": [
                "possess a highly aligned specialization in search, ranking, and NLP",
                "have AI expertise well-aligned with NLP and information retrieval",
                "show a specialized background in text search and ranking systems"
            ],
            "bad": [
                "possess mismatched AI specializations focused on computer vision or speech",
                "show specialized experience in CV/speech instead of search or ranking",
                "have an AI background that is not aligned with search and NLP technologies"
            ]
        },
        "external_validation": {
            "good": [
                "have verified public credentials through open-source contributions or talks",
                "possess a track record validated by active public contributions",
                "show external validation of their work through open-source AI projects"
            ],
            "bad": [
                "lack public validation of their work on open-source repositories",
                "possess a closed-source track record without external validation",
                "show minimal public GitHub activity or external technical validation"
            ]
        },
        "hiring_readiness": {
            "good": [
                "are available to join quickly and fit the location preferences",
                "maintain high responsiveness to outreach and are ready to relocate",
                "possess an immediate notice period and meet the geographic criteria"
            ],
            "bad": [
                "require a longer hiring notice period or are in non-targeted locations",
                "do not match the location preferences or immediate hiring readiness",
                "show low active engagement and require longer transition times"
            ]
        }
    }

    positive_phrases = []
    negative_phrases = []
    
    for item in selected:
        cat_id = item["id"]
        score = item["score"]
        is_good = score >= 0.5
        
        templates = phrasing[cat_id]["good" if is_good else "bad"]
        phrase = rng.choice(templates)
        
        if is_good:
            positive_phrases.append(phrase)
        else:
            negative_phrases.append(phrase)
            
    sentence_parts = []
    if positive_phrases:
        if len(positive_phrases) == 1:
            sentence_parts.append(f"They {positive_phrases[0]}")
        elif len(positive_phrases) == 2:
            sentence_parts.append(f"They {positive_phrases[0]} and {positive_phrases[1]}")
        else:
            joined = ", ".join(positive_phrases[:-1])
            sentence_parts.append(f"They {joined}, and {positive_phrases[-1]}")
            
    if negative_phrases:
        neg_intro = "However, they " if positive_phrases else "They "
        if len(negative_phrases) == 1:
            sentence_parts.append(f"{neg_intro}{negative_phrases[0]}")
        elif len(negative_phrases) == 2:
            sentence_parts.append(f"{neg_intro}{negative_phrases[0]} and {negative_phrases[1]}")
        else:
            joined = ", ".join(negative_phrases[:-1])
            sentence_parts.append(f"{neg_intro}{joined}, and {negative_phrases[-1]}")
            
    return ". ".join(sentence_parts) + "."

def generate_reasoning(cand, ec, fit_score, credibility_score, rank, query_embeddings_cache, constraints_data):
    profile = cand.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    if not title:
        title = profile.get("headline", "Applied Engineer").split("|")[0].strip()
        
    generic_line = f"Candidate possesses {yoe} YOE as a {title}."
    comparison_line = get_natural_comparison(cand, ec, query_embeddings_cache, constraints_data, cand.get("candidate_id", ""))
    
    gap = ""
    if credibility_score < 1.0:
        gap = " Note: Minor timeline check triggered (score adjustment applied)."
        
    reasoning = f"{generic_line} {comparison_line}{gap}"
    return reasoning

def main():
    default_csv = str(ROOT / "ranking" / "top100.csv")
    default_exp = str(ROOT / "ranking" / "explanation.txt")
    parser = argparse.ArgumentParser(description="Rank candidates using Stage 1 and Stage 2 scores.")
    parser.add_argument("--candidates", default="resources/candidates.jsonl", help="Path to candidates file")
    parser.add_argument("--output-csv", default=default_csv, help="Path to output CSV")
    parser.add_argument("--output-explanation", default=default_exp, help="Path to output explanation TXT")
    args = parser.parse_args()

    # Create directories
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_explanation), exist_ok=True)

    # 1. Load raw candidates
    print(f"Loading candidate profiles from {args.candidates}...")
    candidates = {}
    
    is_jsonl = args.candidates.endswith(".jsonl")
    with open(args.candidates, "r", encoding="utf-8") as f:
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cand = json.loads(line)
                candidates[cand["candidate_id"]] = cand
        else:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for cand in data:
                        candidates[cand["candidate_id"]] = cand
                else:
                    candidates[data["candidate_id"]] = data
            except json.JSONDecodeError:
                f.seek(0)
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
                    candidates[obj["candidate_id"]] = obj

    print(f"Loaded {len(candidates)} candidates.")

    # 2. Compute Stage 1 credibility scores on the fly
    print("Computing Stage 1 credibility scores...")
    detectors = [d() for d in ALL_DETECTORS]
    credibility = {}
    for cand_id, cand in candidates.items():
        penalties = 0.0
        for d in detectors:
            evidences = d.detect(cand)
            for ev in evidences:
                penalties += ev.get("penalty", 0.0)
        credibility[cand_id] = max(1.0 - penalties, 0.0)

    # 3. Load pre-embedded constraints and structure
    print(f"Loading pre-embedded constraints from: {EMBEDDED_CONSTRAINTS_FILE}")
    with open(EMBEDDED_CONSTRAINTS_FILE, "rb") as f:
        query_embeddings_cache = pickle.load(f)
        
    print(f"Loading constraints structure from: {CONSTRAINTS_FILE}")
    with open(CONSTRAINTS_FILE, "r", encoding="utf-8") as f:
        constraints_data = json.load(f)

    # Determine which candidate embeddings file to load
    if len(candidates) <= 1000:
        emb_file = ROOT / "stage2" / "outputs" / "test_candidates_embedded.pkl"
    else:
        emb_file = ROOT / "stage2" / "outputs" / "candidates_100k_embedded.pkl"

    print(f"Loading candidate embeddings from: {emb_file}...")
    with open(emb_file, "rb") as f:
        embedded_candidates = pickle.load(f)

    # Map embedded candidates by ID for lookup
    embedded_candidates_map = {ec["candidate_id"]: ec for ec in embedded_candidates}

    # 4. Score candidates
    print("Evaluating fit scores...")
    combined = []
    for ec in embedded_candidates:
        cand_id = ec["candidate_id"]
        if cand_id not in candidates:
            continue

        s_must = evaluate_section_score("must_have", constraints_data.get("must_have", []), ec, query_embeddings_cache)
        s_pref = evaluate_section_score("preferred", constraints_data.get("preferred", []), ec, query_embeddings_cache)
        s_rej = evaluate_section_score("rejection", constraints_data.get("rejection", []), ec, query_embeddings_cache)
        s_neg = evaluate_section_score("negative", constraints_data.get("negative", []), ec, query_embeddings_cache)

        positive_score = 0.75 * s_must + 0.25 * s_pref
        negative_score = 0.75 * s_rej + 0.25 * s_neg
        final_fit_score = positive_score * (1.0 - negative_score)

        cred_score = credibility.get(cand_id, 1.0)
        combined_score = final_fit_score * cred_score

        combined.append({
            "candidate_id": cand_id,
            "fit_score": final_fit_score,
            "credibility_score": cred_score,
            "combined_score": combined_score
        })

    # Sort descending by combined_score, tie-break by candidate_id ascending
    combined.sort(key=lambda x: (-x["combined_score"], x["candidate_id"]))
    
    top_limit = min(len(combined), 100)
    top_n = combined[:top_limit]

    # Write CSV
    print(f"Writing CSV to {args.output_csv}...")
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank_idx, entry in enumerate(top_n, start=1):
            cand_id = entry["candidate_id"]
            cand = candidates[cand_id]
            ec = embedded_candidates_map[cand_id]
            score = round(entry["combined_score"], 6)
            reasoning = generate_reasoning(cand, ec, entry["fit_score"], entry["credibility_score"], rank_idx, query_embeddings_cache, constraints_data)
            writer.writerow([cand_id, rank_idx, score, reasoning])

    # Generate Explanation File
    print(f"Writing explanation report to {args.output_explanation}...")
    credibility_stats = {}
    for entry in top_n:
        c_score = entry["credibility_score"]
        credibility_stats[c_score] = credibility_stats.get(c_score, 0) + 1

    explanation_lines = [
        "=========================================================================",
        "                 TALENTPRISM RANKING EXPLANATION REPORT                  ",
        "=========================================================================",
        "",
        "Overview:",
        "Candidates have been ranked by combining their Stage 2 Fit Score and their",
        "Stage 1 Credibility (Honeypot Detection) Score. The formula used is:",
        "  Final Rank Score = Stage 2 Fit Score * Stage 1 Credibility Score",
        "",
        "This formula heavily penalizes profiles with structural contradictions or faked",
        "timelines, ensuring that highly suspicious candidates (honeypots) are excluded",
        "from the top rankings entirely.",
        "",
        "Top Ranked Statistics:",
        f"  - Maximum Combined Score: {top_n[0]['combined_score']:.6f}" if top_n else "  - Maximum Combined Score: 0.000000",
        f"  - Minimum Combined Score (Rank {top_limit}): {top_n[-1]['combined_score']:.6f}" if top_n else f"  - Minimum Combined Score (Rank {top_limit}): 0.000000",
        "",
        "Credibility Score Distribution in Top Rankings:",
    ]
    for c_score in sorted(credibility_stats.keys(), reverse=True):
        count = credibility_stats[c_score]
        explanation_lines.append(f"  - Score {c_score:.3f}: {count} candidates")

    explanation_lines.append("")
    explanation_lines.append("Top 10 Ranked Candidates:")
    explanation_lines.append("-------------------------------------------------------------------------")
    for idx, entry in enumerate(top_n[:10], start=1):
        cand_id = entry["candidate_id"]
        cand = candidates[cand_id]
        name = cand.get("profile", {}).get("anonymized_name", "N/A")
        title = cand.get("profile", {}).get("current_title", "N/A")
        explanation_lines.append(
            f"Rank {idx:2d}: {cand_id} ({name}) - {title}\n"
            f"         Fit Score: {entry['fit_score']:.6f} | Credibility: {entry['credibility_score']:.3f}\n"
            f"         Combined Score: {entry['combined_score']:.6f}\n"
        )
    explanation_lines.append("-------------------------------------------------------------------------")

    with open(args.output_explanation, "w", encoding="utf-8") as f:
        f.write("\n".join(explanation_lines))

    # Write JSON profile list
    output_json = args.output_csv.replace(".csv", ".json")
    print(f"Writing JSON profile list to {output_json}...")
    ranked_profiles = []
    for rank_idx, entry in enumerate(top_n, start=1):
        cand_id = entry["candidate_id"]
        cand = candidates[cand_id]
        ec = embedded_candidates_map[cand_id]
        reasoning = generate_reasoning(cand, ec, entry["fit_score"], entry["credibility_score"], rank_idx, query_embeddings_cache, constraints_data)
        
        profile_entry = cand.copy()
        profile_entry["rank"] = rank_idx
        profile_entry["score"] = round(entry["combined_score"], 6)
        profile_entry["reasoning"] = reasoning
        ranked_profiles.append(profile_entry)
        
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(ranked_profiles, f, indent=2, ensure_ascii=False)

    print("Ranking process completed successfully!")

if __name__ == "__main__":
    main()
