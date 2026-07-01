import json
import random
import re
from pathlib import Path

class ReasoningGenerator:
    """Selects extreme categories and generates natural explainability text from templates."""
    
    def __init__(self, templates_path=None):
        if templates_path is None:
            templates_path = Path(__file__).resolve().parent / "phrasing_templates.json"
        
        with open(templates_path, "r", encoding="utf-8") as f:
            self.phrasing = json.load(f)

    def generate_reasoning(self, cand, ec, fit_score, credibility_score, rank_idx, fit_evaluator):
        """Generates dynamic, natural 1-2 sentence justification for a candidate."""
        profile = cand.get("profile", {})
        yoe = profile.get("years_of_experience", 0)
        title = profile.get("current_title", "Engineer")
        if not title:
            title = profile.get("headline", "Applied Engineer").split("|")[0].strip()
            
        generic_line = f"Candidate possesses {yoe} YOE as a {title}."
        comparison_line = self.get_natural_comparison(ec, cand.get("candidate_id", ""), fit_evaluator)
        
        gap = ""
        if credibility_score < 1.0:
            gap = " Note: Minor timeline check triggered (score adjustment applied)."
            
        reasoning = f"{generic_line} {comparison_line}{gap}"
        return reasoning

    def get_natural_comparison(self, ec, cand_id, fit_evaluator):
        """Finds the 4 most extreme category scores and maps them to templated phrases."""
        # Seeding randomizer deterministically by candidate ID number
        cand_num = int(re.sub(r'\D', '', cand_id))
        rng = random.Random(cand_num)
        
        # Calculate custom categories using fit_evaluator
        categories = fit_evaluator.get_candidate_categories(ec)
        
        # Calculate extremeness: distance from neutral 0.5
        scored_cats = []
        for item in categories:
            extremeness = abs(item["score"] - 0.5)
            scored_cats.append((item, extremeness))
            
        # Sort by extremeness descending
        scored_cats.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 4
        selected = [x[0] for x in scored_cats[:4]]

        positive_phrases = []
        negative_phrases = []
        
        for item in selected:
            cat_id = item["id"]
            score = item["score"]
            is_good = score >= 0.5
            
            templates = self.phrasing.get(cat_id, {}).get("good" if is_good else "bad", [])
            if templates:
                phrase = rng.choice(templates)
            else:
                phrase = f"satisfy {cat_id}" if is_good else f"lack {cat_id}"
                
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
