import json
import random
import re
from pathlib import Path

class ReasoningGenerator:
    """Generates natural, rank-aware explainability text for candidate rankings."""
    
    JD_YOE_MIN = 5
    JD_YOE_MAX = 9
    
    MISALIGNED_TITLE_KW = ["computer vision", "cv engineer", "speech", "robotics", "frontend", "devops"]
    LEADERSHIP_TITLE_KW = ["director", "vp", "head of", "chief", "manager"]
    
    # Multiple phrasings per tier to avoid repetition
    TIER_QUALIFIERS = {
        "elite": [
            "Strong overall fit",
            "Closely aligned with the role",
            "Top-tier match"
        ],
        "high": [
            "Well-suited for the role",
            "Solid match on core requirements",
            "Strong candidate overall"
        ],
        "solid": [
            "Reasonable fit with some gaps",
            "Decent alignment though not top-tier",
            "Moderate-strength candidate"
        ],
        "borderline": [
            "Partial fit",
            "Some relevant experience but notable gaps",
            "Narrower alignment than higher-ranked candidates"
        ],
        "tail": [
            "Adjacent profile",
            "Tangential fit to the JD's core needs",
            "Limited alignment on key dimensions"
        ]
    }
    
    def __init__(self, templates_path=None):
        if templates_path is None:
            templates_path = Path(__file__).resolve().parent / "phrasing_templates.json"
        with open(templates_path, "r", encoding="utf-8") as f:
            self.phrasing = json.load(f)

    def _get_rank_tier(self, rank_idx):
        """Returns rank tier string for framing logic."""
        if rank_idx <= 5:
            return "elite"
        elif rank_idx <= 20:
            return "high"
        elif rank_idx <= 50:
            return "solid"
        elif rank_idx <= 80:
            return "borderline"
        else:
            return "tail"

    def _build_opening(self, yoe, title, rank_idx, cand_id):
        """Creates a varied, JD-contextualized opening with rank-tier framing.
        
        Varies both qualifier phrasing and sentence structure deterministically
        based on candidate ID to avoid repetitive output across rows.
        Only references the JD's 5-9 year band when YOE falls outside it.
        """
        tier = self._get_rank_tier(rank_idx)
        cand_num = int(re.sub(r'\D', '', cand_id))
        title_lower = title.lower()
        
        # Select qualifier phrase (deterministic by cand_id)
        qualifiers = self.TIER_QUALIFIERS[tier]
        qualifier = qualifiers[cand_num % len(qualifiers)]
        
        # YOE context — only reference JD band when outside range
        if yoe < self.JD_YOE_MIN:
            yoe_part = f"{yoe} YOE (under the JD's 5-9 range)"
        elif yoe > self.JD_YOE_MAX:
            yoe_part = f"{yoe} YOE (above the JD's target range)"
        else:
            yoe_part = f"{yoe} YOE"
        
        # Title alignment flag (only when noteworthy)
        title_flag = ""
        if any(kw in title_lower for kw in self.LEADERSHIP_TITLE_KW):
            title_flag = "; JD prefers hands-on IC"
        elif any(kw in title_lower for kw in self.MISALIGNED_TITLE_KW):
            title_flag = "; focus area outside the JD's NLP/search core"
        
        # Vary sentence structure independently from qualifier
        structures = [
            f"{qualifier} — {yoe_part} as a {title}{title_flag}.",
            f"{title} with {yoe_part}{title_flag}; {qualifier.lower()}.",
            f"{qualifier} — {title}, {yoe_part}{title_flag}.",
        ]
        structure_idx = (cand_num // len(qualifiers)) % len(structures)
        
        return structures[structure_idx]

    def generate_reasoning(self, cand, ec, fit_score, credibility_score, rank_idx, fit_evaluator):
        """Generates dynamic, natural 1-2 sentence justification for a candidate."""
        profile = cand.get("profile", {})
        yoe = profile.get("years_of_experience", 0)
        title = profile.get("current_title", "Engineer")
        if not title:
            title = profile.get("headline", "Applied Engineer").split("|")[0].strip()
        
        cand_id = cand.get("candidate_id", "")
        opening = self._build_opening(yoe, title, rank_idx, cand_id)
        comparison_line = self._build_comparison(ec, cand_id, fit_evaluator, rank_idx)
        
        gap = ""
        if credibility_score < 1.0:
            gap = " Note: timeline inconsistency detected (score adjusted)."
        
        return f"{opening} {comparison_line}{gap}"

    def _build_comparison(self, ec, cand_id, fit_evaluator, rank_idx):
        """Builds comparison phrases with rank-aware negative forcing.
        
        Uses progressively higher weak-score thresholds for lower tiers so that
        borderline and tail candidates always surface honest concerns, even when
        their absolute category scores are modestly above 0.5.
        """
        cand_num = int(re.sub(r'\D', '', cand_id))
        rng = random.Random(cand_num)
        
        categories = fit_evaluator.get_candidate_categories(ec)
        tier = self._get_rank_tier(rank_idx)
        
        by_extreme = sorted(categories, key=lambda x: abs(x["score"] - 0.5), reverse=True)
        by_score = sorted(categories, key=lambda x: x["score"])
        
        # Tier config: (n_top_extreme, n_forced_weak, weak_score_threshold)
        # Higher thresholds for lower tiers ensure negatives surface
        tier_config = {
            "elite":      (4, 0, 0.0),
            "high":       (3, 1, 0.55),
            "solid":      (3, 1, 0.58),
            "borderline": (2, 2, 0.62),
            "tail":       (2, 2, 0.65),
        }
        n_top, n_forced, weak_threshold = tier_config[tier]
        
        # Select top extreme categories
        top_picks = list(by_extreme[:n_top])
        selected_ids = {x["id"] for x in top_picks}
        
        # Force weakest categories that fall below the tier's threshold
        forced_weak = []
        for cat in by_score:
            if len(forced_weak) >= n_forced:
                break
            if cat["id"] not in selected_ids and cat["score"] < weak_threshold:
                forced_weak.append(cat)
                selected_ids.add(cat["id"])
        
        all_selected = top_picks + forced_weak
        forced_weak_ids = {c["id"] for c in forced_weak}

        positive_phrases = []
        negative_phrases = []
        
        for item in all_selected:
            cat_id = item["id"]
            score = item["score"]
            is_good = score >= 0.5 if cat_id not in forced_weak_ids else False
            
            templates = self.phrasing.get(cat_id, {}).get("good" if is_good else "bad", [])
            phrase = rng.choice(templates) if templates else (f"satisfy {cat_id}" if is_good else f"lack {cat_id}")
            
            (positive_phrases if is_good else negative_phrases).append(phrase)
                
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
