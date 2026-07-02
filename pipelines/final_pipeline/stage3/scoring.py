import sys
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.append(str(PARENT))

# Import active detectors from stage1
from stage1.scripts import ALL_DETECTORS

# Import evaluate_section_score, evaluate_category_score from stage2
from stage2.scripts.rank_100k import (
    evaluate_section_score,
    evaluate_category_score
)

class CredibilityEvaluator:
    """Evaluates candidate profiles using Stage 1 credibility and honeypot detectors."""
    
    def __init__(self):
        self.detectors = [d() for d in ALL_DETECTORS]

    def evaluate_credibility(self, candidates):
        """Computes credibility scores for all candidates on the fly."""
        print("Computing Stage 1 credibility scores...")
        credibility = {}
        for cand_id, cand in candidates.items():
            penalties = 0.0
            for d in self.detectors:
                evidences = d.detect(cand)
                for ev in evidences:
                    penalties += ev.get("penalty", 0.0)
            credibility[cand_id] = max(1.0 - penalties, 0.0)
        return credibility

class FitEvaluator:
    """Evaluates Stage 2 fit constraints and custom categories, performs scoring and tie-breaking."""
    
    def __init__(self, candidates, constraints_data, embedded_candidates, query_embeddings_cache, credibility_scores):
        self.candidates = candidates
        self.constraints_data = constraints_data
        self.embedded_candidates = embedded_candidates
        self.query_embeddings_cache = query_embeddings_cache
        self.credibility_scores = credibility_scores

    def evaluate_single_constraint_score(self, c, ec, section_name):
        """Evaluates scoring or penalty metrics for a single constraint."""
        is_negative_section = section_name in ("negative", "rejection")
        sub_constraints = c.get("sub_constraints", [])
        
        if is_negative_section and c.get("type") == "conflicting":
            good_score = 0.0
            bad_score = 0.0
            for sub in sub_constraints:
                sub_item = sub.get("item", "")
                sub_type = sub.get("type", "bad")
                categories = sub.get("categories", [])
                query_emb = self.query_embeddings_cache[sub_item]

                cat_score_sum = 0.0
                cat_weight_sum = sum(cat.get("weight", 0.0) for cat in categories)

                for cat in categories:
                    cat_item = cat.get("item", "")
                    cat_strategy = cat.get("matching_strategy", "")
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
                query_emb = self.query_embeddings_cache[sub_item]

                cat_score_sum = 0.0
                cat_weight_sum = sum(cat.get("weight", 0.0) for cat in categories)

                for cat in categories:
                    cat_item = cat.get("item", "")
                    cat_strategy = cat.get("matching_strategy", "")
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

    def get_candidate_categories(self, ec, mh=None, pref=None, neg=None, rej=None):
        """Computes custom category scores for a given embedded candidate."""
        if mh is None:
            mh = [self.evaluate_single_constraint_score(c, ec, "must_have") for c in self.constraints_data.get("must_have", [])]
        if pref is None:
            pref = [self.evaluate_single_constraint_score(c, ec, "preferred") for c in self.constraints_data.get("preferred", [])]
        if neg is None:
            neg = [self.evaluate_single_constraint_score(c, ec, "negative") for c in self.constraints_data.get("negative", [])]
        if rej is None:
            rej = [self.evaluate_single_constraint_score(c, ec, "rejection") for c in self.constraints_data.get("rejection", [])]

        return [
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

    def evaluate_and_rank(self):
        """Scores and ranks all candidates using fit and credibility components, with logistics tie-breaking."""
        print("Evaluating fit scores...")
        
        must_have = self.constraints_data.get("must_have", [])
        preferred = self.constraints_data.get("preferred", [])
        rejection = self.constraints_data.get("rejection", [])
        negative = self.constraints_data.get("negative", [])

        mh_weights = [c.get("weight", 0.0) for c in must_have]
        pref_weights = [c.get("weight", 0.0) for c in preferred]
        rej_weights = [c.get("weight", 0.0) for c in rejection]
        neg_weights = [c.get("weight", 0.0) for c in negative]

        mh_total_weight = sum(mh_weights)
        pref_total_weight = sum(pref_weights)
        rej_total_weight = sum(rej_weights)
        neg_total_weight = sum(neg_weights)

        combined = []
        for ec in self.embedded_candidates:
            cand_id = ec["candidate_id"]
            if cand_id not in self.candidates:
                continue

            # Evaluate each constraint exactly once
            mh = [self.evaluate_single_constraint_score(c, ec, "must_have") for c in must_have]
            pref = [self.evaluate_single_constraint_score(c, ec, "preferred") for c in preferred]
            rej = [self.evaluate_single_constraint_score(c, ec, "rejection") for c in rejection]
            neg = [self.evaluate_single_constraint_score(c, ec, "negative") for c in negative]

            # Compute section weighted averages
            s_must = float(sum(score * w for score, w in zip(mh, mh_weights)) / mh_total_weight if mh_total_weight > 0.0 else 0.0)
            s_pref = float(sum(score * w for score, w in zip(pref, pref_weights)) / pref_total_weight if pref_total_weight > 0.0 else 0.0)
            s_rej = float(sum(score * w for score, w in zip(rej, rej_weights)) / rej_total_weight if rej_total_weight > 0.0 else 0.0)
            s_neg = float(sum(score * w for score, w in zip(neg, neg_weights)) / neg_total_weight if neg_total_weight > 0.0 else 0.0)

            positive_score = 0.75 * s_must + 0.25 * s_pref
            negative_score = 0.75 * s_rej + 0.25 * s_neg
            final_fit_score = positive_score * (1.0 - negative_score)

            cred_score = self.credibility_scores.get(cand_id, 1.0)
            combined_score = final_fit_score * cred_score

            # Calculate logistics / hiring readiness score directly from pre-computed values
            hiring_readiness = (mh[4] + pref[3] + (1.0 - neg[1])) / 3.0

            combined.append({
                "candidate_id": cand_id,
                "fit_score": final_fit_score,
                "credibility_score": cred_score,
                "combined_score": combined_score,
                "hiring_readiness": hiring_readiness
            })

        # Sort descending by combined_score, tie-break by hiring_readiness descending, then candidate_id ascending
        combined.sort(key=lambda x: (-x["combined_score"], -x["hiring_readiness"], x["candidate_id"]))
        return combined
