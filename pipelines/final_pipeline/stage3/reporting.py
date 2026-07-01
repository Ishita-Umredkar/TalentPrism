import os
import csv
import json

class ReportWriter:
    """Handles generation of CSV rankings, JSON detailed profiles lists, and explanation reports."""
    
    def __init__(self, output_csv, output_explanation):
        self.output_csv = output_csv
        self.output_explanation = output_explanation

    def write_reports(self, top_n, candidates, embedded_candidates_map, reasoning_generator, fit_evaluator):
        """Generates all output files in ranked order."""
        # Create directories
        csv_dir = os.path.dirname(self.output_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
            
        exp_dir = os.path.dirname(self.output_explanation)
        if exp_dir:
            os.makedirs(exp_dir, exist_ok=True)

        # 1. Write CSV
        print(f"Writing CSV to {self.output_csv}...")
        with open(self.output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for rank_idx, entry in enumerate(top_n, start=1):
                cand_id = entry["candidate_id"]
                cand = candidates[cand_id]
                ec = embedded_candidates_map[cand_id]
                score = round(entry["combined_score"], 6)
                reasoning = reasoning_generator.generate_reasoning(
                    cand, ec, entry["fit_score"], entry["credibility_score"], rank_idx, fit_evaluator
                )
                writer.writerow([cand_id, rank_idx, score, reasoning])

        # 2. Write JSON profile list
        output_json = self.output_csv.replace(".csv", ".json")
        print(f"Writing JSON profile list to {output_json}...")
        ranked_profiles = []
        for rank_idx, entry in enumerate(top_n, start=1):
            cand_id = entry["candidate_id"]
            cand = candidates[cand_id]
            ec = embedded_candidates_map[cand_id]
            reasoning = reasoning_generator.generate_reasoning(
                cand, ec, entry["fit_score"], entry["credibility_score"], rank_idx, fit_evaluator
            )
            
            profile_entry = cand.copy()
            profile_entry["rank"] = rank_idx
            profile_entry["score"] = round(entry["combined_score"], 6)
            profile_entry["reasoning"] = reasoning
            ranked_profiles.append(profile_entry)
            
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(ranked_profiles, f, indent=2, ensure_ascii=False)

        # 3. Generate Explanation File
        print(f"Writing explanation report to {self.output_explanation}...")
        credibility_stats = {}
        for entry in top_n:
            c_score = entry["credibility_score"]
            credibility_stats[c_score] = credibility_stats.get(c_score, 0) + 1

        top_limit = len(top_n)
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

        with open(self.output_explanation, "w", encoding="utf-8") as f:
            f.write("\n".join(explanation_lines))
