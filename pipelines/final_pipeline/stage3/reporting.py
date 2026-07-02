import os
import csv
import json


class ReportWriter:
    """Handles generation of the CSV rankings submission file."""
    
    def __init__(self, output_csv):
        self.output_csv = output_csv

    def write_reports(self, top_n, candidates, embedded_candidates_map, reasoning_generator, fit_evaluator):
        """Generates the ranked CSV output file."""
        # Create directory if needed
        csv_dir = os.path.dirname(self.output_csv)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        # Write CSV
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
