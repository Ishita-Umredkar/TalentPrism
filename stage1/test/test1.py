"""
test1.py

Runs Stage 1 Honeypot Detection on the large resources/candidates.jsonl dataset.
Generates candidate credibility classification and defect analysis tables,
saving the results to stage1/outputs/output1.
"""

import os
import json
from datetime import datetime
from stage1.scripts import ALL_DETECTORS

def run_large_scale_analysis():
    input_file = "resources/candidates.jsonl"
    output_file = "stage1/outputs/output1"

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input dataset not found at {input_file}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Instantiate detectors
    detectors = [det_cls() for det_cls in ALL_DETECTORS]
    print(f"Loaded {len(detectors)} detectors for analysis.")

    # Counters for credibility categories
    counts = {
        "Sure Honeypots (0.0 to 0.3)": 0,
        "Likely Honeypots (0.3 to 0.6)": 0,
        "Needs Review (0.6 to 0.85)": 0,
        "Credible Candidates (0.85 to 1.0)": 0
    }

    # Counters for defects per check ID
    defect_counts = {det.check_id: 0 for det in detectors}

    total_processed = 0
    start_time = datetime.now()

    print("Starting processing. This may take a moment due to the large dataset size...")

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                continue

            total_processed += 1
            
            # Run all checks
            total_penalty = 0.0
            for detector in detectors:
                evidence = detector.detect(candidate)
                if evidence:
                    # Increment defect count for this check
                    defect_counts[detector.check_id] += 1
                    total_penalty += sum(ev.get("penalty", 0.0) for ev in evidence)

            # Compute credibility score
            score = max(1.0 - total_penalty, 0.0)
            score = round(score, 3)

            # Classify candidate
            if score <= 0.30:
                counts["Sure Honeypots (0.0 to 0.3)"] += 1
            elif score < 0.60:
                counts["Likely Honeypots (0.3 to 0.6)"] += 1
            elif score < 0.85:
                counts["Needs Review (0.6 to 0.85)"] += 1
            else:
                counts["Credible Candidates (0.85 to 1.0)"] += 1

            # Print progress update
            if total_processed % 5000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"Processed {total_processed} candidates... ({elapsed:.1f}s elapsed)")

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nProcessing complete! Processed {total_processed} candidates in {duration:.1f} seconds.")

    # Write tables to stage1/outputs/output1
    with open(output_file, "w", encoding="utf-8") as out:
        out.write("# Stage 1: Honeypot Detection Analysis Report\n\n")
        out.write(f"Analyzed **{total_processed}** candidates from `{input_file}`.\n")
        out.write(f"Execution duration: **{duration:.2f} seconds**.\n\n")

        # Table 1: Candidate Credibility Category Analysis
        out.write("## 1. Candidate Credibility Distribution\n\n")
        out.write("| Credibility Classification Category | Count | Percentage |\n")
        out.write("|-------------------|-------|------------|\n")
        for category, count in counts.items():
            percentage = (count / total_processed) * 100 if total_processed > 0 else 0
            out.write(f"| {category} | {count} | {percentage:.2f}% |\n")
        out.write("\n")

        # Table 2: Defect Analysis (sorted descending)
        out.write("## 2. Integrity Defect Analysis\n\n")
        out.write("| Defect Check ID | Defect Count | Trigger Rate |\n")
        out.write("|-----------------|--------------|--------------|\n")
        
        sorted_defects = sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)
        for check_id, count in sorted_defects:
            rate = (count / total_processed) * 100 if total_processed > 0 else 0
            out.write(f"| {check_id} | {count} | {rate:.2f}% |\n")
        out.write("\n")

    print(f"Analysis tables successfully written to {output_file}.")

if __name__ == "__main__":
    run_large_scale_analysis()
