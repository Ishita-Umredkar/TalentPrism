"""
run_detection.py

Main execution runner that loads candidates, runs all consistency checks, 
computes overall credibility scores, and outputs reports.
"""

import os
import sys
import json
import argparse
from typing import List, Dict
from pathlib import Path

# Add final_pipeline to sys.path to allow standalone execution
script_dir_path = Path(__file__).resolve().parent
pipeline_root_path = script_dir_path.parents[1]
if str(pipeline_root_path) not in sys.path:
    sys.path.append(str(pipeline_root_path))

# Import all detectors
from stage1.scripts import ALL_DETECTORS

def load_candidates(file_path: str) -> List[Dict]:
    """
    Loads candidates from file_path, supporting:
    1. Standard JSON (array of objects)
    2. Standard JSONL (one JSON object per line)
    3. Multiline concatenated JSON objects (like first100.json)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    # Strategy 1: Attempt to parse as standard JSON array/object
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
    except json.JSONDecodeError:
        pass

    # Strategy 2: Attempt to stream line by line as JSONLines
    candidates = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
        return candidates
    except json.JSONDecodeError:
        pass

    # Strategy 3: Concatenated JSON objects (e.g. multiline JSONLines)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        
        decoder = json.JSONDecoder()
        pos = 0
        candidates = []
        while pos < len(content):
            # Skip whitespace
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break
            try:
                obj, next_pos = decoder.raw_decode(content, pos)
                candidates.append(obj)
                pos = next_pos
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON at position {pos}: {e}")
        return candidates


def run_honeypot_detection(input_path: str, output_path: str) -> List[Dict]:
    """
    Runs honeypot detection on the candidate JSON file.
    """
    candidates = load_candidates(input_path)

    # Instantiate all detectors
    detectors = [det_cls() for det_cls in ALL_DETECTORS]
    print(f"Loaded {len(detectors)} consistency check detectors.")

    results = []

    for candidate in candidates:
        cand_id = candidate.get("candidate_id", "UNKNOWN")
        profile = candidate.get("profile", {})
        name = profile.get("anonymized_name", "Anonymous")
        
        evidence_list = []
        for detector in detectors:
            try:
                evidence = detector.detect(candidate)
                if evidence:
                    evidence_list.extend(evidence)
            except Exception as e:
                print(f"Error running detector '{detector.check_id}' on candidate {cand_id}: {e}")

        # Compute additive credibility score: 1.0 - sum(penalties), capped at 0.0
        total_penalty = sum(ev["penalty"] for ev in evidence_list)
        credibility_score = max(1.0 - total_penalty, 0.0)
        credibility_score = round(credibility_score, 3)

        # Determine overall credibility status
        if credibility_score < 0.60:
            status = "Honeypot/Suspicious"
        elif credibility_score < 0.85:
            status = "Needs Review"
        else:
            status = "Credible"

        results.append({
            "candidate_id": cand_id,
            "name": name,
            "credibility_score": credibility_score,
            "status": status,
            "evidence": evidence_list
        })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Processed {len(candidates)} candidates. Results written to {output_path}")
    return results


def print_summary_table(results: List[Dict]):
    """
    Prints a clean markdown summary table of the detection results.
    """
    print("\n### Honeypot Detection Results Summary")
    print("| Candidate ID | Name | Credibility Score | Status | Triggered Checks |")
    print("|--------------|------|-------------------|--------|------------------|")
    
    for r in results:
        triggered = [ev["check_id"] for ev in r["evidence"]]
        triggered_str = ", ".join(triggered) if triggered else "None"
        
        status_label = r["status"]

        print(f"| {r['candidate_id']} | {r['name']} | {r['credibility_score']:.3f} | {status_label} | {triggered_str} |")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Honeypot Detection Runner")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/test/test_candidates.json", 
        help="Path to input candidates JSON file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="stage1/outputs/honeypot_results.json", 
        help="Path to save output verification results"
    )
    args = parser.parse_args()

    # Determine absolute path to the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))

    input_abs_path = args.input if os.path.isabs(args.input) else os.path.join(project_root, args.input)
    output_abs_path = args.output if os.path.isabs(args.output) else os.path.join(pipeline_root, args.output)

    results = run_honeypot_detection(input_abs_path, output_abs_path)
    print_summary_table(results)
