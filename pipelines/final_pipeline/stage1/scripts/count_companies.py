"""
count_companies.py

A script to extract all company names from candidate career histories,
count their occurrences, and save the aggregated counts in the outputs directory.
"""

import os
import json
import csv
import argparse
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Count company occurrences in candidate profiles")
    parser.add_argument(
        "--input",
        type=str,
        default="resources/candidates.jsonl",
        help="Path to candidates dataset (JSON/JSONL format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="stage1/outputs/company_counts.csv",
        help="Path to save the resulting CSV file"
    )
    return parser.parse_args()

def load_candidates_generator(file_path: str):
    """
    Generator to stream candidates from file_path, supporting:
    1. Standard JSONL (one JSON object per line)
    2. Standard JSON Array
    3. Multiline concatenated JSON objects (like first100.json)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    # Strategy 1: Attempt to stream line by line as JSONLines (memory efficient)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    except json.JSONDecodeError:
        # If any line fails to parse, reset and try other strategies
        pass

    # Strategy 2: Read entire file and handle standard array or raw decode
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return

        # Try parsing as a standard JSON list
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    yield item
                return
            elif isinstance(data, dict):
                yield data
                return
        except json.JSONDecodeError:
            pass

        # Strategy 3: Concatenated JSON objects (e.g. pretty-printed JSONLines without newlines)
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            # Skip whitespace
            while pos < len(content) and content[pos].isspace():
                pos += 1
            if pos >= len(content):
                break
            try:
                obj, next_pos = decoder.raw_decode(content, pos)
                yield obj
                pos = next_pos
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON at position {pos}: {e}")

def main():
    args = parse_args()
    
    # Resolve absolute paths relative to project root / pipeline root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    
    input_path = args.input if os.path.isabs(args.input) else os.path.join(project_root, args.input)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(pipeline_root, args.output)
    
    print(f"Reading candidates from: {input_path}")
    
    company_counter = Counter()
    candidate_count = 0
    
    try:
        for candidate in load_candidates_generator(input_path):
            candidate_count += 1
            
            history = candidate.get("career_history", [])
            companies_in_candidate = []
            
            for job in history:
                comp = job.get("company")
                if comp:
                    comp_clean = comp.strip()
                    if comp_clean:
                        companies_in_candidate.append(comp_clean)
            
            # Fallback to current company if career history has no companies
            if not companies_in_candidate:
                profile = candidate.get("profile", {})
                curr_company = profile.get("current_company")
                if curr_company:
                    curr_company_clean = curr_company.strip()
                    if curr_company_clean:
                        companies_in_candidate.append(curr_company_clean)
                        
            for comp_clean in companies_in_candidate:
                company_counter[comp_clean] += 1
                
            if candidate_count % 10000 == 0:
                print(f"Processed {candidate_count} candidates...")
                
    except Exception as e:
        print(f"Error processing candidate dataset: {e}")
        return

    print(f"Successfully processed {candidate_count} candidates.")
    print(f"Found {len(company_counter)} unique companies.")
    
    # Sort by count descending, then by company name alphabetically
    sorted_companies = sorted(company_counter.items(), key=lambda x: (-x[1], x[0]))
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to CSV
    print(f"Writing company counts to: {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["company", "count"])
        for company, count in sorted_companies:
            writer.writerow([company, count])
            
    print("Done!")

if __name__ == "__main__":
    main()
