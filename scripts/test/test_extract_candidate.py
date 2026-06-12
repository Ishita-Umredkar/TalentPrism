import json

from scripts.candidate_extraction.extract_candidate import (
    extract_candidate
)

INPUT_FILE = "data/test/test_candidates.json"
OUTPUT_FILE = "data/test/extracted_candidates.json"


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    candidates = json.load(f)

results = []

for candidate in candidates:

    extracted = extract_candidate(candidate)

    results.append(extracted)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    f"Saved {len(results)} extracted candidates to {OUTPUT_FILE}"
)