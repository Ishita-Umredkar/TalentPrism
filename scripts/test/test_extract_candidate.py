import json

from scripts.candidate_extraction.extract_candidate import (
    extract_candidate
)


INPUT_FILE = "data/test/test_candidate.json"
OUTPUT_FILE = "data/test/extracted_candidate.json"


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    candidates = json.load(f)

candidate = candidates[0]

result = extract_candidate(candidate)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        result,
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    f"Saved extracted candidate to {OUTPUT_FILE}"
)