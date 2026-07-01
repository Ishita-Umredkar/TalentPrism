from pathlib import Path
import json
from google import genai
import os

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROMPT_FILE = ROOT / "prompts" / "job_extraction.txt"
OUTPUT_DIR = ROOT / "data" / "test"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

# ============================================================
# HELPERS
# ============================================================

def validate_weights(obj):
    """
    Recursively collect all weights and verify they sum to 1.
    """

    weights = []

    def traverse(node):
        if isinstance(node, dict):
            for k, v in node.items():

                if k == "weight" and isinstance(v, (int, float)):
                    weights.append(float(v))

                traverse(v)

        elif isinstance(node, list):
            for item in node:
                traverse(item)

    traverse(obj)

    total = sum(weights)

    return total


def normalize_weights(obj):
    """
    Normalize weights if Gemini is slightly off.
    """

    weights = []

    def collect(node):
        if isinstance(node, dict):
            for k, v in node.items():

                if k == "weight" and isinstance(v, (int, float)):
                    weights.append(v)

                collect(v)

        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(obj)

    total = sum(weights)

    if total == 0:
        return obj

    def normalize(node):
        if isinstance(node, dict):
            for k, v in node.items():

                if k == "weight" and isinstance(v, (int, float)):
                    node[k] = round(v / total, 6)

                else:
                    normalize(v)

        elif isinstance(node, list):
            for item in node:
                normalize(item)

    normalize(obj)

    return obj


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_job_details(jd_file_path: str):

    jd_path = Path(jd_file_path)

    if not jd_path.exists():
        raise FileNotFoundError(f"JD file not found: {jd_path}")

    prompt = PROMPT_FILE.read_text(
        encoding="utf-8"
    )

    jd_text = jd_path.read_text(
        encoding="utf-8"
    )

    final_prompt = f"""
{prompt}

============================================================
JOB DESCRIPTION
============================================================

{jd_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    result_json = json.loads(response.text)

    weight_sum = validate_weights(result_json)

    print(f"Weight Sum Before Normalization: {weight_sum:.6f}")

    if abs(weight_sum - 1.0) > 0.01:
        print("Normalizing weights...")
        result_json = normalize_weights(result_json)

    output_file = OUTPUT_DIR / f"{jd_path.stem}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            result_json,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved: {output_file}")

    return result_json


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    extract_job_details(
        ROOT / "resources" / "job_description.txt"
    )