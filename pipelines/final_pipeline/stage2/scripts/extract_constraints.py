import os
import json
from pathlib import Path
from google import genai

# Define root and path references
ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[4]

JD_FILE = PROJECT_ROOT / "resources" / "job_description.txt"
PROMPT_FILE = ROOT / "stage2" / "prompts" / "extract_constraints _from_jd.txt"
JD_SCHEMA_FILE = ROOT / "stage2" / "schemas" / "jd_constraint_schema.json"
CANDIDATE_SCHEMA_FILE = ROOT / "stage2" / "schemas" / "candidate_categories.json"
OUTPUT_DIR = ROOT / "stage2" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "extracted_constraints.json"

def run_constraint_extraction():
    print("Initializing Gemini API client...")
    # Initialize the client using the GEMINI_API_KEY environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    client = genai.Client(api_key=api_key)

    # Read the job description, prompt, and schemas
    print("Loading job description, prompt, and schema files...")
    if not JD_FILE.exists():
        raise FileNotFoundError(f"Job Description file not found at {JD_FILE}")
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found at {PROMPT_FILE}")
    if not JD_SCHEMA_FILE.exists():
        raise FileNotFoundError(f"JD constraint schema file not found at {JD_SCHEMA_FILE}")
    if not CANDIDATE_SCHEMA_FILE.exists():
        raise FileNotFoundError(f"Candidate categories schema file not found at {CANDIDATE_SCHEMA_FILE}")

    jd_text = JD_FILE.read_text(encoding="utf-8")
    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
    jd_schema_text = JD_SCHEMA_FILE.read_text(encoding="utf-8")
    candidate_schema_text = CANDIDATE_SCHEMA_FILE.read_text(encoding="utf-8")

    # Construct the final prompt with instructions and attached schemas
    final_prompt = f"""{prompt_text}

============================================================
ATTACHED TARGET OUTPUT JSON SCHEMA
============================================================
{jd_schema_text}

============================================================
ATTACHED CANDIDATE SCHEMA / CATEGORIES
============================================================
{candidate_schema_text}

============================================================
JOB DESCRIPTION TO ANALYZE
============================================================
{jd_text}
"""

    print("Calling Gemini 2.5 Flash API to extract constraints...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=final_prompt,
        config={
            "response_mime_type": "application/json"
        }
    )

    raw_response_text = response.text
    print("Response received from LLM.")

    # Parse and validate the response JSON
    try:
        extracted_data = json.loads(raw_response_text)
    except json.JSONDecodeError as e:
        print(f"Error: Response text from LLM is not valid JSON. Raw text:\n{raw_response_text}")
        raise e

    # Create outputs directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save output to stage2/outputs/extracted_constraints.json
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4, ensure_ascii=False)

    print(f"Extraction complete. Results successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_constraint_extraction()
