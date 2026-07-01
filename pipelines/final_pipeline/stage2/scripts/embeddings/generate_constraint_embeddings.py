import os
import json
import pickle
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Define root and path references
ROOT = Path(__file__).resolve().parents[3]
CONSTRAINTS_FILE = ROOT / "stage2" / "outputs" / "extracted_constraints_v2.json"
OUTPUT_DIR = ROOT / "stage2" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "embedded_constraints.pkl"

def main():
    print(f"Reading constraints from: {CONSTRAINTS_FILE}")
    if not CONSTRAINTS_FILE.exists():
        raise FileNotFoundError(f"Extracted constraints file not found at: {CONSTRAINTS_FILE}")

    with open(CONSTRAINTS_FILE, "r", encoding="utf-8") as f:
        constraints_data = json.load(f)

    # Collect unique sub-constraint items
    unique_items = set()
    for sec in ["must_have", "preferred", "negative", "rejection"]:
        for c in constraints_data.get(sec, []):
            for sub in c.get("sub_constraints", []):
                item = sub.get("item", "").strip()
                if item:
                    unique_items.add(item)

    unique_items_list = sorted(list(unique_items))
    print(f"Found {len(unique_items_list)} unique sub-constraints.")
    for idx, item in enumerate(unique_items_list, 1):
        print(f"  {idx}. {item}")

    print("Loading SentenceTransformer model (BAAI/bge-base-en-v1.5)...")
    start_time = time.time()
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print(f"Model loaded successfully in {time.time() - start_time:.2f}s.")

    print("Encoding sub-constraints (normalizing embeddings)...")
    start_embed = time.time()
    embeddings = model.encode(
        unique_items_list,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"Encoding completed in {time.time() - start_embed:.2f}s.")

    # Map text to its embedding vector
    embedded_dict = {text: emb for text, emb in zip(unique_items_list, embeddings)}

    # Ensure outputs directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving embedded constraints dict to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(embedded_dict, f)

    print("Constraint embedding generation completed successfully.")

if __name__ == "__main__":
    main()
