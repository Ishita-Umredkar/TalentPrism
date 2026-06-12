"""
embed.py

Generate embeddings for TalentPrism schemas
and save them as pickle files.

Embeddings are added directly into the existing
schema structure.
"""

import json
import pickle
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CANDIDATES_FILE = (
    ROOT / "data" / "test" / "extracted_candidates.json"
)

JD_FILE = (
    ROOT / "data" / "test" / "job_description.json"
)

CANDIDATES_OUTPUT = (
    ROOT / "data" / "test" / "embedded_candidates.pkl"
)

JD_OUTPUT = (
    ROOT / "data" / "test" / "embedded_jd.pkl"
)


# ============================================================
# MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

print("Model loaded.")


# ============================================================
# HELPERS
# ============================================================

def get_embedding(text: str):

    return model.encode(
        text,
        normalize_embeddings=True
    )


# ============================================================
# SKILLS
# ============================================================

def embed_skills(schema):

    skills = (
        schema
        .get("technical_capability", {})
        .get("technical_skills", [])
    )

    for skill in skills:

        name = skill.get("name", "")

        if name:
            skill["embedding"] = (
                get_embedding(name)
            )


# ============================================================
# ROLES
# ============================================================

def embed_roles(schema):

    roles = (
        schema
        .get("professional_experience", {})
        .get("roles", [])
    )

    for role in roles:

        title = role.get("title", "")

        yoe = role.get("yoe", 0)

        work_done = role.get(
            "work_done",
            []
        )

        text = f"""
Title: {title}

Years of Experience: {yoe}

Work Done:
{chr(10).join(work_done)}
"""

        role["embedding"] = (
            get_embedding(text)
        )


# ============================================================
# ORGANIZATION
# ============================================================

def embed_organization(schema):

    organizations = schema.get(
        "organizational_context",
        []
    )

    if isinstance(organizations, dict):
        organizations = [organizations]

    for org in organizations:
        if not isinstance(org, dict):
            continue

        company = org.get(
            "company",
            {}
        )

        industry = org.get(
            "industry",
            {}
        )

        company_size = org.get(
            "company_size",
            {}
        )

        company_name = (
            company.get("name", "")
        )

        industry_name = (
            industry.get("name", "")
        )

        size_name = (
            company_size.get(
                "size",
                company_size.get(
                    "name",
                    ""
                )
            )
        )

        if company_name:
            company["embedding"] = (
                get_embedding(
                    company_name
                )
            )

        if industry_name:
            industry["embedding"] = (
                get_embedding(
                    industry_name
                )
            )

        if size_name:
            company_size["embedding"] = (
                get_embedding(
                    size_name
                )
            )


# ============================================================
# EDUCATION
# ============================================================

def embed_education(schema):

    education = (
        schema
        .get(
            "education_credentials",
            {}
        )
        .get(
            "education",
            []
        )
    )

    for edu in education:

        text = f"""
Degree: {edu.get('degree', '')}

Field: {edu.get('field', '')}

Institution: {edu.get('institution', '')}

Tier: {edu.get('tier', '')}

Grade: {edu.get('grade', '')}
"""

        edu["embedding"] = (
            get_embedding(text)
        )


# ============================================================
# SCHEMA
# ============================================================

def embed_schema(schema):

    embed_skills(schema)

    embed_roles(schema)

    embed_organization(schema)

    embed_education(schema)

    return schema


# ============================================================
# MAIN
# ============================================================

def main():

    print("Embedding candidates...")

    with open(
        CANDIDATES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        candidates = json.load(f)

    embedded_candidates = []

    for candidate in candidates:

        embedded_candidates.append(
            embed_schema(candidate)
        )

    with open(
        CANDIDATES_OUTPUT,
        "wb"
    ) as f:

        pickle.dump(
            embedded_candidates,
            f
        )

    print(
        f"Saved: {CANDIDATES_OUTPUT}"
    )

    print("Embedding JD...")

    with open(
        JD_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        jd = json.load(f)

    embedded_jd = embed_schema(jd)

    with open(
        JD_OUTPUT,
        "wb"
    ) as f:

        pickle.dump(
            embedded_jd,
            f
        )

    print(
        f"Saved: {JD_OUTPUT}"
    )


if __name__ == "__main__":
    main()