import json
from pathlib import Path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

ROOT = Path(__file__).resolve().parents[2]

FILES = [
    ROOT / "data" / "test" / "jd_embedded.json",
    ROOT / "data" / "test" / "cand_embedded.json"
]


def embed(text: str):
    return model.encode(
        text,
        normalize_embeddings=True
    ).tolist()


def process_skills(data):

    skills = data.get(
        "technical_capability",
        {}
    ).get(
        "technical_skills",
        []
    )

    for skill in skills:

        if "name" in skill:

            skill["embedding"] = embed(
                skill["name"]
            )


def process_roles(data):

    roles = data.get(
        "professional_experience",
        {}
    ).get(
        "roles",
        []
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

        role["embedding"] = embed(text)


def process_organization(data):

    orgs = data.get(
        "organizational_context",
        []
    )

    for org in orgs:

        company = (
            org.get("company", {})
               .get("name", "")
        )

        industry = (
            org.get("industry", {})
               .get("name", "")
        )

        size = (
            org.get("company_size", {})
               .get("size",
                    org.get(
                        "company_size",
                        {}
                    ).get("name", "")
                )
        )

        text = f"""
Company: {company}

Industry: {industry}

Company Size: {size}
"""

        org["embedding"] = embed(text)

        if "company" in org:
            org["company"]["embedding"] = embed(company)

        if "industry" in org:
            org["industry"]["embedding"] = embed(industry)

        if "company_size" in org:

            size_text = (
                org["company_size"].get(
                    "size",
                    org["company_size"].get(
                        "name",
                        ""
                    )
                )
            )

            org["company_size"]["embedding"] = embed(
                size_text
            )


def process_education(data):

    education = data.get(
        "education_credentials",
        {}
    ).get(
        "education",
        []
    )

    for edu in education:

        text = f"""
Degree: {edu.get('degree', '')}

Field: {edu.get('field', '')}

Institution: {edu.get('institution', '')}

Tier: {edu.get('tier', '')}

Grade: {edu.get('grade', '')}
"""

        edu["embedding"] = embed(text)


def process_file(path):

    print(f"Processing: {path}")

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    process_skills(data)
    process_roles(data)
    process_organization(data)
    process_education(data)

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"Saved embeddings to: {path}"
    )


if __name__ == "__main__":

    for file in FILES:

        if file.exists():
            process_file(file)
        else:
            print(
                f"File not found: {file}"
            )