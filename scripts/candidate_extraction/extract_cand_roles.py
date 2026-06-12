"""
extract_cand_roles.py

Extract professional experience roles from a candidate profile.
"""


def extract_work_done(description: str) -> list[str]:
    """
    Convert role description into work chunks.

    Current V1:
    - Split on '.'
    - Trim whitespace
    - Remove empty chunks
    """

    if not description:
        return []

    return [
        chunk.strip()
        for chunk in description.split(".")
        if chunk.strip()
    ]


def extract_roles(candidate: dict) -> list[dict]:
    """
    Returns:

    [
        {
            "title": "Backend Engineer",
            "yoe": 2.3,
            "work_done": [
                "Implemented streaming data pipelines on Kafka and Spark Streaming",
                "Designed the schema-registry integration"
            ]
        }
    ]
    """

    roles = []

    for experience in candidate.get("career_history", []):

        yoe = round(
            experience.get("duration_months", 0) / 12,
            1
        )

        work_done = extract_work_done(
            experience.get("description", "")
        )

        roles.append(
            {
                "title": experience.get("title", ""),
                "yoe": yoe,
                "work_done": work_done
            }
        )

    return roles