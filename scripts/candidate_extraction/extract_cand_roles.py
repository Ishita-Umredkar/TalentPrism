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

    def get_experience_sort_key(exp: dict):
        is_curr = exp.get("is_current", False)
        end_date = exp.get("end_date")
        if is_curr or not end_date:
            end_val = "9999-12-31"
        else:
            end_val = end_date
        start_val = exp.get("start_date") or "0001-01-01"
        return (end_val, start_val)

    sorted_history = sorted(
        candidate.get("career_history", []),
        key=get_experience_sort_key,
        reverse=True
    )

    roles = []

    for experience in sorted_history:

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