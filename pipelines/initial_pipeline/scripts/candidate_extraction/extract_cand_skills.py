"""
extract_cand_skills.py

Extract technical skills from a candidate profile.
"""

import math


PROFICIENCY_MAP = {
    "beginner": 0.4,
    "intermediate": 0.65,
    "advanced": 0.95,
}


def duration_score(duration_months: int) -> float:
    """
    Logarithmic duration curve.

    6 months  -> ~0.43
    12 months -> ~0.60
    24 months -> ~0.76
    36 months -> ~0.86
    48 months -> ~0.94
    60+ months -> 1.00
    """

    return min(
        math.log1p(duration_months) / math.log1p(60),
        1.0
    )


def calculate_skill_score(
    skill: dict,
    assessment_scores: dict
) -> float:
    """
    With assessment:
        0.5 * proficiency
      + 0.3 * duration
      + 0.2 * assessment

    Without assessment:
        0.625 * proficiency
      + 0.375 * duration
    """

    proficiency_score = PROFICIENCY_MAP.get(
        skill.get("proficiency", "").lower(),
        0.0
    )

    duration = duration_score(
        skill.get("duration_months", 0)
    )

    assessment = assessment_scores.get(
        skill["name"]
    )

    if assessment is not None:

        assessment_score = assessment / 100.0

        score = (
            0.5 * proficiency_score
            + 0.3 * duration
            + 0.2 * assessment_score
        )

    else:

        score = (
            0.625 * proficiency_score
            + 0.375 * duration
        )

    return round(score, 3)


def extract_technical_skills(
    candidate: dict
) -> list[dict]:
    """
    Returns:

    [
        {
            "name": "Python",
            "score": 0.87
        },
        {
            "name": "SQL",
            "score": 0.81
        }
    ]
    """

    assessment_scores = (
        candidate
        .get("redrob_signals", {})
        .get("skill_assessment_scores", {})
    )

    technical_skills = []

    for skill in candidate.get("skills", []):

        technical_skills.append(
            {
                "name": skill["name"],
                "score": calculate_skill_score(
                    skill,
                    assessment_scores
                )
            }
        )

    return technical_skills