"""
extract_cand_execution_impact.py

Extract execution impact signals from candidate work history.
"""

import math
import re


PRODUCTION_TERMS = {
    "production",
    "real-time",
    "pipeline",
    "pipelines",
    "deployment",
    "deployments",
    "warehouse",
    "warehouses",
    "monitoring",
    "on-call",
    "customer",
    "customers",
    "user",
    "users",
    "scale",
    "scaled",
    "streaming",
    "distributed",
    "microservice",
    "microservices",
    "sla",
    "uptime",
}


OWNERSHIP_TERMS = {
    "owned",
    "led",
    "managed",
    "architected",
    "responsible",
    "responsibility",
    "mentored",
    "headed",
    "drove",
    "owner",
    "lead",
    "leadership",
    "managed",
    "oversaw",
}


IMPACT_TERMS = {
    "improved",
    "improve",
    "reduced",
    "reduce",
    "increased",
    "increase",
    "optimized",
    "optimize",
    "saved",
    "save",
    "scaled",
    "scale",
    "grew",
    "growth",
}


IMPACT_PATTERNS = [
    r"\d+\s*%",                          # 40%
    r"\$\s*\d+",                         # $500000
    r"\d+\s*(gb|tb|mb)",                # 500GB
    r"\d+\s*(users|customers)",         # 1000 users
    r"\d+\s*(million|billion|k|m)",     # 5 million
]


def normalized_signal_score(
    citation_count: int
) -> float:
    """
    Logarithmic saturation curve.

    Citations:
        0 -> 0.00
        1 -> 0.50
        2 -> 0.79
        3 -> 1.00
        4+ -> 1.00
    """

    return round(
        min(
            math.log1p(citation_count)
            / math.log1p(4),
            1.0
        ),
        3
    )


def get_all_work_chunks(
    candidate: dict
) -> list[str]:
    """
    Collect all work_done chunks across all roles.
    """

    chunks = []

    for role in candidate.get(
        "professional_experience",
        {}
    ).get(
        "roles",
        []
    ):

        chunks.extend(
            role.get(
                "work_done",
                []
            )
        )

    return chunks


def calculate_production_score(
    work_chunks: list[str]
) -> float:

    citation_count = 0

    for chunk in work_chunks:

        text = chunk.lower()

        if any(
            term in text
            for term in PRODUCTION_TERMS
        ):
            citation_count += 1

    return normalized_signal_score(
        citation_count
    )


def calculate_ownership_score(
    work_chunks: list[str]
) -> float:

    citation_count = 0

    for chunk in work_chunks:

        text = chunk.lower()

        if any(
            term in text
            for term in OWNERSHIP_TERMS
        ):
            citation_count += 1

    return normalized_signal_score(
        citation_count
    )


def calculate_impact_score(
    work_chunks: list[str]
) -> float:

    citation_count = 0

    for chunk in work_chunks:

        text = chunk.lower()

        term_match = any(
            term in text
            for term in IMPACT_TERMS
        )

        pattern_match = any(
            re.search(
                pattern,
                text
            )
            for pattern in IMPACT_PATTERNS
        )

        if term_match or pattern_match:
            citation_count += 1

    return normalized_signal_score(
        citation_count
    )


def extract_execution_impact(
    candidate_schema: dict
) -> dict:
    """
    Input:
        Partially processed TalentPrism schema

    Returns:

    {
        "production_experience": 0.79,
        "ownership_leadership": 0.50,
        "impact": 1.00
    }
    """

    work_chunks = get_all_work_chunks(
        candidate_schema
    )

    return {
        "production_experience":
            calculate_production_score(
                work_chunks
            ),

        "ownership_leadership":
            calculate_ownership_score(
                work_chunks
            ),

        "impact":
            calculate_impact_score(
                work_chunks
            ),
    }