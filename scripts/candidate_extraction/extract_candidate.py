"""
extract_candidate.py

Build complete TalentPrism candidate schema.
"""

from .extract_cand_skills import extract_technical_skills
from .extract_cand_roles import extract_roles
from .extract_cand_org_context import (
    extract_organizational_context
)
from .extract_cand_education import extract_education
from .extract_cand_logistics import extract_logistics
from .extract_cand_execution_impact import (
    extract_execution_impact
)


def extract_candidate(candidate: dict) -> dict:

    schema = {

        "technical_capability": {
            "technical_skills":
                extract_technical_skills(candidate)
        },

        "professional_experience": {
            "roles":
                extract_roles(candidate)
        },

        "organizational_context":
            extract_organizational_context(candidate),

        "education_credentials":
            extract_education(candidate),

        "logistics":
            extract_logistics(candidate)
    }

    schema["execution_impact"] = (
        extract_execution_impact(schema)
    )

    return schema