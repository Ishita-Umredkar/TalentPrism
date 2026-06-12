"""
extract_cand_org_context.py

Extract organizational context from a candidate profile.
"""


def extract_organizational_context(
    candidate: dict
) -> list[dict]:
    """
    Returns:

    [
        {
            "company": "Mindtree",
            "industry": "IT Services",
            "company_size": "10001+",
            "years": 2.3
        },
        {
            "company": "Dunder Mifflin",
            "industry": "Paper Products",
            "company_size": "201-500",
            "years": 4.6
        }
    ]
    """

    organizational_context = []

    for experience in candidate.get(
        "career_history",
        []
    ):

        organizational_context.append(
            {
                "company":
                    experience.get(
                        "company",
                        ""
                    ),

                "industry":
                    experience.get(
                        "industry",
                        ""
                    ),

                "company_size":
                    experience.get(
                        "company_size",
                        ""
                    ),

                "years":
                    round(
                        experience.get(
                            "duration_months",
                            0
                        ) / 12,
                        1
                    )
            }
        )

    return organizational_context