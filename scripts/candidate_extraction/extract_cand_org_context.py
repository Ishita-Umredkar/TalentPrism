"""
extract_cand_org_context.py

Extract organizational context from a candidate profile.
"""


def extract_organizational_context(
    candidate: dict
) -> list[dict]:

    organizational_context = []

    for experience in candidate.get(
        "career_history",
        []
    ):

        organizational_context.append(
            {
                "company": {
                    "name":
                        experience.get(
                            "company",
                            ""
                        )
                },

                "industry": {
                    "name":
                        experience.get(
                            "industry",
                            ""
                        )
                },

                "company_size": {
                    "size":
                        experience.get(
                            "company_size",
                            ""
                        )
                },

                "years": {
                    "number_of_years":
                        round(
                            experience.get(
                                "duration_months",
                                0
                            ) / 12,
                            1
                        )
                }
            }
        )

    return organizational_context