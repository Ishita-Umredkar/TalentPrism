"""
extract_cand_org_context.py

Extract organizational context from a candidate profile.
"""


def extract_organizational_context(
    candidate: dict
) -> list[dict]:

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

    organizational_context = []

    for experience in sorted_history:

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