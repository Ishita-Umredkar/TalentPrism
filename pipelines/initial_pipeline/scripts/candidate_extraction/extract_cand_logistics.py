"""
extract_cand_logistics.py

Extract candidate logistics information.
"""


def extract_logistics(candidate: dict) -> dict:
    """
    Returns:

    {
        "location": "Toronto",
        "preferred_work_mode": "onsite",
        "willing_to_relocate": False,
        "notice_period_days": 60,
        "salary_expectation": {
            "min_lpa": 18.7,
            "max_lpa": 36.1
        }
    }
    """

    profile = candidate.get("profile", {})
    signals = candidate.get(
        "redrob_signals",
        {}
    )

    salary_range = signals.get(
        "expected_salary_range_inr_lpa",
        {}
    )

    return {

        "location":
            profile.get(
                "location",
                ""
            ),

        "preferred_work_mode":
            signals.get(
                "preferred_work_mode",
                ""
            ),

        "willing_to_relocate":
            signals.get(
                "willing_to_relocate",
                False
            ),

        "notice_period_days":
            signals.get(
                "notice_period_days",
                0
            ),

        "salary_expectation": {
            "min_lpa":
                salary_range.get(
                    "min",
                    0.0
                ),

            "max_lpa":
                salary_range.get(
                    "max",
                    0.0
                )
        }
    }