"""
extract_cand_education.py

Extract education and certifications from a candidate profile.
"""


def extract_education(candidate: dict) -> dict:
    """
    Returns:

    {
        "education": [...],
        "certifications": [...]
    }
    """

    education = []

    for edu in candidate.get("education", []):

        education.append(
            {
                "degree":
                    edu.get("degree", ""),

                "field":
                    edu.get(
                        "field_of_study",
                        ""
                    ),

                "institution":
                    edu.get(
                        "institution",
                        ""
                    ),

                "tier":
                    edu.get("tier", ""),

                "grade":
                    edu.get("grade", "")
            }
        )

    certifications = []

    for cert in candidate.get(
        "certifications",
        []
    ):

        certifications.append(
            {
                "name":
                    cert.get("name", "")
            }
        )

    return {
        "education": education,
        "certifications": certifications
    }