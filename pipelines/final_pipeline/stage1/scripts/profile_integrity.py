"""
profile_integrity.py

Implements Profile Integrity detectors (Medium Evidence) for candidate profile validation.
"""

from .base_detector import BaseDetector, extract_years_from_text

class HeadlineSummaryContradictionDetector(BaseDetector):
    """
    17. Headline/summary contradicting profile facts.
    Checks if YOE numbers parsed from headline/summary contradict profile.years_of_experience.
    """
    def __init__(self):
        super().__init__(
            check_id="headline_summary_contradiction",
            category="profile_integrity",
            strength="Medium",
            penalty=0.3
        )

    def detect(self, candidate: dict) -> list[dict]:
        profile = candidate.get("profile", {})
        yoe = profile.get("years_of_experience")
        
        if yoe is None:
            return []

        headline = profile.get("headline", "")
        summary = profile.get("summary", "")
        
        # Extract years from text fields
        headline_years = extract_years_from_text(headline)
        summary_years = extract_years_from_text(summary)
        
        all_extracted_years = headline_years + summary_years
        contradictions = []

        for val in all_extracted_years:
            if abs(val - yoe) > 3.0:
                contradictions.append(val)

        if contradictions:
            details = (
                f"Candidate profile states YOE as {yoe:.1f}, but text description(s) "
                f"claim values of {', '.join(f'{x:.1f}' for x in contradictions)} years."
            )
            return [
                self.create_evidence(
                    details,
                    {
                        "profile_yoe": yoe,
                        "contradictory_text_values": contradictions
                    }
                )
            ]
        return []


class SalaryInconsistentWithExperienceDetector(BaseDetector):
    """
    18. Salary expectations wildly inconsistent with experience.
    Checks if expectations are too high for junior or too low for senior candidates.
    """
    def __init__(self):
        super().__init__(
            check_id="salary_inconsistent_with_experience",
            category="profile_integrity",
            strength="Medium",
            penalty=0.3
        )

    def detect(self, candidate: dict) -> list[dict]:
        profile = candidate.get("profile", {})
        yoe = profile.get("years_of_experience")

        if yoe is None:
            return []

        signals = candidate.get("redrob_signals", {})
        salary_range = signals.get("expected_salary_range_inr_lpa")
        
        if not salary_range:
            return []

        min_lpa = salary_range.get("min")
        max_lpa = salary_range.get("max")

        # Check 1: YOE < 2.0 and min expected salary > 25 LPA (extremely high for a junior)
        if yoe < 2.0 and min_lpa is not None and min_lpa > 25.0:
            details = (
                f"Candidate has low experience ({yoe:.1f} YOE) but expects a very high "
                f"minimum salary of {min_lpa:.1f} LPA."
            )
            return [
                self.create_evidence(
                    details,
                    {"profile_yoe": yoe, "min_lpa": min_lpa}
                )
            ]

        # Check 2: YOE > 8.0 and max expected salary < 6.0 LPA (unreasonably low for a senior)
        # Note: We check if max_lpa > 0 to filter out potential -1 or missing/empty indicators
        if yoe > 8.0 and max_lpa is not None and 0 < max_lpa < 6.0:
            details = (
                f"Candidate is senior ({yoe:.1f} YOE) but has a very low maximum expected "
                f"salary of {max_lpa:.1f} LPA."
            )
            return [
                self.create_evidence(
                    details,
                    {"profile_yoe": yoe, "max_lpa": max_lpa}
                )
            ]

        return []
