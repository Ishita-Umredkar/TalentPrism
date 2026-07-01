"""
career_consistency.py

Implements Career Consistency detectors (Medium Evidence) for candidate profile validation.
"""

import re
from .base_detector import BaseDetector, parse_date, calculate_months_between, extract_years_from_text

class ExecutiveTitleMinimalExperienceDetector(BaseDetector):
    """
    C1. Executive Title with Minimal Experience.
    Flags candidates holding executive-level titles such as CEO, CTO, VP, Founder, Chief, etc.,
    while having less than one year of total professional experience.
    """
    def __init__(self):
        super().__init__(
            check_id="executive_title_minimal_experience",
            category="career_consistency",
            strength="Medium",
            penalty=0.4
        )
        self.exec_pattern = re.compile(
            r"\b(ceo|cto|founder|co-founder|vp|president|director|chief|cfo|cio|owner)\b",
            re.IGNORECASE
        )

    def detect(self, candidate: dict) -> list[dict]:
        profile = candidate.get("profile", {})
        yoe = profile.get("years_of_experience")

        if yoe is None or yoe >= 1.0:
            return []

        # Check current title
        current_title = profile.get("current_title", "")
        if self.exec_pattern.search(current_title):
            details = (
                f"Candidate holds an executive current title '{current_title}' "
                f"but has only {yoe:.1f} years of experience."
            )
            return [
                self.create_evidence(
                    details,
                    {"title": current_title, "profile_yoe": yoe}
                )
            ]

        # Check job history titles
        history = candidate.get("career_history", [])
        for idx, job in enumerate(history):
            title = job.get("title", "")
            if self.exec_pattern.search(title):
                details = (
                    f"Candidate has executive title '{title}' in career history (job #{idx+1}) "
                    f"but has only {yoe:.1f} years of experience."
                )
                return [
                    self.create_evidence(
                        details,
                        {"title": title, "profile_yoe": yoe}
                    )
                ]

        return []


class HeadlineSummaryContradictionDetector(BaseDetector):
    """
    C2. Headline vs Experience Contradiction.
    Extracts years of experience mentioned in the profile headline or summary
    and compares them with the declared years_of_experience. A difference greater than 3 years
    is considered inconsistent.
    """
    def __init__(self):
        super().__init__(
            check_id="headline_summary_contradiction",
            category="career_consistency",
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
