"""
career_consistency.py

Implements Career Consistency detectors (Medium Evidence) for candidate profile validation.
"""

import re
from .base_detector import BaseDetector, parse_date, calculate_months_between

class SeniorTitleLowExperienceDetector(BaseDetector):
    """
    13. Senior title with very low experience.
    Checks if a candidate has a senior/lead/manager title but YOE < 2.5.
    """
    def __init__(self):
        super().__init__(
            check_id="senior_title_low_experience",
            category="career_consistency",
            strength="Medium",
            penalty=0.3
        )
        self.senior_pattern = re.compile(
            r"\b(senior|sr|lead|director|vp|chief|head|president)\b",
            re.IGNORECASE
        )

    def detect(self, candidate: dict) -> list[dict]:
        profile = candidate.get("profile", {})
        yoe = profile.get("years_of_experience")
        
        if yoe is None or yoe >= 2.5:
            return []

        # Check current title
        current_title = profile.get("current_title", "")
        if self.senior_pattern.search(current_title):
            details = (
                f"Candidate holds a senior current title '{current_title}' "
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
            if self.senior_pattern.search(title):
                details = (
                    f"Candidate has senior title '{title}' in career history (job #{idx+1}) "
                    f"but has only {yoe:.1f} years of experience."
                )
                return [
                    self.create_evidence(
                        details,
                        {"title": title, "profile_yoe": yoe}
                    )
                ]

        return []


class UnrealisticPromotionVelocityDetector(BaseDetector):
    """
    14. Unrealistic promotion velocity.
    Checks if the candidate transitioned from a junior/intern/associate role 
    to a senior/lead/manager role in less than 18 months.
    """
    def __init__(self):
        super().__init__(
            check_id="unrealistic_promotion_velocity",
            category="career_consistency",
            strength="Medium",
            penalty=0.3
        )
        self.junior_pattern = re.compile(
            r"\b(junior|jr|intern|associate|assistant|trainee|fresher|entry)\b",
            re.IGNORECASE
        )
        self.senior_pattern = re.compile(
            r"\b(senior|sr|lead|director|vp|chief|head|president)\b",
            re.IGNORECASE
        )

    def detect(self, candidate: dict) -> list[dict]:
        history = candidate.get("career_history", [])
        
        # Parse job details and start dates
        parsed_jobs = []
        for job in history:
            start = parse_date(job.get("start_date"))
            title = job.get("title", "")
            if start and title:
                parsed_jobs.append({
                    "company": job.get("company", "Unknown"),
                    "title": title,
                    "start": start
                })

        # Sort jobs chronologically
        parsed_jobs.sort(key=lambda x: x["start"])

        # Check transitions
        for i in range(len(parsed_jobs)):
            for j in range(i + 1, len(parsed_jobs)):
                job_jr = parsed_jobs[i]
                job_sr = parsed_jobs[j]

                # Check if i is junior and j is senior
                if self.junior_pattern.search(job_jr["title"]) and self.senior_pattern.search(job_sr["title"]):
                    months_diff = calculate_months_between(job_jr["start"], job_sr["start"])
                    if 0 <= months_diff < 18:
                        details = (
                            f"Unrealistic promotion velocity: transitioned from junior role "
                            f"'{job_jr['title']}' at {job_jr['company']} to senior role "
                            f"'{job_sr['title']}' at {job_sr['company']} in {months_diff} months (threshold < 18 mos)."
                        )
                        return [
                            self.create_evidence(
                                details,
                                {
                                    "junior_title": job_jr["title"],
                                    "junior_start": str(job_jr["start"]),
                                    "senior_title": job_sr["title"],
                                    "senior_start": str(job_sr["start"]),
                                    "months_difference": months_diff
                                }
                            )
                        ]
        return []


class ExecutiveTitleWithoutProgressionDetector(BaseDetector):
    """
    15. Executive title without supporting career progression.
    Checks if a candidate holds an executive title (CEO, CTO, Founder, VP, President, Director, Chief) 
    but has YOE < 4.0.
    """
    def __init__(self):
        super().__init__(
            check_id="executive_title_no_progression",
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

        if yoe is None or yoe >= 4.0:
            return []

        # Check current title
        current_title = profile.get("current_title", "")
        if self.exec_pattern.search(current_title):
            details = (
                f"Candidate holds an executive title '{current_title}' "
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


class TitleExperienceInconsistencyDetector(BaseDetector):
    """
    16. Major title/experience inconsistencies.
    Checks if the profile headline current_title contradicts the actual current job title.
    """
    def __init__(self):
        super().__init__(
            check_id="title_experience_inconsistency",
            category="career_consistency",
            strength="Medium",
            penalty=0.2
        )
        self.stop_words = {
            "and", "or", "of", "in", "at", "for", "the", "a", "an", "to", "with", "by", 
            "co", "role", "position", "job", "at", "on", "specialist", "professional"
        }

    def _clean_title_words(self, title: str) -> set[str]:
        if not title:
            return set()
        # Remove punctuation, convert to lower, split by word
        words = re.findall(r"\b[a-z0-9\-]+\b", title.lower())
        return {w for w in words if w not in self.stop_words}

    def detect(self, candidate: dict) -> list[dict]:
        profile = candidate.get("profile", {})
        current_title = profile.get("current_title", "")
        
        if not current_title:
            return []

        history = candidate.get("career_history", [])
        
        # Find current job(s)
        current_jobs = []
        for job in history:
            is_curr = job.get("is_current", False) or job.get("end_date") is None
            if is_curr and job.get("title"):
                current_jobs.append(job.get("title"))

        # If no active current job is marked, use the latest job by start date
        if not current_jobs and history:
            # Sort to find the latest start date job
            def get_start_date(j):
                dt = parse_date(j.get("start_date"))
                return dt if dt else date(1970, 1, 1)
            sorted_history = sorted(history, key=get_start_date, reverse=True)
            if sorted_history[0].get("title"):
                current_jobs.append(sorted_history[0].get("title"))

        if not current_jobs:
            return []

        profile_words = self._clean_title_words(current_title)
        if not profile_words:
            return []

        # Check if the profile current title overlaps with ANY current job title
        for job_title in current_jobs:
            job_words = self._clean_title_words(job_title)
            # If there is some word overlap, we consider it consistent
            if profile_words & job_words:
                return []

        # If no overlap found across any current jobs
        details = (
            f"Profile current title '{current_title}' does not match "
            f"any current job title(s) in career history: {', '.join(current_jobs)}."
        )
        return [
            self.create_evidence(
                details,
                {
                    "profile_current_title": current_title,
                    "career_current_titles": current_jobs
                }
            )
        ]
