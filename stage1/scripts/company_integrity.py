"""
company_integrity.py

Implements Company Integrity detector (Strong Evidence) for candidate profile validation.
"""

from .base_detector import BaseDetector, parse_date, get_last_active_date

class CompanyExperienceVsAgeDetector(BaseDetector):
    """
    Checks if the duration of the candidate's job at a company exceeds the company's age,
    or if the candidate claims to have worked at a company before it was founded.
    """
    def __init__(self):
        super().__init__(
            check_id="company_experience_vs_age",
            category="company_integrity",
            strength="Strong",
            penalty=0.5
        )

    def detect(self, candidate: dict) -> list[dict]:
        last_active = get_last_active_date(candidate)
        history = candidate.get("career_history", [])
        offending_jobs = []

        for idx, job in enumerate(history):
            company = job.get("company", f"Company {idx+1}")
            start_date = parse_date(job.get("start_date"))
            duration_months = job.get("duration_months", 0)
            job_years = duration_months / 12.0
            
            founded_year = job.get("company_founded_year")
            
            if founded_year is not None:
                # 1. Check if candidate started working before company was founded
                if start_date and start_date.year < founded_year:
                    offending_jobs.append(
                        f"'{company}' (started working in {start_date.year}, but company founded in {founded_year})"
                    )
                else:
                    # 2. Check if experience duration exceeds company age at last active date
                    company_age_at_active = last_active.year - founded_year
                    # Allow a small buffer of 1.0 year for round-offs or transition boundaries
                    if job_years > company_age_at_active + 1.0:
                        offending_jobs.append(
                            f"'{company}' (experience: {job_years:.2f} yrs, company age: {company_age_at_active} yrs, founded: {founded_year})"
                        )

        if offending_jobs:
            details = (
                f"Candidate's career timeline is inconsistent with the company's founding year for: "
                f"{', '.join(offending_jobs)}."
            )
            return [
                self.create_evidence(
                    details,
                    {"offending_jobs": offending_jobs}
                )
            ]
        return []
