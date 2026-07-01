"""
company_integrity.py

Implements Company Integrity detector (Strong Evidence) for candidate profile validation.
"""

import os
import json
from .base_detector import BaseDetector, parse_date

# Locate companies.json relative to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPANIES_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "companies.json"))

try:
    with open(COMPANIES_PATH, "r", encoding="utf-8") as f:
        COMPANIES_MAP = json.load(f)
except Exception:
    COMPANIES_MAP = {}

# Prepare a case-insensitive map for robustness
COMPANIES_MAP_LOWER = {k.lower().strip(): v for k, v in COMPANIES_MAP.items()}

def get_company_founding_year(company_name: str) -> int:
    if not company_name:
        return -1
    company_name_clean = company_name.strip()
    if company_name_clean in COMPANIES_MAP:
        return COMPANIES_MAP[company_name_clean]
    company_name_lower = company_name_clean.lower()
    if company_name_lower in COMPANIES_MAP_LOWER:
        return COMPANIES_MAP_LOWER[company_name_lower]
    return -1


class EmploymentBeforeCompanyExistedDetector(BaseDetector):
    """
    I1. Employment Before Company Existed.
    Detects candidates claiming employment before the company's founding date.
    """
    def __init__(self):
        super().__init__(
            check_id="employment_before_company_existed",
            category="company_integrity",
            strength="Strong",
            penalty=0.8
        )

    def detect(self, candidate: dict) -> list[dict]:
        history = candidate.get("career_history", [])
        offending_jobs = []

        for idx, job in enumerate(history):
            company = job.get("company", "").strip()
            if not company:
                continue

            founded_year = get_company_founding_year(company)
            if founded_year <= 0:
                continue

            start_date = parse_date(job.get("start_date"))

            if start_date and start_date.year < founded_year:
                diff_years = founded_year - start_date.year
                job_penalty = 0.40 if diff_years == 1 else 0.80
                offending_jobs.append({
                    "details": f"'{company}' (started working in {start_date.year}, but company founded in {founded_year})",
                    "penalty": job_penalty
                })

        if offending_jobs:
            total_penalty = sum(job["penalty"] for job in offending_jobs)
            effective_penalty = min(total_penalty, 0.80)
            
            details_list = [job["details"] for job in offending_jobs]
            details = (
                f"Candidate's career timeline is inconsistent with the company's founding year for: "
                f"{', '.join(details_list)}."
            )
            evidence = self.create_evidence(
                details,
                {"offending_jobs": details_list}
            )
            evidence["penalty"] = effective_penalty
            return [evidence]

        return []
