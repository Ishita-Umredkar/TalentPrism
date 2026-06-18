"""
temporal_consistency.py

Implements Temporal Consistency detectors (Strong Evidence) for candidate profile validation.
"""

from datetime import date
from .base_detector import (
    BaseDetector,
    parse_date,
    get_last_active_date,
    calculate_months_between
)

class JobDurationVsDateDiffDetector(BaseDetector):
    """
    1. Job duration_months vs actual date difference.
    Checks if the duration_months matches the difference between start_date and end_date.
    """
    def __init__(self):
        super().__init__(
            check_id="job_duration_vs_date_difference",
            category="temporal_consistency",
            strength="Strong",
            penalty=0.3
        )

    def detect(self, candidate: dict) -> list[dict]:
        last_active = get_last_active_date(candidate)
        history = candidate.get("career_history", [])
        offending_jobs = []

        for idx, job in enumerate(history):
            start = parse_date(job.get("start_date"))
            if not start:
                continue
            
            end = parse_date(job.get("end_date"))
            if not end and job.get("is_current", False):
                end = last_active
            
            if not end:
                continue

            actual_months = calculate_months_between(start, end)
            reported_months = job.get("duration_months", 0)

            if abs(actual_months - reported_months) > 18:
                company = job.get("company", f"Company {idx+1}")
                offending_jobs.append(
                    f"{company} (reported: {reported_months} mos, calculated: {actual_months} mos)"
                )

        if offending_jobs:
            # Additive penalty: 0.3 per offending job, capped at 0.8
            effective_penalty = min(0.3 * len(offending_jobs), 0.8)
            details = f"Reported job durations do not match date calculations for: {', '.join(offending_jobs)}."
            evidence = self.create_evidence(details, {"offending_count": len(offending_jobs)})
            evidence["penalty"] = effective_penalty
            return [evidence]
        return []


class CareerDurationVsYoeDetector(BaseDetector):
    """
    2. Sum of career durations vs years_of_experience.
    Checks if the sum of all career durations matches profile.years_of_experience.
    """
    def __init__(self):
        super().__init__(
            check_id="career_duration_vs_yoe",
            category="temporal_consistency",
            strength="Strong",
            penalty=0.4
        )

    def detect(self, candidate: dict) -> list[dict]:
        history = candidate.get("career_history", [])
        profile = candidate.get("profile", {})
        yoe = profile.get("years_of_experience")

        if yoe is None:
            return []

        total_reported_months = sum(job.get("duration_months", 0) for job in history)
        total_reported_years = total_reported_months / 12.0

        if abs(total_reported_years - yoe) > 1.5:
            details = (
                f"Sum of career durations ({total_reported_years:.1f} years) "
                f"contradicts profile years of experience ({yoe:.1f} years)."
            )
            return [
                self.create_evidence(
                    details,
                    {
                        "total_reported_years": round(total_reported_years, 2),
                        "profile_yoe": yoe,
                        "difference": round(abs(total_reported_years - yoe), 2)
                    }
                )
            ]
        return []


class ExperienceSinceEducationDetector(BaseDetector):
    """
    3. Experience possible since education timeline.
    Checks if jobs start way before education start, or if YOE exceeds time since education start.
    """
    def __init__(self):
        super().__init__(
            check_id="experience_since_education",
            category="temporal_consistency",
            strength="Strong",
            penalty=0.5
        )

    def detect(self, candidate: dict) -> list[dict]:
        education = candidate.get("education", [])
        history = candidate.get("career_history", [])
        profile = candidate.get("profile", {})
        yoe = profile.get("years_of_experience")
        last_active = get_last_active_date(candidate)

        # Find earliest education start year
        edu_starts = []
        for edu in education:
            start_yr = edu.get("start_year")
            end_yr = edu.get("end_year")
            if start_yr:
                edu_starts.append(start_yr)
            elif end_yr:
                # Fallback to end_year - 4
                edu_starts.append(end_yr - 4)

        if not edu_starts:
            return []

        earliest_edu_start = min(edu_starts)

        # Find earliest job start year
        job_starts = []
        for job in history:
            start_dt = parse_date(job.get("start_date"))
            if start_dt:
                job_starts.append(start_dt.year)

        # Check 1: Job starts way before education starts
        if job_starts:
            earliest_job_start = min(job_starts)
            # Allow a buffer of 4 years (e.g. working full time before bachelor's or high-school jobs)
            if earliest_job_start < earliest_edu_start - 4:
                details = (
                    f"First job started in {earliest_job_start}, which is unreasonably early "
                    f"compared to education starting in {earliest_edu_start}."
                )
                return [
                    self.create_evidence(
                        details,
                        {
                            "earliest_job_start_year": earliest_job_start,
                            "earliest_edu_start_year": earliest_edu_start
                        }
                    )
                ]

        # Check 2: Total YOE exceeds time since education start + buffer
        if yoe is not None:
            max_possible_years = last_active.year - earliest_edu_start
            if yoe > max_possible_years + 2:
                details = (
                    f"Claimed years of experience ({yoe:.1f}) exceeds the maximum possible "
                    f"years since education timeline started in {earliest_edu_start} ({max_possible_years} years)."
                )
                return [
                    self.create_evidence(
                        details,
                        {
                            "profile_yoe": yoe,
                            "earliest_edu_start_year": earliest_edu_start,
                            "max_possible_years": max_possible_years
                        }
                    )
                ]

        return []


class EducationOrderConsistencyDetector(BaseDetector):
    """
    4. Education order consistency.
    Higher degrees (PhD > Master > Bachelor) should end after lower degrees.
    """
    def __init__(self):
        super().__init__(
            check_id="education_order_consistency",
            category="temporal_consistency",
            strength="Strong",
            penalty=0.4
        )

    def _get_degree_level(self, degree_str: str) -> int:
        if not degree_str:
            return 0
        d_lower = degree_str.lower()
        # Doctorate / Ph.D.
        if "ph" in d_lower or "doctor" in d_lower or "postdoc" in d_lower:
            return 3
        # Master's
        if "master" in d_lower or "m.s" in d_lower or "m.t" in d_lower or "m.e" in d_lower or "mba" in d_lower or "pg" in d_lower:
            return 2
        # Bachelor's
        if "bach" in d_lower or "b.s" in d_lower or "b.a" in d_lower or "b.t" in d_lower or "b.e" in d_lower or "b.c" in d_lower or "ug" in d_lower:
            return 1
        return 0

    def detect(self, candidate: dict) -> list[dict]:
        education = candidate.get("education", [])
        degrees = []

        for edu in education:
            deg_name = edu.get("degree")
            end_yr = edu.get("end_year")
            if not deg_name or not end_yr:
                continue
            lvl = self._get_degree_level(deg_name)
            if lvl > 0:
                degrees.append({
                    "degree": deg_name,
                    "level": lvl,
                    "end_year": end_yr,
                    "institution": edu.get("institution", ""),
                    "field_of_study": edu.get("field_of_study", "")
                })

        # Compare pairs
        for i in range(len(degrees)):
            for j in range(i + 1, len(degrees)):
                d1 = degrees[i]
                d2 = degrees[j]
                
                same_inst = d1["institution"].lower().strip() == d2["institution"].lower().strip()
                same_field = d1["field_of_study"].lower().strip() == d2["field_of_study"].lower().strip()

                if same_inst and same_field:
                    # If d1 level > d2 level but d1 ended earlier
                    if d1["level"] > d2["level"] and d1["end_year"] < d2["end_year"]:
                        details = (
                            f"Higher degree '{d1['degree']}' ended in {d1['end_year']} "
                            f"which is before lower degree '{d2['degree']}' ended in {d2['end_year']} "
                            f"at the same institution ({d1['institution']}) and major ({d1['field_of_study']})."
                        )
                        return [
                            self.create_evidence(
                                details,
                                {
                                    "higher_degree": d1["degree"],
                                    "higher_degree_end": d1["end_year"],
                                    "lower_degree": d2["degree"],
                                    "lower_degree_end": d2["end_year"],
                                    "institution": d1["institution"],
                                    "field_of_study": d1["field_of_study"]
                                }
                            )
                        ]
                    # If d2 level > d1 level but d2 ended earlier
                    elif d2["level"] > d1["level"] and d2["end_year"] < d1["end_year"]:
                        details = (
                            f"Higher degree '{d2['degree']}' ended in {d2['end_year']} "
                            f"which is before lower degree '{d1['degree']}' ended in {d1['end_year']} "
                            f"at the same institution ({d2['institution']}) and major ({d2['field_of_study']})."
                        )
                        return [
                            self.create_evidence(
                                details,
                                {
                                    "higher_degree": d2["degree"],
                                    "higher_degree_end": d2["end_year"],
                                    "lower_degree": d1["degree"],
                                    "lower_degree_end": d1["end_year"],
                                    "institution": d2["institution"],
                                    "field_of_study": d2["field_of_study"]
                                }
                            )
                        ]

        return []


class OverlappingJobsDetector(BaseDetector):
    """
    5. Overlapping jobs.
    Checks if job timelines overlap significantly (more than 90 days).
    """
    def __init__(self):
        super().__init__(
            check_id="overlapping_jobs",
            category="temporal_consistency",
            strength="Strong",
            penalty=0.5
        )

    def detect(self, candidate: dict) -> list[dict]:
        history = candidate.get("career_history", [])
        last_active = get_last_active_date(candidate)
        parsed_jobs = []

        for idx, job in enumerate(history):
            start = parse_date(job.get("start_date"))
            if not start:
                continue
            
            end = parse_date(job.get("end_date"))
            if not end and job.get("is_current", False):
                end = last_active
            
            if not end:
                continue

            parsed_jobs.append({
                "company": job.get("company", f"Company {idx+1}"),
                "start": start,
                "end": end
            })

        overlapping_pairs = []
        for i in range(len(parsed_jobs)):
            for j in range(i + 1, len(parsed_jobs)):
                j1 = parsed_jobs[i]
                j2 = parsed_jobs[j]

                # Check overlap: start1 < end2 and start2 < end1
                if j1["start"] < j2["end"] and j2["start"] < j1["end"]:
                    overlap_start = max(j1["start"], j2["start"])
                    overlap_end = min(j1["end"], j2["end"])
                    overlap_days = (overlap_end - overlap_start).days
                    
                    if overlap_days > 90:
                        overlapping_pairs.append(
                            f"'{j1['company']}' and '{j2['company']}' overlap by {overlap_days} days"
                        )

        if overlapping_pairs:
            details = f"Significant overlaps (>90 days) detected between jobs: {'; '.join(overlapping_pairs)}."
            return [
                self.create_evidence(
                    details,
                    {"overlaps": overlapping_pairs}
                )
            ]

        return []


class MultipleCurrentJobsDetector(BaseDetector):
    """
    6. Multiple current jobs.
    Checks if multiple jobs are marked as current.
    """
    def __init__(self):
        super().__init__(
            check_id="multiple_current_jobs",
            category="temporal_consistency",
            strength="Strong",
            penalty=0.3
        )

    def detect(self, candidate: dict) -> list[dict]:
        history = candidate.get("career_history", [])
        current_jobs = []

        for idx, job in enumerate(history):
            is_curr = job.get("is_current", False)
            end_date = job.get("end_date")
            
            if is_curr or (end_date is None and job.get("start_date") is not None):
                current_jobs.append(job.get("company", f"Company {idx+1}"))

        if len(current_jobs) > 1:
            details = f"Multiple jobs are marked as currently active: {', '.join(current_jobs)}."
            return [
                self.create_evidence(
                    details,
                    {"current_jobs": current_jobs}
                )
            ]
        return []


class CurrentJobChronologyDetector(BaseDetector):
    """
    7. Current job chronology.
    Verify if the current job has the latest start date.
    """
    def __init__(self):
        super().__init__(
            check_id="current_job_chronology",
            category="temporal_consistency",
            strength="Strong",
            penalty=0.2
        )

    def detect(self, candidate: dict) -> list[dict]:
        history = candidate.get("career_history", [])
        
        current_jobs_starts = []
        other_jobs_starts = []

        for job in history:
            start = parse_date(job.get("start_date"))
            if not start:
                continue
            is_curr = job.get("is_current", False) or job.get("end_date") is None
            if is_curr:
                current_jobs_starts.append((job.get("company"), start))
            else:
                other_jobs_starts.append((job.get("company"), start))

        if not current_jobs_starts or not other_jobs_starts:
            return []

        # Find the earliest start date among current jobs
        earliest_current_start = min(s[1] for s in current_jobs_starts)
        
        # Check if any non-current job started AFTER the current job started
        violating_jobs = []
        for name, start in other_jobs_starts:
            if start > earliest_current_start:
                violating_jobs.append(f"'{name}' (started {start})")

        if violating_jobs:
            details = (
                f"Current job started on {earliest_current_start}, but other subsequent jobs "
                f"started later: {', '.join(violating_jobs)}."
            )
            return [
                self.create_evidence(
                    details,
                    {"violating_jobs": violating_jobs}
                )
            ]

        return []


class FutureDatesDetector(BaseDetector):
    """
    8. Future dates.
    Checks if any date is after the candidate's last_active_date.
    """
    def __init__(self):
        super().__init__(
            check_id="future_dates",
            category="temporal_consistency",
            strength="Strong",
            penalty=0.6
        )

    def detect(self, candidate: dict) -> list[dict]:
        current_date = date(2026, 6, 13)
        history = candidate.get("career_history", [])
        education = candidate.get("education", [])
        signals = candidate.get("redrob_signals", {})
        signup = parse_date(signals.get("signup_date"))
        last_active = parse_date(signals.get("last_active_date"))
        
        future_elements = []

        # Check signup date
        if signup and signup > current_date:
            future_elements.append(f"Signup date ({signup}) is in the future relative to {current_date}")

        # Check last active date
        if last_active and last_active > current_date:
            future_elements.append(f"Last active date ({last_active}) is in the future relative to {current_date}")

        # Check job dates
        for idx, job in enumerate(history):
            start = parse_date(job.get("start_date"))
            end = parse_date(job.get("end_date"))
            company = job.get("company", f"Company {idx+1}")

            if start and start > current_date:
                future_elements.append(f"Job start at {company} ({start}) is after current date ({current_date})")
            if end and end > current_date:
                future_elements.append(f"Job end at {company} ({end}) is after current date ({current_date})")

        # Check education years
        for idx, edu in enumerate(education):
            start_yr = edu.get("start_year")
            end_yr = edu.get("end_year")
            inst = edu.get("institution", f"Institution {idx+1}")

            # Allow a reasonable future graduation buffer if candidate is still a student, 
            # but if start_year is beyond current year, it is a flag.
            if start_yr and start_yr > current_date.year:
                future_elements.append(f"Education start at {inst} ({start_yr}) is in the future")
            if end_yr and end_yr > current_date.year + 6: # Unreasonably far in the future
                future_elements.append(f"Education graduation at {inst} ({end_yr}) is unreasonably far in the future")

        if future_elements:
            details = f"Future dates detected relative to current year 2026: {'; '.join(future_elements)}."
            return [
                self.create_evidence(
                    details,
                    {"future_elements": future_elements}
                )
            ]
        return []


class LastActiveBeforeSignupDetector(BaseDetector):
    """
    9. Last active date before signup date.
    Verify signup_date <= last_active_date.
    """
    def __init__(self):
        super().__init__(
            check_id="last_active_before_signup",
            category="temporal_consistency",
            strength="Strong",
            penalty=0.8
        )

    def detect(self, candidate: dict) -> list[dict]:
        signals = candidate.get("redrob_signals", {})
        signup = parse_date(signals.get("signup_date"))
        last_active = parse_date(signals.get("last_active_date"))

        if signup and last_active and last_active < signup:
            details = f"Last active date ({last_active}) is before signup date ({signup})."
            return [
                self.create_evidence(
                    details,
                    {
                        "signup_date": str(signup),
                        "last_active_date": str(last_active)
                    }
                )
            ]
        return []
