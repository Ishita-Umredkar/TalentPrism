"""
skill_consistency.py

Implements Skill Consistency detectors (Strong Evidence) for candidate profile validation.
"""

from .base_detector import BaseDetector, parse_date, get_last_active_date, calculate_months_between

class ExpertSkillZeroDurationDetector(BaseDetector):
    """
    S1. Expert Skill with Zero Experience.
    Flags skills marked as Advanced or Expert while having 0 months of experience.
    """
    def __init__(self):
        super().__init__(
            check_id="expert_skill_zero_duration",
            category="skill_consistency",
            strength="Strong",
            penalty=0.4
        )

    def detect(self, candidate: dict) -> list[dict]:
        skills = candidate.get("skills", [])
        offending_skills = []

        for skill in skills:
            name = skill.get("name")
            prof = skill.get("proficiency", "").lower()
            duration = skill.get("duration_months")

            if prof in ["advanced", "expert"] and duration == 0:
                offending_skills.append(name)

        if offending_skills:
            effective_penalty = min(0.4 * len(offending_skills), 0.8)
            details = (
                f"Candidate claimed 'advanced' or 'expert' proficiency but with 0 months "
                f"duration for skills: {', '.join(offending_skills)}."
            )
            evidence = self.create_evidence(
                details,
                {"offending_skills": offending_skills}
            )
            evidence["penalty"] = effective_penalty
            return [evidence]
        return []


class SkillDurationVsTimelineDetector(BaseDetector):
    """
    S2. Skill Duration Exceeds Career Timeline.
    Detects skills whose reported duration exceeds the candidate's total possible career timeline (allowing a 12-month buffer).
    """
    def __init__(self):
        super().__init__(
            check_id="skill_duration_greater_than_timeline",
            category="skill_consistency",
            strength="Strong",
            penalty=0.5
        )

    def detect(self, candidate: dict) -> list[dict]:
        from datetime import date

        history = candidate.get("career_history", [])
        education = candidate.get("education", [])
        skills = candidate.get("skills", [])
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

        earliest_edu_start_date = None
        if edu_starts:
            earliest_edu_year = min(edu_starts)
            # Use January 1st of that year as a safe lower bound start date
            earliest_edu_start_date = date(earliest_edu_year, 1, 1)

        # Find earliest job start date
        job_starts = []
        for job in history:
            start = parse_date(job.get("start_date"))
            if start:
                job_starts.append(start)

        earliest_job_start_date = min(job_starts) if job_starts else None

        # Determine the start of the possible timeline
        timeline_starts = []
        if earliest_edu_start_date:
            timeline_starts.append(earliest_edu_start_date)
        if earliest_job_start_date:
            timeline_starts.append(earliest_job_start_date)

        if not timeline_starts:
            span_months = 0
        else:
            earliest_start = min(timeline_starts)
            span_months = calculate_months_between(earliest_start, last_active)

        # Calculate sum of all job durations (as a fallback in case it's larger)
        sum_job_months = sum(job.get("duration_months", 0) for job in history)

        total_career_months = max(sum_job_months, span_months)
        max_allowed_months = total_career_months + 12 # 12 months buffer
        offending_skills = []

        for skill in skills:
            name = skill.get("name")
            duration = skill.get("duration_months", 0)
            if duration > max_allowed_months:
                offending_skills.append(
                    f"{name} ({duration} mos vs limit {max_allowed_months} mos)"
                )

        if offending_skills:
            effective_penalty = min(0.5 * len(offending_skills), 1.0)
            details = (
                f"Skill duration exceeds combined career history timeline "
                f"(max allowed: {max_allowed_months} months) for: {', '.join(offending_skills)}."
            )
            evidence = self.create_evidence(
                details,
                {
                    "total_career_months": total_career_months,
                    "max_allowed_months": max_allowed_months,
                    "offending_skills": offending_skills
                }
            )
            evidence["penalty"] = effective_penalty
            return [evidence]
        return []


class AssessmentContradictsProficiencyDetector(BaseDetector):
    """
    12. Assessment score contradicting claimed proficiency.
    Checks if an expert/advanced skill has a very low assessment score (<40.0).
    """
    def __init__(self):
        super().__init__(
            check_id="assessment_contradicts_proficiency",
            category="skill_consistency",
            strength="Strong",
            penalty=0.25 # Reduced from 0.5 per user feedback
        )

    def detect(self, candidate: dict) -> list[dict]:
        signals = candidate.get("redrob_signals", {})
        assessments = signals.get("skill_assessment_scores", {})
        skills = candidate.get("skills", [])
        
        if not assessments:
            return []

        offending_skills = []
        for skill in skills:
            name = skill.get("name")
            prof = skill.get("proficiency", "").lower()
            
            # Use case-insensitive matching for assessment scores key
            assessment_score = None
            for key, val in assessments.items():
                if key.lower() == name.lower():
                    assessment_score = val
                    break

            if assessment_score is not None:
                if prof in ["advanced", "expert"] and assessment_score < 40.0:
                    offending_skills.append(
                        f"{name} (claimed: {prof}, score: {assessment_score:.1f})"
                    )

        if offending_skills:
            details = (
                f"Claimed expertise contradicted by low assessment scores: {', '.join(offending_skills)}."
            )
            return [
                self.create_evidence(
                    details,
                    {"offending_skills": offending_skills}
                )
            ]
        return []
