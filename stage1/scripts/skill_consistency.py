"""
skill_consistency.py

Implements Skill Consistency detectors (Strong Evidence) for candidate profile validation.
"""

from .base_detector import BaseDetector

class ExpertSkillZeroDurationDetector(BaseDetector):
    """
    10. Expert/Advanced skill with zero duration.
    Checks if a skill with high proficiency claims zero duration.
    """
    def __init__(self):
        super().__init__(
            check_id="expert_skill_zero_duration",
            category="skill_consistency",
            strength="Strong",
            penalty=0.3
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
            details = (
                f"Candidate claimed 'advanced' or 'expert' proficiency but with 0 months "
                f"duration for skills: {', '.join(offending_skills)}."
            )
            return [
                self.create_evidence(
                    details,
                    {"offending_skills": offending_skills}
                )
            ]
        return []


class SkillDurationVsTimelineDetector(BaseDetector):
    """
    11. Skill duration greater than career + education duration.
    Checks if any skill duration exceeds the combined sum of career and education timelines (plus a 1-year buffer).
    """
    def __init__(self):
        super().__init__(
            check_id="skill_duration_greater_than_timeline",
            category="skill_consistency",
            strength="Strong",
            penalty=0.5
        )

    def detect(self, candidate: dict) -> list[dict]:
        history = candidate.get("career_history", [])
        education = candidate.get("education", [])
        skills = candidate.get("skills", [])

        # Total career months
        total_career_months = sum(job.get("duration_months", 0) for job in history)

        # Total education months
        total_education_months = 0
        for edu in education:
            start = edu.get("start_year")
            end = edu.get("end_year")
            if start and end:
                duration_yrs = max(end - start, 0)
                total_education_months += duration_yrs * 12
            elif end:
                # Default 3 years (36 months) if only graduation year is present
                total_education_months += 36

        max_allowed_months = total_career_months + total_education_months + 12 # 12 months buffer
        offending_skills = []

        for skill in skills:
            name = skill.get("name")
            duration = skill.get("duration_months", 0)
            if duration > max_allowed_months:
                offending_skills.append(
                    f"{name} ({duration} mos vs limit {max_allowed_months} mos)"
                )

        if offending_skills:
            details = (
                f"Skill duration exceeds combined career history and education timelines "
                f"(max allowed: {max_allowed_months} months) for: {', '.join(offending_skills)}."
            )
            return [
                self.create_evidence(
                    details,
                    {
                        "total_career_months": total_career_months,
                        "total_education_months": total_education_months,
                        "max_allowed_months": max_allowed_months,
                        "offending_skills": offending_skills
                    }
                )
            ]
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
