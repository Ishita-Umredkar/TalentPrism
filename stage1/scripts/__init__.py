"""
__init__.py

Package initialization for honeypot check scripts.
"""

from .base_detector import BaseDetector
from .temporal_consistency import (
    JobDurationVsDateDiffDetector,
    CareerDurationVsYoeDetector,
    ExperienceSinceEducationDetector,
    EducationOrderConsistencyDetector,
    OverlappingJobsDetector,
    MultipleCurrentJobsDetector,
    CurrentJobChronologyDetector,
    FutureDatesDetector,
    LastActiveBeforeSignupDetector
)
from .skill_consistency import (
    ExpertSkillZeroDurationDetector,
    SkillDurationVsTimelineDetector
)
from .career_consistency import (
    SeniorTitleLowExperienceDetector,
    UnrealisticPromotionVelocityDetector,
    ExecutiveTitleWithoutProgressionDetector,
    TitleExperienceInconsistencyDetector
)
from .profile_integrity import (
    HeadlineSummaryContradictionDetector,
    SalaryInconsistentWithExperienceDetector
)
from .company_integrity import CompanyExperienceVsAgeDetector

# Export all detector classes for easy registry loading
ALL_DETECTORS = [
    JobDurationVsDateDiffDetector,
    CareerDurationVsYoeDetector,
    ExperienceSinceEducationDetector,
    EducationOrderConsistencyDetector,
    OverlappingJobsDetector,
    MultipleCurrentJobsDetector,
    CurrentJobChronologyDetector,
    FutureDatesDetector,
    LastActiveBeforeSignupDetector,
    ExpertSkillZeroDurationDetector,
    SkillDurationVsTimelineDetector,
    SeniorTitleLowExperienceDetector,
    UnrealisticPromotionVelocityDetector,
    ExecutiveTitleWithoutProgressionDetector,
    TitleExperienceInconsistencyDetector,
    HeadlineSummaryContradictionDetector,
    SalaryInconsistentWithExperienceDetector
]
