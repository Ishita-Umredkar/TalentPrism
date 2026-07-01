"""
__init__.py

Package initialization for honeypot check scripts.
"""

from .base_detector import BaseDetector
from .temporal_consistency import (
    JobDurationVsDateDiffDetector,
    CareerDurationVsYoeDetector,
    EducationOrderConsistencyDetector,
    MultipleCurrentJobsDetector,
    FutureDatesDetector
)
from .skill_consistency import (
    ExpertSkillZeroDurationDetector,
    SkillDurationVsTimelineDetector
)
from .career_consistency import (
    ExecutiveTitleMinimalExperienceDetector,
    HeadlineSummaryContradictionDetector
)
from .company_integrity import EmploymentBeforeCompanyExistedDetector

# Export all 10 active detector classes for the registry (T3 and T7 removed)
ALL_DETECTORS = [
    JobDurationVsDateDiffDetector,
    CareerDurationVsYoeDetector,
    EducationOrderConsistencyDetector,
    MultipleCurrentJobsDetector,
    FutureDatesDetector,
    ExpertSkillZeroDurationDetector,
    SkillDurationVsTimelineDetector,
    ExecutiveTitleMinimalExperienceDetector,
    HeadlineSummaryContradictionDetector,
    EmploymentBeforeCompanyExistedDetector
]
