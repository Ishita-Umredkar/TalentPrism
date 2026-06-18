"""
test_honeypot_detectors.py

Automated unit tests for the Stage 1 Honeypot Detection system.
"""

import unittest
from datetime import date
from stage1.scripts.temporal_consistency import (
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
from stage1.scripts.skill_consistency import (
    ExpertSkillZeroDurationDetector,
    SkillDurationVsTimelineDetector,
    AssessmentContradictsProficiencyDetector
)
from stage1.scripts.career_consistency import (
    SeniorTitleLowExperienceDetector,
    UnrealisticPromotionVelocityDetector,
    ExecutiveTitleWithoutProgressionDetector,
    TitleExperienceInconsistencyDetector
)
from stage1.scripts.profile_integrity import (
    HeadlineSummaryContradictionDetector,
    SalaryInconsistentWithExperienceDetector
)
from stage1.scripts.company_integrity import CompanyExperienceVsAgeDetector


class TestTemporalConsistency(unittest.TestCase):

    def test_job_duration_vs_date_difference(self):
        detector = JobDurationVsDateDiffDetector()

        # Case 1: Matching duration
        candidate_ok = {
            "redrob_signals": {"last_active_date": "2026-05-20"},
            "career_history": [
                {
                    "company": "Company A",
                    "start_date": "2024-03-08",
                    "end_date": "2024-06-08",
                    "duration_months": 3,
                    "is_current": False
                }
            ]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Mismatch duration (mismatch > 18 months)
        candidate_bad = {
            "redrob_signals": {"last_active_date": "2026-05-20"},
            "career_history": [
                {
                    "company": "Company B",
                    "start_date": "2024-03-08",
                    "end_date": "2024-06-08",
                    "duration_months": 25,  # Mismatch by 22 months (> 18 threshold)
                    "is_current": False
                }
            ]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "job_duration_vs_date_difference")

    def test_career_duration_vs_yoe(self):
        detector = CareerDurationVsYoeDetector()

        # Case 1: Match (sum of job months / 12 = 1.0, YOE = 1.0)
        candidate_ok = {
            "profile": {"years_of_experience": 1.0},
            "career_history": [{"duration_months": 12}]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Mismatch (sum of job months = 1.0 year, YOE = 3.5 years)
        candidate_bad = {
            "profile": {"years_of_experience": 3.5},
            "career_history": [{"duration_months": 12}]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "career_duration_vs_yoe")

    def test_experience_since_education(self):
        detector = ExperienceSinceEducationDetector()

        # Case 1: Valid timeline
        candidate_ok = {
            "profile": {"years_of_experience": 2.0},
            "redrob_signals": {"last_active_date": "2026-05-20"},
            "education": [{"start_year": 2020, "end_year": 2024}],
            "career_history": [{"start_date": "2024-06-01"}]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Experience starts before education (e.g. 10 years before)
        candidate_bad_start = {
            "profile": {"years_of_experience": 1.0},
            "redrob_signals": {"last_active_date": "2026-05-20"},
            "education": [{"start_year": 2020, "end_year": 2024}],
            "career_history": [{"start_date": "2010-06-01"}]
        }
        evidence = detector.detect(candidate_bad_start)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "experience_since_education")

    def test_education_order_consistency(self):
        detector = EducationOrderConsistencyDetector()

        # Case 1: Bachelor ends before Master
        candidate_ok = {
            "education": [
                {"degree": "Bachelor of Engineering", "end_year": 2018},
                {"degree": "Master of Science", "end_year": 2020}
            ]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Master ends before Bachelor at same college and major
        candidate_bad = {
            "education": [
                {"degree": "Bachelor of Engineering", "end_year": 2020, "institution": "LPU", "field_of_study": "CS"},
                {"degree": "Master of Science", "end_year": 2018, "institution": "LPU", "field_of_study": "CS"}
            ]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "education_order_consistency")

        # Case 3: Master ends before Bachelor but at DIFFERENT college and major
        candidate_diff_ok = {
            "education": [
                {"degree": "Bachelor of Engineering", "end_year": 2020, "institution": "LPU", "field_of_study": "CS"},
                {"degree": "Master of Science", "end_year": 2018, "institution": "IIT", "field_of_study": "EE"}
            ]
        }
        self.assertEqual(detector.detect(candidate_diff_ok), [])

    def test_overlapping_jobs(self):
        detector = OverlappingJobsDetector()

        # Case 1: No major overlap
        candidate_ok = {
            "redrob_signals": {"last_active_date": "2026-05-20"},
            "career_history": [
                {"company": "A", "start_date": "2020-01-01", "end_date": "2020-12-31", "is_current": False},
                {"company": "B", "start_date": "2021-01-01", "end_date": "2021-12-31", "is_current": False}
            ]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Significant overlap (>90 days)
        candidate_bad = {
            "redrob_signals": {"last_active_date": "2026-05-20"},
            "career_history": [
                {"company": "A", "start_date": "2020-01-01", "end_date": "2020-12-31", "is_current": False},
                {"company": "B", "start_date": "2020-06-01", "end_date": "2021-06-01", "is_current": False} # Overlaps from June to Dec
            ]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "overlapping_jobs")

    def test_multiple_current_jobs(self):
        detector = MultipleCurrentJobsDetector()

        # Case 1: One current job
        candidate_ok = {
            "career_history": [
                {"company": "A", "is_current": True, "start_date": "2024-01-01"},
                {"company": "B", "is_current": False, "start_date": "2022-01-01", "end_date": "2023-12-31"}
            ]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Multiple current jobs
        candidate_bad = {
            "career_history": [
                {"company": "A", "is_current": True, "start_date": "2024-01-01"},
                {"company": "B", "is_current": True, "start_date": "2025-01-01"}
            ]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "multiple_current_jobs")

    def test_current_job_chronology(self):
        detector = CurrentJobChronologyDetector()

        # Case 1: Current job started after previous job
        candidate_ok = {
            "career_history": [
                {"company": "A", "is_current": True, "start_date": "2024-01-01"},
                {"company": "B", "is_current": False, "start_date": "2020-01-01", "end_date": "2023-12-31"}
            ]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Current job started BEFORE non-current job
        candidate_bad = {
            "career_history": [
                {"company": "A", "is_current": True, "start_date": "2020-01-01"},
                {"company": "B", "is_current": False, "start_date": "2024-01-01", "end_date": "2025-01-01"}
            ]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "current_job_chronology")

    def test_future_dates(self):
        detector = FutureDatesDetector()

        # Case 1: Valid dates
        candidate_ok = {
            "redrob_signals": {"last_active_date": "2026-05-20", "signup_date": "2025-10-16"},
            "career_history": [
                {"company": "A", "start_date": "2024-01-01", "end_date": "2026-01-01"}
            ],
            "education": [{"start_year": 2020, "end_year": 2024}]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Future date in career history
        candidate_bad = {
            "redrob_signals": {"last_active_date": "2026-05-20", "signup_date": "2025-10-16"},
            "career_history": [
                {"company": "A", "start_date": "2027-01-01", "end_date": "2028-01-01"}
            ]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "future_dates")

    def test_last_active_before_signup(self):
        detector = LastActiveBeforeSignupDetector()

        # Case 1: Last active after signup
        candidate_ok = {
            "redrob_signals": {"last_active_date": "2026-05-20", "signup_date": "2025-10-16"}
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Last active before signup
        candidate_bad = {
            "redrob_signals": {"last_active_date": "2025-05-20", "signup_date": "2025-10-16"}
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "last_active_before_signup")


class TestSkillConsistency(unittest.TestCase):

    def test_expert_skill_zero_duration(self):
        detector = ExpertSkillZeroDurationDetector()

        # Case 1: Advanced skill with >0 duration
        candidate_ok = {
            "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 12}]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Advanced skill with 0 duration
        candidate_bad = {
            "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 0}]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "expert_skill_zero_duration")

    def test_skill_duration_vs_timeline(self):
        detector = SkillDurationVsTimelineDetector()

        # Case 1: Skill duration is inside career + edu bounds
        candidate_ok = {
            "career_history": [{"duration_months": 12}],
            "education": [{"start_year": 2018, "end_year": 2022}], # 48 months
            # Total allowed = 12 + 48 + 12 (buffer) = 72 months
            "skills": [{"name": "Python", "duration_months": 60}]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Skill duration exceeds timeline
        candidate_bad = {
            "career_history": [{"duration_months": 12}],
            "education": [{"start_year": 2018, "end_year": 2022}], # 48 months
            # Total allowed = 12 + 48 + 12 = 72 months
            "skills": [{"name": "Python", "duration_months": 90}]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "skill_duration_greater_than_timeline")

    def test_assessment_contradicts_proficiency(self):
        detector = AssessmentContradictsProficiencyDetector()

        # Case 1: Consistent (Advanced with score 80)
        candidate_ok = {
            "skills": [{"name": "Python", "proficiency": "advanced"}],
            "redrob_signals": {"skill_assessment_scores": {"Python": 80.0}}
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Contradictory (Advanced with score 20)
        candidate_bad = {
            "skills": [{"name": "Python", "proficiency": "advanced"}],
            "redrob_signals": {"skill_assessment_scores": {"Python": 20.0}}
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "assessment_contradicts_proficiency")


class TestCareerConsistency(unittest.TestCase):

    def test_senior_title_low_experience(self):
        detector = SeniorTitleLowExperienceDetector()

        # Case 1: Senior with high experience (5.0 YOE)
        candidate_ok = {
            "profile": {"years_of_experience": 5.0, "current_title": "Senior Engineer"}
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Senior with low experience (1.0 YOE)
        candidate_bad = {
            "profile": {"years_of_experience": 1.0, "current_title": "Senior Engineer"}
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "senior_title_low_experience")

    def test_unrealistic_promotion_velocity(self):
        detector = UnrealisticPromotionVelocityDetector()

        # Case 1: Natural promotion (e.g. 24 months)
        candidate_ok = {
            "career_history": [
                {"title": "Junior Engineer", "start_date": "2020-01-01"},
                {"title": "Senior Engineer", "start_date": "2022-01-01"}
            ]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Unrealistic promotion (e.g. 6 months)
        candidate_bad = {
            "career_history": [
                {"title": "Junior Engineer", "start_date": "2020-01-01"},
                {"title": "Senior Engineer", "start_date": "2020-07-01"}
            ]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "unrealistic_promotion_velocity")

    def test_executive_title_no_progression(self):
        detector = ExecutiveTitleWithoutProgressionDetector()

        # Case 1: Executive with high experience
        candidate_ok = {
            "profile": {"years_of_experience": 6.0, "current_title": "CTO"}
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Executive with low experience (< 4.0 YOE)
        candidate_bad = {
            "profile": {"years_of_experience": 2.0, "current_title": "CTO"}
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "executive_title_no_progression")

    def test_title_experience_inconsistency(self):
        detector = TitleExperienceInconsistencyDetector()

        # Case 1: Matching/overlapping words
        candidate_ok = {
            "profile": {"current_title": "Backend Engineer"},
            "career_history": [
                {"title": "Senior Backend Developer", "is_current": True}
            ]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Completely mismatched words
        candidate_bad = {
            "profile": {"current_title": "Backend Engineer"},
            "career_history": [
                {"title": "Product Marketing Manager", "is_current": True}
            ]
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "title_experience_inconsistency")


class TestProfileIntegrity(unittest.TestCase):

    def test_headline_summary_contradiction(self):
        detector = HeadlineSummaryContradictionDetector()

        # Case 1: Matching YOE
        candidate_ok = {
            "profile": {
                "years_of_experience": 5.0,
                "headline": "Developer | 5 years experience",
                "summary": "Software professional with 5.2+ yrs."
            }
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Contradicting YOE (profile=5.0, text claims 10 years)
        candidate_bad = {
            "profile": {
                "years_of_experience": 5.0,
                "headline": "Developer | 10 years experience",
                "summary": "Professional with 10+ yrs."
            }
        }
        evidence = detector.detect(candidate_bad)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "headline_summary_contradiction")

    def test_salary_inconsistent_with_experience(self):
        detector = SalaryInconsistentWithExperienceDetector()

        # Case 1: Consistent expectations
        candidate_ok = {
            "profile": {"years_of_experience": 1.0},
            "redrob_signals": {
                "expected_salary_range_inr_lpa": {"min": 8.0, "max": 12.0}
            }
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: High expectation for junior (min = 30 LPA for 1 YOE)
        candidate_bad_junior = {
            "profile": {"years_of_experience": 1.0},
            "redrob_signals": {
                "expected_salary_range_inr_lpa": {"min": 30.0, "max": 40.0}
            }
        }
        evidence = detector.detect(candidate_bad_junior)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "salary_inconsistent_with_experience")

        # Case 3: Low expectation for senior (max = 5 LPA for 10 YOE)
        candidate_bad_senior = {
            "profile": {"years_of_experience": 10.0},
            "redrob_signals": {
                "expected_salary_range_inr_lpa": {"min": 3.0, "max": 5.0}
            }
        }
        evidence = detector.detect(candidate_bad_senior)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "salary_inconsistent_with_experience")


class TestCompanyIntegrity(unittest.TestCase):

    def test_company_experience_vs_age(self):
        detector = CompanyExperienceVsAgeDetector()

        # Case 1: Job starts after company is founded and duration is valid
        candidate_ok = {
            "redrob_signals": {"last_active_date": "2026-05-20"},
            "career_history": [
                {
                    "company": "A",
                    "start_date": "2022-01-01",
                    "duration_months": 24,
                    "company_founded_year": 2020 # Company is 6 years old, job is 2 years
                }
            ]
        }
        self.assertEqual(detector.detect(candidate_ok), [])

        # Case 2: Job starts before company was founded
        candidate_bad_start = {
            "redrob_signals": {"last_active_date": "2026-05-20"},
            "career_history": [
                {
                    "company": "B",
                    "start_date": "2018-01-01",
                    "duration_months": 24,
                    "company_founded_year": 2020 # Started working 2 years before company founded
                }
            ]
        }
        evidence = detector.detect(candidate_bad_start)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "company_experience_vs_age")
        self.assertIn("started working in 2018, but company founded in 2020", evidence[0]["details"])

        # Case 3: Job duration exceeds company age
        candidate_bad_duration = {
            "redrob_signals": {"last_active_date": "2022-05-20"},
            "career_history": [
                {
                    "company": "C",
                    "start_date": "2020-01-01",
                    "duration_months": 72, # 6 years of experience
                    "company_founded_year": 2020 # Company is 2 years old in 2022
                }
            ]
        }
        evidence = detector.detect(candidate_bad_duration)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["check_id"], "company_experience_vs_age")
        self.assertIn("experience: 6.00 yrs, company age: 2 yrs", evidence[0]["details"])


if __name__ == "__main__":
    unittest.main()
