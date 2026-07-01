"""
base_detector.py

Defines the BaseDetector abstract class and shared utility functions for honeypot detection.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
import re

class BaseDetector(ABC):
    """
    Abstract Base Class for all Honeypot Detectors.
    """
    def __init__(self, check_id: str, category: str, strength: str, penalty: float):
        self.check_id = check_id
        self.category = category
        self.strength = strength
        self.penalty = penalty

    @abstractmethod
    def detect(self, candidate: dict) -> list[dict]:
        """
        Runs the consistency check on the candidate.
        
        Args:
            candidate (dict): The candidate profile data structure.
            
        Returns:
            list[dict]: A list of evidence dicts. If the candidate passes, returns an empty list [].
                        Each evidence dict should follow this structure:
                        {
                            "check_id": str,
                            "category": str,
                            "strength": str,
                            "penalty": float,
                            "details": str,
                            "metrics": dict
                        }
        """
        pass

    def create_evidence(self, details: str, metrics: dict = None) -> dict:
        """
        Helper to construct a standardized evidence dictionary.
        """
        return {
            "check_id": self.check_id,
            "category": self.category,
            "strength": self.strength,
            "penalty": self.penalty,
            "details": details,
            "metrics": metrics or {}
        }


# --- Shared Date & Utility Helper Functions ---

def parse_date(date_str: str) -> date:
    """
    Parses a YYYY-MM-DD date string into a datetime.date object.
    Returns None if date_str is empty, null, or malformed.
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def get_last_active_date(candidate: dict) -> date:
    """
    Retrieves the candidate's last_active_date from redrob_signals.
    Defaults to 2026-06-13 (current local date) if missing or malformed.
    """
    signals = candidate.get("redrob_signals", {})
    last_active_str = signals.get("last_active_date")
    parsed = parse_date(last_active_str)
    if parsed:
        return parsed
    # Fallback default date as of current timeframe
    return date(2026, 6, 13)


def calculate_months_between(start_date: date, end_date: date) -> int:
    """
    Calculates the duration in months between two dates using rounding based on days.
    """
    if not start_date or not end_date:
        return 0
    delta_days = (end_date - start_date).days
    if delta_days <= 0:
        return 0
    return round(delta_days / 30.4375)


def extract_years_from_text(text: str) -> list[float]:
    """
    Extracts numbers describing years of experience from text descriptions
    e.g., "6.9 years", "10+ years", "12.5+ yrs".
    """
    if not text:
        return []
    # Match patterns like: 10+, 10.5, 6.9 followed by space/zero-or-more-chars then "year", "yr", "years", "yrs"
    pattern = r"\b(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:year|yr)s?\b"
    matches = re.findall(pattern, text.lower())
    return [float(x) for x in matches]
