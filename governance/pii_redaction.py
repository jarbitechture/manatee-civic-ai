"""
PII Redaction Module
Detects and redacts Personally Identifiable Information (PII) for compliance
Designed for government/public sector data protection requirements
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from datetime import datetime


class PIIType(Enum):
    """Types of PII that can be detected"""

    SSN = "social_security_number"
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone_number"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVERS_LICENSE = "drivers_license"
    ADDRESS = "physical_address"
    NAME = "person_name"
    ZIP_CODE = "zip_code"


@dataclass
class PIIMatch:
    """Represents a detected PII instance"""

    pii_type: PIIType
    original_value: str
    redacted_value: str
    start_pos: int
    end_pos: int
    confidence: float


class PIIRedactor:
    """
    Comprehensive PII detection and redaction system
    Compliant with NIST 800-53, HIPAA, and government data protection standards
    """

    def __init__(self, redaction_char: str = "X", preserve_format: bool = True):
        """
        Initialize PII Redactor

        Args:
            redaction_char: Character to use for redaction (default: X)
            preserve_format: Preserve format of redacted data (e.g., XXX-XX-XXXX for SSN)
        """
        self.redaction_char = redaction_char
        self.preserve_format = preserve_format
        self.redaction_log: List[PIIMatch] = []

        # Compile regex patterns for efficiency
        self.patterns = {
            PIIType.SSN: re.compile(r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),
            PIIType.CREDIT_CARD: re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
            PIIType.EMAIL: re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            PIIType.PHONE: re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            PIIType.IP_ADDRESS: re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            PIIType.DATE_OF_BIRTH: re.compile(
                r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](\d{2}|\d{4})\b"
            ),
            PIIType.ZIP_CODE: re.compile(r"\b\d{5}(?:-\d{4})?\b"),
        }

    def detect_pii(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII instances in text

        Args:
            text: Input text to scan for PII

        Returns:
            List of PIIMatch objects
        """
        matches = []

        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                original = match.group()
                redacted = self._redact_value(original, pii_type)

                pii_match = PIIMatch(
                    pii_type=pii_type,
                    original_value=original,
                    redacted_value=redacted,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=self._calculate_confidence(original, pii_type),
                )
                matches.append(pii_match)

        return matches

    def _redact_value(self, value: str, pii_type: PIIType) -> str:
        """Redact a PII value while optionally preserving format"""
        if not self.preserve_format:
            return f"[REDACTED_{pii_type.value.upper()}]"

        if pii_type == PIIType.SSN:
            return f"{self.redaction_char*3}-{self.redaction_char*2}-{self.redaction_char*4}"
        elif pii_type == PIIType.CREDIT_CARD:
            return f"{self.redaction_char*4}-{self.redaction_char*4}-{self.redaction_char*4}-{self.redaction_char*4}"
        elif pii_type == PIIType.EMAIL:
            parts = value.split("@")
            if len(parts) == 2:
                return f"{self.redaction_char*len(parts[0])}@{parts[1]}"
        elif pii_type == PIIType.PHONE:
            digits = re.sub(r"\D", "", value)
            if len(digits) == 10:
                return f"({self.redaction_char*3}) {self.redaction_char*3}-{self.redaction_char*4}"
            elif len(digits) == 11:
                return f"+{self.redaction_char} ({self.redaction_char*3}) {self.redaction_char*3}-{self.redaction_char*4}"
        elif pii_type == PIIType.ZIP_CODE:
            if "-" in value:
                return f"{self.redaction_char*5}-{self.redaction_char*4}"
            return self.redaction_char * 5

        return self.redaction_char * len(value)

    def _calculate_confidence(self, value: str, pii_type: PIIType) -> float:
        """Calculate confidence score for PII detection"""
        # Basic confidence calculation - can be enhanced with ML models
        if pii_type == PIIType.SSN:
            # High confidence if matches SSN format exactly
            if re.match(r"^\d{3}-\d{2}-\d{4}$", value):
                return 0.95
            return 0.75
        elif pii_type == PIIType.CREDIT_CARD:
            # Validate using Luhn algorithm
            if self._luhn_check(value):
                return 0.95
            return 0.60
        elif pii_type == PIIType.EMAIL:
            # High confidence for emails
            return 0.90
        elif pii_type == PIIType.PHONE:
            return 0.85
        else:
            return 0.70

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        digits = [int(d) for d in re.sub(r"\D", "", card_number)]
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0

    def redact_text(self, text: str, min_confidence: float = 0.7) -> Tuple[str, List[PIIMatch]]:
        """
        Redact all PII from text

        Args:
            text: Input text
            min_confidence: Minimum confidence threshold (0.0-1.0)

        Returns:
            Tuple of (redacted_text, list_of_matches)
        """
        matches = self.detect_pii(text)

        # Filter by confidence
        matches = [m for m in matches if m.confidence >= min_confidence]

        # Sort by position (reverse) to maintain indices during replacement
        matches.sort(key=lambda x: x.start_pos, reverse=True)

        redacted_text = text
        for match in matches:
            redacted_text = (
                redacted_text[: match.start_pos]
                + match.redacted_value
                + redacted_text[match.end_pos :]
            )

        # Log redactions
        self.redaction_log.extend(matches)

        return redacted_text, matches

    def get_pii_summary(self, text: str) -> Dict[str, int]:
        """Get summary of PII types detected in text"""
        matches = self.detect_pii(text)
        summary = {}
        for match in matches:
            pii_name = match.pii_type.value
            summary[pii_name] = summary.get(pii_name, 0) + 1
        return summary

    def generate_audit_report(self) -> Dict:
        """Generate audit report of all redactions"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_redactions": len(self.redaction_log),
            "by_type": {},
            "redactions": [],
        }

        for match in self.redaction_log:
            pii_name = match.pii_type.value
            report["by_type"][pii_name] = report["by_type"].get(pii_name, 0) + 1

            # Hash original value for audit trail (not storing actual PII)
            hashed = hashlib.sha256(match.original_value.encode()).hexdigest()[:16]

            report["redactions"].append(
                {
                    "type": pii_name,
                    "hash": hashed,
                    "confidence": match.confidence,
                    "position": f"{match.start_pos}-{match.end_pos}",
                }
            )

        return report

    def save_audit_report(self, filepath: str):
        """Save audit report to file"""
        report = self.generate_audit_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

    def clear_log(self):
        """Clear redaction log"""
        self.redaction_log = []


def quick_redact(text: str) -> str:
    """Quick utility function to redact PII from text"""
    redactor = PIIRedactor()
    redacted_text, _ = redactor.redact_text(text)
    return redacted_text


# Example usage and testing
if __name__ == "__main__":
    # Test PII redaction
    test_text = """
    John Smith's SSN is 123-45-6789 and his email is john.smith@example.com.
    His phone number is (555) 123-4567 and he lives at 12345 zip code.
    His credit card is 4532-1234-5678-9010.
    His IP address is 192.168.1.1.
    Date of birth: 01/15/1985
    """

    print("=" * 80)
    print("PII REDACTION TEST")
    print("=" * 80)
    print("\nOriginal Text:")
    print(test_text)

    redactor = PIIRedactor(preserve_format=True)

    print("\n" + "=" * 80)
    print("PII DETECTION")
    print("=" * 80)
    summary = redactor.get_pii_summary(test_text)
    for pii_type, count in summary.items():
        print(f"  {pii_type}: {count} instance(s)")

    print("\n" + "=" * 80)
    print("REDACTED TEXT")
    print("=" * 80)
    redacted_text, matches = redactor.redact_text(test_text)
    print(redacted_text)

    print("\n" + "=" * 80)
    print("AUDIT REPORT")
    print("=" * 80)
    report = redactor.generate_audit_report()
    print(json.dumps(report, indent=2))

    print("\n✅ PII Redaction Module Test Complete")
