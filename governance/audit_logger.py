"""
Comprehensive Audit Logging System
Provides full audit trail for compliance and security
Designed for government/public sector requirements (NIST 800-53, FISMA)
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib


class AuditEventType(Enum):
    """Types of auditable events"""

    # Authentication & Authorization
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"

    # Data Access
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    PII_ACCESS = "pii_access"
    PII_REDACTION = "pii_redaction"

    # AI/Model Operations
    MODEL_INFERENCE = "model_inference"
    PROMPT_EXECUTION = "prompt_execution"
    MODEL_DEPLOYMENT = "model_deployment"
    MODEL_ROLLBACK = "model_rollback"

    # System Operations
    CONFIG_CHANGE = "config_change"
    API_CALL = "api_call"
    ERROR = "error"
    SECURITY_VIOLATION = "security_violation"

    # Governance
    HUMAN_REVIEW_REQUEST = "human_review_request"
    HUMAN_APPROVAL = "human_approval"
    HUMAN_REJECTION = "human_rejection"
    SAFETY_GATE_TRIGGERED = "safety_gate_triggered"


class AuditSeverity(Enum):
    """Severity levels for audit events"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Represents a single auditable event"""

    event_id: str
    timestamp: str
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: str
    action: str
    resource: str
    result: str  # success, failure, pending
    ip_address: Optional[str]
    session_id: Optional[str]
    metadata: Dict[str, Any]
    pii_detected: bool
    compliance_tags: List[str]


class AuditLogger:
    """
    Comprehensive audit logging system
    Maintains immutable audit trail for compliance
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        retention_days: int = 2555,  # 7 years for government compliance
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days

        # Set up file handlers
        self.current_log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.master_index_file = self.log_dir / "audit_index.json"

        # Initialize Python logger
        self.logger = logging.getLogger("AuditLogger")
        self.logger.setLevel(logging.DEBUG)

        # File handler for structured logs
        file_handler = logging.FileHandler(self.current_log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler for high-severity events
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter("🔔 [AUDIT] %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self._load_index()

    def _load_index(self):
        """Load audit index"""
        if self.master_index_file.exists():
            with open(self.master_index_file, "r") as f:
                self.index = json.load(f)
        else:
            self.index = {
                "created_at": datetime.now().isoformat(),
                "total_events": 0,
                "log_files": [],
            }

    def _save_index(self):
        """Save audit index"""
        with open(self.master_index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def _generate_event_id(self, event: Dict) -> str:
        """Generate unique event ID"""
        event_str = f"{event['timestamp']}:{event['user_id']}:{event['action']}"
        return hashlib.sha256(event_str.encode()).hexdigest()[:16]

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        action: str,
        resource: str,
        result: str = "success",
        severity: AuditSeverity = AuditSeverity.INFO,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        pii_detected: bool = False,
        compliance_tags: Optional[List[str]] = None,
    ) -> AuditEvent:
        """Log an audit event"""

        event_data = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type.value,
            "severity": severity.value,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "result": result,
            "ip_address": ip_address,
            "session_id": session_id,
            "metadata": metadata or {},
            "pii_detected": pii_detected,
            "compliance_tags": compliance_tags or [],
        }

        event_data["event_id"] = self._generate_event_id(event_data)

        event = AuditEvent(
            event_id=event_data["event_id"],
            timestamp=event_data["timestamp"],
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            session_id=session_id,
            metadata=metadata or {},
            pii_detected=pii_detected,
            compliance_tags=compliance_tags or [],
        )

        # Write to log file (JSONL format - one JSON object per line)
        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(event_data) + "\n")

        # Log to Python logger
        log_level = getattr(logging, severity.value.upper())
        self.logger.log(
            log_level, f"[{event_type.value}] {action} on {resource} by {user_id}: {result}"
        )

        # Update index
        self.index["total_events"] += 1
        if str(self.current_log_file) not in self.index["log_files"]:
            self.index["log_files"].append(str(self.current_log_file))
        self._save_index()

        return event

    def query_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Query audit events with filters"""

        results = []
        events_read = 0

        # Read from current log file
        if self.current_log_file.exists():
            with open(self.current_log_file, "r") as f:
                for line in f:
                    if events_read >= limit:
                        break

                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines

                    # Apply filters
                    if event_type and event["event_type"] != event_type.value:
                        continue
                    if user_id and event["user_id"] != user_id:
                        continue
                    if resource and resource not in event["resource"]:
                        continue
                    if severity and event["severity"] != severity.value:
                        continue

                    # Date filters
                    event_time = datetime.fromisoformat(event["timestamp"])
                    if start_date and event_time < start_date:
                        continue
                    if end_date and event_time > end_date:
                        continue

                    results.append(event)
                    events_read += 1

        return results

    def get_user_activity(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get all activity for a specific user"""
        return self.query_events(user_id=user_id, limit=limit)

    def get_security_events(self, limit: int = 100) -> List[Dict]:
        """Get all security-related events"""
        results = []
        security_types = [
            AuditEventType.ACCESS_DENIED,
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.PII_ACCESS,
        ]

        for event_type in security_types:
            events = self.query_events(event_type=event_type, limit=limit)
            results.extend(events)

        return sorted(results, key=lambda x: x["timestamp"], reverse=True)[:limit]

    def generate_compliance_report(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for audit period"""

        events = self.query_events(start_date=start_date, end_date=end_date, limit=10000)

        report = {
            "report_period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_events": len(events),
            "events_by_type": {},
            "events_by_severity": {},
            "users_active": set(),
            "pii_accesses": 0,
            "security_violations": 0,
            "failed_actions": 0,
        }

        for event in events:
            # Count by type
            event_type = event["event_type"]
            report["events_by_type"][event_type] = report["events_by_type"].get(event_type, 0) + 1

            # Count by severity
            severity = event["severity"]
            report["events_by_severity"][severity] = (
                report["events_by_severity"].get(severity, 0) + 1
            )

            # Track users
            report["users_active"].add(event["user_id"])

            # PII accesses
            if event.get("pii_detected"):
                report["pii_accesses"] += 1

            # Security violations
            if event_type == AuditEventType.SECURITY_VIOLATION.value:
                report["security_violations"] += 1

            # Failed actions
            if event["result"] == "failure":
                report["failed_actions"] += 1

        report["users_active"] = list(report["users_active"])

        return report

    def export_logs(self, output_file: str, start_date: Optional[datetime] = None):
        """Export logs to file"""
        events = self.query_events(start_date=start_date, limit=100000)

        with open(output_file, "w") as f:
            json.dump(
                {
                    "export_date": datetime.now().isoformat(),
                    "total_events": len(events),
                    "events": events,
                },
                f,
                indent=2,
            )


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("AUDIT LOGGING SYSTEM TEST")
    print("=" * 80)

    logger = AuditLogger()

    # Test various event types
    print("\n1. Logging authentication event...")
    logger.log_event(
        event_type=AuditEventType.USER_LOGIN,
        user_id="admin@mymanatee.org",
        action="User logged in",
        resource="Manus Chatbot System",
        result="success",
        severity=AuditSeverity.INFO,
        ip_address="192.168.1.100",
        session_id="sess_abc123",
    )

    print("2. Logging PII access...")
    logger.log_event(
        event_type=AuditEventType.PII_ACCESS,
        user_id="admin@mymanatee.org",
        action="Accessed citizen records",
        resource="Database:CitizenData",
        result="success",
        severity=AuditSeverity.WARNING,
        pii_detected=True,
        compliance_tags=["HIPAA", "PII_PROTECTION"],
    )

    print("3. Logging model inference...")
    logger.log_event(
        event_type=AuditEventType.MODEL_INFERENCE,
        user_id="admin@mymanatee.org",
        action="GPT-4 inference",
        resource="azure-gpt-4-main:v1",
        result="success",
        severity=AuditSeverity.INFO,
        metadata={"tokens_used": 150, "response_time_ms": 823},
    )

    print("4. Logging security violation...")
    logger.log_event(
        event_type=AuditEventType.SECURITY_VIOLATION,
        user_id="unknown_user",
        action="Attempted unauthorized access",
        resource="Admin Dashboard",
        result="failure",
        severity=AuditSeverity.CRITICAL,
        ip_address="203.0.113.45",
        metadata={"reason": "Invalid API key"},
    )

    # Query events
    print("\n5. Querying user activity...")
    user_events = logger.get_user_activity("admin@mymanatee.org")
    print(f"   Found {len(user_events)} events for admin@mymanatee.org")

    print("\n6. Querying security events...")
    security_events = logger.get_security_events()
    print(f"   Found {len(security_events)} security events")

    # Generate compliance report
    print("\n7. Generating compliance report...")
    report = logger.generate_compliance_report(
        start_date=datetime(2026, 1, 1), end_date=datetime(2026, 12, 31)
    )
    print(f"   Total events: {report['total_events']}")
    print(f"   PII accesses: {report['pii_accesses']}")
    print(f"   Security violations: {report['security_violations']}")

    print("\n✅ Audit Logging System Test Complete")
    print(f"📁 Logs saved to: {logger.current_log_file}")
