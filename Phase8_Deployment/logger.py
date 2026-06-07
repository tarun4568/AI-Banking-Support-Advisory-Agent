"""
Phase 8: Structured Logger with PII Masking
═════════════════════════════════════════════
Provides a PII-safe logging layer for the NovaTrust AI Banking Agent.

PII MASKING RULES:
  - Account numbers (10–16 digits): ACCT-XXXXXXXX
  - Phone numbers (10 digits starting 6–9): +91-XXXXXXXXXX
  - Email addresses: u***@domain.com
  - Aadhaar (12 digits starting 2–9): XXXX-XXXX-XXXX
  - PAN card (alphanumeric 10-char): XXXXXXXXXX
  - Credit/debit card numbers (16 digits): XXXX-XXXX-XXXX-XXXX

All logs written to Phase8_Deployment/logs/ as structured JSON.
Console output uses human-readable format.
"""

import re
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

# ─────────────────────────────────────────────
# PII Masking Patterns
# ─────────────────────────────────────────────
PII_PATTERNS = [
    # 16-digit card number (4 groups of 4 or 16 continuous)
    (re.compile(r"\b(\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})\b"), "XXXX-XXXX-XXXX-XXXX"),
    # Aadhaar (12 digits, may be space-separated)
    (re.compile(r"\b([2-9]\d{3}\s?\d{4}\s?\d{4})\b"), "XXXX-XXXX-XXXX"),
    # PAN card (ABCDE1234F format)
    (re.compile(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"), "XXXXXXXXXX"),
    # Indian mobile number (10 digits, starts with 6-9)
    (re.compile(r"\b([6-9]\d{9})\b"), "+91-XXXXXXXXXX"),
    # IFSC + account number combo
    (re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b"), "IFSC-XXXXXXXX"),
    # Bank account numbers (9–18 digits, standalone)
    (re.compile(r"\b(\d{9,18})\b"), "ACCT-XXXXXXXX"),
    # Email address
    (re.compile(r"\b([a-zA-Z0-9._%+\-]{1})[a-zA-Z0-9._%+\-]*@([a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b"),
     r"\1***@\2"),
]


def mask_pii(text: str) -> str:
    """Return text with all detected PII patterns replaced with safe placeholders."""
    if not text:
        return text
    masked = text
    for pattern, replacement in PII_PATTERNS:
        if callable(replacement):
            masked = pattern.sub(replacement, masked)
        else:
            masked = pattern.sub(replacement, masked)
    return masked


# ─────────────────────────────────────────────
# Structured Logger
# ─────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


class AgentLogger:
    """Thread-safe structured logger with PII masking for the banking agent."""

    def __init__(self, agent_name: str = "novatrust_agent", log_level: str = "INFO"):
        os.makedirs(LOG_DIR, exist_ok=True)

        self.agent_name = agent_name
        self.session_id = f"sess-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_file = os.path.join(LOG_DIR, f"agent_{datetime.now().strftime('%Y%m%d')}.jsonl")
        self.metrics_file = os.path.join(LOG_DIR, "session_metrics.json")

        # Python logger for console output
        self._logger = logging.getLogger(agent_name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            ))
            self._logger.addHandler(handler)
        self._logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        self.session_metrics = {
            "session_id": self.session_id,
            "started": datetime.now().isoformat(),
            "turn_count": 0,
            "escalations": 0,
            "refusals": 0,
            "errors": 0,
            "total_latency_s": 0.0,
            "feedback_received": 0,
        }

    def log_turn(
        self,
        user_input: str,
        response: str,
        intent_or_topic: Optional[str],
        latency_s: float,
        tools_used: Optional[list] = None,
        chunks_retrieved: Optional[int] = None,
        was_refusal: bool = False,
        was_escalation: bool = False,
    ) -> None:
        """Log one conversation turn with PII masking."""
        # Mask PII from user input and response before logging
        safe_input = mask_pii(user_input)
        # We only log a snippet of the response (not full text, to save space)
        safe_response_snippet = mask_pii(response[:200])

        record = {
            "type": "TURN",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "turn": self.session_metrics["turn_count"] + 1,
            "input_snippet": safe_input[:120],
            "response_snippet": safe_response_snippet,
            "topic": intent_or_topic,
            "latency_s": round(latency_s, 3),
            "tools_used": tools_used or [],
            "chunks_retrieved": chunks_retrieved,
            "was_refusal": was_refusal,
            "was_escalation": was_escalation,
        }

        self._write_jsonl(record)

        # Update session metrics
        self.session_metrics["turn_count"] += 1
        self.session_metrics["total_latency_s"] += latency_s
        if was_refusal:
            self.session_metrics["refusals"] += 1
        if was_escalation:
            self.session_metrics["escalations"] += 1

        # Console log (INFO level)
        self._logger.info(
            "TURN=%d | TOPIC=%s | LATENCY=%.2fs | REFUSAL=%s | ESCALATION=%s",
            self.session_metrics["turn_count"],
            intent_or_topic,
            latency_s,
            was_refusal,
            was_escalation,
        )

    def log_feedback(self, turn: int, thumbs: Optional[str], rating: Optional[int], category: Optional[str]) -> None:
        record = {
            "type": "FEEDBACK",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "for_turn": turn,
            "thumbs": thumbs,
            "rating": rating,
            "category": category,
        }
        self._write_jsonl(record)
        self.session_metrics["feedback_received"] += 1
        self._logger.info("FEEDBACK | TURN=%d | THUMBS=%s | RATING=%s | CATEGORY=%s",
                         turn, thumbs, rating, category)

    def log_error(self, error_type: str, message: str, recoverable: bool = True) -> None:
        record = {
            "type": "ERROR",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "error_type": error_type,
            "message": mask_pii(message[:300]),
            "recoverable": recoverable,
        }
        self._write_jsonl(record)
        self.session_metrics["errors"] += 1
        self._logger.error("ERROR | TYPE=%s | MSG=%s | RECOVERABLE=%s",
                          error_type, message[:100], recoverable)

    def log_escalation(self, reason: str, ticket_id: str) -> None:
        record = {
            "type": "ESCALATION",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "reason_snippet": mask_pii(reason[:200]),
            "ticket_id": ticket_id,
        }
        self._write_jsonl(record)
        self.session_metrics["escalations"] += 1
        self._logger.warning("ESCALATION | TICKET=%s | REASON=%s", ticket_id, reason[:80])

    def close_session(self) -> dict:
        """Finalise session metrics and write to metrics file."""
        self.session_metrics["ended"] = datetime.now().isoformat()
        avg_latency = (
            self.session_metrics["total_latency_s"] / self.session_metrics["turn_count"]
            if self.session_metrics["turn_count"] > 0 else 0
        )
        self.session_metrics["avg_latency_s"] = round(avg_latency, 3)

        # Append to metrics file
        existing = []
        if os.path.exists(self.metrics_file):
            try:
                with open(self.metrics_file) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        existing.append(self.session_metrics)
        with open(self.metrics_file, "w") as f:
            json.dump(existing, f, indent=2)

        self._logger.info(
            "SESSION END | TURNS=%d | AVG_LATENCY=%.2fs | REFUSALS=%d | ESCALATIONS=%d",
            self.session_metrics["turn_count"],
            avg_latency,
            self.session_metrics["refusals"],
            self.session_metrics["escalations"],
        )
        return self.session_metrics

    def _write_jsonl(self, record: dict) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except IOError as e:
            self._logger.error("Failed to write log: %s", e)


def get_logger(agent_name: str = "novatrust_agent") -> AgentLogger:
    """Factory function to create a new AgentLogger for a session."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    return AgentLogger(agent_name, log_level)


# ─────────────────────────────────────────────
# PII Masking Tests (run directly to verify)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_strings = [
        "My account number is 1234567890 and email is user@example.com",
        "PAN card: ABCDE1234F, Aadhaar: 2345 6789 0123",
        "Credit card 4111 1111 1111 1111 expires soon",
        "Call me on 9876543210 or at +91-9876543210",
        "I have ₹50,000 in account 987654321012",
        "Normal banking query about FD rates",  # no PII — should pass through
    ]

    print("PII Masking Test Results:")
    print("─" * 60)
    for s in test_strings:
        masked = mask_pii(s)
        print(f"ORIGINAL: {s}")
        print(f"MASKED  : {masked}")
        print()
