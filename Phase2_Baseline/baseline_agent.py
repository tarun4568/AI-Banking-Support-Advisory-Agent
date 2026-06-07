"""
Phase 2: Baseline Agent — Rules-Based / Keyword-Matching
═══════════════════════════════════════════════════════════
NovaTrust AI Banking Support Agent — Baseline Version

This agent uses simple keyword matching and template responses.
No LLM, no retrieval, no context memory.

PURPOSE:
  Demonstrate how a naive rule-based system behaves and highlight its
  limitations compared to later LLM + RAG-enabled versions.

KNOWN LIMITATIONS (deliberately left in for Phase 2 evidence):
  1. Brittle matching: "show me my amount" won't match "balance" template.
  2. No context: follow-up questions ("What about savings?") fail.
  3. Over-broad matching: "What's the fraud process?" incorrectly hits "transfer".
  4. No ranked responses when multiple keywords match.
  5. Cannot handle compound questions ("FD rate and minimum amount?").

Run:
    python baseline_agent.py
"""

import re
import logging
import time
import json
from datetime import datetime

# ─────────────────────────────────────────────
# Logging setup (PII-safe from the start)
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("phase2_baseline")


# ─────────────────────────────────────────────
# Intent → Keyword mapping
# ─────────────────────────────────────────────
INTENT_KEYWORDS: dict[str, list[str]] = {
    "balance_inquiry": [
        "balance", "how much money", "account amount", "funds available",
        "what is in my account",
    ],
    "fd_rates": [
        "fixed deposit", "fd rate", "fd interest", "fd maturity",
        "deposit rate", "tax saver fd",
    ],
    "home_loan": [
        "home loan", "house loan", "mortgage", "property loan",
        "home loan rate", "housing loan",
    ],
    "personal_loan": [
        "personal loan", "pl rate", "loan for personal", "quick loan",
    ],
    "credit_card": [
        "credit card", "card limit", "cc limit", "credit limit",
        "card charges", "card annual fee",
    ],
    "transfer_request": [
        "transfer", "send money", "wire", "neft", "rtgs", "imps",
        "pay money", "move funds",
    ],
    "account_opening": [
        "open account", "new account", "savings account", "current account",
        "how to open",
    ],
    "netbanking_issue": [
        "net banking", "netbanking", "internet banking", "login problem",
        "password reset", "forgot password",
    ],
    "card_block": [
        "block card", "lost card", "stolen card", "debit card lost",
        "hotlist card",
    ],
    "complaint": [
        "complaint", "problem", "issue", "wrong charge", "incorrect debit",
        "fraud", "dispute",
    ],
    "branch_hours": [
        "branch hours", "branch timing", "when is branch open", "working hours",
        "branch open",
    ],
    "helpline": [
        "customer care", "helpline", "contact number", "phone number",
        "call center", "support number",
    ],
    "loan_approval": [
        "approve loan", "loan approval", "will i get loan", "loan sanctioned",
        "loan eligible",
    ],
    "legal_tax": [
        "legal", "lawyer", "tax advice", "income tax", "file itr",
        "tax saving", "section 80c",
    ],
}

# ─────────────────────────────────────────────
# Template responses per intent
# ─────────────────────────────────────────────
RESPONSES: dict[str, str] = {
    "balance_inquiry": (
        "I'm unable to access your account balance directly. "
        "Please check your balance via:\n"
        "• NovaTrust Mobile App → Dashboard\n"
        "• NetBanking at www.novatrust.in\n"
        "• Missed call: 1800-NOVA-BAL from your registered mobile\n"
        "• Any NovaTrust ATM → Balance Enquiry (free)"
    ),
    "fd_rates": (
        "NovaTrust Fixed Deposit interest rates (April 2026):\n"
        "• 1 year to < 2 years: 7.00% (General) | 7.50% (Senior)\n"
        "• 2 years to < 3 years: 7.25% (General) | 7.75% (Senior)\n"
        "• 3 to 5 years: 7.00% (General) | 7.50% (Senior)\n"
        "• 6 months to < 1 year: 6.00% (General) | 6.50% (Senior)\n"
        "For a full rate chart, visit: www.novatrust.in/fd-rates"
    ),
    "home_loan": (
        "NovaTrust Home Loan details:\n"
        "• Interest Rate: 8.40%–9.50% p.a. (floating)\n"
        "• Loan Amount: ₹5 lakh to ₹5 crore\n"
        "• Tenure: Up to 30 years\n"
        "• Processing Fee: 0.5% of loan amount + GST\n"
        "For eligibility and application, visit your nearest branch "
        "or www.novatrust.in/home-loan."
    ),
    "personal_loan": (
        "NovaTrust Personal Loan:\n"
        "• Interest Rate: 10.99%–18.00% p.a.\n"
        "• Loan Amount: ₹50,000 to ₹25 lakh\n"
        "• Tenure: 12–60 months\n"
        "Apply online or at any branch. For pre-approved offers, "
        "check NovaTrust Mobile App."
    ),
    "credit_card": (
        "NovaTrust Credit Cards:\n"
        "• RuPay Classic: ₹499 annual fee; 1 reward point/₹100\n"
        "• Visa Gold: ₹999 annual fee; 3x points on dining/travel\n"
        "• Platinum Rewards: ₹2,499 annual fee; 5x points all categories\n"
        "Apply at any branch or www.novatrust.in/credit-cards."
    ),
    "transfer_request": (
        "⚠️ I'm not able to process or initiate any fund transfers.\n\n"
        "For transfers, please use:\n"
        "• NovaTrust Mobile App (NEFT/RTGS/IMPS/UPI)\n"
        "• NetBanking at www.novatrust.in\n"
        "• Visit any NovaTrust branch\n"
        "This is a non-transactional support assistant."
    ),
    "account_opening": (
        "To open a NovaTrust Savings Account:\n"
        "• Online: www.novatrust.in → Open Account (eCKYC via Aadhaar)\n"
        "• Branch: Visit with Aadhaar, PAN, and a passport-size photo\n"
        "Minimum Balance: ₹5,000 (urban) | ₹2,000 (semi-urban) | ₹500 (rural)\n"
        "Classic Savings earns 3.5% p.a. interest."
    ),
    "netbanking_issue": (
        "For NetBanking password reset:\n"
        "• Online: www.novatrust.in → Forgot Password → OTP on registered mobile\n"
        "• Branch: Visit with photo ID for instant reset\n"
        "Locked accounts auto-unlock after 24 hours or immediately at a branch.\n"
        "Helpline: 1800-NOVA-123"
    ),
    "card_block": (
        "To immediately BLOCK a lost/stolen card:\n"
        "• NovaTrust App → Cards → Block Card (fastest)\n"
        "• Call 1800-NOVA-123 (24/7)\n"
        "• NetBanking → Card Management → Block\n"
        "Zero liability on fraud reported within 3 business days."
    ),
    "complaint": (
        "To register a complaint:\n"
        "• Mobile App → Raise Complaint\n"
        "• Call 1800-NOVA-123 (24/7)\n"
        "• Email: customercare@novatrust.in\n"
        "• Branch: Speak to the branch manager\n"
        "You'll receive a Complaint Reference Number (CRN) for tracking.\n"
        "Resolution time: 5 business days for most issues."
    ),
    "branch_hours": (
        "NovaTrust Branch Hours:\n"
        "• Mon–Fri: 9:30 AM – 3:30 PM (counter) | 5:00 PM (extended)\n"
        "• Saturday: 9:30 AM – 12:30 PM (1st & 3rd Sat; alternate Sat closed)\n"
        "• Sunday & public holidays: Closed\n"
        "Digital banking is available 24/7."
    ),
    "helpline": (
        "NovaTrust Customer Service:\n"
        "• 24/7 Toll-Free: 1800-NOVA-123\n"
        "• Email: customercare@novatrust.in\n"
        "• Fraud Hotline: 1800-NOVA-FRAUD (24/7)\n"
        "• WhatsApp: +91 98765 43210 (9 AM–9 PM)\n"
        "• International: +91-22-6682-1234"
    ),
    "loan_approval": (
        "⚠️ I'm not able to approve, reject, or pre-assess loan applications.\n\n"
        "For loan eligibility:\n"
        "• Use the online calculator: www.novatrust.in/loan-eligibility\n"
        "• Visit your nearest NovaTrust branch\n"
        "• Call 1800-NOVA-123 to speak with a loan officer\n"
        "Eligibility depends on income, credit score (CIBIL), and existing liabilities."
    ),
    "legal_tax": (
        "⚠️ I'm not able to provide legal or tax advice.\n\n"
        "For tax-related queries:\n"
        "• Please consult a Chartered Accountant (CA) or tax advisor\n"
        "• I can share factual product information (e.g., Section 80C benefit "
        "on Tax-Saver FD) — just ask and I'll share the product details."
    ),
    "unknown": (
        "I'm sorry, I didn't quite understand your query. "
        "Could you please rephrase? I can help with:\n"
        "• Account information and features\n"
        "• Loan products and rates\n"
        "• Fixed/Recurring Deposits\n"
        "• Card services\n"
        "• Complaints and escalations\n\n"
        "Or call our helpline: 1800-NOVA-123 (24/7)."
    ),
}


# ─────────────────────────────────────────────
# Intent detection (keyword matching)
# ─────────────────────────────────────────────
def detect_intent(user_input: str) -> str:
    """Return the best-matching intent string based on keyword overlap."""
    text = user_input.lower()
    scores: dict[str, int] = {}

    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[intent] = score

    if not scores:
        return "unknown"

    # Return intent with highest keyword hit count
    return max(scores, key=lambda k: scores[k])


def respond(user_input: str) -> tuple[str, str]:
    """Detect intent and return (response_text, detected_intent)."""
    intent = detect_intent(user_input)
    response = RESPONSES.get(intent, RESPONSES["unknown"])
    return response, intent


# ─────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────
def log_interaction(turn: int, user_input: str, intent: str, response: str, latency_ms: float) -> None:
    """Log interaction without PII (no account numbers, phone, email in user input logged)."""
    # Simple PII mask: redact potential account numbers (10 digit sequences)
    safe_input = re.sub(r"\b\d{9,12}\b", "[REDACTED]", user_input)
    logger.info(
        "TURN=%d | INTENT=%s | LATENCY=%.1fms | INPUT_SNIPPET=%s",
        turn, intent, latency_ms, safe_input[:80],
    )


# ─────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────
def run_demo_mode() -> None:
    """Run predefined prompts that expose limitations of the baseline agent."""
    demo_inputs = [
        # ── Normal cases (should work) ──
        ("Q1 – Product info",         "What are the FD interest rates?"),
        ("Q2 – Process query",        "How do I block a lost debit card?"),
        ("Q3 – Safety refusal",       "Transfer 5000 to account 9876543210"),
        # ── Limitation cases ──
        ("Q4 – Limitation 1: wording variation",
         "Can you show me how much money I have?"),   # won't match "balance"
        ("Q5 – Limitation 2: compound question",
         "What are FD rates and what is the minimum amount I can invest?"),
        ("Q6 – Limitation 3: no context (follow-up fails)",
         "What about the senior citizen rate?"),      # no memory of Q1 context
        ("Q7 – Limitation 4: keyword collision",
         "What is the fraud reporting process for a failed transfer?"),
    ]

    print("\n" + "═" * 60)
    print("  NovaTrust AI Banking — Phase 2: Baseline Agent  ")
    print("  (Rules-based; No LLM, No Memory)")
    print("═" * 60)

    interaction_log = []

    for label, user_input in demo_inputs:
        print(f"\n[{label}]")
        print(f"USER: {user_input}")
        t0 = time.time()
        response, intent = respond(user_input)
        latency_ms = (time.time() - t0) * 1000
        log_interaction(len(interaction_log) + 1, user_input, intent, response, latency_ms)
        print(f"INTENT DETECTED: {intent}")
        print(f"AGENT: {response}")
        print(f"LATENCY: {latency_ms:.1f}ms")
        interaction_log.append({
            "label": label,
            "input": user_input,
            "intent": intent,
            "response": response,
            "latency_ms": round(latency_ms, 1),
        })

    # Save log
    log_path = "Phase2_Baseline/baseline_interaction_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "agent_version": "baseline_v1.0",
            "interactions": interaction_log,
        }, f, indent=2)
    print(f"\n[LOG] Interactions saved to {log_path}")

    # Print limitation summary
    print("\n" + "═" * 60)
    print("  BASELINE LIMITATIONS DEMONSTRATED")
    print("═" * 60)
    limitations = [
        "1. Brittle keyword matching: 'show me how much money I have' missed "
        "the 'balance' intent because none of the exact keywords appeared.",
        "2. No semantic understanding: Compound questions only partially matched "
        "— the second part (minimum amount) was silently dropped.",
        "3. Zero conversation memory: After answering Q1 about FD rates, "
        "the follow-up 'What about the senior citizen rate?' had no context "
        "and returned 'unknown' intent.",
        "4. Keyword collision: A fraud-reporting question containing the word "
        "'transfer' incorrectly triggered the transfer-refusal response.",
        "5. No confidence scoring: The agent cannot express uncertainty or "
        "distinguish between high/low confidence matches.",
    ]
    for lim in limitations:
        print(f"\n  ⚠ {lim}")
    print("\n→ These limitations justify upgrading to an LLM-based agent in Phase 3.")


def run_interactive_mode() -> None:
    print("\n" + "═" * 60)
    print("  NovaTrust AI Banking — Phase 2: Baseline Agent")
    print("  Type 'quit' to exit | Type 'demo' for demo mode")
    print("═" * 60)
    turn = 0
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "demo":
            run_demo_mode()
            break
        turn += 1
        t0 = time.time()
        response, intent = respond(user_input)
        latency_ms = (time.time() - t0) * 1000
        log_interaction(turn, user_input, intent, response, latency_ms)
        print(f"\nAgent [{intent}]: {response}")
        print(f"(Latency: {latency_ms:.1f}ms)")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_demo_mode()
    else:
        run_interactive_mode()
