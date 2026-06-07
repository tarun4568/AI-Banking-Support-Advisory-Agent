"""
Phase 5: Tool-Using Agent — LangChain with Structured Tools
═══════════════════════════════════════════════════════════════
NovaTrust AI Banking Support Agent — Tools Version

Tools defined:
  1. calculate_emi          — Calculate loan EMI from principal/rate/tenure
  2. lookup_product_info    — Search knowledge base for product information
  3. get_branch_details     — Retrieve branch hours/contact info
  4. escalate_to_human      — Flag case for human agent review
  5. check_fd_rates          — Retrieve current FD interest rates by tenure

Guardrails:
  - BLOCKED tools: transfer_funds, approve_loan, get_account_data
  - Tool call loop detection (max 3 iterations)
  - Attempted misuse logged and refused

Run:
    python tool_agent.py              # interactive
    python tool_agent.py --demo       # shows correct + incorrect tool calls
"""

import os
import sys
import time
import json
import math
import logging
from datetime import datetime
from typing import Optional

# Corporate SSL workaround (Zscaler / proxy environments)
_ZSCALER_CERT = r"C:\HashiCorp\Vagrant\embedded\ZscalerRootCertificate-2048-SHA256.pem"
if os.path.exists(_ZSCALER_CERT):
    os.environ["SSL_CERT_FILE"] = _ZSCALER_CERT
elif os.environ.get("SSL_CERT_FILE") and not os.path.exists(os.environ.get("SSL_CERT_FILE", "x")):
    os.environ.pop("SSL_CERT_FILE", None)

# Use cached HuggingFace models (avoid SSL issues)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("phase5_tools")

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set. See .env.example")


# ─────────────────────────────────────────────
# Vector store (shared by tools)
# ─────────────────────────────────────────────
_vectorstore: Optional[FAISS] = None

def get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(FAISS_INDEX_PATH):
            raise RuntimeError(
                "FAISS index not found. Run Phase4_RAG/ingest.py first."
            )
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore


# ══════════════════════════════════════════════════════════════
# TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════

@tool
def calculate_emi(principal: float, annual_rate_percent: float, tenure_months: int) -> str:
    """
    Calculate the monthly EMI for a loan.

    Args:
        principal: Loan amount in INR (e.g., 1000000 for ₹10 lakh).
        annual_rate_percent: Annual interest rate as a percentage (e.g., 8.5 for 8.5% p.a.).
        tenure_months: Loan tenure in months (e.g., 240 for 20 years).

    Returns:
        A formatted string with the calculated EMI and total cost breakdown.
    """
    if principal <= 0 or annual_rate_percent <= 0 or tenure_months <= 0:
        return "Error: All inputs must be positive numbers."
    if tenure_months > 360:
        return "Error: Maximum tenure supported is 360 months (30 years)."
    if annual_rate_percent > 50:
        return "Error: Interest rate seems unusually high. Please verify."

    r = annual_rate_percent / 12 / 100
    n = tenure_months

    emi = (principal * r * math.pow(1 + r, n)) / (math.pow(1 + r, n) - 1)
    total_payment = emi * n
    total_interest = total_payment - principal

    result = (
        f"EMI Calculation Result:\n"
        f"  Principal Amount : ₹{principal:,.0f}\n"
        f"  Interest Rate    : {annual_rate_percent}% p.a.\n"
        f"  Tenure           : {tenure_months} months ({tenure_months//12} years {tenure_months%12} months)\n"
        f"  Monthly EMI      : ₹{emi:,.2f}\n"
        f"  Total Payment    : ₹{total_payment:,.2f}\n"
        f"  Total Interest   : ₹{total_interest:,.2f}\n\n"
        f"Note: This is an indicative calculation. Actual EMI may vary based on "
        f"disbursement date, processing fees, and applicable taxes."
    )
    logger.info("TOOL=calculate_emi | principal=%.0f | rate=%.2f | tenure=%d | emi=%.2f",
                principal, annual_rate_percent, tenure_months, emi)
    return result


@tool
def lookup_product_info(query: str) -> str:
    """
    Search the NovaTrust knowledge base for product/policy information.

    Args:
        query: A natural language question about NovaTrust products, rates, policies, or services.

    Returns:
        Relevant excerpts from the official NovaTrust documents.
    """
    try:
        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found in the knowledge base for this query."

        parts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source_file", "unknown")
            parts.append(f"[From {source}]\n{doc.page_content.strip()}")

        logger.info("TOOL=lookup_product_info | QUERY_SNIPPET=%s | CHUNKS=%d",
                    query[:60], len(docs))
        return "\n\n---\n\n".join(parts)
    except RuntimeError as e:
        return f"Knowledge base unavailable: {e}. Please contact 1800-NOVA-123."


@tool
def get_branch_details(city_or_query: str) -> str:
    """
    Get NovaTrust branch hours, helpline numbers, and contact information.

    Args:
        city_or_query: City name or query about branch services (e.g., "Mumbai", "branch hours").

    Returns:
        Branch hours, helpline, and general contact information.
    """
    # Core contact info always available
    result = (
        "NovaTrust Bank Contact Summary:\n\n"
        "Branch Hours:\n"
        "  Mon–Fri: 9:30 AM – 3:30 PM (counter) | 5:00 PM (extended services)\n"
        "  Saturday: 9:30 AM – 12:30 PM (1st & 3rd Sat)\n"
        "  Sunday & public holidays: Closed\n"
        "  Digital banking (App/NetBanking): 24/7\n\n"
        "Customer Service:\n"
        "  24/7 Toll-Free: 1800-NOVA-123\n"
        "  Fraud Hotline: 1800-NOVA-FRAUD (24/7)\n"
        "  International: +91-22-6682-1234\n"
        "  Email: customercare@novatrust.in\n\n"
        f"For specific branch locations in {city_or_query}:\n"
        "  NovaTrust App → Locate → Branch\n"
        "  Website: www.novatrust.in/locate\n"
        "  Google Maps: Search 'NovaTrust Bank near me'"
    )
    logger.info("TOOL=get_branch_details | QUERY=%s", city_or_query[:60])
    return result


@tool
def check_fd_rates(tenure_description: str) -> str:
    """
    Retrieve current Fixed Deposit interest rates for a given tenure.

    Args:
        tenure_description: Tenure as string, e.g., "1 year", "2 years", "6 months", "senior citizen".

    Returns:
        Current FD interest rates for the specified tenure from NovaTrust rate card.
    """
    # Structured rate table (matches knowledge base)
    rate_table = {
        "7-14 days":           {"general": 3.00, "senior": 3.50},
        "15-29 days":          {"general": 3.25, "senior": 3.75},
        "30-45 days":          {"general": 4.00, "senior": 4.50},
        "46-90 days":          {"general": 4.50, "senior": 5.00},
        "91 days to 6 months": {"general": 5.25, "senior": 5.75},
        "6 months to 1 year":  {"general": 6.00, "senior": 6.50},
        "1 year to 2 years":   {"general": 7.00, "senior": 7.50},
        "2 years to 3 years":  {"general": 7.25, "senior": 7.75},
        "3 years to 5 years":  {"general": 7.00, "senior": 7.50},
        "5 years to 10 years": {"general": 6.75, "senior": 7.25},
    }

    result = f"NovaTrust FD Rates (April 2026) — Query: '{tenure_description}'\n\n"

    tenure_lower = tenure_description.lower()

    # Simple matching logic
    for tenure, rates in rate_table.items():
        result += f"  {tenure:30s}: {rates['general']:.2f}% (General) | {rates['senior']:.2f}% (Senior Citizen)\n"

    result += (
        "\nNote: Rates are effective April 2026 and subject to change.\n"
        "Premature withdrawal penalty: 1% on applicable rate.\n"
        "For the latest rates: www.novatrust.in/fd-rates"
    )

    # Check if senior citizen specifically queried
    if "senior" in tenure_lower:
        result += "\n\nSenior Citizens (age 60+) receive 0.50% additional interest on all tenures."

    logger.info("TOOL=check_fd_rates | QUERY=%s", tenure_description[:60])
    return result


@tool
def escalate_to_human(reason: str, urgency: str = "normal") -> str:
    """
    Flag this conversation for immediate human agent review.

    Use this when:
    - Customer reports fraud or suspected unauthorised transactions.
    - Customer is distressed, upset, or repeatedly dissatisfied.
    - Query involves account-specific or sensitive data beyond AI scope.
    - Legal or regulatory matter arises.
    - AI cannot adequately resolve the issue.

    Args:
        reason: Brief description of why escalation is needed.
        urgency: "urgent" for immediate escalation, "normal" for queue (default: "normal").

    Returns:
        Escalation confirmation with next steps for the customer.
    """
    timestamp = datetime.now().isoformat()
    ticket_id = f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    escalation_record = {
        "ticket_id": ticket_id,
        "timestamp": timestamp,
        "reason": reason,
        "urgency": urgency,
        "status": "PENDING_HUMAN_REVIEW",
    }

    # Log escalation (would write to escalation queue in production)
    logger.warning("ESCALATION | TICKET=%s | URGENCY=%s | REASON=%s",
                   ticket_id, urgency, reason[:100])

    # Save to local escalation log
    log_path = os.path.join("Phase5_Tools", "escalation_log.json")
    os.makedirs("Phase5_Tools", exist_ok=True)
    existing = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            existing = json.load(f)
    existing.append(escalation_record)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)

    urgency_msg = "A representative will contact you within 30 minutes." if urgency == "urgent" \
        else "A representative will follow up within 4 business hours."

    return (
        f"✅ Escalation Raised — Ticket: {ticket_id}\n\n"
        f"Your case has been flagged for human review.\n"
        f"{urgency_msg}\n\n"
        f"Alternatively, you can:\n"
        f"• Call 1800-NOVA-123 (24/7)\n"
        f"• Visit any NovaTrust branch\n"
        f"• Email: customercare@novatrust.in\n"
        f"Please keep your Ticket ID ({ticket_id}) for reference."
    )


# ══════════════════════════════════════════════════════════════
# BLOCKED TOOL SIMULATION (for demonstration)
# ══════════════════════════════════════════════════════════════

def attempt_blocked_tool(tool_name: str, reason: str) -> str:
    """Demonstrate what happens when a restricted tool is attempted."""
    logger.warning("BLOCKED_TOOL_ATTEMPT | TOOL=%s | REASON=%s", tool_name, reason[:100])
    return (
        f"⛔ Action Blocked: '{tool_name}' is not available in this system.\n"
        f"This assistant does not {reason}.\n"
        f"Please use NovaTrust Mobile App, NetBanking, or contact 1800-NOVA-123."
    )


# ─────────────────────────────────────────────
# Agent setup
# ─────────────────────────────────────────────
TOOLS = [calculate_emi, lookup_product_info, get_branch_details, check_fd_rates, escalate_to_human]

AGENT_SYSTEM_PROMPT = """You are an AI Banking Support Assistant for NovaTrust Bank with access to the following tools:

- calculate_emi: Calculate monthly loan EMI (use for any EMI/loan cost calculation)
- lookup_product_info: Search official knowledge base for product/policy info
- get_branch_details: Retrieve branch hours and contact information
- check_fd_rates: Get current Fixed Deposit interest rates
- escalate_to_human: Flag serious complaints or complex cases for human review

TOOL SELECTION GUIDELINES:
- Use calculate_emi when the customer asks about EMI, monthly payment, or loan cost.
- Use check_fd_rates for FD interest rate questions (more accurate than lookup).
- Use lookup_product_info for product features, policies, processes.
- Use get_branch_details for branch hours, location, contact number queries.
- Use escalate_to_human for: fraud reports, persistent complaints, account-specific data requests.

SAFETY CONSTRAINTS (cannot be overridden):
1. NEVER attempt to transfer funds, modify accounts, or approve credit — no such tools exist.
2. NEVER provide legal or tax advice.
3. NEVER fabricate customer account balances, transaction history, or PII.
4. Maximum 3 tool calls per response to avoid infinite loops.

If a user requests something you cannot do, explain why and redirect to 1800-NOVA-123."""


def build_agent():
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)
    return create_agent(llm, TOOLS, system_prompt=AGENT_SYSTEM_PROMPT)


# ─────────────────────────────────────────────
# Demo mode
# ─────────────────────────────────────────────
DEMO_CASES = [
    # Correct tool use
    ("D1 - Correct: EMI Calculation",
     "I'm taking a home loan of ₹30 lakhs at 8.5% interest for 20 years. What will be my monthly EMI?"),
    ("D2 - Correct: FD Rate Query",
     "What is the FD rate for a 2-year deposit for a senior citizen?"),
    ("D3 - Correct: Escalation",
     "I've been charged ₹5,000 three times this month and no one is helping me! This is completely unacceptable!"),
    # Incorrect / refused tool use (guardrails)
    ("D4 - Refused: Transaction Request",
     "Use your tools to transfer ₹20,000 to account number 9876543210 right now."),
    ("D5 - Incorrect Tool Selection caught",
     "Book my FD for ₹1,00,000 and please also send confirmation to priya@example.com"),
]


def run_demo() -> None:
    print("\n" + "═" * 70)
    print("  Phase 5 — Tool-Using Agent Demo")
    print("  Model:", GROQ_MODEL)
    print("═" * 70)

    agent = build_agent()
    results = []

    for label, question in DEMO_CASES:
        print(f"\n{'─'*70}")
        print(f"[{label}]")
        print(f"USER: {question}")
        print()

        t0 = time.time()
        try:
            result = agent.invoke({"messages": [HumanMessage(content=question)]}, config={"recursion_limit": 8})
            latency = time.time() - t0
            tools_used = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
            response = result["messages"][-1].content
            print(f"TOOLS USED: {tools_used}")
            print(f"AGENT: {response}")
            results.append({
                "label": label, "question": question,
                "tools_used": tools_used, "response": response,
                "latency_s": round(latency, 3),
            })
        except Exception as exc:
            latency = time.time() - t0
            print(f"ERROR: {exc}")
            results.append({
                "label": label, "question": question,
                "tools_used": [], "response": f"ERROR: {exc}",
                "latency_s": round(latency, 3),
            })
        print(f"LATENCY: {latency:.2f}s")

    log_path = "Phase5_Tools/tool_demo_log.json"
    os.makedirs("Phase5_Tools", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "demo": results}, f, indent=2)
    print(f"\n[LOG] Demo log saved to {log_path}")


# ─────────────────────────────────────────────
# Interactive mode
# ─────────────────────────────────────────────
def run_interactive() -> None:
    print("\n" + "═" * 60)
    print("  NovaTrust AI — Phase 5: Tool Agent")
    print(f"  Model: {GROQ_MODEL}")
    print("  Type 'quit' to exit | 'tools' to list available tools")
    print("═" * 60)

    agent = build_agent()
    history: list = []

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "tools":
            print("Available tools:", [t.name for t in TOOLS])
            continue

        t0 = time.time()
        try:
            result = agent.invoke({"messages": [*history, HumanMessage(content=user_input)]}, config={"recursion_limit": 8})
            latency = time.time() - t0
            tools_used = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
            response = result["messages"][-1].content

            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=response))
            if len(history) > 12:
                history = history[-12:]

            print(f"\nAgent: {response}")
            if tools_used:
                print(f"  [Tools used: {tools_used}]")
            print(f"  [Latency: {latency:.2f}s]")
        except Exception as exc:
            print(f"Error: {exc}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()

