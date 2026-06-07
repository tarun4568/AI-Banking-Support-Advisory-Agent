"""
Phase 8: Full Deployment — NovaTrust AI Banking Agent (Streamlit)
══════════════════════════════════════════════════════════════════
Complete integrated agent combining ALL phases:
  • Groq LLM (Phase 3)
  • RAG from FAISS knowledge base (Phase 4)
  • Tools: EMI, FD rates, escalation, product lookup (Phase 5)
  • Short-term memory (Phase 6)
  • Adaptive behaviour via feedback store (Phase 7)
  • PII-safe structured logging, latency tracing (Phase 8)

Run:
    streamlit run app.py

Prerequisites:
    1. Run: pip install -r requirements.txt
    2. Run: python Phase4_RAG/ingest.py   (builds FAISS index)
    3. Ensure .env has GROQ_API_KEY set
"""

import os
import sys
import re
import time
import math
import json
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

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Add project root to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Phase8_Deployment.logger import get_logger, mask_pii

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
BANK_NAME        = os.getenv("BANK_NAME", "NovaTrust Bank")
BANK_HELPLINE    = os.getenv("BANK_HELPLINE", "1800-NOVA-123")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
FEEDBACK_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "Phase7_Adaptive", "feedback_store.json")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# ─────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title=f"{BANK_NAME} — AI Support",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
# CACHED RESOURCES (load once per session)
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading knowledge base…")
def load_vectorstore():
    """Load FAISS vector store (cached across reruns)."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.warning(f"Could not load knowledge base: {e}. Responses will use LLM knowledge only.")
        return None


@st.cache_resource(show_spinner="Loading feedback store…")
def load_feedback_store():
    """Load adaptive feedback store."""
    DEFAULT = {
        "schema_version": "1.0", "total_interactions": 0,
        "response_length_score": 0.5, "topics": {}, "negative_feedback_streak": 0,
        "behaviour_adjustments": {
            "max_response_tokens": 512, "add_caveats": False,
            "proactive_escalation_threshold": 0.7,
        }, "feedback_history": [],
    }
    if os.path.exists(FEEDBACK_STORE_PATH):
        try:
            with open(FEEDBACK_STORE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT


def save_feedback_store(store: dict) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_STORE_PATH), exist_ok=True)
    with open(FEEDBACK_STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)


# ══════════════════════════════════════════════════════════════
# TOOL DEFINITIONS (integrated from Phase 5)
# ══════════════════════════════════════════════════════════════

@tool
def calculate_emi(principal: float, annual_rate_percent: float, tenure_months: int) -> str:
    """Calculate the monthly EMI for a loan given principal, annual rate %, and tenure in months."""
    if principal <= 0 or annual_rate_percent <= 0 or tenure_months <= 0:
        return "Error: All inputs must be positive."
    r = annual_rate_percent / 12 / 100
    emi = (principal * r * math.pow(1 + r, tenure_months)) / (math.pow(1 + r, tenure_months) - 1)
    total = emi * tenure_months
    return (
        f"EMI: ₹{emi:,.2f}/month | Total: ₹{total:,.2f} | "
        f"Interest: ₹{total-principal:,.2f} "
        f"(₹{principal:,.0f} at {annual_rate_percent}% for {tenure_months} months)"
    )


@tool
def check_fd_rates(tenure_description: str) -> str:
    """Get NovaTrust Fixed Deposit interest rates for a given tenure."""
    rates = {
        "7-14 days": (3.00, 3.50), "15-29 days": (3.25, 3.75), "30-45 days": (4.00, 4.50),
        "46-90 days": (4.50, 5.00), "91 days-6 months": (5.25, 5.75),
        "6 months-1 year": (6.00, 6.50), "1 year-2 years": (7.00, 7.50),
        "2 years-3 years": (7.25, 7.75), "3 years-5 years": (7.00, 7.50),
        "5 years-10 years": (6.75, 7.25),
    }
    result = "NovaTrust FD Rates (April 2026):\n"
    for tenure, (gen, sr) in rates.items():
        result += f"  {tenure}: {gen:.2f}% (General) | {sr:.2f}% (Senior)\n"
    result += "\nPremature withdrawal penalty: 1%. Verify at www.novatrust.in/fd-rates"
    return result


@tool
def lookup_product_info(query: str) -> str:
    """Search NovaTrust knowledge base for product, policy, FAQ, or service information."""
    vs = load_vectorstore()
    if vs is None:
        return "Knowledge base not available. Please contact 1800-NOVA-123."
    docs = vs.similarity_search(query, k=3)
    parts = [f"[{d.metadata.get('source_file','KB')}]\n{d.page_content.strip()}" for d in docs]
    return "\n\n---\n\n".join(parts) if parts else "No relevant information found."


@tool
def escalate_to_human(reason: str, urgency: str = "normal") -> str:
    """
    Escalate this case to a human agent. Use for: fraud reports, distressed customers,
    account-specific issues, unresolved complaints, legal/complex matters.
    urgency: 'urgent' or 'normal'
    """
    ticket_id = f"ESC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    eta = "30 minutes" if urgency == "urgent" else "4 business hours"
    return (
        f"✅ Escalated — Ticket: {ticket_id}\n"
        f"A NovaTrust representative will contact you within {eta}.\n"
        f"Reason: {reason[:100]}\n"
        f"Alternatively: Call 1800-NOVA-123 | Email customercare@novatrust.in"
    )


TOOLS = [calculate_emi, check_fd_rates, lookup_product_info, escalate_to_human]

# ══════════════════════════════════════════════════════════════
# AGENT BUILDER
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an AI Banking Support Assistant for {bank_name}.

ADAPTIVE BEHAVIOUR:
{adaptive_note}

TOOLS AVAILABLE:
- calculate_emi: Use for any EMI, monthly payment, or loan cost calculation.
- check_fd_rates: Use for FD interest rate questions.
- lookup_product_info: Use for product features, policies, processes, FAQs.
- escalate_to_human: Use for fraud, distress, persistent complaints, or unresolvable issues.

STRICT SAFETY RULES:
1. NEVER process fund transfers, wire transfers, payments, or account modifications.
2. NEVER approve, reject, or assess loan applications.
3. NEVER provide legal, tax, or comparative investment advice.
4. NEVER fabricate customer account numbers, balances, or transaction history.
5. After 2 failed attempts to resolve, proactively offer escalation.

WHEN REFUSING: Be brief, explain why, redirect to {helpline} or www.novatrust.in.

{caveat_instruction}"""

CAVEAT = (
    "Always add: 'Please verify at www.novatrust.in or call {helpline} as rates may have changed.'"
)


def build_agent(feedback_store: dict):
    adj = feedback_store.get("behaviour_adjustments", {})
    max_tokens = adj.get("max_response_tokens", 512)
    add_caveats = adj.get("add_caveats", False)

    adaptive_note = (
        f"Response length target: {'brief' if max_tokens <= 350 else 'moderate' if max_tokens <= 500 else 'detailed'}. "
        f"Max tokens: {max_tokens}."
    )
    caveat_instruction = CAVEAT.format(helpline=BANK_HELPLINE) if add_caveats else ""

    system_content = SYSTEM_PROMPT.format(
        bank_name=BANK_NAME,
        helpline=BANK_HELPLINE,
        adaptive_note=adaptive_note,
        caveat_instruction=caveat_instruction,
    )

    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0, max_tokens=max_tokens)
    return create_agent(llm, TOOLS, system_prompt=system_content)


# ══════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════

def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_logger" not in st.session_state:
        st.session_state.agent_logger = get_logger("novatrust_streamlit")
    if "feedback_store" not in st.session_state:
        st.session_state.feedback_store = load_feedback_store()
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now()
    if "latency_history" not in st.session_state:
        st.session_state.latency_history = []
    if "tools_used_session" not in st.session_state:
        st.session_state.tools_used_session = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = build_agent(st.session_state.feedback_store)


# ══════════════════════════════════════════════════════════════
# HELPER: detect refusal / escalation in response
# ══════════════════════════════════════════════════════════════

REFUSAL_SIGNALS = [
    "not able to process", "cannot initiate", "cannot transfer",
    "not in a position to", "i'm unable to", "i cannot approve",
    "not provide legal", "not provide tax",
]
ESCALATION_SIGNALS = ["ESC-", "escalat", "human agent", "representative will contact"]

def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in REFUSAL_SIGNALS)

def is_escalation(text: str) -> bool:
    return any(sig.lower() in text.lower() for sig in ESCALATION_SIGNALS)


# ══════════════════════════════════════════════════════════════
# UPDATE FEEDBACK STORE
# ══════════════════════════════════════════════════════════════

def update_feedback(store: dict, thumbs: Optional[str], rating: Optional[int], category: Optional[str]) -> dict:
    store["total_interactions"] = store.get("total_interactions", 0) + 1
    if category == "too_verbose":
        store["response_length_score"] = max(0.0, store.get("response_length_score", 0.5) - 0.1)
    elif thumbs == "up":
        store["response_length_score"] = min(1.0, store.get("response_length_score", 0.5) + 0.05)

    if thumbs == "down" or (rating and rating <= 2):
        store["negative_feedback_streak"] = store.get("negative_feedback_streak", 0) + 1
    else:
        store["negative_feedback_streak"] = 0

    # Recalculate behaviour
    score = store.get("response_length_score", 0.5)
    adj = store.setdefault("behaviour_adjustments", {})
    adj["max_response_tokens"] = 300 if score < 0.3 else 400 if score < 0.45 else 650 if score > 0.75 else 512
    streak = store.get("negative_feedback_streak", 0)
    adj["proactive_escalation_threshold"] = 0.4 if streak >= 3 else 0.7

    save_feedback_store(store)
    return store


# ══════════════════════════════════════════════════════════════
# MAIN STREAMLIT UI
# ══════════════════════════════════════════════════════════════

def main() -> None:
    if not GROQ_API_KEY:
        st.error("⚠️ GROQ_API_KEY not set. Please configure .env file and restart.")
        st.stop()

    init_session()

    # ── Header ──────────────────────────────────
    st.title(f"🏦 {BANK_NAME} — AI Support Assistant")
    st.caption(
        "Non-transactional AI assistant. Cannot process transfers, approve loans, or give legal advice. "
        f"For urgent help: **{BANK_HELPLINE}** (24/7)"
    )

    # ── Sidebar ─────────────────────────────────
    with st.sidebar:
        st.header("📊 Session Dashboard")

        session_dur = (datetime.now() - st.session_state.session_start).seconds
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", len(st.session_state.messages))
        with col2:
            st.metric("Session (min)", f"{session_dur // 60}")

        if st.session_state.latency_history:
            avg_lat = sum(st.session_state.latency_history) / len(st.session_state.latency_history)
            p95 = sorted(st.session_state.latency_history)[int(len(st.session_state.latency_history) * 0.95)]
            st.metric("Avg Latency", f"{avg_lat:.2f}s")
            st.metric("P95 Latency", f"{p95:.2f}s")

        st.divider()
        st.subheader("🛠 Tools Used This Session")
        if st.session_state.tools_used_session:
            for tool_event in st.session_state.tools_used_session[-5:]:
                st.text(f"• {tool_event}")
        else:
            st.caption("No tools used yet.")

        st.divider()
        st.subheader("🧠 Adaptive Behaviour")
        adj = st.session_state.feedback_store.get("behaviour_adjustments", {})
        st.text(f"Max Tokens: {adj.get('max_response_tokens', 512)}")
        st.text(f"Caveats: {'On' if adj.get('add_caveats') else 'Off'}")
        st.text(f"Escalation: {adj.get('proactive_escalation_threshold', 0.7)}")

        st.divider()
        st.subheader("⚙️ Controls")
        if st.button("🗑 Clear Conversation"):
            st.session_state.messages = []
            st.session_state.latency_history = []
            st.session_state.tools_used_session = []
            st.rerun()
        if st.button("🔄 Rebuild Agent (after feedback)"):
            st.session_state.agent_executor = build_agent(st.session_state.feedback_store)
            st.success("Agent rebuilt with updated behaviour!")

        st.divider()
        st.subheader("📋 Knowledge Base")
        if os.path.exists(FAISS_INDEX_PATH):
            summary_path = os.path.join(FAISS_INDEX_PATH, "ingestion_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    kb_info = json.load(f)
                st.text(f"Chunks: {kb_info.get('total_chunks', 'N/A')}")
                st.text(f"Docs: {kb_info.get('documents_loaded', 'N/A')}")
                st.text(f"Updated: {kb_info.get('timestamp', 'N/A')[:10]}")
            else:
                st.success("✓ Index loaded")
        else:
            st.warning("⚠️ FAISS index missing.\nRun: python Phase4_RAG/ingest.py")

    # ── Chat display ────────────────────────────
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🏦"):
            st.markdown(msg["content"])

            # Show metadata for assistant messages
            if msg["role"] == "assistant":
                meta = msg.get("meta", {})

                # Expandable: tools used
                if meta.get("tools"):
                    with st.expander(f"🛠 Tools used: {meta['tools']}", expanded=False):
                        for step in meta.get("steps", []):
                            st.code(f"Tool: {step.get('tool')}\nInput: {step.get('input')}\nOutput: {step.get('output', '')[:200]}", language="text")

                # Expandable: retrieved context
                if meta.get("retrieved_context"):
                    with st.expander("📄 Knowledge base context (retrieved)", expanded=False):
                        st.text(meta["retrieved_context"][:600])

                col_a, col_b = st.columns([3, 1])
                with col_b:
                    st.caption(f"⏱ {meta.get('latency', 0):.2f}s")
                    if meta.get("is_refusal"):
                        st.caption("🚫 Refusal")
                    if meta.get("is_escalation"):
                        st.caption("📞 Escalated")

                # Feedback buttons
                fb_key = f"fb_{i}"
                if not msg.get("feedback_given"):
                    cols = st.columns(6)
                    if cols[0].button("👍", key=f"up_{i}"):
                        st.session_state.feedback_store = update_feedback(
                            st.session_state.feedback_store, "up", None, None
                        )
                        st.session_state.messages[i]["feedback_given"] = True
                        st.session_state.agent_logger.log_feedback(i + 1, "up", None, None)
                        st.rerun()
                    if cols[1].button("👎", key=f"dn_{i}"):
                        st.session_state.feedback_store = update_feedback(
                            st.session_state.feedback_store, "down", None, None
                        )
                        st.session_state.messages[i]["feedback_given"] = True
                        st.session_state.agent_logger.log_feedback(i + 1, "down", None, None)
                        st.rerun()
                    if cols[2].button("📝 Too long", key=f"tl_{i}"):
                        st.session_state.feedback_store = update_feedback(
                            st.session_state.feedback_store, "down", None, "too_verbose"
                        )
                        st.session_state.messages[i]["feedback_given"] = True
                        st.session_state.agent_logger.log_feedback(i + 1, "down", None, "too_verbose")
                        st.rerun()

    # ── Chat input ──────────────────────────────
    if user_input := st.chat_input("Ask about accounts, loans, FDs, cards…"):
        # Safety: basic input sanitisation
        user_input = user_input.strip()[:1000]  # limit input length

        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Build chat history for agent
        history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))

        # Agent response
        with st.chat_message("assistant", avatar="🏦"):
            with st.spinner("Thinking…"):
                t0 = time.time()
                try:
                    result = st.session_state.agent_executor.invoke({
                        "messages": [*history, HumanMessage(content=user_input)],
                    }, config={"recursion_limit": 8})
                    latency = time.time() - t0
                    response = result["messages"][-1].content
                    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]

                    # Extract tool usage info
                    tools_used = [m.name for m in tool_msgs]
                    step_details = [
                        {"tool": m.name, "input": "", "output": m.content[:200]}
                        for m in tool_msgs
                    ]

                    # Retrieved context (if lookup_product_info was used)
                    retrieved_ctx = ""
                    for m in tool_msgs:
                        if m.name == "lookup_product_info":
                            retrieved_ctx = m.content[:500]
                            break

                    refusal = is_refusal(response)
                    escalation = is_escalation(response)

                    # Log
                    st.session_state.agent_logger.log_turn(
                        user_input=user_input,
                        response=response,
                        intent_or_topic=None,
                        latency_s=latency,
                        tools_used=tools_used,
                        chunks_retrieved=1 if retrieved_ctx else 0,
                        was_refusal=refusal,
                        was_escalation=escalation,
                    )

                    # Update session metrics
                    st.session_state.latency_history.append(latency)
                    for t in tools_used:
                        st.session_state.tools_used_session.append(
                            f"{t} ({datetime.now().strftime('%H:%M:%S')})"
                        )

                    st.markdown(response)

                    # Metadata display
                    if tools_used:
                        with st.expander(f"🛠 Tools used: {tools_used}", expanded=False):
                            for sd in step_details:
                                st.code(f"Tool: {sd['tool']}\nInput: {sd['input']}\nOutput: {sd['output']}", language="text")
                    if retrieved_ctx:
                        with st.expander("📄 Knowledge base context", expanded=False):
                            st.text(retrieved_ctx)

                    col_x, col_y = st.columns([3, 1])
                    with col_y:
                        st.caption(f"⏱ {latency:.2f}s")
                        if refusal:
                            st.caption("🚫 Refusal")
                        if escalation:
                            st.caption("📞 Escalated")

                    # Store message
                    meta = {
                        "latency": latency,
                        "tools": tools_used,
                        "steps": step_details,
                        "retrieved_context": retrieved_ctx,
                        "is_refusal": refusal,
                        "is_escalation": escalation,
                    }
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "meta": meta,
                        "feedback_given": False,
                    })

                except Exception as exc:
                    latency = time.time() - t0
                    err_msg = (
                        "I'm experiencing a temporary issue. Please try again or "
                        f"contact {BANK_HELPLINE} for immediate assistance."
                    )
                    st.error(err_msg)
                    st.session_state.agent_logger.log_error(
                        "AGENT_EXECUTION", str(exc), recoverable=True
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "meta": {"latency": latency, "tools": [], "is_refusal": False, "is_escalation": False},
                        "feedback_given": True,
                    })

        st.rerun()


if __name__ == "__main__":
    main()

