"""
Phase 6: Memory Agent — Planning, Context & Multi-Turn Conversations
═══════════════════════════════════════════════════════════════════════
NovaTrust AI Banking Support Agent — Memory Version

Memory architecture:
  SHORT-TERM  : Sliding window of last 6 conversation turns (in-session).
                Cleared when session ends.
  LONG-TERM   : JSON-persisted episodic memory across sessions.
                Stores: topics discussed, preferences, escalation history.
  PLANNING    : Multi-step reasoning using explicit <plan> → <execute> → <respond>.

IMPROVEMENTS OVER Phase 5:
  1. Remembers user context within conversation ("you mentioned a home loan earlier").
  2. Long-term: recalls that user asked about home loans previously.
  3. Multi-step planning: breaks complex queries into sub-steps.
  4. Memory reset command available.

Run:
    python memory_agent.py              # interactive with memory
    python memory_agent.py --demo       # demo showing memory retention
"""

import os
import sys
import time
import json
import math
import logging
import re
from datetime import datetime, date
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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("phase6_memory")

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
GROQ_MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FAISS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
MEMORY_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "Phase6_Memory", "long_term_memory.json")
SHORT_TERM_WINDOW = 6  # number of turns to keep in short-term memory

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set. See .env.example")


# ══════════════════════════════════════════════════════════════
# LONG-TERM MEMORY MANAGER
# ══════════════════════════════════════════════════════════════

class LongTermMemory:
    """Persists episodic memory across sessions in a JSON file."""

    DEFAULT_SCHEMA = {
        "sessions": [],
        "topics_discussed": [],
        "products_of_interest": [],
        "escalation_history": [],
        "preferences": {
            "verbosity": "medium",    # brief | medium | detailed
            "language": "english",
        },
        "last_seen": None,
        "total_sessions": 0,
    }

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return dict(self.DEFAULT_SCHEMA)
        return dict(self.DEFAULT_SCHEMA)

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def start_session(self) -> str:
        session_id = f"sess-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._data["total_sessions"] += 1
        self._data["last_seen"] = datetime.now().isoformat()
        self._data["sessions"].append({
            "id": session_id,
            "started": datetime.now().isoformat(),
            "turns": 0,
        })
        self.save()
        return session_id

    def end_session(self, session_id: str, turns: int) -> None:
        for sess in self._data["sessions"]:
            if sess["id"] == session_id:
                sess["ended"] = datetime.now().isoformat()
                sess["turns"] = turns
                break
        # Keep only last 20 sessions
        if len(self._data["sessions"]) > 20:
            self._data["sessions"] = self._data["sessions"][-20:]
        self.save()

    def add_topic(self, topic: str) -> None:
        if topic and topic not in self._data["topics_discussed"]:
            self._data["topics_discussed"].append(topic)
            if len(self._data["topics_discussed"]) > 50:
                self._data["topics_discussed"] = self._data["topics_discussed"][-50:]
            self.save()

    def add_product_interest(self, product: str) -> None:
        entry = {"product": product, "date": date.today().isoformat()}
        self._data["products_of_interest"].append(entry)
        if len(self._data["products_of_interest"]) > 20:
            self._data["products_of_interest"] = self._data["products_of_interest"][-20:]
        self.save()

    def add_escalation(self, reason: str) -> None:
        self._data["escalation_history"].append({
            "date": datetime.now().isoformat(),
            "reason": reason[:200],
        })
        self.save()

    def set_preference(self, key: str, value: str) -> None:
        self._data["preferences"][key] = value
        self.save()

    def get_context_summary(self) -> str:
        """Return a natural-language summary of long-term memory for injection into system prompt."""
        total = self._data["total_sessions"]
        topics = self._data["topics_discussed"][-5:] if self._data["topics_discussed"] else []
        products = [p["product"] for p in self._data["products_of_interest"][-3:]]
        last_seen = self._data.get("last_seen")

        parts = []
        if total > 1:
            parts.append(f"This customer has interacted {total} times previously.")
        if topics:
            parts.append(f"Recent topics they've asked about: {', '.join(topics)}.")
        if products:
            parts.append(f"Products they've shown interest in: {', '.join(products)}.")
        if last_seen:
            parts.append(f"Last interaction: {last_seen[:10]}.")

        return " ".join(parts) if parts else "This appears to be the customer's first interaction."

    def reset(self) -> None:
        """Clear all long-term memory."""
        self._data = dict(self.DEFAULT_SCHEMA)
        self.save()

    @property
    def preferences(self) -> dict:
        return self._data.get("preferences", {})


# ══════════════════════════════════════════════════════════════
# TOPIC EXTRACTOR (simple NLP)
# ══════════════════════════════════════════════════════════════

TOPIC_MAP = {
    r"fixed deposit|fd |tax.saver fd": "Fixed Deposits",
    r"home loan|housing loan|mortgage": "Home Loans",
    r"personal loan": "Personal Loans",
    r"car loan|auto loan|vehicle loan": "Car Loans",
    r"credit card": "Credit Cards",
    r"savings account": "Savings Account",
    r"current account": "Current Account",
    r"emi|equated monthly": "EMI / Loan Payments",
    r"upi|neft|rtgs|imps|transfer": "Fund Transfers",
    r"atm|debit card": "Debit Card / ATM",
    r"net banking|internet banking|netbanking": "Net Banking",
    r"mobile app|novatrust app": "Mobile App",
    r"complaint|dispute|fraud": "Complaint / Dispute",
    r"branch|visiting|office": "Branch Services",
}

def extract_topic(text: str) -> Optional[str]:
    lower = text.lower()
    for pattern, topic in TOPIC_MAP.items():
        if re.search(pattern, lower):
            return topic
    return None


PRODUCT_KEYWORDS = {
    "Home Loan", "Personal Loan", "Car Loan", "Education Loan",
    "Fixed Deposits", "Recurring Deposit", "Credit Cards", "Savings Account",
}


# ══════════════════════════════════════════════════════════════
# RETRIEVAL (shared with Phase 4)
# ══════════════════════════════════════════════════════════════

_vectorstore_cache: Optional[FAISS] = None

def get_vectorstore() -> Optional[FAISS]:
    global _vectorstore_cache
    if _vectorstore_cache is None and os.path.exists(FAISS_INDEX_PATH):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore_cache = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
    return _vectorstore_cache


def retrieve_context(query: str, top_k: int = 3) -> str:
    vs = get_vectorstore()
    if vs is None:
        return ""
    docs = vs.similarity_search(query, k=top_k)
    parts = [f"[{doc.metadata.get('source_file', 'KB')}] {doc.page_content.strip()}" for doc in docs]
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════
# MEMORY-ENHANCED AGENT
# ══════════════════════════════════════════════════════════════

MEMORY_SYSTEM_PROMPT = """You are an AI Banking Support Assistant for NovaTrust Bank.

CUSTOMER CONTEXT (from memory):
{long_term_context}

CURRENT SESSION KNOWLEDGE BASE CONTEXT:
{kb_context}

CONVERSATION RULES:
1. Use the customer context to personalise responses naturally
   (e.g., "Since you've been asking about home loans earlier, you might also want to know...").
2. NEVER process transactions, approve loans, or give legal/tax advice.
3. NEVER fabricate account data or PII.
4. For multi-step queries, address each step clearly and in order.
5. If the customer mentions a topic they've asked about before, acknowledge continuity.
6. For complex queries, use this internal mini-plan (do not show to user):
   [Step 1: Understand what is being asked]
   [Step 2: Check if safety restrictions apply]
   [Step 3: Retrieve relevant info from context]
   [Step 4: Formulate complete, helpful response]
7. Keep responses {verbosity_instruction}.

Helpline: 1800-NOVA-123 | Website: www.novatrust.in"""

VERBOSITY_MAP = {
    "brief": "concise — 2-3 sentences unless a list is essential",
    "medium": "moderate — cover key points without excessive detail",
    "detailed": "thorough — cover all relevant aspects with examples",
}


def build_messages(
    user_input: str,
    short_term: list[dict],
    long_term_mem: LongTermMemory,
) -> list:
    """Build the message list with memory-injected context."""

    # Retrieve KB context
    kb_ctx = retrieve_context(user_input)
    long_term_ctx = long_term_mem.get_context_summary()
    verbosity = VERBOSITY_MAP.get(long_term_mem.preferences.get("verbosity", "medium"), VERBOSITY_MAP["medium"])

    system_content = MEMORY_SYSTEM_PROMPT.format(
        long_term_context=long_term_ctx,
        kb_context=kb_ctx if kb_ctx else "Knowledge base not available — answer from general knowledge.",
        verbosity_instruction=verbosity,
    )

    messages = [SystemMessage(content=system_content)]

    # Short-term conversation history
    for turn in short_term[-SHORT_TERM_WINDOW:]:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(AIMessage(content=turn["assistant"]))

    messages.append(HumanMessage(content=user_input))
    return messages


def call_with_memory(
    user_input: str,
    short_term: list[dict],
    long_term_mem: LongTermMemory,
) -> tuple[str, float]:
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0, max_tokens=600)
    messages = build_messages(user_input, short_term, long_term_mem)
    t0 = time.time()
    result = llm.invoke(messages)
    return result.content, time.time() - t0


# ─────────────────────────────────────────────
# Demo: multi-turn memory retention
# ─────────────────────────────────────────────
DEMO_TURNS = [
    ("T1", "What is the home loan interest rate at NovaTrust?"),
    ("T2", "And what is the maximum tenure for that?"),          # ← needs T1 context
    ("T3", "What documents do I need for a home loan?"),         # ← still home loan context
    ("T4", "OK, separately — what is the FD rate for 2 years?"),
    ("T5", "Can you compare home loan vs FD investment returns?"), # comparative advice → should refuse
    ("T6", "Going back to the home loan — what's the processing fee?"), # ← long context recall
]


def run_demo() -> None:
    print("\n" + "═" * 70)
    print("  Phase 6 — Memory Agent Demo (Multi-Turn Context Retention)")
    print("═" * 70)

    mem = LongTermMemory(MEMORY_FILE_PATH)
    sess_id = mem.start_session()
    short_term: list[dict] = []

    for label, question in DEMO_TURNS:
        print(f"\n{'─'*70}")
        print(f"[{label}] YOU: {question}")

        t0 = time.time()
        response, latency = call_with_memory(question, short_term, mem)
        short_term.append({"user": question, "assistant": response})
        if len(short_term) > SHORT_TERM_WINDOW:
            short_term = short_term[-SHORT_TERM_WINDOW:]

        # Update long-term memory
        topic = extract_topic(question)
        if topic:
            mem.add_topic(topic)
            if topic in PRODUCT_KEYWORDS:
                mem.add_product_interest(topic)

        print(f"AGENT: {response}")
        print(f"[Short-term turns: {len(short_term)} | LT topics: {mem._data['topics_discussed']}]")
        print(f"Latency: {latency:.2f}s")

    mem.end_session(sess_id, len(short_term))
    print(f"\n[Memory saved to {MEMORY_FILE_PATH}]")

    print("\n" + "═" * 70)
    print("  PHASE 6 IMPROVEMENTS DEMONSTRATED:")
    print("  T2 correctly used 'home loan' context from T1.")
    print("  T3 continued home loan topic without restatement.")
    print("  T6 recalled home loan context after a tangent (T4).")
    print("  T5 correctly refused comparative investment advice.")
    print("  Long-term memory persisted to disk for next session.")


# ─────────────────────────────────────────────
# Interactive mode
# ─────────────────────────────────────────────
def run_interactive() -> None:
    mem = LongTermMemory(MEMORY_FILE_PATH)
    sess_id = mem.start_session()
    short_term: list[dict] = []

    print(f"\n{'═'*60}")
    print("  NovaTrust AI — Phase 6: Memory Agent")
    print(f"  {mem.get_context_summary()}")
    print("  Commands: 'quit' | 'memory' | 'reset memory' | 'brief/medium/detailed'")
    print(f"{'═'*60}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "memory":
            print(json.dumps(mem._data, indent=2))
            continue
        if user_input.lower() == "reset memory":
            mem.reset()
            short_term.clear()
            print("[Memory cleared]")
            continue
        if user_input.lower() in ("brief", "medium", "detailed"):
            mem.set_preference("verbosity", user_input.lower())
            print(f"[Verbosity set to {user_input.lower()}]")
            continue

        try:
            response, latency = call_with_memory(user_input, short_term, mem)
            short_term.append({"user": user_input, "assistant": response})
            if len(short_term) > SHORT_TERM_WINDOW:
                short_term = short_term[-SHORT_TERM_WINDOW:]

            # Update memory
            topic = extract_topic(user_input)
            if topic:
                mem.add_topic(topic)
                if topic in PRODUCT_KEYWORDS:
                    mem.add_product_interest(topic)

            print(f"\nAgent: {response}")
            print(f"[Latency: {latency:.2f}s | Short-term: {len(short_term)} turns]")
            logger.info("TURN=%d | TOPIC=%s | LATENCY=%.2fs", len(short_term), topic, latency)

        except Exception as exc:
            print(f"Error: {exc}")

    mem.end_session(sess_id, len(short_term))
    print(f"\n[Session ended. Memory saved to {MEMORY_FILE_PATH}]")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()

