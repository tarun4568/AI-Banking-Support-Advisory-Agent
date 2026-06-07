"""
Phase 7: Adaptive Agent — Feedback-Driven Behaviour Modification
═════════════════════════════════════════════════════════════════
NovaTrust AI Banking Support Agent — Adaptive Version

ADAPTATION MECHANISMS:
  1. Response length:    Negative "too verbose" feedback → reduces max_tokens.
                         Positive feedback → maintains or slightly increases.
  2. Caveat injection:   After multiple confusion signals → auto-adds KB disclaimers.
  3. Escalation threshold: High frustration signals → lowers threshold for human handoff.
  4. Proactive suggestions: After successful patterns → adds related product prompts.

FEEDBACK TYPES COLLECTED:
  • thumbs_up / thumbs_down  — simple polarity
  • rating 1–5               — granular quality score
  • category: "too_verbose" | "not_helpful" | "wrong_info" | "perfect" | "needs_escalation"

Run:
    python adaptive_agent.py              # interactive with adaptive behaviour
    python adaptive_agent.py --demo       # before/after feedback demonstration
"""

import os
import sys
import time
import json
import math
import logging
import re
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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("phase7_adaptive")

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
GROQ_MODEL        = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FAISS_INDEX_PATH  = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
FEEDBACK_STORE    = os.path.join(os.path.dirname(__file__), "feedback_store.json")

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set. See .env.example")


# ══════════════════════════════════════════════════════════════
# FEEDBACK STORE MANAGER
# ══════════════════════════════════════════════════════════════

DEFAULT_STORE = {
    "schema_version": "1.0",
    "total_interactions": 0,
    "response_length_score": 0.5,       # 0=shorten, 1=lengthen, 0.5=neutral
    "topics": {},
    "refusal_feedback": {"appropriate_count": 0, "inappropriate_count": 0},
    "escalation_feedback": {"appropriate_count": 0, "inappropriate_count": 0},
    "overall_rating_sum": 0,
    "overall_rating_count": 0,
    "negative_feedback_streak": 0,
    "behaviour_adjustments": {
        "max_response_tokens": 512,
        "add_caveats": False,
        "proactive_escalation_threshold": 0.7,
    },
    "feedback_history": [],
}


class FeedbackStore:
    def __init__(self, path: str):
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return dict(DEFAULT_STORE)

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    def record_feedback(
        self,
        query: str,
        response: str,
        topic: Optional[str],
        thumbs: Optional[str],         # "up" | "down" | None
        rating: Optional[int],          # 1–5 | None
        category: Optional[str],        # "too_verbose" | "not_helpful" | "wrong_info" | "perfect" | "needs_escalation"
    ) -> None:
        """Store feedback and trigger behaviour update."""
        self._data["total_interactions"] += 1
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query_snippet": query[:80],
            "topic": topic,
            "thumbs": thumbs,
            "rating": rating,
            "category": category,
        }
        self._data["feedback_history"].append(entry)
        # Keep only last 100 entries
        if len(self._data["feedback_history"]) > 100:
            self._data["feedback_history"] = self._data["feedback_history"][-100:]

        # Update topic-level stats
        if topic:
            if topic not in self._data["topics"]:
                self._data["topics"][topic] = {"positive": 0, "negative": 0, "total": 0}
            self._data["topics"][topic]["total"] += 1
            if thumbs == "up" or (rating and rating >= 4):
                self._data["topics"][topic]["positive"] += 1
            elif thumbs == "down" or (rating and rating <= 2):
                self._data["topics"][topic]["negative"] += 1

        # Track overall rating
        if rating:
            self._data["overall_rating_sum"] += rating
            self._data["overall_rating_count"] += 1

        # Track verbosity signals
        if category == "too_verbose":
            self._data["response_length_score"] = max(0.0, self._data["response_length_score"] - 0.1)
        elif category == "perfect" or thumbs == "up":
            self._data["response_length_score"] = min(1.0, self._data["response_length_score"] + 0.05)

        # Track negative streak for escalation threshold
        if thumbs == "down" or (rating and rating <= 2):
            self._data["negative_feedback_streak"] += 1
        else:
            self._data["negative_feedback_streak"] = 0

        # Trigger behaviour adjustments
        self._update_behaviour()
        self.save()

        logger.info(
            "FEEDBACK | THUMBS=%s | RATING=%s | CATEGORY=%s | TOPIC=%s",
            thumbs, rating, category, topic,
        )

    def _update_behaviour(self) -> None:
        """Recalculate behaviour parameters based on aggregated feedback."""
        adj = self._data["behaviour_adjustments"]

        # Response length adaptation
        score = self._data["response_length_score"]
        if score < 0.3:
            adj["max_response_tokens"] = 300    # shorten significantly
        elif score < 0.45:
            adj["max_response_tokens"] = 400
        elif score > 0.75:
            adj["max_response_tokens"] = 650
        else:
            adj["max_response_tokens"] = 512    # default

        # Caveat injection: if 3+ interactions flag "wrong_info"
        wrong_info_count = sum(
            1 for e in self._data["feedback_history"]
            if e.get("category") == "wrong_info"
        )
        adj["add_caveats"] = wrong_info_count >= 3

        # Escalation threshold: lower if multiple unresolved frustration signals
        streak = self._data["negative_feedback_streak"]
        if streak >= 3:
            adj["proactive_escalation_threshold"] = 0.4   # more aggressive escalation
        elif streak == 0:
            adj["proactive_escalation_threshold"] = 0.7   # default

    @property
    def behaviour(self) -> dict:
        return self._data["behaviour_adjustments"]

    @property
    def avg_rating(self) -> Optional[float]:
        if self._data["overall_rating_count"] == 0:
            return None
        return round(self._data["overall_rating_sum"] / self._data["overall_rating_count"], 2)

    def get_summary(self) -> str:
        return (
            f"Interactions: {self._data['total_interactions']} | "
            f"Avg Rating: {self.avg_rating or 'N/A'} | "
            f"Length Score: {self._data['response_length_score']:.2f} | "
            f"Max Tokens: {self.behaviour['max_response_tokens']} | "
            f"Caveats: {self.behaviour['add_caveats']} | "
            f"Escalation Threshold: {self.behaviour['proactive_escalation_threshold']}"
        )


# ══════════════════════════════════════════════════════════════
# RAG HELPER
# ══════════════════════════════════════════════════════════════

_vs_cache: Optional[FAISS] = None

def get_vectorstore() -> Optional[FAISS]:
    global _vs_cache
    if _vs_cache is None and os.path.exists(FAISS_INDEX_PATH):
        emb = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vs_cache = FAISS.load_local(FAISS_INDEX_PATH, emb, allow_dangerous_deserialization=True)
    return _vs_cache


def retrieve(query: str, k: int = 3) -> str:
    vs = get_vectorstore()
    if vs is None:
        return ""
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(d.page_content.strip() for d in docs)


# ══════════════════════════════════════════════════════════════
# ADAPTIVE SYSTEM PROMPT BUILDER
# ══════════════════════════════════════════════════════════════

BASE_SYSTEM = """You are an AI Banking Support Assistant for NovaTrust Bank.

KNOWLEDGE BASE CONTEXT:
{kb_context}

{caveat_instruction}

RULES:
1. NEVER process transactions, approve loans, or give legal/tax advice.
2. NEVER fabricate account data or PII.
3. Be {length_instruction}.
4. For complaints or unresolvable issues, suggest calling 1800-NOVA-123.
{escalation_hint}
NovaTrust Helpline: 1800-NOVA-123 | Website: www.novatrust.in"""

CAVEAT_INSTRUCTION = (
    "IMPORTANT: Always add this caveat when stating product rates or policies:\n"
    "\"Please verify the latest details at www.novatrust.in or contact 1800-NOVA-123 "
    "as rates and policies are subject to change.\""
)

ESCALATION_HINT = (
    "5. This customer has shown frustration recently. Consider proactively offering "
    "human escalation if the query seems complex or if you cannot fully resolve it."
)

LENGTH_MAP = {
    (None, 350): "very concise — 1–2 sentences; no lists unless essential",
    (350, 450):  "brief — 2–3 sentences; lists only for multi-step processes",
    (450, 580):  "moderate — cover key points clearly",
    (580, None): "thorough — address all aspects; use lists for clarity",
}

def length_instruction(max_tokens: int) -> str:
    for (lo, hi), label in LENGTH_MAP.items():
        if (lo is None or max_tokens > lo) and (hi is None or max_tokens <= hi):
            return label
    return "moderate"


def build_adaptive_prompt(query: str, feedback_store: FeedbackStore) -> str:
    kb_ctx = retrieve(query)
    adj = feedback_store.behaviour

    caveat = CAVEAT_INSTRUCTION if adj["add_caveats"] else ""
    esc_hint = ESCALATION_HINT if adj["proactive_escalation_threshold"] < 0.6 else ""

    return BASE_SYSTEM.format(
        kb_context=kb_ctx if kb_ctx else "Not available — use general knowledge carefully.",
        caveat_instruction=caveat,
        length_instruction=length_instruction(adj["max_response_tokens"]),
        escalation_hint=esc_hint,
    )


# ══════════════════════════════════════════════════════════════
# TOPIC EXTRACTOR
# ══════════════════════════════════════════════════════════════

TOPIC_PATTERNS = {
    r"fixed deposit|fd |tax.saver": "Fixed Deposits",
    r"home loan|housing loan|mortgage": "Home Loans",
    r"personal loan": "Personal Loans",
    r"credit card|cc limit": "Credit Cards",
    r"emi|equated monthly": "EMI",
    r"transfer|neft|rtgs|upi": "Fund Transfers",
}

def extract_topic(text: str) -> Optional[str]:
    lower = text.lower()
    for pat, topic in TOPIC_PATTERNS.items():
        if re.search(pat, lower):
            return topic
    return None


# ══════════════════════════════════════════════════════════════
# MAIN CALL
# ══════════════════════════════════════════════════════════════

def adaptive_respond(
    query: str,
    history: list[dict],
    feedback_store: FeedbackStore,
) -> tuple[str, float]:
    adj = feedback_store.behaviour
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=adj["max_response_tokens"],
    )

    system = build_adaptive_prompt(query, feedback_store)
    messages = [SystemMessage(content=system)]
    for turn in history[-6:]:
        messages.append(HumanMessage(content=turn["user"]))
        messages.append(AIMessage(content=turn["assistant"]))
    messages.append(HumanMessage(content=query))

    t0 = time.time()
    result = llm.invoke(messages)
    return result.content, time.time() - t0


# ── Before/After Demo ──────────────────────────────────────────

DEMO_QUERY = "What are the FD interest rates at NovaTrust and how is the interest taxed?"

def run_demo() -> None:
    print("\n" + "═" * 70)
    print("  Phase 7 — Adaptive Agent: Before vs After Feedback")
    print("═" * 70)

    # State A: Initial (neutral feedback store)
    store_before = FeedbackStore(FEEDBACK_STORE)
    print(f"\n[BEFORE] Behaviour: {store_before.get_summary()}")
    resp_before, lat_before = adaptive_respond(DEMO_QUERY, [], store_before)
    print(f"\nQUERY: {DEMO_QUERY}")
    print(f"\n[BEFORE — max_tokens={store_before.behaviour['max_response_tokens']}]")
    print(resp_before)
    print(f"Latency: {lat_before:.2f}s | Length: {len(resp_before)} chars")

    # Simulate 5 "too verbose" feedbacks
    print(f"\n{'─'*70}")
    print("  Simulating 5 'too_verbose' feedback signals...")
    for _ in range(5):
        store_before.record_feedback(
            query=DEMO_QUERY, response=resp_before, topic="Fixed Deposits",
            thumbs="down", rating=2, category="too_verbose"
        )

    print(f"\n[AFTER FEEDBACK] Behaviour: {store_before.get_summary()}")
    resp_after, lat_after = adaptive_respond(DEMO_QUERY, [], store_before)
    print(f"\n[AFTER — max_tokens={store_before.behaviour['max_response_tokens']}]")
    print(resp_after)
    print(f"Latency: {lat_after:.2f}s | Length: {len(resp_after)} chars")

    print(f"\n{'═'*70}")
    print(f"  LENGTH CHANGE: {len(resp_before)} chars → {len(resp_after)} chars")
    print(f"  TOKEN LIMIT:   {512} → {store_before.behaviour['max_response_tokens']}")
    print("  EXPLANATION:")
    print("  - 5 consecutive 'too_verbose' signals reduced response_length_score to ≤ 0.3")
    print("  - This triggered a reduction of max_tokens from 512 → 300")
    print("  - The LLM produced a shorter, more direct response")
    print("  - Without changing the question — only feedback changed the behaviour")

    # Reset after demo
    store_before.record_feedback(None, None, None, "up", 5, "perfect")
    store_before.record_feedback(None, None, None, "up", 5, "perfect")
    print("\n[Demo complete. Feedback store partially restored with positive signals.]")


def run_interactive() -> None:
    store = FeedbackStore(FEEDBACK_STORE)
    history: list[dict] = []

    print(f"\n{'═'*60}")
    print("  NovaTrust AI — Phase 7: Adaptive Agent")
    print(f"  {store.get_summary()}")
    print("  Commands: 'quit' | 'stats' | 'feedback' | 'reset stats'")
    print(f"{'═'*60}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            print(store.get_summary())
            continue
        if user_input.lower() == "reset stats":
            store._data = dict(DEFAULT_STORE)
            store.save()
            print("[Feedback stats reset]")
            continue
        if user_input.lower() == "feedback":
            print("Rate the last response:")
            try:
                thumbs = input("  Thumbs (up/down/skip): ").strip().lower()
                rating_str = input("  Rating (1–5/skip): ").strip()
                rating = int(rating_str) if rating_str.isdigit() else None
                category = input("  Category (too_verbose/not_helpful/wrong_info/perfect/needs_escalation/skip): ").strip().lower()
                topic = extract_topic(history[-1]["user"]) if history else None
                store.record_feedback(
                    query=history[-1]["user"] if history else "",
                    response=history[-1]["assistant"] if history else "",
                    topic=topic,
                    thumbs=thumbs if thumbs in ("up", "down") else None,
                    rating=rating if rating and 1 <= rating <= 5 else None,
                    category=category if category not in ("skip", "") else None,
                )
                print(f"[Feedback recorded. Behaviour: {store.get_summary()}]")
            except (IndexError, ValueError):
                print("[Feedback requires at least one interaction first]")
            continue

        try:
            response, latency = adaptive_respond(user_input, history, store)
            history.append({"user": user_input, "assistant": response})
            if len(history) > 6:
                history = history[-6:]

            print(f"\nAgent: {response}")
            print(f"[Latency: {latency:.2f}s | MaxTokens: {store.behaviour['max_response_tokens']} | Caveats: {store.behaviour['add_caveats']}]")
            print("Type 'feedback' to rate this response.")

        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_interactive()

