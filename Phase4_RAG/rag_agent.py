"""
Phase 4: RAG Agent — Retrieval-Augmented Generation
═══════════════════════════════════════════════════════════════
NovaTrust AI Banking Support Agent — RAG Version

Combines FAISS semantic search over the banking knowledge base
with Groq LLM to produce grounded, accurate responses.

IMPROVEMENTS OVER Phase 3 (LLM only):
  1. Answers grounded in official NovaTrust documents.
  2. Retrieves specific product rates and policy details.
  3. Source chunk transparency (shows where answer came from).
  4. Handles questions that would cause hallucination without retrieval.

Run:
    python rag_agent.py              # interactive mode
    python rag_agent.py --compare    # with vs without retrieval comparison
"""

import os
import sys
import time
import json
import logging
import re
from datetime import datetime

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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("phase4_rag")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
GROQ_MODEL        = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH  = os.path.join(os.path.dirname(__file__), "..", "faiss_index")
TOP_K             = 4    # number of chunks to retrieve per query

if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not set. See .env.example")


# ─────────────────────────────────────────────
# Load FAISS vector store
# ─────────────────────────────────────────────
def load_vectorstore() -> FAISS:
    if not os.path.exists(FAISS_INDEX_PATH):
        raise RuntimeError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'.\n"
            "Run Phase4_RAG/ingest.py first to build the knowledge base index."
        )
    print("[RAG] Loading embedding model and FAISS index...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[RAG] ✓ Index loaded. Ready to retrieve from NovaTrust knowledge base.")
    return store


# ─────────────────────────────────────────────
# RAG system prompt
# ─────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are an AI Banking Support Assistant for NovaTrust Bank.
Answer the customer's question using ONLY the information in the RETRIEVED CONTEXT below.

RULES:
1. If the answer is clearly in the context, provide it accurately and concisely.
2. If the context does not contain enough information, say:
   "I don't have enough information in my knowledge base for that. Please contact
   1800-NOVA-123 or visit www.novatrust.in for the most up-to-date details."
3. NEVER make up rates, fees, phone numbers, processes, or account data.
4. NEVER process transactions, approve loans, or provide legal/tax advice.
5. Always cite which aspect of the context you used (e.g., "According to our FD rates guide,").
6. For complex queries or complaints, recommend speaking with a NovaTrust representative.

RETRIEVED CONTEXT:
{context}
"""


# ─────────────────────────────────────────────
# Core RAG chain
# ─────────────────────────────────────────────
def retrieve_and_respond(
    query: str,
    vectorstore: FAISS,
    history: list[dict] | None = None,
) -> tuple[str, list[dict], float]:
    """
    Returns:
      (response_text, retrieved_chunks_info, latency_seconds)
    """
    # Step 1: Retrieve relevant chunks
    t0 = time.time()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    docs = retriever.invoke(query)

    # Build context string with source labels
    context_parts = []
    chunks_info = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source_file", "unknown")
        content = doc.page_content.strip()
        context_parts.append(f"[Source {i+1}: {source}]\n{content}")
        chunks_info.append({"source": source, "snippet": content[:120] + "…"})

    context = "\n\n".join(context_parts)

    # Step 2: Build messages
    system_content = RAG_SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_content)]

    if history:
        for turn in history:
            messages.append(HumanMessage(content=turn["user"]))
            messages.append(AIMessage(content=turn["assistant"]))

    messages.append(HumanMessage(content=query))

    # Step 3: Call LLM
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0, max_tokens=600)
    result = llm.invoke(messages)
    latency = time.time() - t0

    return result.content, chunks_info, latency


# ─────────────────────────────────────────────
# With vs Without Retrieval Comparison
# ─────────────────────────────────────────────
COMPARISON_QUESTIONS = [
    {
        "id": "R1",
        "q": "What is the interest rate on a 3-year Fixed Deposit for senior citizens?",
        "note": "Specific rate — must come from KB; LLM alone likely to hallucinate wrong number.",
    },
    {
        "id": "R2",
        "q": "What is the minimum balance requirement for a Nova Premium Savings Account?",
        "note": "Specific product detail — should be retrieved from products KB.",
    },
    {
        "id": "R3",
        "q": "How many business days does NovaTrust take to resolve a disputed ATM transaction?",
        "note": "Specific policy detail — available in policies KB.",
    },
    {
        "id": "R4",
        "q": "What is the process to add a nomination to my savings account?",
        "note": "Process detail — should be retrieved from policies KB.",
    },
    {
        "id": "R5",
        "q": "What is the forex markup on the NovaTrust Visa Gold Credit Card for foreign transactions?",
        "note": "Specific card feature — in products KB.",
    },
]

PLAIN_LLM_SYSTEM = """You are an AI Banking Support Assistant for NovaTrust Bank.
Answer questions based on your general banking knowledge. Be accurate and concise."""


def run_comparison(vectorstore: FAISS) -> None:
    """Compare responses with retrieval vs without retrieval."""
    print("\n" + "═" * 70)
    print("  Phase 4 — With vs Without Retrieval Comparison")
    print("═" * 70)

    llm_plain = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0, max_tokens=400)
    results = []

    for test in COMPARISON_QUESTIONS:
        print(f"\n{'─'*70}")
        print(f"[{test['id']}] {test['q']}")
        print(f"Note: {test['note']}")

        # Without retrieval
        msgs_plain = [
            SystemMessage(content=PLAIN_LLM_SYSTEM),
            HumanMessage(content=test["q"]),
        ]
        t0 = time.time()
        plain_resp = llm_plain.invoke(msgs_plain).content
        plain_latency = time.time() - t0

        # With retrieval
        rag_resp, chunks, rag_latency = retrieve_and_respond(test["q"], vectorstore)

        print(f"\n  WITHOUT retrieval ({plain_latency:.2f}s):")
        for line in plain_resp.split("\n"):
            print(f"    {line}")

        print(f"\n  WITH retrieval ({rag_latency:.2f}s) [{len(chunks)} chunks]:")
        for line in rag_resp.split("\n"):
            print(f"    {line}")

        print(f"\n  Retrieved from: {[c['source'] for c in chunks]}")

        results.append({
            "id": test["id"],
            "question": test["q"],
            "without_rag": {"response": plain_resp, "latency_s": round(plain_latency, 3)},
            "with_rag": {"response": rag_resp, "latency_s": round(rag_latency, 3), "chunks": chunks},
        })

    log_path = "Phase4_RAG/rag_comparison_log.json"
    os.makedirs("Phase4_RAG", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)
    print(f"\n[LOG] Comparison saved to {log_path}")


# ─────────────────────────────────────────────
# Interactive mode
# ─────────────────────────────────────────────
def run_interactive(vectorstore: FAISS) -> None:
    history: list[dict] = []
    print(f"\n{'═'*60}")
    print("  NovaTrust AI — Phase 4: RAG Agent")
    print(f"  Model: {GROQ_MODEL} | Index: {FAISS_INDEX_PATH}")
    print("  Type 'quit' to exit | 'clear' to reset history")
    print(f"{'═'*60}")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[History cleared]")
            continue

        try:
            response, chunks, latency = retrieve_and_respond(user_input, vectorstore, history)
            history.append({"user": user_input, "assistant": response})
            if len(history) > 6:
                history = history[-6:]

            print(f"\nAgent: {response}")
            print(f"\n  Sources: {[c['source'] for c in chunks]}")
            print(f"  Latency: {latency:.2f}s")
            logger.info(
                "TURN=%d | LATENCY=%.2fs | CHUNKS=%d | SOURCES=%s",
                len(history), latency, len(chunks),
                str([c['source'] for c in chunks])[:80],
            )
        except Exception as exc:
            print(f"Error: {exc}")
            logger.error("RAG call failed: %s", exc)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        vs = load_vectorstore()
    except RuntimeError as e:
        print(f"\n⚠️  {e}")
        sys.exit(1)

    if "--compare" in sys.argv:
        run_comparison(vs)
    else:
        run_interactive(vs)

