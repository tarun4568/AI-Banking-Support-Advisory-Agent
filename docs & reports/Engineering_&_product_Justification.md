# Engineering & Product Justification
## NovaTrust AI Banking Support Agent — Design Decisions & Tradeoffs

**Author:** Tarun Khaneja | **Date:** April 2026

---

## 1. Architecture Overview

```
User Query
    │
    ▼
[Input Sanitisation]  ← max 1000 chars; no shell characters
    │
    ▼
[Safety Pre-Screen]   ← Prompt-level: DETAILED system prompt with 6 rules
    │
    ├─── Restricted? ──► [Refusal Response + Redirect]
    │
    ▼
[Agent Router]        ← LangChain create_tool_calling_agent
    │
    ├─► [calculate_emi]         Tools (Phase 5)
    ├─► [check_fd_rates]
    ├─► [lookup_product_info] ──► [FAISS Retriever] ──► [KB Chunks]
    └─► [escalate_to_human]
    │
    ▼
[Groq LLM]            ← llama3-8b-8192, temp=0, max_tokens=adaptive
    │
    ▼
[Response]
    │
    ├─► [PII-Safe Logger]    → logs/agent_YYYYMMDD.jsonl
    ├─► [Feedback Store]     → Phase7_Adaptive/feedback_store.json
    └─► [Streamlit UI]       → User
```

---

## 2. Key Design Decisions

### 2.1 LLM Provider: Groq (llama3-8b-8192)

**Decision:** Use Groq's hosted llama3-8b-8192 via langchain-groq.

**Rationale:**
- **Latency:** Groq's inference is the fastest available for open-weight models (~200–400 tokens/second vs. ~50–100 tokens/second for standard cloud providers). Average response: 2.1s.
- **Cost:** Free tier sufficient for capstone; enterprise pricing is 90% cheaper than equivalent GPT-4o calls.
- **Safety:** llama3-8b follows system prompt instructions reliably at temperature=0.
- **Availability:** No waitlist; API key issued immediately.

**Alternatives considered:**
- OpenAI GPT-4o-mini: Higher quality but 10× more expensive; API key not freely available.
- Local Ollama: No internet dependency but requires 8GB+ RAM and slow on CPU.
- Groq llama3-70b: Better reasoning but 4× more latency; not needed for this use case.

**Tradeoff accepted:** Smaller model (8B) may occasionally miss multi-step reasoning in complex financial calculations → mitigated by delegating math to the `calculate_emi` tool.

---

### 2.2 Embeddings: HuggingFace sentence-transformers (all-MiniLM-L6-v2)

**Decision:** Use local HuggingFace embeddings, not OpenAI/Cohere API embeddings.

**Rationale:**
- **No API key required:** Runs entirely locally after initial download (~22MB).
- **Privacy:** Customer queries never sent to a third-party embedding service.
- **Speed:** Embedding 600-char chunks: ~5ms/chunk on CPU.
- **Quality:** all-MiniLM-L6-v2 achieves 78.9 on SBERT benchmark, sufficient for banking domain retrieval.

**Alternatives considered:**
- OpenAI text-embedding-3-small: Better accuracy (+5%) but requires API key and costs $0.02/1M tokens.
- Cohere embed-english-v3: Commercial API, data privacy concerns for financial data.

**Tradeoff accepted:** Slightly lower retrieval precision vs. OpenAI embeddings → acceptable for this domain; compensated by large chunk overlap (100 chars) and top-k=4 retrieval.

---

### 2.3 Vector Store: FAISS (local)

**Decision:** Use FAISS local index, not a hosted vector database (Pinecone, Qdrant, Weaviate).

**Rationale:**
- **No infrastructure:** Runs from disk, no server to manage or pay for.
- **Latency:** Local FAISS similarity search: <5ms for our 300-chunk index.
- **Simplicity:** Single `vectorstore.save_local()` call; no connection management.
- **Scale:** Our KB has ~300 chunks total — FAISS is optimally suited for <1M vectors without a cluster.

**Alternatives considered:**
- Qdrant: Better for production (persistent, scalable, filtering support) but adds operational complexity.
- Chroma: Python-native, good default but slower than FAISS for pure similarity search.
- Pinecone: Managed, excellent scalability, but requires paid API key and network calls.

**Production note:** For a real deployment with 10,000+ documents and multiple concurrent users, Qdrant or Pinecone would be the preferred choice. FAISS is entirely appropriate for this capstone scope.

---

### 2.4 Prompt Strategy: DETAILED (Selected Default)

**Decision:** Use DETAILED system prompt as default; COT retained as optional fallback.

**Rationale (from Phase 3 comparison):**
- BASIC prompt failed 2/5 safety tests (gave legal advice; ambiguous framing on transfer).
- DETAILED achieved 100% safety compliance across all 20 evaluation tests.
- DETAILED is ~30% shorter in output tokens vs. COT, reducing latency and cost.
- COT occasionally exposed reasoning steps to users (UX issue fixed by routing ambiguous queries to COT only).

**When COT is better:** High-ambiguity compound queries ("Compare my options for a 5-year investment"). In production, a lightweight intent classifier could route to COT for these cases.

---

### 2.5 Memory Architecture: Short-term + JSON Long-term

**Decision:** Use a sliding window for short-term memory (6 turns) + JSON file for long-term.

**Rationale:**
- **Short-term:** 6 turns chosen because home loan discussions rarely need more context than 3 back-and-forths; 6 allows for a full topic exploration without token overflow.
- **Long-term:** JSON file is sufficient for prototype; in production, a Redis or PostgreSQL store would handle concurrent sessions.
- **No LangChain memory wrapper used:** Transparent manual management gives finer control over what is stored and when reset occurs.

**Why not ConversationSummaryMemory?**
- Adds LLM call to generate summary → doubles latency per turn. Not justified for customer support use case where topics are usually narrowly focused.

---

### 2.6 Tool Design: Single-Responsibility + No Sensitive Data

**Decision:** Each tool does exactly one thing and accesses no sensitive customer data.

**Tool Charter:**
- `calculate_emi`: Pure math. No I/O except inputs. Deterministic.
- `check_fd_rates`: Read-only from hardcoded rate table (updated quarterly).
- `lookup_product_info`: Read-only FAISS search. Never writes.
- `escalate_to_human`: Writes to local escalation log only. No customer PII stored.

**Blocked tools:** `transfer_funds`, `get_account_balance`, `approve_loan` — these tools were intentionally NOT created. The system's safety is not just prompt-based; the restricted capabilities simply do not exist as code.

**Anti-loop protection:** `max_iterations=3` in AgentExecutor prevents runaway tool-calling loops.

---

### 2.7 Adaptive Behaviour: Feedback-Driven Token Budget

**Decision:** Adjust `max_tokens` based on aggregated user feedback signals (verbosity score).

**Rationale:**
- Users who give "too verbose" feedback consistently prefer shorter answers. Adapting reduces bounce rate.
- The adaptation happens at the session level, not per-message, preventing erratic behaviour.
- The feedback_store.json persists across sessions for long-term learning.
- Score decay is conservative (–0.1 per negative, +0.05 per positive) to prevent overcorrection.

**Safety constraint on adaptation:** Adaptation CANNOT lower safety rules. Only `max_tokens` and caveat text are adapted; the safety guardrails in the system prompt are fixed.

---

## 3. Safety-First Architecture

**Three-layer safety design:**

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| Layer 1: System Prompt | 6 explicit rules + STRICT RULES header | LLM generating dangerous content |
| Layer 2: No Dangerous Tools | Restricted tools don't exist | Accidental tool execution of blocked actions |
| Layer 3: PII-Safe Logging | Regex masking before any log write | PII leakage via logs or audit trails |

**Escalation triggers (defined in escalate_to_human tool docstring):**
- Fraud reports
- Distressed customers (urgency signal words)
- Account-specific data requests
- Legal/regulatory matters
- 3+ unresolved conversation turns on same topic

---

## 4. Deployment Assumptions & Limitations

**Assumptions:**
1. Single-user sessions (no multi-tenancy required for prototype).
2. FAISS index fits in memory (<500MB).
3. Groq API has <1% downtime (aligns with Groq SLA).
4. Knowledge base updated manually by admin quarterly.
5. English-only inputs (Hindi support is roadmap item).

**Known Limitations:**
1. **No real-time data:** FD rates are static in KB; live rates require API integration.
2. **No account access:** Agent cannot look up real customer accounts (by design — non-transactional).
3. **FAISS not concurrent:** Multiple simultaneous users would need FAISS index in shared memory or migration to Qdrant/Redis.
4. **Session statefulness:** Streamlit session state is in-memory; server restart clears all conversations.
5. **Model hallucination risk:** llama3-8b may hallucinate on very specific product details not in KB → mitigated by RAG and caveat instructions.

---

## 5. Framework Justification: LangChain

**Why LangChain:**
- Native tool calling with `create_tool_calling_agent` — avoids custom agent loop code.
- Compatible with Groq, HuggingFace embeddings, FAISS out of the box.
- `AgentExecutor` provides built-in max_iterations (anti-loop) and error handling.
- Well-documented; large community; easy to extend to LangGraph for more complex flows.

**What was NOT used (and why):**
- **LangGraph:** Overkill for linear question-answering workflow. LangGraph shines for multi-agent or conditional branching flows.
- **LangChain memory classes:** Replaced with manual conversation history for transparency and control.
- **LangChain Hub prompts:** Used custom prompts for full control over safety rules.

---

## 6. Product Justification

**Why this agent, for this user, in this bank:**

The target user (Priya Mehta, retail banking customer) currently spends 8–12 minutes per helpline call for queries that take 30 seconds with this agent. At 45,000 calls/month, with 62% being informational (28,000 calls), deflecting even 40% to the AI agent saves:

```
28,000 × 0.4 = 11,200 calls/month
11,200 × ₹100 avg cost = ₹11.2 lakh/month in ops savings
= ₹1.34 crore/year
```

Against an estimated infrastructure cost of ₹8–12 lakh/year (Groq API + hosting), the ROI is **11× in Year 1**.

Beyond cost, the agent provides:
- **24/7 availability** — most helpline calls are between 8 PM–11 PM (post-work hours).
- **Consistent** information — no agent variability or training gaps.
- **Instant responses** — 2.1s average vs. 8–12 minutes on phone.
- **Safety-first** — never makes a bad financial decision on the customer's behalf.
