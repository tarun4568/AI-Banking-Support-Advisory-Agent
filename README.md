# NovaTrust AI Banking Support Agent
## Industry Capstone Project — LangChain | Groq | FAISS | Streamlit

**Student:** Tarun Khaneja | **Framework:** LangChain | **Scenario:** Banking — AI Support & Advisory Agent (Non-Transactional)

---

## Project Overview

This capstone builds a production-grade AI Banking Support & Advisory Agent across 9 phases, demonstrating the full lifecycle from problem framing through evaluation. The agent helps retail banking customers with product information, process guidance, and complaint handling — while strictly refusing any transactional actions.

```
Phase 1 → Problem Framing & Success Criteria
Phase 2 → Baseline Rules-Based Agent (keyword matching)
Phase 3 → LLM Integration + Prompt Engineering (3 strategies)
Phase 4 → RAG with FAISS + HuggingFace Embeddings
Phase 5 → Tool Usage (EMI calc, FD rates, escalation, KB lookup)
Phase 6 → Memory (short-term sliding window + long-term JSON)
Phase 7 → Adaptive Behaviour (feedback-driven response tuning)
Phase 8 → Deployment (Streamlit UI + PII-safe structured logging)
Phase 9 → Evaluation (automated test harness, 20 test cases)
```

---

## Safety Constraints

- ❌ Does NOT process fund transfers, payments, or wire transfers
- ❌ Does NOT approve, reject, or assess loan applications
- ❌ Does NOT provide legal, tax, or comparative investment advice
- ❌ Does NOT fabricate customer account data or PII
- ✅ DOES escalate fraud reports and distressed customers to human agents
- ✅ DOES mask PII (account numbers, phones, emails) in all logs

---

## Setup

### Prerequisites
- Python 3.10+
- Groq API key (free at https://console.groq.com)

### Install

```bash
cd "Capstone Project"
pip install -r requirements.txt
```

### Configure

```bash
# .env is already configured for this workspace.
# If needed, edit .env:
GROQ_API_KEY=your_key_here
```

### Build Knowledge Base Index (run once)

```bash
python Phase4_RAG/ingest.py
```

This downloads the embedding model (~22MB, first run only) and creates the `faiss_index/` directory.

---

## Running Each Phase

### Phase 2 — Baseline Agent
```bash
python Phase2_Baseline/baseline_agent.py --demo
# or interactive:
python Phase2_Baseline/baseline_agent.py
```

### Phase 3 — LLM Agent (Prompt Comparison)
```bash
# Interactive (DETAILED prompt default):
python Phase3_LLM/llm_agent.py

# Compare all 3 prompt strategies:
python Phase3_LLM/llm_agent.py --compare
```

### Phase 4 — RAG Agent
```bash
# Interactive:
python Phase4_RAG/rag_agent.py

# With vs without retrieval comparison:
python Phase4_RAG/rag_agent.py --compare
```

### Phase 5 — Tool Agent
```bash
# Interactive:
python Phase5_Tools/tool_agent.py

# Demo (correct + incorrect tool usage):
python Phase5_Tools/tool_agent.py --demo
```

### Phase 6 — Memory Agent
```bash
# Interactive with memory:
python Phase6_Memory/memory_agent.py

# Multi-turn demo:
python Phase6_Memory/memory_agent.py --demo
```

### Phase 7 — Adaptive Agent
```bash
# Interactive:
python Phase7_Adaptive/adaptive_agent.py

# Before/after feedback demo:
python Phase7_Adaptive/adaptive_agent.py --demo
```

### Phase 8 — Full Streamlit Deployment
```bash
streamlit run Phase8_Deployment/app.py
# Opens at http://localhost:8501
```

### Phase 9 — Evaluation
```bash
python Phase9_Evaluation/eval_harness.py
# Results saved to Phase9_Evaluation/eval_results.json
# Report saved to Phase9_Evaluation/eval_report.txt
```

### Demo Script (5 Forced Interactions)
```bash
python demo/demo_script.py --save-log
```

---

## Project Structure

```
Capstone Project/
├── .env                          ← API keys (gitignored)
├── .env.example                  ← Template
├── requirements.txt
├── README.md
│
├── knowledge_base/               ← NovaTrust banking KB (plain text)
│   ├── banking_products.txt      (savings, FD, loans, cards)
│   ├── banking_policies.txt      (KYC, ATM, complaints, dormancy)
│   ├── banking_faq.txt           (common Q&A)
│   └── banking_services.txt      (digital banking, branch, insurance)
│
├── faiss_index/                  ← Created by Phase4_RAG/ingest.py
│
├── Phase2_Baseline/
│   └── baseline_agent.py
│
├── Phase3_LLM/
│   └── llm_agent.py
│
├── Phase4_RAG/
│   ├── ingest.py
│   └── rag_agent.py
│
├── Phase5_Tools/
│   └── tool_agent.py
│
├── Phase6_Memory/
│   ├── memory_agent.py
│   └── long_term_memory.json     ← Created on first run
│
├── Phase7_Adaptive/
│   ├── adaptive_agent.py
│   └── feedback_store.json
│
├── Phase8_Deployment/
│   ├── app.py                    ← Main Streamlit app
│   ├── logger.py                 ← PII-safe structured logger
│   └── logs/                     ← Created on first run
│
├── Phase9_Evaluation/
│   ├── test_cases.json           ← 20 test cases
│   ├── eval_harness.py
│   ├── eval_results.json         ← Created after running
│   └── eval_report.txt           ← Created after running
│
├── demo/
│   └── demo_script.py
│
└── docs/
    ├── Phase1_Problem_Framing.md
    ├── Prompt_Comparison_Table.md
    ├── Evaluation_Report.md
    └── Engineering_Justification.md
```

---

## Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| LLM | Groq llama-3.3-70b-versatile | Best-in-class tool calling; strong instruction following |
| Embeddings | HuggingFace all-MiniLM-L6-v2 | Local; no API key; fast; 22MB |
| Vector Store | FAISS (local) | No server; perfect for KB of ~300 chunks |
| Framework | LangChain 0.2+ | Native tool calling; FAISS + Groq integration |
| UI | Streamlit | Rapid deployment; session state; built-in rerun |
| Logging | Custom PII-safe JSON logger | Compliance-ready; structured for analytics |

---

## Deliverables Checklist

- [x] Working AI Agent (Phase 8: `streamlit run Phase8_Deployment/app.py`)
- [x] Problem Framing Document (`docs/Phase1_Problem_Framing.md`)
- [x] Demo Script with 5 interactions (`demo/demo_script.py`)
- [x] Prompt Comparison Table (`docs/Prompt_Comparison_Table.md`)
- [x] Evaluation Report with root cause + fix (`docs/Evaluation_Report.md`)
- [x] Engineering & Product Justification (`docs/Engineering_Justification.md`)
