"""
Phase 4: RAG — Document Ingestion
═══════════════════════════════════════════════════════════════
Loads NovaTrust banking knowledge base (plain-text files),
splits into chunks, embeds using HuggingFace local model,
and stores in FAISS vector index.

Run ONCE (or whenever knowledge base is updated):
    python ingest.py

Output:
    ../faiss_index/   — FAISS vector store (persisted to disk)
"""

import os
import glob
import json
import time
from datetime import datetime

# Workaround for corporate SSL interceptors (e.g., Zscaler) that set SSL_CERT_FILE
# to a certificate that Python's ssl module cannot load on Windows.
_ZSCALER_CERT = r"C:\HashiCorp\Vagrant\embedded\ZscalerRootCertificate-2048-SHA256.pem"
if os.path.exists(_ZSCALER_CERT):
    os.environ["SSL_CERT_FILE"] = _ZSCALER_CERT
elif os.environ.get("SSL_CERT_FILE") and not os.path.exists(os.environ.get("SSL_CERT_FILE", "x")):
    os.environ.pop("SSL_CERT_FILE", None)

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
FAISS_INDEX_PATH   = os.path.join(os.path.dirname(__file__), "..", "faiss_index")

# HuggingFace model (local, no API key needed, ~22MB)
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings — tuned for banking policy documents
CHUNK_SIZE         = 600    # characters per chunk
CHUNK_OVERLAP      = 100    # overlap to preserve context between chunks


# ─────────────────────────────────────────────
# Step 1: Load text documents
# ─────────────────────────────────────────────
print("\n" + "═" * 60)
print("  NovaTrust Bank — Knowledge Base Ingestion")
print("═" * 60)

kb_files = glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt"))
if not kb_files:
    raise FileNotFoundError(
        f"No .txt files found in '{KNOWLEDGE_BASE_DIR}'. "
        "Ensure the knowledge_base/ folder contains the bank documents."
    )

print(f"\n[1/4] Loading {len(kb_files)} document(s) from knowledge_base/...")
all_docs = []
for path in sorted(kb_files):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    # Add source metadata to each document
    for doc in docs:
        doc.metadata["source_file"] = os.path.basename(path)
    all_docs.extend(docs)
    print(f"   ✓ {os.path.basename(path)}  ({sum(len(d.page_content) for d in docs):,} chars)")

print(f"\n   Total characters loaded: {sum(len(d.page_content) for d in all_docs):,}")


# ─────────────────────────────────────────────
# Step 2: Split into chunks
# ─────────────────────────────────────────────
print(f"\n[2/4] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "?", "!", " "],
    length_function=len,
)

chunks = splitter.split_documents(all_docs)
print(f"   Total chunks created: {len(chunks)}")
print(f"   Average chunk size  : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")


# ─────────────────────────────────────────────
# Step 3: Create embeddings
# ─────────────────────────────────────────────
print(f"\n[3/4] Loading embedding model: {EMBEDDING_MODEL}...")
print("   (First run downloads ~22MB model; subsequent runs use cache)")

t0 = time.time()
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
embed_load_time = time.time() - t0
print(f"   Embedding model loaded in {embed_load_time:.1f}s")


# ─────────────────────────────────────────────
# Step 4: Build and persist FAISS index
# ─────────────────────────────────────────────
print(f"\n[4/4] Building FAISS index from {len(chunks)} chunks...")
t0 = time.time()

vectorstore = FAISS.from_documents(chunks, embeddings)
index_time = time.time() - t0

os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
vectorstore.save_local(FAISS_INDEX_PATH)

print(f"   FAISS index built in {index_time:.1f}s")
print(f"   Index saved to: {FAISS_INDEX_PATH}/")


# ─────────────────────────────────────────────
# Ingestion summary
# ─────────────────────────────────────────────
summary = {
    "timestamp": datetime.now().isoformat(),
    "documents_loaded": len(kb_files),
    "total_chunks": len(chunks),
    "embedding_model": EMBEDDING_MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "index_path": FAISS_INDEX_PATH,
}

summary_path = os.path.join(FAISS_INDEX_PATH, "ingestion_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "═" * 60)
print("  Ingestion Complete!")
print(f"  Documents : {len(kb_files)}")
print(f"  Chunks    : {len(chunks)}")
print(f"  Index     : {FAISS_INDEX_PATH}/")
print("  Next step : Run rag_agent.py")
print("═" * 60)
