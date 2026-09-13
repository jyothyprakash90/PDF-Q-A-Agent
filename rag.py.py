"""Simple RAG pipeline: PDF -> chunks -> embeddings -> Chroma -> answer."""

import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500      # words per chunk
CHUNK_OVERLAP = 50    # overlap between chunks
TOP_K = 4             # chunks to retrieve

# --- Clients (created once) ---
_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_chroma = chromadb.Client()
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)


# ---------- Tool 1: read_page ----------
def read_page(pdf_file, page_number: int) -> str:
    """Extract text from a specific page (1-indexed)."""
    reader = PdfReader(pdf_file)
    if page_number < 1 or page_number > len(reader.pages):
        return ""
    return reader.pages[page_number - 1].extract_text() or ""


def load_pdf(pdf_file) -> list[dict]:
    """Extract all pages as {text, page} dicts."""
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"text": text, "page": i, "source": pdf_file.name})
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ---------- Vector store ----------
def build_index(pdf_files) -> chromadb.Collection:
    """Index a list of uploaded PDFs into a fresh Chroma collection."""
    # Fresh collection each time
    try:
        _chroma.delete_collection("pdf_qa")
    except Exception:
        pass
    collection = _chroma.create_collection(
        name="pdf_qa", embedding_function=_embed_fn
    )

    ids, docs, metas = [], [], []
    for pdf in pdf_files:
        for page in load_pdf(pdf):
            for chunk in chunk_text(page["text"]):
                idx = len(ids)
                ids.append(f"chunk_{idx}")
                docs.append(chunk)
                metas.append({"page": page["page"], "source": page["source"]})

    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metas)

    return collection


# ---------- Tool 2: search_chunks ----------
def search_chunks(collection, query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve top-k most similar chunks."""
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        hits.append({"text": doc, "page": meta["page"], "source": meta["source"]})
    return hits


# ---------- Answer ----------
def answer_question(collection, question: str, top_k: int = TOP_K) -> dict:
    """Retrieve chunks and generate a cited answer."""
    chunks = search_chunks(collection, question, top_k)

    if not chunks:
        return {"answer": "No documents indexed yet.", "sources": []}

    # Build context with numbered sources
    context = "\n\n".join(
        f"[Source {i}] (page {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )

    prompt = f"""Answer the question using ONLY the context below.
Cite sources like [Source 1], [Source 2].
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    response = _openai.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful document Q&A assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": chunks,
    }