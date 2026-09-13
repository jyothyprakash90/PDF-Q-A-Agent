# 📄 PDF Q&A Agent (Mini-RAG)

A simple RAG (Retrieval-Augmented Generation) app: upload a PDF, ask questions, get answers with **page numbers**.

## ✨ Features

- Upload any text-based PDF
- Automatic chunking and embedding
- Semantic search with ChromaDB
- Answers from GPT-4o-mini with citations
- Clean Streamlit chat UI

## 🧠 How it works

```
PDF → chunks → embeddings → ChromaDB
                                 ↓
Question → embed → top-4 chunks → GPT-4o-mini → cited answer
```

## 🚀 Run locally

```bash
# 1. Clone
git clone https://github.com/<you>/pdf-qa-mini-rag.git
cd pdf-qa-mini-rag

# 2. Install
pip install -r requirements.txt

# 3. Set your OpenAI key
cp .env.example .env
# edit .env and paste your key

# 4. Run
streamlit run app.py
```

Open http://localhost:8501

## 📦 Stack

| Component | Tool |
|-----------|------|
| UI | Streamlit |
| PDF parsing | pypdf |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| LLM | OpenAI GPT-4o-mini |

## 📂 Project structure

```
pdf-qa-mini-rag/
├── app.py              # Streamlit UI
├── rag.py              # RAG logic (chunk, embed, search, answer)
├── requirements.txt
├── README.md
└── .env.example
```

## 🔧 Tools explained

The two core tools this app uses:

- **`search_chunks(query)`** — finds the most relevant chunks from the PDF
- **`read_page(page_number)`** — reads a specific page from the PDF

## ⚠️ Limitations

- Only works with **text-based PDFs** (no OCR for scanned docs)
- Index resets when the app restarts
- English-focused embedding model
- Requires an OpenAI API key

## 🛣️ Future improvements

- Add a reranker for better retrieval
- Persist the index to disk
- Support multiple PDFs at once
- Add streaming responses

## 📜 License

MIT
