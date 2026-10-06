# Multi-Modal RAG Assignment

This workspace contains notebooks and helper scripts to build a multi-modal Retrieval-Augmented Generation (RAG) pipeline using Azure Document Intelligence for PDF extraction and Milvus as the primary vector database. The work is split into several notebooks for clarity and reproducibility.

Notebooks:
- notebooks/1_finish_existing_rag.ipynb — Complete the existing RAG notebook.
- notebooks/2_ingest_azure_document_intelligence.ipynb — PDF ingestion using Azure Document Intelligence (Document Intelligence / Form Recognizer).
- notebooks/3_chunking_embeddings.ipynb — Semantic chunking and embedding pipeline.
- notebooks/4_vector_db_indexing.ipynb — Vector DB ingestion and index experiments (Flat, HNSW, IVF) using Milvus.
- notebooks/5_retriever_benchmarking.ipynb — Retriever pipeline and latency benchmarking.
- notebooks/6_reranking_prompting.ipynb — Reranking (BM25 and MMR) and prompt design for LLM.
- notebooks/7_export_docx.ipynb — Render LLM outputs and citations into DOCX.

Quick start

1. Create and activate your virtualenv (you've done this already):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set Azure Document Intelligence credentials in environment variables:

```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="https://<your-endpoint>"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="<your-key>"
```

3. Configure Milvus (run locally with Docker or use a managed instance). See `notebooks/4_vector_db_indexing.ipynb` for connection details.

What I can do next:
- Scaffold the notebooks and helper scripts (already created).
- Implement Azure ingest runner and a small demo extraction on sample PDFs (I need sample PDFs or a path).
- Wire embeddings → Milvus and run sample index experiments.

Tell me which step you want me to run next (e.g., implement Azure ingest code, process sample PDFs, or finish an existing notebook).