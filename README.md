# Multi-Modal RAG Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) pipeline over technical PDFs containing text, tables, and figures. It extracts layout-aware content with Azure Document Intelligence, captions figures with GPT-4o, indexes semantically chunked content in Milvus, compares retrieval strategies, generates cited answers, and evaluates them with RAGAS.

The pipeline is built as a sequence of Jupyter notebooks so each stage can be inspected, re-run, and cached independently.

## Results

Evaluated on 1,300+ chunks from three public IPCC AR6 climate-science report chapters (WGI Chapters 7–9), using a synthetic ground-truth question set of 20 questions spanning text, table, and image sources.

| Area | Finding |
|---|---|
| Vector indexes (Milvus) | FLAT, HNSW, and IVF_FLAT returned comparable accuracy; approximate indexes were ~2.5× faster (p50/p95 latency measured) |
| Retrieval strategy | Hybrid retrieval (vector + BM25, fused via Reciprocal Rank Fusion) achieved the best Recall@5 (**0.90**) vs. baseline vector search and MMR reranking |
| Generation (RAGAS) | Faithfulness **0.82**, answer relevancy **0.82**, context precision **0.93**, context recall **0.74** |

Context recall was the weakest metric, which points to top-k context breadth and figure-description quality as the next areas to improve.

## Pipeline

| # | Notebook | What it does |
|---|---|---|
| 1 | `notebooks/1_data_ingestion.ipynb` | Analyzes PDFs with Azure Document Intelligence (`prebuilt-layout`), caches raw responses, extracts text, tables (as Markdown), and figure crops |
| 2 | `notebooks/2_clean_and_organize.ipynb` | Removes boilerplate (headers/footers/page numbers), de-hyphenates text, drops degenerate tables and tiny image crops, validates metadata |
| 3 | `notebooks/3_chunking_embeddings.ipynb` | Captions figures with GPT-4o vision, performs per-page semantic chunking (sentence-embedding breakpoints), embeds all chunks with Azure OpenAI |
| 4 | `notebooks/4_milvus_indexing.ipynb` | Creates three Milvus collections (FLAT, HNSW, IVF_FLAT) with identical data and different vector indexes |
| 5 | `notebooks/5_retrieval_benchmarking.ipynb` | Builds a synthetic eval set with known ground truth; benchmarks latency (p50/p95) and Recall@5 across indexes |
| 6 | `notebooks/6_mmr_reranking.ipynb` | Compares baseline, MMR, and hybrid (BM25 + RRF) retrieval on recall, diversity, and chunk-type coverage |
| 7 | `notebooks/7_generation.ipynb` | Generates grounded answers with GPT-4o using hybrid retrieval, returning structured JSON with citations |
| 8 | `notebooks/8_ragas_evaluation.ipynb` | Scores faithfulness, answer relevancy, context precision, and context recall with RAGAS |

Each stage writes outputs to `data/` (intermediate artifacts are git-ignored; small evaluation artifacts such as `eval_set.jsonl` and `generations.jsonl` are tracked).

**Not yet implemented:** DOCX export of answers and citations.

## Requirements

- Python 3.13
- Docker Desktop (for Milvus)
- Azure resources:
  - Azure AI Document Intelligence
  - Azure OpenAI with two deployments: a GPT-4o model and an embedding model (e.g. `text-embedding-3-small`)

## Setup

### 1. Install dependencies

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

`pyproject.toml` and `uv.lock` are the source of truth for dependencies. `requirements.txt` is kept as a plain list for reference.

> **Note on RAGAS:** `ragas` is pinned to `0.3.1` because later releases require an `openai` version that conflicts with this project. Do not upgrade it without checking that compatibility. Notebook 8 also includes a guarded import workaround for a `langchain-community` module that newer versions removed.

### 2. Environment variables

Create a `.env` file in the project root (it is git-ignored):

```bash
# Azure AI Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<your-di-resource>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=<your-key>

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-openai-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2024-08-01-preview
AZURE_OPENAI_GPT4O_DEPLOYMENT=<your-gpt4o-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<your-embedding-deployment-name>
```

### 3. Start Milvus

```bash
docker compose up -d
```

This starts Milvus standalone (with etcd and MinIO) on `localhost:19530`. Milvus data is stored in `./volumes/`. An Attu web UI is available at `http://localhost:8000` for browsing collections.

### 4. Add the source PDFs

The source PDFs are not included in this repository. Download the three public IPCC AR6 WGI chapters and place them in `data/`:

- `IPCC_AR6_WGI_Chapter07.pdf`
- `IPCC_AR6_WGI_Chapter08.pdf`
- `IPCC_AR6_WGI_Chapter09.pdf`

(Available from the IPCC AR6 report website.)

### 5. Run the notebooks

Run notebooks 1 through 8 in order. Each notebook depends on outputs from the previous one. Inside each notebook, run cells top to bottom; if you restart the kernel, re-run from the first cell.

Caching: Document Intelligence responses, figure captions, and generated questions are cached on disk, so re-running a notebook does not repeat billed API calls for inputs it has already processed.

## Project structure

```
multi-modal-rag/
├── notebooks/          # Pipeline stages 1–8
├── data/               # Generated artifacts (most git-ignored)
│   ├── cleaned/        # Stage 2 output
│   ├── chunks/         # Stage 3 output
│   ├── logs/           # Usage and benchmark logs
│   ├── eval_set.jsonl  # Synthetic eval questions (tracked)
│   └── generations.jsonl
├── volumes/            # Milvus data (git-ignored)
├── docker-compose.yml  # Milvus + etcd + MinIO + Attu
├── pyproject.toml      # Dependencies (uv)
├── uv.lock
└── requirements.txt
```

## Cost and rate limits

- Document Intelligence and GPT-4o calls are billed per page or per token.
- Embedding and generation calls are subject to Azure OpenAI TPM/RPM limits per deployment. If you see `429` errors, raise the deployment's capacity in the Azure portal.
- RAGAS evaluation makes several LLM calls per question internally, so a full run is noticeably more expensive than a single generation pass.

## Limitations and future work

- Figure descriptions are generated as prose captions. Exact numbers in charts can be paraphrased, which affects retrieval for numeric questions. Structured or multi-modal (e.g. CLIP-style) embeddings are a natural next step.
- Evaluation uses a small synthetic question set (20 questions) with the expected chunk's raw content as the reference. Results are directional, not statistically conclusive.
- Retrieval scale was tested at ~1,300 chunks; index trade-offs (FLAT vs. approximate) typically become more pronounced at larger scale.
