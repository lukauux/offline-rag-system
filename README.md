# Multimodal RAG System (Offline Prototype)

Hackathon-ready prototype for **Smart India Hackathon 2025 – Problem Statement 25231**. The system ingests PDFs, DOC/DOCX, images, and audio files, stores unified CLIP embeddings in FAISS, and serves grounded answers with citations while running fully offline after the first model download.

## Key Capabilities
- **Unified knowledge base** - drag & drop documents, screenshots, and recordings into a single store.
- **Offline-first pipeline** - CLIP (`clip-ViT-B-32`) for text/image embeddings, Whisper for audio transcription, FAISS for vector search.
- **Cross-modal retrieval** - text queries surface relevant pages, audio segments (with timestamps), and image metadata.
- **Grounded answers** - stitched summary with explicit citations (filename, page, timestamp, chunk id).
- **Local LLM option** - plug in a llama.cpp GGUF checkpoint for richer answer synthesis.
- **Modern UI** - Next.js 15 chat interface with search controls (MMR, similarity, multi-query) and citation explorer.

## Project Structure
```
multimodal-rag-system/
+- app/                         # Next.js App Router entrypoints
¦  +- api/rag/ingest/route.ts   # Upload ? Python pipeline bridge
¦  +- api/rag/query/route.ts    # Query ? Python pipeline bridge
+- components/                  # UI building blocks (chat, sidebar, results)
+- scripts/                     # Python ML pipeline & server
¦  +- enhanced_rag_pipeline.py  # Main pipeline (ingest, query, save)
¦  +- document_processor.py     # PDF/DOC/TXT extraction + chunking
¦  +- image_processor.py        # CLIP image embeddings
¦  +- audio_processor.py        # Whisper transcription
¦  +- vector_database.py        # FAISS cosine index + metadata store
¦  +- run_rag_server.py         # Lightweight HTTP bridge (port 8000)
¦  +- setup_rag_system.py       # Creates .data workspace & config
+- .data/                       # Local storage (created at runtime)
+- package.json                 # Next.js dependencies
+- requirements.txt             # Python dependencies
```

## Prerequisites
- **Node.js 18+** (Next.js 15, React 19)
- **Python 3.11 / 3.12**
- **Git LFS disabled** (large models downloaded via SentenceTransformers / Whisper caches)
- ~6 GB free disk (model caches + FAISS index)

## Setup & Run
1. **Install frontend deps**
   ```bash
   npm install
   ```

2. **Create Python virtual environment & install deps**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Prep local workspace** (creates `.data`, config, upload folder)
   ```bash
   python scripts/setup_rag_system.py
   ```

4. **Start the offline RAG server** (downloads CLIP + Whisper on first run)
   ```bash
   python scripts/run_rag_server.py
   ```
   - Runs on `http://127.0.0.1:8000`
   - Endpoints: `POST /ingest`, `POST /query`, `POST /stats`

5. **Run Next.js frontend** (separate terminal)
   ```bash
   npm run dev
   ```
   Visit [http://localhost:3000](http://localhost:3000)

> Optional: set `RAG_SERVER_URL` in `.env.local` if the Python server runs on a different host/port.

## Usage Flow
1. **Ingest files** – drag & drop PDFs, DOC/DOCX, images, or audio (MP3/WAV/M4A/OGG).
2. **Indexing** – files are saved to `.data/uploads`, processed by the Python server, and embedded into FAISS.
3. **Ask questions** – use the chat box to issue natural language queries.
4. **Review evidence** – answers include the stitched reasoning plus source cards (page, timestamp, chunk id, file path).

## Retrieval Strategies
- **Similarity** – top-k cosine matches.
- **MMR** – diversity filter across different files/pages.
- **Multi-query** – auto-expands the question with simple paraphrases for wider recall.

## Optional: Enable Local LLM
1. Download a compatible GGUF model (e.g., `mistral-7b-instruct.gguf`) and place it on disk.
2. Update `.data/config.json` after running the setup script:
   ```json
   "llm": {
     "model_path": "C:/models/mistral-7b-instruct.gguf",
     "threads": 8,
     "context_window": 4096,
     "max_tokens": 384
   }
   ```
3. Restart `python scripts/run_rag_server.py` so the LLM loads (requires `llama-cpp-python`, already in `requirements.txt`).

With no `model_path` configured the pipeline falls back to extractive summaries, so the prototype remains usable without an LLM.

## Offline Behaviour
- All models (CLIP + Whisper) are cached in standard SentenceTransformers/Whisper cache folders.
- FAISS index + metadata persist locally in `.data/index` – restart-safe.
- Once caches exist, both Python and Next.js runtimes work without internet access.

## Troubleshooting
| Issue | Fix |
| --- | --- |
| `whisper` import errors | Ensure `ffmpeg` is installed and available on PATH. |
| Python server reports “No text extracted” | Verify the document contains selectable text; scanned PDFs need OCR. |
| Query returns empty results | Confirm files were indexed (sidebar shows `Indexed`). Increase `k` or enable multi-query. |
| Cross-origin errors | Keep the Python server on `localhost` or update `RAG_SERVER_URL` accordingly. |

## Roadmap Ideas
- Integrate on-device LLM (llama.cpp) for richer generation.
- Add OCR for scanned PDFs and handwriting.
- Implement hybrid sparse + dense retrieval with BM25.
- Stream responses and partial citations back to the UI.

Built as an offline-first reference implementation for Smart India Hackathon 2025. MIT licensed – remix freely for your team.
"# offline-rag-system" 
