# Project Status Handover (2026-04-17)

## 1) Current snapshot
- Branch/worktree is dirty because of runtime data files:
  - `D faiss_index/chat_history.json`
  - `M faiss_index/index.faiss`
  - `M faiss_index/index.pkl`
  - `?? session_memory/`
- UI file was updated to show model labels and prefer 7B Q4 option:
  - `M frontend/ui.py`

## 2) What is already done
- Added/kept Advanced pipeline in backend (`ask_question_advanced`) with:
  - self-check step
  - optional retry with `better_query`
  - confidence output
- Controller supports `advanced_mode` switch from UI.
- UI has Advanced toggle in sidebar (`Advanced RAG (self-check + confidence)`).
- Added BM25 dependency to requirements (`rank-bm25`).
- Added BM25 self-healing + fallback to FAISS when BM25 is unavailable.
- Added CrossEncoder load fallback when `torch_dtype` is unsupported.
- Chat memory is now stored separately in `session_memory/chat_history.json`.

## 3) Confirmed environment facts
- Local Ollama model currently available:
  - `qwen2.5:7b`
- Quantization of local `qwen2.5:7b` is confirmed as `Q4_K_M`.
- UI now displays model labels:
  - `qwen2.5:7b (Q4_K_M - khuyen nghi)`
  - `qwen2.5:14b (chat luong cao hon, ton RAM)`

## 4) Known issues (not fixed yet)
- Retrieval scope leak between old/new files:
  - New upload may append into existing DB, so answers can mix file A and older file B.
  - Root cause: no per-question file filter passed from UI + append behavior in vector/BM25 DB.
- Advanced mode can be RAM heavy:
  - first answer + self-check + optional retry + reranker cost.
- Confidence warning still appears sometimes:
  - `The truth value of an array with more than one element is ambiguous`
  - likely from boolean check on CrossEncoder score array in confidence logic.

## 5) Practical run guidance (stable mode)
- Use `qwen2.5:7b (Q4_K_M)` for low RAM machines.
- Before testing a new file set, click `Xoa vector / tai lieu` to avoid mixed context.
- Keep `Top-k` low (2-3).
- Turn off Advanced mode for long sessions or weak RAM.
- Use Advanced mode only when needed for self-check/confidence.

## 6) Advanced mode capability status
- Self-RAG self-evaluation: PARTIAL (basic self-check prompt + JSON parse).
- Query rewriting: PARTIAL (history-aware rewrite + better_query retry path).
- Multi-hop reasoning: PARTIAL (single extra retrieval retry, not full multi-hop decomposition).
- Confidence scoring: YES (heuristic + optional self-check confidence).

## 7) Priority next tasks
1. Add strict file-level filter from UI -> controller -> retriever for each question.
2. Add "replace dataset" option when uploading a new file (not always append).
3. Fix CrossEncoder score array check in confidence function.
4. Add low-RAM mode for advanced flow (skip retry/reranker when memory is tight).

## 8) Key files to continue from
- `/home/catouis/Project-Qwen-Rag-Agent/backend/rag_pipeline.py`
- `/home/catouis/Project-Qwen-Rag-Agent/backend/controller.py`
- `/home/catouis/Project-Qwen-Rag-Agent/frontend/ui.py`
- `/home/catouis/Project-Qwen-Rag-Agent/documentation/project_targets_section8_status.md`
- `/home/catouis/Project-Qwen-Rag-Agent/README.md`
