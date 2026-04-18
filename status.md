# Project Status Handover (2026-04-18, post-merge main)

## 1) Current snapshot
- Current HEAD: `4d1399a` (`Reduce cost on RAM`), same commit as `main` và `origin/main`.
- Branch working on: `vankhanh` (ahead `origin/vankhanh` by 3 commits).
- Working tree at update time: modified `status.md` (this handover refresh).
- Runtime data currently present on disk:
  - `faiss_index/index.faiss`
  - `faiss_index/index.pkl`
  - `session_memory/chat_history.json`

## 2) Delta introduced after merging main
- RAM-related adjustments are now in codebase:
  - `backend/controller.py`: Ollama context reduced from `num_ctx=3072` to `num_ctx=2048`.
  - `backend/rag_pipeline.py`: CrossEncoder float16 path now pins `device="cpu"`.
  - `backend/rag_pipeline.py`: extra `gc.collect()` is called after pipeline init.
- Existing advanced features remain:
  - advanced flow (`ask_question_advanced`) with self-check, optional retry via `better_query`, and confidence output.
  - confidence array-check bug fix is still in place (`raw_scores is not None and len(raw_scores) > 0`).

## 3) Confirmed done in code
- Controller supports `advanced_mode` routing.
- UI has sidebar toggle `Advanced RAG (self-check + confidence)` and displays current mode in query form.
- UI model labels:
  - `qwen2.5:7b (Q4_K_M - khuyen nghi)`
  - `qwen2.5:14b (chat luong cao hon, ton RAM)`
- Hybrid retrieval hardening:
  - `rank-bm25` present in `requirements.txt`
  - BM25 auto-rebuild + fallback to FAISS when unavailable
- Chat memory is persisted in `session_memory/chat_history.json` and separate from vector-store cleanup.
- Backend supports dataset replacement path via `process_new_document(..., clear_old=True)`.

## 4) Open gaps / known limitations
- UI upload flow still calls `process_new_document(chunks)` without exposing `clear_old=True`, so default behavior remains append.
- UI still does not pass `filter_dict` per question, so strict per-file retrieval filtering is not active in normal flow.
- Advanced mode is lighter than before, but still relatively RAM-heavy due to self-check + optional retry + reranker.
- `tests/` are mostly manual scripts; no stable automated test command in repo yet.

## 5) Verification notes for this refresh
- Key commands checked:
  - `git status --short --branch` -> `vankhanh...origin/vankhanh [ahead 3]`, modified `status.md`.
  - `git log --oneline --decorate -n 12` -> HEAD `4d1399a` aligns with `main`/`origin/main`.
  - `git show --name-status --oneline -n 1` -> latest merge delta touches:
    - `backend/controller.py`
    - `backend/rag_pipeline.py`
- Runtime test tooling:
  - `pytest -q` still unavailable in current environment (`pytest: command not found`).

## 6) Practical run guidance
- For low-RAM machines, keep using `qwen2.5:7b`; current defaults are already reduced (`num_ctx=2048`).
- Keep `Top-k` around 2-3.
- Use Standard mode as default; turn on Advanced only when self-check/confidence is required.
- Before loading a fully new dataset, click `Xóa vector / tài liệu` to avoid mixed retrieval context.

## 7) Priority next tasks
1. Add UI option `Replace existing dataset` and pass `clear_old=True`.
2. Add UI metadata/file filter and pass `filter_dict` per query.
3. Add explicit low-RAM profile toggle for Advanced flow (skip retry/reranker when needed).
4. Convert scripts in `tests/` into automated tests and provide a repeatable test command.
5. Sync `documentation/project_targets_section8_status.md` with this post-merge snapshot.

## 8) Key files to continue from
- `/home/catouis/Project-Qwen-Rag-Agent/backend/rag_pipeline.py`
- `/home/catouis/Project-Qwen-Rag-Agent/backend/controller.py`
- `/home/catouis/Project-Qwen-Rag-Agent/frontend/ui.py`
- `/home/catouis/Project-Qwen-Rag-Agent/documentation/project_targets_section8_status.md`
- `/home/catouis/Project-Qwen-Rag-Agent/README.md`
