# Project Status Handover (2026-04-21)

## 1) Snapshot hiện tại
- Current HEAD: `5c5c4a0` (`fix ui`), đồng bộ với `main` và `origin/main`.
- Branch đang làm việc: `vankhanh` (ahead `origin/vankhanh` 2 commits).
- Working tree hiện tại đang dirty do runtime file:
  - `M session_memory/chat_history.json`
- Runtime data đang có trên disk:
  - `faiss_index/index.faiss`
  - `faiss_index/index.pkl`
  - `session_memory/chat_history.json`

## 2) Dự án đã làm được gì (đến hiện tại)
- RAG backend hoàn chỉnh luồng hỏi đáp chuẩn + Advanced:
  - `ask_question()` (standard conversational RAG)
  - `ask_question_advanced()` (self-check + retry `better_query` + `confidence`)
- Retrieval đã có hybrid và hardening:
  - FAISS + BM25
  - tự rebuild BM25 khi thiếu state
  - fallback sang FAISS khi BM25 lỗi/chưa sẵn sàng
  - reranker CrossEncoder có fallback compatibility
- Quản lý memory/session đã tách rõ:
  - lịch sử chat lưu ở `session_memory/chat_history.json`
  - xóa vector store không xóa lịch sử chat session
- Tối ưu RAM backend đã có:
  - `num_ctx` giảm từ `3072` xuống `2048`
  - CrossEncoder chạy `device="cpu"` ở nhánh float16
  - có `gc.collect()` sau init pipeline
- UI đã nâng cấp đáng kể:
  - thêm model `qwen2.5:3b` và đặt mặc định là `3b` (hợp máy RAM thấp)
  - giữ `qwen2.5:7b`
  - toggle Advanced mode + badge trạng thái pipeline
  - ô lọc metadata theo `file_name` và đã truyền `filter_dict` xuống backend khi hỏi
  - lịch sử chat có nút chọn nhanh và highlight message
  - modal xác nhận khi xóa lịch sử/xóa vector
  - avatar user/assistant bằng ảnh (`images/user.jpg`, `images/AI.jpg`)

## 3) Hạn chế còn tồn tại
- Upload flow vẫn đang append mặc định:
  - UI gọi `process_new_document(chunks)` mà chưa expose `clear_old=True`.
- Bộ lọc metadata hiện còn tối giản:
  - mới có text input theo `file_name`, chưa có UI dropdown/tập giá trị khả dụng.
- Chưa có test automation chuẩn:
  - `pytest` chưa có trong môi trường runtime hiện tại.
  - thư mục `tests/` chủ yếu là script kiểm thử thủ công.
- Với máy RAM thấp, model `qwen2.5:7b` vẫn có thể fail do thiếu bộ nhớ.

## 4) Verification notes (đợt cập nhật này)
- `git status --short --branch` -> `vankhanh...origin/vankhanh [ahead 2]`, dirty do `session_memory/chat_history.json`.
- `git log --oneline --decorate -n 12` -> HEAD `5c5c4a0` (`fix ui`), có merge PR #2.
- `frontend/ui.py` đã xác nhận:
  - model options hiện là `3b` + `7b`
  - default model là `3b`
  - có `cfg_filter_filename` + truyền `filter_dict`
- `pytest -q` -> `pytest: command not found`.

## 5) Hướng dẫn chạy ổn định (máy RAM thấp)
- Ưu tiên `qwen2.5:3b`.
- Giữ `Top-k` khoảng 2-3.
- Chỉ bật Advanced mode khi thật sự cần self-check/confidence.
- Trước khi nạp bộ tài liệu hoàn toàn mới, bấm `Xóa vector / tài liệu` để tránh trộn ngữ cảnh.

## 6) Priority next tasks
1. Thêm tùy chọn UI `Replace existing dataset` và truyền `clear_old=True`.
2. Nâng cấp UI filter metadata (dropdown/tự gợi ý file thay vì nhập tay).
3. Bổ sung xử lý lỗi thân thiện cho trường hợp Ollama thiếu RAM (auto-gợi ý chuyển 3b).
4. Chuẩn hóa automated tests (cài pytest + test command ổn định).
5. Đồng bộ lại `documentation/project_targets_section8_status.md` theo trạng thái mới.

## 7) Key files để tiếp tục
- `/home/catouis/Project-Qwen-Rag-Agent/backend/rag_pipeline.py`
- `/home/catouis/Project-Qwen-Rag-Agent/backend/controller.py`
- `/home/catouis/Project-Qwen-Rag-Agent/frontend/ui.py`
- `/home/catouis/Project-Qwen-Rag-Agent/frontend/styles.py`
- `/home/catouis/Project-Qwen-Rag-Agent/documentation/project_targets_section8_status.md`
