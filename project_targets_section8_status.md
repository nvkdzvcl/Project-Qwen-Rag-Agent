# Project Targets & Section 8 Status (Updated)

Cập nhật lần cuối: `2026-04-17`  
Phạm vi: đối chiếu yêu cầu mục `8` trong `project_report_final.pdf` với code và kết quả test web hiện tại.

## 1) Snapshot hiện tại

### Đã xác nhận chạy được
- App web chạy được end-to-end trên Linux khi Ollama hoạt động ổn.
- Luồng hỏi đáp chính chạy ổn.
- BM25 thiếu dependency đã được xử lý bằng cách thêm `rank-bm25` vào [requirements.txt](/home/catouis/Project-Qwen-Rag-Agent/requirements.txt:13).
- Trường hợp BM25 mất state không còn làm app văng lỗi; backend tự rebuild hoặc fallback vector.

### Vấn đề còn tồn tại (đã quan sát thực tế)
- Toggle `Advanced RAG` chưa thấy trên giao diện runtime dù code đã có.
- UI hiện vẫn chưa có luồng multi-document và metadata filter thực sự cho người dùng cuối.

## 2) Cập nhật kỹ thuật mới trong code

### 2.1 Advanced mode backend
- Controller đã nhận tham số `advanced_mode` tại [backend/controller.py](/home/catouis/Project-Qwen-Rag-Agent/backend/controller.py:92).
- Pipeline đã có `ask_question_advanced` với self-check, retry theo `better_query` và trả `confidence` tại [backend/rag_pipeline.py](/home/catouis/Project-Qwen-Rag-Agent/backend/rag_pipeline.py:627).

### 2.2 BM25 ổn định hơn
- Thêm cơ chế tự đảm bảo BM25 bằng `_ensure_bm25_retriever()` tại [backend/rag_pipeline.py](/home/catouis/Project-Qwen-Rag-Agent/backend/rag_pipeline.py:160).
- Khi BM25 chưa sẵn sàng, hệ thống không throw lỗi nữa mà fallback vector trong `get_retriever()` tại [backend/rag_pipeline.py](/home/catouis/Project-Qwen-Rag-Agent/backend/rag_pipeline.py:218).

### 2.3 Reranker compatibility
- Load Cross-Encoder đã có fallback khi môi trường không hỗ trợ `torch_dtype` tại [backend/rag_pipeline.py](/home/catouis/Project-Qwen-Rag-Agent/backend/rag_pipeline.py:100).

### 2.4 Advanced toggle ở UI (code-level)
- Toggle có trong code tại [frontend/ui.py](/home/catouis/Project-Qwen-Rag-Agent/frontend/ui.py:325).
- Kết quả `confidence` có hiển thị nếu trả về hợp lệ tại [frontend/ui.py](/home/catouis/Project-Qwen-Rag-Agent/frontend/ui.py:291).
- Tuy nhiên runtime bạn test chưa thấy toggle, nên trạng thái thực tế vẫn là “chưa usable trên UI”.

## 3) Trạng thái 10 câu hỏi phần 8

| Câu | Nội dung | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | Hỗ trợ DOCX | Đã có một phần | Backend loader có, UI upload vẫn thiên về PDF |
| 2 | Lưu lịch sử hội thoại | Đã có | Session + backend memory + save JSON |
| 3 | Nút xóa history/vector | Đã có một phần | Có nút xóa, chưa có confirmation rõ ràng |
| 4 | Chunk strategy | Đã có một phần | Có slider, thiếu benchmark/report hệ thống |
| 5 | Citation/source tracking | Đã có một phần | Có snippet/citation, chưa click-context/highlight thật |
| 6 | Conversational RAG | Đã có | History-aware retriever + memory theo session |
| 7 | Hybrid search | Đã có một phần | Core đã có; UI chưa expose mode/benchmark |
| 8 | Multi-doc + metadata filtering | Đã có một phần | Backend hỗ trợ nhiều phần, UI chưa hoàn thiện |
| 9 | Re-ranking Cross-Encoder | Đã có một phần | Core có + fallback compatibility, UI chưa có control rõ |
| 10 | Self-RAG / Advanced | Đã có một phần | Backend advanced đã có, UI toggle runtime chưa thấy |

## 4) Phân tích riêng cho Advanced mode (vì đang vướng UI)

### Code đã có
- `st.toggle("Advanced RAG (self-check + confidence)")` trong form hỏi đáp.
- `advanced_mode` được truyền qua controller xuống pipeline.

### Runtime thực tế
- Bạn đã test web và chưa thấy toggle.
- Điều này cho thấy có độ lệch giữa “code hiện tại” và “UI render thực tế”.

### Giả thuyết kỹ thuật hợp lý
1. UI đang chạy chưa reload đúng phiên bản file mới.
2. Toggle trong `st.form` không render đúng trong layout/theme hiện tại.
3. CSS/DOM behavior đang ảnh hưởng widget toggle (dù chưa thấy rule ẩn trực tiếp).

### Kết luận tạm thời
- Advanced mode: `backend-ready`, `ui-not-confirmed`.

## 5) Việc nên làm tiếp (ưu tiên ngắn hạn)

1. Ưu tiên xác nhận toggle ở UI trước.
   - Nếu cần, chuyển toggle ra sidebar hoặc ra ngoài `st.form` để loại trừ lỗi render.
2. Sau khi thấy toggle, test 5-10 câu follow-up thật trên web.
   - Ghi lại `confidence`, `used_retry`, độ ổn định câu trả lời.
3. Chốt lại status Câu 10.
   - Khi toggle dùng được và confidence hiển thị ổn định, có thể nâng Câu 10 lên mức gần hoàn thành.
4. Dọn nợ UI cho Câu 7-8-9.
   - Expose retriever mode, metadata filter, và benchmark output cho người dùng.

## 6) Kết luận ngắn

Dự án đã tiến xa ở backend (hybrid, reranker, advanced pipeline, cơ chế self-healing BM25), và web flow chính đã chạy được. Điểm nghẽn còn lại để “chốt điểm phần 8” là hoàn thiện mặt hiển thị/điều khiển trên UI, đặc biệt là xác nhận và sử dụng được toggle `Advanced RAG` ngay trong giao diện.
