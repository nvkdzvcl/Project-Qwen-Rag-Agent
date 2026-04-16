# SmartDoc AI (RAG + Qwen + Ollama)

SmartDoc AI là hệ thống hỏi đáp tài liệu bằng RAG, chạy local với Streamlit UI, FAISS/BM25 retrieval và LLM qua Ollama.

## Tính năng chính

- Upload và phân tích tài liệu PDF.
- Hỏi đáp theo ngữ cảnh tài liệu (Conversational RAG).
- Hybrid retrieval: FAISS + BM25 (có fallback an toàn khi BM25 chưa sẵn sàng).
- Re-ranking với Cross-Encoder.
- Trích dẫn nguồn trong câu trả lời.
- Chế độ `Advanced RAG` (self-check + confidence) trên UI.

## Yêu cầu môi trường

- Linux (khuyến nghị Ubuntu).
- Python `3.10+` (đã test với 3.12).
- Ollama đã cài và chạy local.
- Tối thiểu RAM khả dụng: 8GB (khuyến nghị 16GB nếu dùng model lớn).

## Cài đặt dự án

```bash
cd /home/catouis/Project-Qwen-Rag-Agent
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Cài và chạy Ollama

Ví dụ với snap:

```bash
sudo snap install ollama
sudo snap start ollama
```

Kiểm tra Ollama đang hoạt động:

```bash
curl http://127.0.0.1:11434/api/tags
```

Tải model:

```bash
ollama pull qwen2.5:7b
ollama list
```

## Chạy ứng dụng web

```bash
cd /home/catouis/Project-Qwen-Rag-Agent
source venv/bin/activate
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

## Luồng sử dụng cơ bản

1. Upload file PDF.
2. Bấm `Bắt đầu phân tích tài liệu`.
3. Đặt câu hỏi trong khung chat.
4. Bật `Advanced RAG (self-check + confidence)` ở sidebar nếu muốn dùng chế độ nâng cao.

## Ghi chú quan trọng

- Luôn chạy app trong `venv` của project để tránh lỗi thiếu package.
- Nên chạy Ollama cùng môi trường Linux hiện tại (tránh lệch host Windows/Linux).
- Nếu thấy lỗi BM25 thiếu dependency, kiểm tra đã cài `rank-bm25` qua `requirements.txt`.

## Troubleshooting nhanh

- `Connection refused localhost:11434`:
  - Ollama chưa chạy hoặc chạy sai môi trường.
  - Cách xử lý: kiểm tra `curl http://127.0.0.1:11434/api/tags`.

- `Remote end closed connection`:
  - Ollama đóng kết nối giữa chừng (thường do service/model chưa ổn định).
  - Cách xử lý: restart service Ollama, warm-up model bằng một request ngắn.

- `gio ... Operation not supported`:
  - Không nghiêm trọng, chỉ là hệ thống không tự mở browser.
  - Mở thủ công `http://localhost:8501`.

## Cấu trúc thư mục chính

- `frontend/`: Streamlit UI.
- `backend/`: RAG pipeline, controller, loader, splitter.
- `tests/`: script test và kiểm tra chức năng.
- `documentation/`: tài liệu báo cáo và trạng thái triển khai.
