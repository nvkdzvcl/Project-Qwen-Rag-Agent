import streamlit as st
import os
import time

# CẤU HÌNH TRANG 
st.set_page_config(page_title="SmartDoc AI", layout="wide")

# MOCK CONTROLLER (GIẢ LẬP ĐỂ CHỈ HIỆN GIAO DIỆN) 
class MockRAGController:
    def process_new_document(self, documents):
        # Giả lập thời gian xử lý file để hiện Progress Bar
        self.mock_documents = documents
        return True

    def answer_question(self, question):
        # Trả về kết quả mẫu để test UI
        return {
            "answer": f"Đây là câu trả lời mẫu cho câu hỏi: '{question}'. Hiện tại hệ thống đang chạy ở chế độ 'Chỉ Giao diện' (UI Only).",
            "context": [
                type('obj', (object,), {
                    'page_content': 'Đoạn văn bản mẫu được trích xuất từ tài liệu để kiểm tra hiển thị nguồn trích dẫn.',
                    'metadata': {'page': 1, 'source': 'tai_lieu_mau.pdf'}
                })
            ]
        }


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; }

    [data-testid="stSidebar"] { background-color: #2C2F33; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #FFC107 !important; }

    .stButton > button {
        background-color: #007BFF;
        color: #FFFFFF;
        border: none;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover { background-color: #0056b3; color: #FFFFFF; }

    [data-testid="stFileUploader"] {
        border: 2px dashed #FFC107;
        border-radius: 8px;
        background-color: #fffdf0;
        padding: 1rem;
    }

    .main-header {
        background: linear-gradient(135deg, #007BFF, #0056b3);
        color: #FFFFFF;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; color: #FFFFFF; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.88; font-size: 0.95rem; }

    .answer-box {
        background-color: #FFFFFF;
        border-left: 4px solid #007BFF;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        color: #212529;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }

    .citation-box {
        background-color: #f0f4ff;
        border: 1px solid #c8d8ff;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-top: 0.4rem;
        font-size: 0.83rem;
        color: #212529;
    }

    .history-item {
        background-color: #3a3f44;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.83rem;
    }
    .history-q { color: #FFC107 !important; font-weight: 600; }
    .history-a { color: #cccccc !important; }

    p, label, .stMarkdown { color: #212529; }
</style>
""", unsafe_allow_html=True)

if "rag_controller" not in st.session_state:
    # SỬA TẠI ĐÂY: Dùng Mock thay vì RAGController thật
    st.session_state.rag_controller = MockRAGController() 

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "app_logs" not in st.session_state:
    st.session_state.app_logs = []

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## SmartDoc AI")
    st.markdown("---")

    st.markdown("### Huong dan su dung")
    st.markdown("""
1. **Tai len** tai lieu PDF
2. **Cho** he thong xu ly tai lieu
3. **Dat cau hoi** ve noi dung
""")
    st.markdown("---")

    st.markdown("### Cau hinh he thong")
    st.markdown("""
- **LLM:** Qwen2.5:7b (Ollama)
- **Embedding:** MPNet 768-dim
- **Search:** FAISS + MMR
- **Ngon ngu:** Viet / Anh
""")
    st.markdown("---")

    st.markdown("### Tai lieu hien tai")
    if st.session_state.doc_processed and st.session_state.uploaded_filename:
        st.success(st.session_state.uploaded_filename)
    else:
        st.info("Chua co tai lieu nao duoc tai len.")
    st.markdown("---")

    st.markdown("### Lich su hoi thoai")
    if st.session_state.chat_history:
        for item in reversed(st.session_state.chat_history):
            q_short = item["question"][:60] + ("..." if len(item["question"]) > 60 else "")
            a_short = item["answer"][:80] + ("..." if len(item["answer"]) > 80 else "")
            st.markdown(f"""
<div class="history-item">
  <div class="history-q">Q: {q_short}</div>
  <div class="history-a">A: {a_short}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown("<small>Chua co cau hoi nao.</small>", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("Xem log he thong"):
        if st.session_state.app_logs:
            for log_line in st.session_state.app_logs[-12:]:
                st.write(log_line)
        else:
            st.write("Chua co log nao.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Xoa lich su", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("Xoa tai lieu", use_container_width=True):
            st.session_state.doc_processed = False
            st.session_state.uploaded_filename = None
            st.session_state.rag_controller = MockRAGController()
            st.rerun()

# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>SmartDoc AI</h1>
    <p>Intelligent Document Q&A System — Ho tro tieng Viet va tieng Anh</p>
</div>
""", unsafe_allow_html=True)

# ── SECTION 1: UPLOAD ─────────────────────────────────────────────────────────
st.markdown("### Tai tai lieu len")

uploaded_file = st.file_uploader(
    "Keo tha hoac chon file PDF",
    type=["pdf"],
    help="Chi ho tro dinh dang PDF. Kich thuoc khuyen nghi duoi 50MB."
)

if uploaded_file is not None:
    if uploaded_file.name != st.session_state.uploaded_filename:
        st.info(f"Da chon: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        if st.button("Xu ly tai lieu", type="primary"):
            progress = st.progress(0, text="Dang bat dau xu ly...")
            st.session_state.app_logs.append(f"[INFO] Bat dau xu ly file {uploaded_file.name}")
            try:
                progress.progress(15, text="Dang doc file...")
                st.session_state.app_logs.append("[INFO] Doc file PDF...")
                time.sleep(0.15)

                progress.progress(35, text="Dang trich xuat noi dung PDF (demo)...")
                st.session_state.app_logs.append("[INFO] Trich xuat noi dung (demo)...")
                time.sleep(0.15)

                progress.progress(55, text="Dang chia nho van ban (demo)...")
                st.session_state.app_logs.append("[INFO] Chia nho van ban (demo)...")
                time.sleep(0.15)

                progress.progress(75, text="Dang tao vector embeddings (demo)...")
                st.session_state.app_logs.append("[INFO] Tao embeddings (demo)...")
                time.sleep(0.15)

                documents = [
                    type('obj', (object,), {
                        'page_content': 'Noi dung demo tu file PDF',
                        'metadata': {'page': 1, 'source': uploaded_file.name}
                    })
                ]
                st.session_state.rag_controller.process_new_document(documents)

                progress.progress(100, text="Hoan tat!")
                st.session_state.app_logs.append("[INFO] Hoan tat xu ly tai lieu.")
                st.session_state.doc_processed = True
                st.session_state.uploaded_filename = uploaded_file.name

                st.success(f"Da xu ly **{uploaded_file.name}** thanh cong! ({len(documents)} chunks)")
                st.rerun()

            except Exception as e:
                progress.empty()
                st.session_state.app_logs.append(f"[ERROR] Loi khi xu ly tai lieu: {str(e)}")
                st.error(f"Loi khi xu ly tai lieu: {str(e)}")
                st.warning("Goi y: Kiem tra file PDF khong bi hong hoac duoc bao ve bang mat khau.")
    else:
        st.success(f"Tai lieu **{uploaded_file.name}** da san sang de hoi dap.")
else:
    if not st.session_state.doc_processed:
        st.markdown("""
<div style="text-align:center; padding:2rem; color:#6c757d;">
    <h3>Hay tai len mot tai lieu PDF de bat dau</h3>
    <p>He thong ho tro tieng Viet va tieng Anh</p>
</div>
""", unsafe_allow_html=True)

# ── SECTION 2: HOI DAP 
if st.session_state.doc_processed:
    st.markdown("---")
    st.markdown("### Dat cau hoi")

    question = st.text_input(
        "Nhap cau hoi cua ban",
        placeholder="Vi du: Noi dung chinh cua tai lieu la gi?",
        label_visibility="collapsed"
    )

    if question:
        with st.spinner("Dang xu ly cau hoi..."):
            try:
                answer = st.session_state.rag_controller.answer_question(question)

                # Luu vao lich su
                sources = []
                try:
                    # Lay sources neu controller tra ve dict
                    if isinstance(answer, dict):
                        sources = answer.get("context", [])
                        answer = answer.get("answer", "")
                except Exception:
                    pass

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })

                # Hien thi cau tra loi
                st.markdown("### Cau tra loi")
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

                # Hien thi citation neu co
                if sources:
                    with st.expander("Xem nguon trich dan"):
                        for i, doc in enumerate(sources, 1):
                            page = doc.metadata.get("page", "N/A")
                            source_file = doc.metadata.get("source", "tai lieu")
                            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                            st.markdown(f"""
<div class="citation-box">
  <strong>Nguon {i}</strong> — Trang: {page} | File: {os.path.basename(str(source_file))}<br>
  <em>{content_preview}</em>
</div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Loi xu ly cau hoi: {str(e)}")
                st.warning("Vui long kiem tra Ollama dang chay va thu lai.")

# ── SECTION 3: LICH SU DAY DU 
if st.session_state.chat_history:
    st.markdown("---")
    with st.expander("Xem toan bo lich su hoi thoai"):
        for i, item in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Cau hoi {i}:** {item['question']}")
            st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
            st.markdown("")
