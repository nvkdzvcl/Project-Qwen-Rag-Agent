"""
Giao diện Streamlit SmartDoc AI.

Tổ chức: cấu hình trang → state → CSS (styles.py) → sidebar → từng vùng nội dung.
"""

from __future__ import annotations

import uuid
from typing import Dict, List

import pdfplumber
import streamlit as st
from langchain_core.documents import Document

from backend.controller import RAGController
from backend.splitter import SmartDocSplitter
from frontend.constants import MAX_PDF_BYTES
from frontend.styles import inject_global_styles

# Local Ollama tags and their user-facing labels.
MODEL_LABELS = {
    "qwen2.5:3b": "qwen2.5:3b (it RAM, uu tien may yeu)",
    "qwen2.5:7b": "qwen2.5:7b (Q4_K_M - khuyen nghi)",
}
MODEL_OPTIONS = list(MODEL_LABELS.keys())

# --- Cấu hình trang (gọi một lần khi import) ---
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "rag_controller" not in st.session_state:
        with st.spinner("Đang khởi tạo hệ thống AI (lần đầu có thể mất vài phút)…"):
            st.session_state.rag_controller = RAGController()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict] = []
    if "vector_ready" not in st.session_state:
        # Nếu pipeline đã khôi phục FAISS từ disk, đánh dấu sẵn sàng luôn
        st.session_state.vector_ready = (
            st.session_state.rag_controller.pipeline.vector_store is not None
        )
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False

    # Giá trị mặc định cho slider (tránh KeyError khi parse_and_split chạy trước render)
    if "cfg_chunk" not in st.session_state:
        st.session_state.cfg_chunk = 1000
    if "cfg_overlap" not in st.session_state:
        st.session_state.cfg_overlap = 200
    if "cfg_topk" not in st.session_state:
        st.session_state.cfg_topk = 3
    if "cfg_model" not in st.session_state:
        st.session_state.cfg_model = "qwen2.5:7b"
    if "cfg_advanced_mode" not in st.session_state:
        st.session_state.cfg_advanced_mode = False


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _parse_and_split(uploaded_file) -> List[Document]:
    """Đọc PDF → list[Document] rồi split bằng SmartDocSplitter."""
    raw_docs: List[Document] = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                raw_docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name, "page": i},
                    )
                )

    if not raw_docs:
        return []

    splitter = SmartDocSplitter(
        chunk_size=st.session_state.cfg_chunk,
        chunk_overlap=st.session_state.cfg_overlap,
    )
    return splitter.split_documents(raw_docs)


def _sources_to_citations(sources: List[Dict]) -> List[Dict]:
    """Map cấu trúc sources từ backend sang citations cho UI."""
    return [
        {
            "page": s.get("page", "?"),
            "location": s.get("file_name", ""),
            "snippet": s.get("content_snippet", ""),
        }
        for s in sources
    ]


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## SmartDoc AI")
        st.caption("Hỏi đáp tài liệu thông minh (RAG + Qwen)")

        st.markdown("### Hướng dẫn")
        st.markdown("1. Tải lên PDF (tối đa 50 MB)")
        st.markdown("2. Bấm **Phân tích tài liệu**")
        st.markdown("3. Đặt câu hỏi và xem trích dẫn")

        st.markdown("### Cấu hình")
        if st.session_state.cfg_model not in MODEL_OPTIONS:
            st.session_state.cfg_model = MODEL_OPTIONS[0]

        model_choice = st.selectbox(
            "Mô hình LLM",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state.cfg_model),
            format_func=lambda model_tag: MODEL_LABELS.get(model_tag, model_tag),
            key="cfg_model_select",
        )
        # Swap model nếu người dùng đổi
        if model_choice != st.session_state.cfg_model:
            st.session_state.cfg_model = model_choice
            st.session_state.rag_controller.setup_llm(model_name=model_choice)
            st.success(f"Đã chuyển sang {model_choice}")

        st.slider("Chunk size", min_value=500, max_value=3000, value=st.session_state.cfg_chunk,
                  step=100, key="cfg_chunk")
        st.slider("Chunk overlap", min_value=50, max_value=300, value=st.session_state.cfg_overlap,
                  step=10, key="cfg_overlap")
        st.slider("Top-k", min_value=1, max_value=10, value=st.session_state.cfg_topk,
                  step=1, key="cfg_topk")
        st.checkbox(
            "Advanced RAG (self-check + confidence)",
            value=st.session_state.cfg_advanced_mode,
            key="cfg_advanced_mode",
            help="Bật để dùng luồng tự kiểm tra câu trả lời và trả về confidence score.",
        )

        st.markdown("### Lịch sử (gần đây)")
        hist = st.session_state.chat_history
        if not hist:
            st.caption("Chưa có hội thoại.")
        else:
            for i, item in enumerate(reversed(hist[-8:]), start=1):
                q = item.get("question", "")[:56]
                st.markdown(f"**{i}.** {q}…")

        if st.button("Xóa lịch sử chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.rag_controller.clear_chat_history(
                session_id=st.session_state.session_id
            )
            st.success("Đã xóa lịch sử.")

        if st.button("Xóa vector / tài liệu", use_container_width=True):
            ok, msg = st.session_state.rag_controller.clear_vector_store()
            st.session_state.vector_ready = False
            st.session_state.processing_done = False
            st.session_state.uploaded_filename = ""
            if ok:
                st.warning("Đã xóa toàn bộ dữ liệu tài liệu.")
            else:
                st.error(msg)


# ---------------------------------------------------------------------------
# HEADER & WORKFLOW
# ---------------------------------------------------------------------------

def render_page_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin-bottom: 0.35rem;">SmartDoc AI</h1>
            <p style="margin: 0; color: #6c757d;">Hỏi đáp tài liệu thông minh — RAG + Qwen2.5</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_strip() -> None:
    st.markdown(
        """
        <div class="feature-section">
            <div class="workflow-flow">
                <div class="workflow-step"><h4>Tải tài liệu</h4><p>PDF tối đa 50 MB</p></div>
                <div class="workflow-step-arrow">→</div>
                <div class="workflow-step"><h4>Phân tích</h4><p>Vector hóa nội dung</p></div>
                <div class="workflow-step-arrow">→</div>
                <div class="workflow-step"><h4>Truy vấn</h4><p>Đặt câu hỏi</p></div>
                <div class="workflow-step-arrow">→</div>
                <div class="workflow-step"><h4>Trích dẫn</h4><p>Nguồn trong PDF</p></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------------------------

def render_upload_section() -> None:
    with st.container():
        st.markdown(
            """
            <div class="upload-banner">
                <div class="upload-banner-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24"
                         fill="none" stroke="currentColor" stroke-width="2.2"
                         stroke-linecap="round" stroke-linejoin="round">
                        <line x1="12" y1="19" x2="12" y2="5"/>
                        <polyline points="5 12 12 5 19 12"/>
                    </svg>
                </div>
                <div class="upload-banner-title">Tải lên tài liệu</div>
                <div class="upload-banner-desc">Sau khi tải, bấm nút phân tích để chuẩn bị hỏi đáp.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Tải file PDF",
            type=["pdf"],
            help=f"PDF tối đa {MAX_PDF_BYTES // (1024 * 1024)} MB.",
            label_visibility="collapsed",
        )
        st.markdown(
            '<p class="upload-note">Kéo–thả hoặc chọn file. Hiển thị tiến trình khi phân tích.</p>',
            unsafe_allow_html=True,
        )

        if uploaded_file is None:
            st.info("Vui lòng chọn file PDF để bắt đầu.")
            return

        size = getattr(uploaded_file, "size", None) or 0
        if size > MAX_PDF_BYTES:
            st.error(f"File vượt quá {MAX_PDF_BYTES // (1024 * 1024)} MB. Chọn file nhỏ hơn.")
            return

        st.session_state.uploaded_filename = uploaded_file.name
        st.success(f"Đã chọn: **{uploaded_file.name}**")

        if st.session_state.vector_ready:
            st.success(f"Sẵn sàng hỏi đáp: {uploaded_file.name}")
            return

        if st.button("Bắt đầu phân tích tài liệu", type="secondary",
                     use_container_width=True, key="btn_process_doc"):
            with st.spinner("Đang đọc và vector hóa tài liệu…"):
                chunks = _parse_and_split(uploaded_file)
                if not chunks:
                    st.error("Không đọc được nội dung từ file PDF. Vui lòng thử file khác.")
                    return
                ok, msg = st.session_state.rag_controller.process_new_document(chunks)

            if ok:
                st.session_state.vector_ready = True
                st.session_state.processing_done = True
                st.success(f"Xử lý xong {len(chunks)} chunks. Bạn có thể đặt câu hỏi!")
            else:
                st.error(msg)


# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------

def render_chat_section() -> None:
    st.markdown('<p class="section-title">💬 Hỏi đáp tài liệu</p>', unsafe_allow_html=True)

    # Hiển thị lịch sử hội thoại
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                '<div class="answer-empty">Lịch sử hội thoại sẽ hiển thị tại đây.</div>',
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(msg["question"])
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["answer"])
                confidence = msg.get("confidence")
                if isinstance(confidence, (int, float)):
                    st.caption(f"Confidence: {float(confidence):.2f}")
                cites = msg.get("citations", [])
                if cites:
                    with st.expander(f"📎 {len(cites)} trích dẫn nguồn"):
                        citations_html = '<div class="citations-wrap">'
                        for i, c in enumerate(cites, start=1):
                            citations_html += f"""
                            <div class="citation-card">
                                <div class="citation-top">
                                    <span class="cite-index">{i}</span>
                                    <span class="cite-page">Trang {c.get('page', '?')}</span>
                                    <span class="cite-location">{c.get('location', '')}</span>
                                </div>
                                <blockquote class="cite-snippet">"{c.get('snippet', '')}"</blockquote>
                            </div>"""
                        citations_html += "</div>"
                        st.markdown(citations_html, unsafe_allow_html=True)

    # Form nhập câu hỏi
    if not st.session_state.vector_ready:
        st.markdown(
            '<p class="qa-hint">⚠️ Cần phân tích tài liệu trước khi đặt câu hỏi.</p>',
            unsafe_allow_html=True,
        )
        return

    with st.form(key="query_form", clear_on_submit=True):
        question = st.text_input(
            "Câu hỏi",
            placeholder="Ví dụ: Tóm tắt mục tiêu chính của tài liệu?",
            label_visibility="collapsed",
        )
        advanced_mode = bool(st.session_state.cfg_advanced_mode)
        st.caption(
            "Che do hien tai: Advanced RAG"
            if advanced_mode
            else "Che do hien tai: Standard RAG"
        )
        submitted = st.form_submit_button("🔍 Gửi câu hỏi", type="primary")

        if submitted and question.strip():
            with st.spinner("Đang suy luận…"):
                result = st.session_state.rag_controller.answer_question(
                    question=question.strip(),
                    session_id=st.session_state.session_id,
                    advanced_mode=bool(advanced_mode),
                )
            answer = result.get("answer", "")
            citations = _sources_to_citations(result.get("sources", []))
            confidence = result.get("confidence")
            st.session_state.chat_history.append({
                "question": question.strip(),
                "answer": answer,
                "citations": citations,
                "confidence": confidence if isinstance(confidence, (int, float)) else None,
            })
            st.rerun()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    init_state()
    inject_global_styles()
    render_sidebar()
    render_page_header()
    render_workflow_strip()
    render_upload_section()
    render_chat_section()
