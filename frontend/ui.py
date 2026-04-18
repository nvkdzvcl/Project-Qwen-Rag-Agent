"""
Giao diện Streamlit SmartDoc AI.

Tổ chức: cấu hình trang → state → CSS (styles.py) → sidebar → từng vùng nội dung.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List

import pdfplumber
import streamlit as st
from langchain_core.documents import Document

from backend.controller import RAGController
from backend.splitter import SmartDocSplitter
from frontend.constants import MAX_PDF_BYTES
from frontend.styles import inject_global_styles

# Avatar images
_BASE = Path(__file__).parent.parent
AVATAR_USER = str(_BASE / "images" / "user.jpg")
AVATAR_AI   = str(_BASE / "images" / "AI.jpg")

# Local Ollama tags and their user-facing labels.
MODEL_LABELS = {
    "qwen2.5:3b": "qwen2.5:3b (it RAM, uu tien may yeu)",
    "qwen2.5:7b": "qwen2.5:7b (Q4_K_M - khuyen nghi)",
}
MODEL_OPTIONS = list(MODEL_LABELS.keys())

# --- Cấu hình trang được gọi trong main() ---


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "rag_controller" not in st.session_state:
        try:
            with st.spinner("Đang khởi tạo hệ thống AI (lần đầu có thể mất vài phút)…"):
                st.session_state.rag_controller = RAGController()
        except Exception as e:
            st.error(f"Lỗi khởi tạo hệ thống: {e}")
            st.session_state.rag_controller = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "vector_ready" not in st.session_state:
        ctrl = st.session_state.get("rag_controller")
        try:
            st.session_state.vector_ready = (
                ctrl is not None
                and ctrl.pipeline is not None
                and ctrl.pipeline.vector_store is not None
            )
        except Exception:
            st.session_state.vector_ready = False

    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False
    if "cfg_chunk" not in st.session_state:
        st.session_state.cfg_chunk = 1000
    if "cfg_overlap" not in st.session_state:
        st.session_state.cfg_overlap = 200
    if "cfg_topk" not in st.session_state:
        st.session_state.cfg_topk = 3
    if "cfg_model" not in st.session_state:
        st.session_state.cfg_model = "qwen2.5:3b"
    if "cfg_advanced_mode" not in st.session_state:
        st.session_state.cfg_advanced_mode = False
    if "cfg_filter_filename" not in st.session_state:
        st.session_state.cfg_filter_filename = ""
    if "confirm_clear_history" not in st.session_state:
        st.session_state.confirm_clear_history = False
    if "confirm_clear_vector" not in st.session_state:
        st.session_state.confirm_clear_vector = False
    if "selected_chat_idx" not in st.session_state:
        st.session_state.selected_chat_idx = None


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
# DIALOGS — dùng session_state flag + render inline
# ---------------------------------------------------------------------------

def _render_confirm_dialogs() -> None:
    """Hiển thị modal confirm nếu cần — gọi ở ngoài sidebar."""

    if st.session_state.get("confirm_clear_history"):
        st.markdown(
            """
            <div class="modal-overlay">
                <div class="modal-box">
                    <div class="modal-title"> Xóa lịch sử chat?</div>
                    <div class="modal-body">Toàn bộ lịch sử hội thoại sẽ bị xóa vĩnh viễn.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([3, 1, 1])
        with col2:
            if st.button("Xóa", type="primary", key="dlg_yes_hist", use_container_width=True):
                st.session_state.chat_history = []
                ctrl = st.session_state.get("rag_controller")
                if ctrl:
                    ctrl.clear_chat_history(session_id=st.session_state.session_id)
                st.session_state.confirm_clear_history = False
                st.rerun()
        with col3:
            if st.button("Huỷ", key="dlg_no_hist", use_container_width=True):
                st.session_state.confirm_clear_history = False
                st.rerun()

    if st.session_state.get("confirm_clear_vector"):
        st.markdown(
            """
            <div class="modal-overlay">
                <div class="modal-box">
                    <div class="modal-title">Xóa toàn bộ tài liệu?</div>
                    <div class="modal-body">Vector store và dữ liệu tài liệu sẽ bị xóa vĩnh viễn.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([3, 1, 1])
        with col2:
            if st.button("Xóa", type="primary", key="dlg_yes_vec", use_container_width=True):
                ctrl = st.session_state.get("rag_controller")
                if ctrl:
                    ctrl.clear_vector_store()
                st.session_state.vector_ready = False
                st.session_state.processing_done = False
                st.session_state.uploaded_filename = ""
                st.session_state.confirm_clear_vector = False
                st.rerun()
        with col3:
            if st.button("Huỷ", key="dlg_no_vec", use_container_width=True):
                st.session_state.confirm_clear_vector = False
                st.rerun()


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown('''
        <style>
        [data-testid="stSidebar"] {
            color: white !important;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        </style>
        ''', unsafe_allow_html=True)
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

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.number_input(
                "Chunk size", min_value=100, max_value=5000,
                value=st.session_state.cfg_chunk, step=100, key="cfg_chunk",
            )
        with col_c2:
            st.number_input(
                "Overlap", min_value=0, max_value=1000,
                value=st.session_state.cfg_overlap, step=10, key="cfg_overlap",
            )
        st.number_input(
            "Top-k", min_value=1, max_value=20,
            value=st.session_state.cfg_topk, step=1, key="cfg_topk",
        )
        st.markdown('<div style="color: white;">', unsafe_allow_html=True)
        st.checkbox(
            "Advanced RAG (self-check + confidence)",
            value=st.session_state.cfg_advanced_mode,
            key="cfg_advanced_mode",
            help="Bật để dùng luồng tự kiểm tra câu trả lời và trả về confidence score.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Badge trạng thái pipeline hiện tại
        adv = st.session_state.cfg_advanced_mode
        badge_color = "#28a745" if adv else "#007BFF"
        badge_text = "Advanced RAG ✓" if adv else "Standard RAG"
        st.markdown(
            f'<div style="background:{badge_color};color:#fff;border-radius:6px;'
            f'padding:4px 10px;font-size:0.78rem;text-align:center;margin-top:4px;">'
            f'{badge_text} · Hybrid+Reranker</div>',
            unsafe_allow_html=True,
        )

        # Metadata filter (Câu 8)
        st.markdown("### Lọc tài liệu (tuỳ chọn)")
        st.text_input(
            "Chỉ tìm trong file",
            placeholder="vd: chuong1.pdf",
            key="cfg_filter_filename",
            help="Để trống = tìm toàn bộ. Nhập tên file để lọc theo metadata.",
        )

        st.markdown("### Lịch sử (gần đây)")
        hist = st.session_state.chat_history
        if not hist:
            st.caption("Chưa có hội thoại.")
        else:
            # Hiển thị 8 câu gần nhất, index thật trong chat_history
            recent = list(enumerate(hist))[-8:]
            for real_idx, item in reversed(recent):
                q = item.get("question", "")[:50]
                label = f" {q}…" if len(item.get("question","")) > 50 else f" {q}"
                if st.button(label, key=f"hist_btn_{real_idx}", use_container_width=True):
                    st.session_state.selected_chat_idx = real_idx

        if st.button("Xóa lịch sử chat", use_container_width=True):
            st.session_state.confirm_clear_history = True

        if st.button("Xóa vector / tài liệu", use_container_width=True):
            st.session_state.confirm_clear_vector = True


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
    if st.session_state.get("rag_controller") is None:
        st.error("Hệ thống AI chưa khởi tạo được. Kiểm tra Ollama đang chạy và thử reload trang.")
        return
    with st.container():
        st.markdown(
            """
            <div class="upload-banner">
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
    if st.session_state.get("rag_controller") is None:
        st.error("Hệ thống AI chưa khởi tạo. Không thể hỏi đáp.")
        return
    st.markdown('<p class="section-title">Hỏi đáp tài liệu</p>', unsafe_allow_html=True)

    # Hiển thị lịch sử hội thoại
    selected = st.session_state.get("selected_chat_idx")
    chat_container = st.container(height=500)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                '<div class="answer-empty">Lịch sử hội thoại sẽ hiển thị tại đây.</div>',
                unsafe_allow_html=True,
            )
        for idx, msg in enumerate(st.session_state.chat_history):
            # Anchor để scroll đến
            st.markdown(f'<div id="chat-msg-{idx}"></div>', unsafe_allow_html=True)

            # Highlight nếu được chọn từ sidebar
            is_selected = (selected == idx)
            if is_selected:
                st.markdown('<div class="chat-highlight">', unsafe_allow_html=True)

            with st.chat_message("user", avatar=AVATAR_USER):
                st.write(msg["question"])
            with st.chat_message("assistant", avatar=AVATAR_AI):
                st.write(msg["answer"])
                confidence = msg.get("confidence")
                if isinstance(confidence, (int, float)):
                    st.caption(f"Confidence: {float(confidence):.2f}")
                cites = msg.get("citations", [])
                if cites:
                    with st.expander(f"{len(cites)} trích dẫn nguồn"):
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

            if is_selected:
                st.markdown('</div>', unsafe_allow_html=True)

        # Auto-scroll đến item được chọn
        if selected is not None:
            st.markdown(
                f"""<script>
                    const el = document.getElementById('chat-msg-{selected}');
                    if(el) el.scrollIntoView({{behavior:'smooth', block:'center'}});
                </script>""",
                unsafe_allow_html=True,
            )

    # Ô nhập câu hỏi — st.chat_input tự dính cứng dưới màn hình
    advanced_mode = bool(st.session_state.cfg_advanced_mode)
    mode_label = "Advanced RAG" if advanced_mode else "Standard RAG"

    if not st.session_state.vector_ready:
        st.markdown(
            '<p class="qa-hint"> Cần phân tích tài liệu trước khi đặt câu hỏi.</p>',
            unsafe_allow_html=True,
        )
        return

    question = st.chat_input(
        placeholder=f"Nhập câu hỏi… ({mode_label})",
    )
    if question and question.strip():
        with st.spinner("Đang suy luận…"):
            filter_filename = st.session_state.get("cfg_filter_filename", "").strip()
            filter_dict = {"file_name": filter_filename} if filter_filename else None
            result = st.session_state.rag_controller.answer_question(
                question=question.strip(),
                session_id=st.session_state.session_id,
                advanced_mode=advanced_mode,
                filter_dict=filter_dict,
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
    st.set_page_config(
        page_title="SmartDoc AI",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()
    inject_global_styles()
    render_sidebar()
    _render_confirm_dialogs()
    render_page_header()
    render_upload_section()
    render_chat_section()
