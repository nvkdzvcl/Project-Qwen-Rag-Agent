"""
Giao diện Streamlit SmartDoc AI.

Tổ chức: cấu hình trang → state → CSS (styles.py) → sidebar → từng vùng nội dung.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import streamlit as st

from frontend.constants import MAX_PDF_BYTES
from frontend.styles import inject_global_styles

# --- Cấu hình trang (gọi một lần khi import) ---
st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # type: List[Dict[str, str]]
    if "vector_ready" not in st.session_state:
        st.session_state.vector_ready = False
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_citations" not in st.session_state:
        st.session_state.last_citations = []  # type: List[Dict[str, str]]
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False


def _demo_process_document() -> None:
    progress_ph = st.empty()
    status_ph = st.empty()
    status_ph.info("Đang phân tích tài liệu và tạo vector store…")
    bar = progress_ph.progress(0)
    for i in range(0, 101, 12):
        time.sleep(0.05)
        bar.progress(min(i, 100))
    st.session_state.vector_ready = True
    st.session_state.processing_done = True
    status_ph.success("Xử lý xong. Bạn có thể đặt câu hỏi về nội dung.")


def _demo_answer(question: str) -> Dict[str, Any]:
    short = question.strip()[:140] if question.strip() else "câu hỏi"
    return {
        "answer": (
            f"[Demo] Trả lời mẫu cho: «{short}». "
            "Khi kết nối backend RAG + Ollama, đây sẽ là câu trả lời thật từ PDF."
        ),
        "citations": [
            {
                "page": "1",
                "location": "Đoạn trích",
                "snippet": "Ví dụ trích dẫn — kết nối RAG để hiển thị trang và đoạn thật.",
            },
        ],
    }


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## SmartDoc AI")
        st.caption("Hỏi đáp tài liệu thông minh (RAG + Qwen)")

        st.markdown("### Hướng dẫn")
        st.markdown("1. Tải lên PDF (tối đa 50MB)")
        st.markdown("2. Bấm **Phân tích tài liệu**")
        st.markdown("3. Đặt câu hỏi và xem trích dẫn")

        st.markdown("### Cấu hình")
        st.selectbox("Mô hình LLM", ["Qwen2.5:7b", "Qwen2.5:14b"], index=0)
        st.slider("Chunk size", min_value=500, max_value=3000, value=500, step=100, key="cfg_chunk")
        st.slider("Chunk overlap", min_value=50, max_value=300, value=50, step=10, key="cfg_overlap")
        st.slider("Top-k", min_value=1, max_value=10, value=3, step=1, key="cfg_topk")

        st.markdown("### Lịch sử (gần đây)")
        hist: List[Dict[str, str]] = st.session_state.chat_history
        if not hist:
            st.caption("Chưa có hội thoại.")
        else:
            for i, item in enumerate(reversed(hist[-8:]), start=1):
                q = item.get("question", "")[:56]
                st.markdown(f"**{i}.** {q}…")

        if st.button("Xóa lịch sử", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_answer = ""
            st.session_state.last_citations = []
            st.success("Đã xóa lịch sử.")

        if st.button("Xóa vector / tài liệu", use_container_width=True):
            st.session_state.vector_ready = False
            st.session_state.processing_done = False
            st.session_state.uploaded_filename = ""
            st.warning("Đã xóa trạng thái tài liệu (demo).")


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
                <div class="workflow-step">
                    <h4>Tải tài liệu</h4>
                    <p>nhiều loại file, tối đa 50MB</p>
                </div>
                <div class="workflow-step-arrow">→</div>
                <div class="workflow-step">
                    <h4>Phân tích</h4>
                    <p>Vector hóa nội dung</p>
                </div>
                <div class="workflow-step-arrow">→</div>
                <div class="workflow-step">
                    <h4>Truy vấn</h4>
                    <p>Đặt câu hỏi</p>
                </div>
                <div class="workflow-step-arrow">→</div>
                <div class="workflow-step">
                    <h4>Trích dẫn</h4>
                    <p>Nguồn trong PDF</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_upload_section() -> None:
   

    with st.container():
        st.markdown(
            """
            <div class="upload-banner">
                <div class="upload-banner-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="12" y1="19" x2="12" y2="5"/>
                        <polyline points="5 12 12 5 19 12"/>
                    </svg>
                </div>
                <div class="upload-banner-title">Tải lên tài liệu</div>
                <div class="upload-banner-desc">
                     <b></b> Sau khi tải, bấm nút phân tích để chuẩn bị hỏi đáp.
                </div>
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
        if size and size > MAX_PDF_BYTES:
            st.error(f"File vượt quá {MAX_PDF_BYTES // (1024 * 1024)} MB. Chọn file nhỏ hơn.")
            return

        st.session_state.uploaded_filename = uploaded_file.name
        st.success(f"Đã chọn: **{uploaded_file.name}**")

        if not st.session_state.vector_ready:
            if st.button(
                "Bắt đầu phân tích tài liệu",
                type="secondary",
                use_container_width=True,
                key="btn_process_doc",
            ):
                _demo_process_document()
        else:
            st.success(f"Sẵn sàng hỏi đáp: {uploaded_file.name}")


def render_query_section() -> None:
    st.markdown('<p class="section-title">Câu hỏi</p>', unsafe_allow_html=True)
    question = st.text_input(
        "Nội dung câu hỏi",
        placeholder="Ví dụ: Tóm tắt mục tiêu chính của tài liệu?",
        label_visibility="collapsed",
        key="user_question",
    )
    can_ask = st.session_state.vector_ready and bool(question and question.strip())
    if st.button("Gửi câu hỏi", type="primary", disabled=not can_ask, use_container_width=False):
        with st.spinner("Đang suy luận…"):
            time.sleep(0.35)
            out = _demo_answer(question)
            st.session_state.last_answer = str(out["answer"])
            st.session_state.last_citations = list(out["citations"])
            st.session_state.chat_history.append(
                {"question": question.strip(), "answer": st.session_state.last_answer}
            )
        st.success("Đã nhận phản hồi.")

    if not st.session_state.vector_ready:
        st.caption("Cần phân tích tài liệu trước khi gửi câu hỏi.")


def render_response_section() -> None:
    st.markdown('<p class="section-title">Câu trả lời</p>', unsafe_allow_html=True)
    if st.session_state.last_answer:
        st.write(st.session_state.last_answer)
    else:
        st.info("Câu trả lời sẽ hiển thị tại đây.")

    st.markdown('<p class="section-title">Trích dẫn</p>', unsafe_allow_html=True)
    cites: List[Dict[str, str]] = st.session_state.last_citations
    if cites:
        for c in cites:
            st.markdown(
                f"""
                <div class="citation-item">
                    <strong>Trang {c.get("page", "?")}</strong> — {c.get("location", "")}<br/>
                    <span>{c.get("snippet", "")}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("Chưa có trích dẫn. Gửi câu hỏi sau khi đã phân tích tài liệu.")


def main() -> None:
    init_state()
    inject_global_styles()
    render_sidebar()
    render_page_header()
    render_workflow_strip()
    render_upload_section()
    render_query_section()
    render_response_section()
