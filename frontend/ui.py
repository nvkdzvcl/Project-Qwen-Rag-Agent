"""
Giao diện Streamlit SmartDoc AI.

Phong cách #4 + #5:
- UI chat + quản lý tài liệu rõ ràng
- Backend local Ollama/LangChain giữ nguyên
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

_BASE = Path(__file__).parent.parent
_USER_AVATAR = _BASE / "images" / "user.jpg"
_AI_AVATAR = _BASE / "images" / "AI.jpg"

MODEL_LABELS = {
    "qwen2.5:3b": "qwen2.5:3b (nhẹ RAM, khuyến nghị)",
    "qwen2.5:7b": "qwen2.5:7b (chất lượng cao hơn, tốn RAM hơn)",
}
MODEL_OPTIONS = list(MODEL_LABELS.keys())


def _avatar_or_fallback(path: Path, fallback: str) -> str:
    return str(path) if path.exists() else fallback


AVATAR_USER = _avatar_or_fallback(_USER_AVATAR, "👤")
AVATAR_AI = _avatar_or_fallback(_AI_AVATAR, "🤖")


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "rag_controller" not in st.session_state:
        try:
            with st.spinner("Đang khởi tạo hệ thống AI..."):
                st.session_state.rag_controller = RAGController(default_model="qwen2.5:3b")
        except Exception as e:
            st.error(f"Lỗi khởi tạo hệ thống: {e}")
            st.session_state.rag_controller = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict] = []

    if "vector_ready" not in st.session_state:
        ctrl = st.session_state.get("rag_controller")
        st.session_state.vector_ready = bool(
            ctrl is not None
            and getattr(ctrl, "pipeline", None) is not None
            and getattr(ctrl.pipeline, "vector_store", None) is not None
        )

    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files: List[str] = []
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
    if "cfg_replace_dataset" not in st.session_state:
        st.session_state.cfg_replace_dataset = True


def _parse_and_split(uploaded_file) -> List[Document]:
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
    return [
        {
            "page": s.get("page", "?"),
            "location": s.get("file_name", ""),
            "snippet": s.get("content_snippet", ""),
        }
        for s in sources
    ]


def _get_controller() -> RAGController | None:
    return st.session_state.get("rag_controller")


def _process_document(uploaded_file) -> None:
    ctrl = _get_controller()
    if ctrl is None:
        st.error("Backend chưa sẵn sàng, không thể xử lý tài liệu.")
        return

    size = getattr(uploaded_file, "size", None) or 0
    if size > MAX_PDF_BYTES:
        st.error(f"File vượt quá {MAX_PDF_BYTES // (1024 * 1024)} MB.")
        return

    with st.spinner("Đang đọc và vector hóa tài liệu..."):
        chunks = _parse_and_split(uploaded_file)
        if not chunks:
            st.error("Không đọc được nội dung PDF.")
            return
        ok, msg = ctrl.process_new_document(
            chunks,
            clear_old=bool(st.session_state.cfg_replace_dataset),
        )

    if not ok:
        st.error(msg)
        return

    filename = uploaded_file.name
    if st.session_state.cfg_replace_dataset:
        st.session_state.uploaded_files = [filename]
    elif filename not in st.session_state.uploaded_files:
        st.session_state.uploaded_files.append(filename)

    st.session_state.uploaded_filename = filename
    st.session_state.vector_ready = True
    st.session_state.processing_done = True
    st.success(f"Phân tích xong {len(chunks)} chunks từ `{filename}`.")


def _clear_vector_store() -> None:
    ctrl = _get_controller()
    if ctrl is not None:
        ok, msg = ctrl.clear_vector_store()
        if not ok:
            st.error(msg)
            return

    st.session_state.vector_ready = False
    st.session_state.processing_done = False
    st.session_state.uploaded_filename = ""
    st.session_state.uploaded_files = []
    st.warning("Đã xóa toàn bộ vector/tài liệu.")


def _clear_chat_history() -> None:
    ctrl = _get_controller()
    st.session_state.chat_history = []
    if ctrl is not None:
        ctrl.clear_chat_history(session_id=st.session_state.session_id)
    st.info("Đã xóa lịch sử chat.")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="side-brand">
                <h2>SmartDoc AI</h2>
                <p>RAG local với Ollama + LangChain</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Tài liệu")
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
            help=f"PDF tối đa {MAX_PDF_BYTES // (1024 * 1024)} MB.",
        )

        st.checkbox(
            "Replace dataset khi upload",
            key="cfg_replace_dataset",
            help="Bật: xóa dữ liệu cũ trước khi nạp file mới.",
        )

        if uploaded_file is not None:
            st.caption(f"Đã chọn: `{uploaded_file.name}`")
            if st.button("Phân tích tài liệu", type="primary", use_container_width=True):
                _process_document(uploaded_file)

        if st.session_state.uploaded_files:
            st.markdown("#### Danh sách tài liệu")
            for doc_name in st.session_state.uploaded_files:
                st.markdown(
                    f"<div class='doc-chip'>{doc_name}</div>",
                    unsafe_allow_html=True,
                )
            st.caption("Xóa từng file chưa hỗ trợ; hiện chỉ xóa toàn bộ vector store.")
        else:
            st.caption("Chưa có tài liệu nào trong phiên hiện tại.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Xóa tài liệu", use_container_width=True):
                _clear_vector_store()
        with col_b:
            if st.button("Xóa chat", use_container_width=True):
                _clear_chat_history()

        st.markdown("---")
        st.markdown("### Cấu hình")
        if st.session_state.cfg_model not in MODEL_OPTIONS:
            st.session_state.cfg_model = MODEL_OPTIONS[0]

        model_choice = st.selectbox(
            "Model",
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state.cfg_model),
            format_func=lambda tag: MODEL_LABELS[tag],
        )
        if model_choice != st.session_state.cfg_model:
            st.session_state.cfg_model = model_choice
            ctrl = _get_controller()
            if ctrl is not None:
                ok = ctrl.setup_llm(model_name=model_choice)
                if ok:
                    st.success(f"Đã chuyển model: {model_choice}")
                else:
                    st.error("Không chuyển được model. Kiểm tra Ollama.")

        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Chunk", min_value=200, max_value=5000, step=100, key="cfg_chunk")
        with col2:
            st.number_input("Overlap", min_value=0, max_value=1000, step=10, key="cfg_overlap")

        st.slider("Top-k", min_value=1, max_value=20, key="cfg_topk")
        st.toggle("Advanced RAG", key="cfg_advanced_mode")
        st.text_input(
            "Lọc theo file",
            key="cfg_filter_filename",
            placeholder="vd: chuong1.pdf",
            help="Để trống để tìm trên toàn bộ dữ liệu.",
        )


def render_header() -> None:
    mode_label = "Advanced" if st.session_state.cfg_advanced_mode else "Standard"
    doc_count = len(st.session_state.uploaded_files)
    msg_count = len(st.session_state.chat_history)
    st.markdown(
        f"""
        <section class="hero-card">
            <p class="hero-kicker">Template #4 + #5</p>
            <h1>SmartDoc AI Workspace</h1>
            <p>Chat với tài liệu PDF local bằng Ollama, có bộ lọc file và trích dẫn nguồn.</p>
            <div class="hero-meta">
                <span>Mode: {mode_label}</span>
                <span>Model: {st.session_state.cfg_model}</span>
                <span>Docs: {doc_count}</span>
                <span>Messages: {msg_count}</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_chat() -> None:
    if _get_controller() is None:
        st.error("Không thể kết nối backend. Kiểm tra Ollama rồi reload trang.")
        return

    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-state">
                <h3>Sẵn sàng hỏi đáp</h3>
                <p>Upload tài liệu ở sidebar, bấm phân tích, sau đó hỏi ở ô chat phía dưới.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for msg in st.session_state.chat_history:
        with st.chat_message("user", avatar=AVATAR_USER):
            st.write(msg["question"])
        with st.chat_message("assistant", avatar=AVATAR_AI):
            st.write(msg["answer"])
            confidence = msg.get("confidence")
            if isinstance(confidence, (int, float)):
                st.caption(f"Confidence: {float(confidence):.2f}")
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"{len(citations)} trích dẫn"):
                    for i, c in enumerate(citations, start=1):
                        st.markdown(
                            f"""
                            <div class="citation-row">
                                <strong>[{i}]</strong> Trang {c.get("page", "?")} · {c.get("location", "")}<br/>
                                <span>{c.get("snippet", "")}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    if not st.session_state.vector_ready:
        st.info("Cần phân tích tài liệu trước khi hỏi.")
        return

    question = st.chat_input("Nhập câu hỏi về tài liệu...")
    if not question or not question.strip():
        return

    filter_filename = st.session_state.cfg_filter_filename.strip()
    filter_dict = {"file_name": filter_filename} if filter_filename else None

    with st.spinner("Đang suy luận..."):
        result = st.session_state.rag_controller.answer_question(
            question=question.strip(),
            session_id=st.session_state.session_id,
            filter_dict=filter_dict,
            advanced_mode=bool(st.session_state.cfg_advanced_mode),
        )

    answer = result.get("answer", "")
    citations = _sources_to_citations(result.get("sources", []))
    confidence = result.get("confidence")
    st.session_state.chat_history.append(
        {
            "question": question.strip(),
            "answer": answer,
            "citations": citations,
            "confidence": confidence if isinstance(confidence, (int, float)) else None,
        }
    )
    st.rerun()


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
    render_header()
    render_chat()
