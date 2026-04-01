import time
from typing import Dict, List

import streamlit as st


st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, str]] = []
    if "vector_ready" not in st.session_state:
        st.session_state.vector_ready = False
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = ""
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""
    if "last_citations" not in st.session_state:
        st.session_state.last_citations: List[Dict[str, str]] = []
    if "processing_done" not in st.session_state:
        st.session_state.processing_done = False


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --primary: #007BFF;
                --secondary: #FFC107;
                --bg-main: #F8F9FA;
                --bg-sidebar: #2C2F33;
                --text-main: #212529;
                --text-light: #FFFFFF;
            }

            .stApp {
                background-color: var(--bg-main);
                color: var(--text-main);
            }

            [data-testid="stSidebar"] {
                background-color: var(--bg-sidebar) !important;
            }

            [data-testid="stSidebar"] * {
                color: var(--text-light) !important;
            }

            [data-testid="stSidebar"] .stButton > button {
                border: 1px solid rgba(255, 255, 255, 0.25);
            }

            h1, h2, h3, h4, h5, h6 {
                color: var(--text-main);
            }

            .main-header {
                margin-top: 0.5rem;
                color: #007BFF;
                margin-bottom: 1.5rem;
                border-bottom: 1px solid #e3e6ea;
                padding-bottom: 0.75rem;
            }

            .card {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
                padding: 1rem 1.1rem;
                margin-bottom: 0.9rem;
            }

            .step-chip {
                display: inline-block;
                font-size: 0.84rem;
                padding: 0.25rem 0.65rem;
                border-radius: 999px;
                border: 1px solid #ced4da;
                margin-right: 0.35rem;
                margin-bottom: 0.4rem;
                background: #fff;
            }

            .citation-item {
                background: #fffef4;
                border-left: 4px solid var(--secondary);
                padding: 0.5rem 0.75rem;
                margin-bottom: 0.45rem;
                border-radius: 0 8px 8px 0;
                color: #4a4a4a;
                font-size: 0.95rem;
            }

            .stButton > button[kind="primary"] {
                background-color: var(--primary) !important;
                border-color: var(--primary) !important;
                color: #fff !important;
            }

            div[data-testid="stFileUploaderDropzone"] {
                border: 2px dashed #ced4da;
                border-radius: 12px;
                background: #ffffff;
                padding: 0.8rem;
            }

            .upload-note {
                color: #6c757d;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fake_process_document() -> None:
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    status_placeholder.info("Dang phan tich tai lieu va tao vector store...")
    progress = progress_placeholder.progress(0)

    for i in range(1, 101, 10):
        time.sleep(0.08)
        progress.progress(i)

    st.session_state.vector_ready = True
    st.session_state.processing_done = True
    status_placeholder.success("Xu ly tai lieu thanh cong. San sang hoi dap.")


def fake_answer(question: str) -> Dict[str, object]:
    short = question.strip()[:140] if question.strip() else "Noi dung cau hoi"
    answer = (
        f"Day la cau tra loi mau cho: '{short}'. "
        "Phan nay se duoc thay bang ket qua RAG + LLM khi ban ket noi backend."
    )
    citations = [
        {"page": "3", "location": "Doan 2", "snippet": "Noi dung lien quan den chu de trong cau hoi."},
        {"page": "7", "location": "Bang 1", "snippet": "Du lieu ho tro cho ket luan trong cau tra loi."},
    ]
    return {"answer": answer, "citations": citations}


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## SmartDoc AI")
        st.caption("Hỏi đáp tài liệu thông minh")

        
        st.markdown("### Hướng Dẫn ")
        st.markdown("1. Tải lên file PDF")
        st.markdown("2. Đợi hệ thống xử lý")
        st.markdown("3. Đặt câu hỏi và phản hồi")

        st.markdown("### Cấu hình hệ thống")
        model_name = st.selectbox("Mô hình LLM", ["Qwen2.5:7b", "Qwen2.5:14b"], index=0)
        chunk_size = st.slider("Chunk Size", min_value=500, max_value=3000, value=500, step=500)
        chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=300, value=50, step=50)
        top_k = st.slider("Top-k", min_value=1, max_value=10, value=3, step=1)

        st.caption(
            f"cấu hình hiện tại: {model_name} | chunk={chunk_size} | overlap={chunk_overlap} | top_k={top_k}"
        )

        
        st.markdown("### lịch sử hội thoại")
        if not st.session_state.chat_history:
            st.caption("Chưa có hội thoại")
        else:
            for i, item in enumerate(reversed(st.session_state.chat_history[-8:]), start=1):
                st.markdown(f"**{i}.** {item['question'][:60]}...")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Xóa lịch sử", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.last_answer = ""
                st.session_state.last_citations = []
                st.success("Đã xóa lịch sử")
        with col2:
            if st.button("Xóa Vector Store", use_container_width=True):
                st.session_state.vector_ready = False
                st.session_state.processing_done = False
                st.session_state.uploaded_filename = ""
                st.warning("Đã xóa tài liệu và vector store")


def render_main() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1 style="margin-bottom: 0.25rem;">SmartDoc AI</h1>
            <p style="margin: 0; color: #6c757d;">Hoi dap tai lieu thong minh voi RAG + Qwen2.5</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin-bottom: 0.8rem;">
            <span class="step-chip">Landing</span>
            <span class="step-chip">Upload</span>
            <span class="step-chip">Processing</span>
            <span class="step-chip">Query</span>
            <span class="step-chip">Response</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Tai len tai lieu PDF")
    uploaded_file = st.file_uploader(
        "Keo tha file PDF vao day hoac bam de chon file",
        type=["pdf"],
        help="Ho tro dinh dang PDF, toi da 50MB.",
    )
    st.markdown('<p class="upload-note">Ho tro keo-tha, hiem thi tien trinh va thong bao loi than thien.</p>', unsafe_allow_html=True)

    if uploaded_file is not None:
        st.session_state.uploaded_filename = uploaded_file.name
        st.success(f"Da tai len: {uploaded_file.name}")
        if st.button("Bat dau xu ly tai lieu", type="primary"):
            fake_process_document()
    else:
        st.info("Vui long tai len file PDF de bat dau.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Nhap cau hoi")
    question = st.text_input(
        "Dat cau hoi ve noi dung tai lieu",
        placeholder="Vi du: Tom tat muc tieu chinh cua tai lieu la gi?",
    )
    ask_disabled = not st.session_state.vector_ready or not question.strip()
    if st.button("Gui cau hoi", type="primary", disabled=ask_disabled):
        with st.spinner("Mo hinh dang suy luan, vui long doi..."):
            time.sleep(0.8)
            response = fake_answer(question)
            st.session_state.last_answer = str(response["answer"])
            st.session_state.last_citations = list(response["citations"])
            st.session_state.chat_history.append(
                {"question": question.strip(), "answer": st.session_state.last_answer}
            )
        st.success("Da nhan duoc cau tra loi.")

    if not st.session_state.vector_ready:
        st.caption("Can xu ly tai lieu truoc khi dat cau hoi.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Cau tra loi")
    if st.session_state.last_answer:
        st.write(st.session_state.last_answer)
    else:
        st.info("Cau tra loi se hien thi o day sau khi ban gui cau hoi.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Trich dan (Citation)")
    if st.session_state.last_citations:
        for c in st.session_state.last_citations:
            st.markdown(
                f"""
                <div class="citation-item">
                    <strong>Trang {c["page"]}</strong> - {c["location"]}<br/>
                    <span>{c["snippet"]}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("Chua co citation. Hay gui cau hoi de xem vi tri thong tin trong PDF.")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    init_state()
    inject_styles()
    render_sidebar()
    render_main()
