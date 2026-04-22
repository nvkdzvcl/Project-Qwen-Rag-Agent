"""
Giao diện Streamlit SmartDoc AI.

Tổ chức: cấu hình trang → state → CSS (styles.py) → sidebar → từng vùng nội dung.
"""

from __future__ import annotations

import uuid
import tempfile
import os
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import streamlit as st
from langchain_core.documents import Document

from backend.controller import RAGController
from backend.loader import SmartDocLoader
from backend.splitter import SmartDocSplitter
from frontend.constants import MAX_PDF_BYTES
from frontend.styles import inject_global_styles

logger = logging.getLogger(__name__)

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

    # Quản lý nhiều cuộc hội thoại
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}  # {conversation_id: {name, messages, created_at, uploaded_files}}
    
    if "active_conversation_id" not in st.session_state:
        # Tạo cuộc hội thoại đầu tiên
        first_conv_id = str(uuid.uuid4())
        st.session_state.conversations[first_conv_id] = {
            "name": "Hội thoại 1",
            "messages": [],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "uploaded_files": []  # Danh sách file đã upload trong cuộc hội thoại này
        }
        st.session_state.active_conversation_id = first_conv_id
    
    # Backward compatibility - migrate old chat_history if exists
    if "chat_history" in st.session_state and st.session_state.chat_history:
        active_id = st.session_state.active_conversation_id
        if active_id in st.session_state.conversations:
            st.session_state.conversations[active_id]["messages"] = st.session_state.chat_history
            # Thêm uploaded_files nếu chưa có
            if "uploaded_files" not in st.session_state.conversations[active_id]:
                st.session_state.conversations[active_id]["uploaded_files"] = []
        del st.session_state.chat_history

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
    if "confirm_clear_vector" not in st.session_state:
        st.session_state.confirm_clear_vector = False
    if "selected_chat_idx" not in st.session_state:
        st.session_state.selected_chat_idx = None
    
    if "conversation_counter" not in st.session_state:
        st.session_state.conversation_counter = 1


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _parse_and_split(uploaded_file) -> List[Document]:
    """Đọc file (PDF/DOCX/DOC/TXT) → list[Document] rồi split bằng SmartDocSplitter."""
    try:
        # Lưu file tạm để loader có thể đọc
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Xác định loại file
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        raw_docs: List[Document] = []
        
        if file_ext in ['pdf', 'docx']:
            # Dùng SmartDocLoader cho PDF và DOCX (hỗ trợ OCR, table extraction)
            loader = SmartDocLoader()
            raw_docs = loader.load(tmp_path, doc_type="general")
            logger.info(f"✅ Đã load {len(raw_docs)} pages/sections từ {uploaded_file.name} bằng SmartDocLoader")
        
        elif file_ext == 'txt':
            # Đọc text thuần
            with open(tmp_path, 'r', encoding='utf-8') as f:
                text = f.read()
            if text.strip():
                raw_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "file_name": uploaded_file.name,
                            "page": 1,
                            "doc_type": "txt"
                        }
                    )
                )
            logger.info(f"✅ Đã load file TXT: {uploaded_file.name}")
        
        else:
            # Fallback cho các định dạng khác
            logger.warning(f"⚠️ Định dạng file {file_ext} chưa được hỗ trợ đầy đủ, thử đọc như text")
            with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            if text.strip():
                raw_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "file_name": uploaded_file.name,
                            "page": 1,
                            "doc_type": file_ext
                        }
                    )
                )
        
        # Xóa file tạm
        os.unlink(tmp_path)
        
        if not raw_docs:
            return []

        # Split documents thành chunks
        splitter = SmartDocSplitter(
            chunk_size=st.session_state.cfg_chunk,
            chunk_overlap=st.session_state.cfg_overlap,
        )
        return splitter.split_documents(raw_docs)
    
    except Exception as e:
        logger.error(f"❌ Lỗi khi xử lý file {uploaded_file.name}: {str(e)}")
        raise e


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
        st.markdown("1. Tải lên tài liệu (PDF/DOCX/DOC/TXT, tối đa 50 MB)")
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

        # Nút tạo cuộc hội thoại mới
        st.markdown("### Cuộc hội thoại")
        if st.button("+ Hội thoại mới", use_container_width=True, type="primary"):
            new_conv_id = str(uuid.uuid4())
            st.session_state.conversation_counter += 1
            st.session_state.conversations[new_conv_id] = {
                "name": f"Hội thoại {st.session_state.conversation_counter}",
                "messages": [],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "uploaded_files": []
            }
            st.session_state.active_conversation_id = new_conv_id
            st.session_state.selected_chat_idx = None
            st.rerun()
        
        st.markdown("---")
        
        # Hiển thị danh sách các cuộc hội thoại
        active_id = st.session_state.active_conversation_id
        conversations = st.session_state.conversations
        
        # Sắp xếp theo thời gian tạo (mới nhất trước)
        sorted_convs = sorted(
            conversations.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )
        
        if not sorted_convs:
            st.caption("Chưa có hội thoại nào.")
        else:
            for conv_id, conv_data in sorted_convs:
                msg_count = len(conv_data["messages"])
                conv_name = conv_data["name"]
                file_count = len(conv_data.get("uploaded_files", []))
                
                # Highlight cuộc hội thoại đang active
                is_active = (conv_id == active_id)
                button_type = "secondary" if is_active else "tertiary"
                
                # Tạo label với số tin nhắn và số file
                label = f"{conv_name} ({msg_count} tin, {file_count} file)"
                if is_active:
                    label = f"• {label}"
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(label, key=f"conv_{conv_id}", use_container_width=True, type=button_type):
                        if conv_id != active_id:
                            st.session_state.active_conversation_id = conv_id
                            st.session_state.selected_chat_idx = None
                            st.rerun()
                
                with col2:
                    # Nút xóa cuộc hội thoại (không cho xóa nếu chỉ còn 1)
                    if len(conversations) > 1:
                        if st.button("X", key=f"del_{conv_id}", help="Xóa cuộc hội thoại này"):
                            del st.session_state.conversations[conv_id]
                            # Nếu xóa cuộc hội thoại đang active, chuyển sang cuộc khác
                            if conv_id == active_id:
                                st.session_state.active_conversation_id = list(conversations.keys())[0]
                            st.rerun()
        
        st.markdown("---")
        
        # Hiển thị danh sách file của cuộc hội thoại hiện tại
        active_conv = conversations.get(active_id, {})
        uploaded_files_list = active_conv.get("uploaded_files", [])
        
        if uploaded_files_list:
            st.markdown("### Tài liệu trong cuộc hội thoại này")
            for file_info in uploaded_files_list:
                st.caption(f"• {file_info['name']} ({file_info['size']})")
        
        st.markdown("---")

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
                <div class="upload-banner-desc">Hỗ trợ PDF, DOCX, DOC, TXT. Sau khi tải, bấm nút phân tích để chuẩn bị hỏi đáp.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Tải file tài liệu",
            type=["pdf", "docx", "doc", "txt"],
            help=f"Hỗ trợ PDF, DOCX, DOC, TXT. Tối đa {MAX_PDF_BYTES // (1024 * 1024)} MB mỗi file.",
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        st.markdown(
            '<p class="upload-note">Kéo–thả hoặc chọn nhiều file PDF, DOCX, DOC, TXT. Hiển thị tiến trình khi phân tích.</p>',
            unsafe_allow_html=True,
        )

        if not uploaded_files:
            st.info("Vui lòng chọn file tài liệu (PDF, DOCX, DOC, TXT) để bắt đầu.")
            return

        # Kiểm tra kích thước từng file
        valid_files = []
        for uploaded_file in uploaded_files:
            size = getattr(uploaded_file, "size", None) or 0
            if size > MAX_PDF_BYTES:
                st.warning(f"File **{uploaded_file.name}** vượt quá {MAX_PDF_BYTES // (1024 * 1024)} MB, bỏ qua.")
            else:
                valid_files.append(uploaded_file)
        
        if not valid_files:
            st.error("Không có file hợp lệ để xử lý.")
            return

        # Hiển thị danh sách file đã chọn
        file_names = ", ".join([f.name for f in valid_files])
        st.session_state.uploaded_filename = file_names
        st.success(f"Đã chọn {len(valid_files)} file: **{file_names}**")

        # Luôn hiển thị nút phân tích (cho phép upload thêm)
        button_label = "Thêm tài liệu vào hệ thống" if st.session_state.vector_ready else "Bắt đầu phân tích tài liệu"
        
        if st.button(button_label, type="secondary",
                     use_container_width=True, key="btn_process_doc"):
            all_chunks = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(valid_files):
                status_text.text(f"Đang xử lý file {idx + 1}/{len(valid_files)}: {uploaded_file.name}...")
                try:
                    chunks = _parse_and_split(uploaded_file)
                    if chunks:
                        all_chunks.extend(chunks)
                        logger.info(f"✅ Đã xử lý {uploaded_file.name}: {len(chunks)} chunks")
                    else:
                        st.warning(f"Không đọc được nội dung từ {uploaded_file.name}")
                except Exception as e:
                    st.warning(f"Lỗi khi xử lý {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(valid_files))
            
            status_text.text("Đang vector hóa tất cả tài liệu...")
            
            if not all_chunks:
                st.error("Không đọc được nội dung từ bất kỳ file nào. Vui lòng thử lại.")
                return
            
            try:
                ok, msg = st.session_state.rag_controller.process_new_document(all_chunks)
            except Exception as e:
                st.error(f"Lỗi khi lưu vào database: {str(e)}")
                return

            if ok:
                st.session_state.vector_ready = True
                st.session_state.processing_done = True
                
                # Lưu thông tin file vào cuộc hội thoại hiện tại
                active_id = st.session_state.active_conversation_id
                if active_id in st.session_state.conversations:
                    # Đảm bảo uploaded_files tồn tại
                    if "uploaded_files" not in st.session_state.conversations[active_id]:
                        st.session_state.conversations[active_id]["uploaded_files"] = []
                    
                    # Thêm thông tin file mới
                    for uploaded_file in valid_files:
                        file_size_mb = uploaded_file.size / (1024 * 1024)
                        file_info = {
                            "name": uploaded_file.name,
                            "size": f"{file_size_mb:.2f} MB",
                            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        # Kiểm tra trùng lặp
                        if not any(f["name"] == file_info["name"] for f in st.session_state.conversations[active_id]["uploaded_files"]):
                            st.session_state.conversations[active_id]["uploaded_files"].append(file_info)
                
                st.success(f"Xử lý xong {len(all_chunks)} chunks từ {len(valid_files)} file. Bạn có thể đặt câu hỏi!")
            else:
                st.error(msg)


# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------

def render_chat_section() -> None:
    if st.session_state.get("rag_controller") is None:
        st.error("Hệ thống AI chưa khởi tạo. Không thể hỏi đáp.")
        return
    
    # Lấy cuộc hội thoại hiện tại
    active_id = st.session_state.active_conversation_id
    active_conv = st.session_state.conversations.get(active_id, {"name": "Unknown", "messages": [], "uploaded_files": []})
    
    # Hiển thị tên cuộc hội thoại và số file
    file_count = len(active_conv.get("uploaded_files", []))
    title = f'{active_conv["name"]} ({file_count} tài liệu)' if file_count > 0 else active_conv["name"]
    st.markdown(f'<p class="section-title">{title}</p>', unsafe_allow_html=True)
    
    # Hiển thị danh sách file trong cuộc hội thoại này (dạng compact)
    uploaded_files_list = active_conv.get("uploaded_files", [])
    if uploaded_files_list:
        with st.expander(f"Xem {len(uploaded_files_list)} tài liệu đã upload"):
            for idx, file_info in enumerate(uploaded_files_list, 1):
                st.text(f"{idx}. {file_info['name']} - {file_info['size']} - {file_info.get('uploaded_at', 'N/A')}")

    # Hiển thị lịch sử hội thoại
    selected = st.session_state.get("selected_chat_idx")
    chat_container = st.container(height=500)
    with chat_container:
        current_messages = active_conv["messages"]
        if not current_messages:
            st.markdown(
                '<div class="answer-empty">Bắt đầu cuộc hội thoại bằng cách đặt câu hỏi bên dưới.</div>',
                unsafe_allow_html=True,
            )
        for idx, msg in enumerate(current_messages):
            # Anchor để scroll đến
            st.markdown(f'<div id="chat-msg-{idx}"></div>', unsafe_allow_html=True)

            # Highlight nếu được chọn từ sidebar
            is_selected = (selected == idx)
            if is_selected:
                st.markdown('<div class="chat-highlight">', unsafe_allow_html=True)

            with st.chat_message("user", avatar=AVATAR_USER):
                st.write(msg["question"])
            
            # Hiển thị 2 cột: Standard RAG vs Advanced RAG
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Standard RAG**")
                with st.chat_message("assistant", avatar=AVATAR_AI):
                    st.write(msg.get("answer_standard", msg.get("answer", "")))
                    cites_std = msg.get("citations_standard", msg.get("citations", []))
                    if cites_std:
                        with st.expander(f"{len(cites_std)} trích dẫn nguồn"):
                            citations_html = '<div class="citations-wrap">'
                            for i, c in enumerate(cites_std, start=1):
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
            
            with col2:
                st.markdown("**Advanced RAG**")
                with st.chat_message("assistant", avatar=AVATAR_AI):
                    st.write(msg.get("answer_advanced", msg.get("answer", "")))
                    confidence = msg.get("confidence")
                    if isinstance(confidence, (int, float)):
                        st.caption(f"Confidence: {float(confidence):.2f}")
                    
                    # Hiển thị metadata của Advanced RAG
                    advanced_meta = msg.get("advanced_meta", {})
                    if advanced_meta.get("used_retry"):
                        st.caption(f"🔄 Đã retry với query: {advanced_meta.get('better_query', 'N/A')}")
                    
                    cites_adv = msg.get("citations_advanced", msg.get("citations", []))
                    if cites_adv:
                        with st.expander(f"{len(cites_adv)} trích dẫn nguồn"):
                            citations_html = '<div class="citations-wrap">'
                            for i, c in enumerate(cites_adv, start=1):
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
    if not st.session_state.vector_ready:
        st.markdown(
            '<p class="qa-hint"> Cần phân tích tài liệu trước khi đặt câu hỏi.</p>',
            unsafe_allow_html=True,
        )
        return

    question = st.chat_input(
        placeholder="Nhập câu hỏi để so sánh Standard RAG vs Advanced RAG...",
    )
    if question and question.strip():
        # Sử dụng active_conversation_id làm session_id - CÙNG lịch sử cho cả 2
        active_id = st.session_state.active_conversation_id
        
        # Chạy Standard RAG trước - KHÔNG lưu vào memory
        with st.spinner("Đang chạy Standard RAG..."):
            result_standard = st.session_state.rag_controller.answer_question(
                question=question.strip(),
                session_id=active_id,
                advanced_mode=False,
                filter_dict=None,  # Không lọc
                save_to_memory=False  # Không lưu
            )
        
        answer_standard = result_standard.get("answer", "")
        citations_standard = _sources_to_citations(result_standard.get("sources", []))
        
        # Chạy Advanced RAG sau - LÀM lưu vào memory (chỉ lưu 1 lần)
        with st.spinner("Đang chạy Advanced RAG (self-check + retry)..."):
            result_advanced = st.session_state.rag_controller.answer_question(
                question=question.strip(),
                session_id=active_id,
                advanced_mode=True,
                filter_dict=None,  # Không lọc
                save_to_memory=True  # Lưu vào memory
            )
        
        answer_advanced = result_advanced.get("answer", "")
        citations_advanced = _sources_to_citations(result_advanced.get("sources", []))
        confidence = result_advanced.get("confidence")
        advanced_meta = result_advanced.get("advanced_meta", {})
        
        # Lưu vào cuộc hội thoại hiện tại với cả 2 kết quả
        if active_id in st.session_state.conversations:
            st.session_state.conversations[active_id]["messages"].append({
                "question": question.strip(),
                "answer_standard": answer_standard,
                "citations_standard": citations_standard,
                "answer_advanced": answer_advanced,
                "citations_advanced": citations_advanced,
                "confidence": confidence if isinstance(confidence, (int, float)) else None,
                "advanced_meta": advanced_meta,
            })
        
        st.rerun()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="SmartDoc AI",
        page_icon="",
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
