"""Global CSS cho giao diện Streamlit."""


def inject_global_styles() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
            :root {
                --ink-900: #0e1117;
                --ink-700: #2c3342;
                --ink-500: #546179;
                --line: #d9e0ec;
                --surface: #ffffff;
                --surface-soft: #f4f7fc;
                --brand: #0f4c81;
                --accent: #f0b429;
                --ok: #1f8b4c;
            }

            .stApp {
                background:
                    radial-gradient(circle at 0% 0%, #e9f1ff 0%, transparent 28%),
                    radial-gradient(circle at 100% 0%, #fff2d3 0%, transparent 20%),
                    #f3f6fb;
                color: var(--ink-900);
            }

            section[data-testid="stMain"] .block-container {
                max-width: 1080px;
                padding-top: 1.1rem;
                padding-bottom: 1.3rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f1728 0%, #17263d 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
            }

            .side-brand {
                padding: 0.35rem 0.2rem 0.5rem;
                border-bottom: 1px solid rgba(255, 255, 255, 0.16);
                margin-bottom: 0.8rem;
            }

            .side-brand h2 {
                margin: 0;
                color: #f3f8ff;
                font-size: 1.42rem;
                letter-spacing: 0.3px;
            }

            .side-brand p {
                margin: 0.3rem 0 0;
                color: #a9bbd9;
                font-size: 0.84rem;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] .stCaption,
            [data-testid="stSidebar"] .stMarkdown {
                color: #eef3ff !important;
            }

            [data-testid="stSidebar"] .stButton > button {
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                background: rgba(255, 255, 255, 0.06);
                color: #eff5ff !important;
                transition: all 0.2s ease;
            }

            [data-testid="stSidebar"] .stButton > button:hover {
                border-color: rgba(255, 255, 255, 0.45);
                background: rgba(255, 255, 255, 0.13);
            }

            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #0f67b6 0%, #0d4f8e 100%);
                border-color: #0f67b6;
                color: #ffffff !important;
            }

            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] textarea,
            [data-testid="stSidebar"] [data-baseweb="select"] {
                background: #f7faff !important;
                color: #10151f !important;
                border-radius: 10px !important;
                border: 1px solid #c7d4ec !important;
            }

            [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
                background: #f0b429 !important;
            }

            /* File uploader trong sidebar: giữ tương phản tốt cho cả light/dark */
            [data-testid="stSidebar"] [data-testid="stFileUploader"] {
                background: rgba(10, 20, 36, 0.42) !important;
                border: 1px solid rgba(255, 255, 255, 0.14) !important;
                border-radius: 12px !important;
                padding: 0.58rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                background: rgba(5, 12, 24, 0.9) !important;
                border: 1px dashed rgba(255, 255, 255, 0.34) !important;
                border-radius: 10px !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
                color: #eef4ff !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
            [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stCaptionContainer"],
            [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
                color: #e6eeff !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
                border-radius: 9px !important;
                border: 1px solid rgba(255, 255, 255, 0.22) !important;
                background: rgba(255, 255, 255, 0.12) !important;
                color: #f4f8ff !important;
            }

            .doc-chip {
                border: 1px solid rgba(255, 255, 255, 0.28);
                border-radius: 999px;
                padding: 0.25rem 0.62rem;
                font-size: 0.78rem;
                margin-bottom: 0.36rem;
                background: rgba(255, 255, 255, 0.08);
                color: #f4f8ff;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }

            .hero-card {
                background:
                    linear-gradient(110deg, rgba(15, 103, 182, 0.95) 0%, rgba(12, 58, 107, 0.96) 62%),
                    linear-gradient(145deg, #0d62ad 0%, #0a3f74 100%);
                border-radius: 18px;
                padding: 1.15rem 1.2rem;
                color: #ffffff;
                margin-bottom: 0.85rem;
                box-shadow: 0 16px 34px rgba(9, 34, 68, 0.23);
            }

            .hero-card h1 {
                margin: 0.12rem 0 0.4rem;
                font-size: clamp(1.35rem, 2.8vw, 1.85rem);
                line-height: 1.2;
                color: #ffffff;
                letter-spacing: 0.2px;
            }

            .hero-kicker {
                margin: 0;
                font-size: 0.76rem;
                letter-spacing: 0.6px;
                text-transform: uppercase;
                color: #ffd56e;
                font-weight: 700;
            }

            .hero-card p {
                margin: 0;
                color: #eaf2ff;
                max-width: 760px;
            }

            .hero-meta {
                margin-top: 0.72rem;
                display: flex;
                flex-wrap: wrap;
                gap: 0.38rem;
            }

            .hero-meta span {
                background: rgba(255, 255, 255, 0.18);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 999px;
                padding: 0.17rem 0.56rem;
                font-size: 0.76rem;
                color: #ffffff;
            }

            .empty-state {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 1rem 1rem;
                margin-bottom: 0.8rem;
                box-shadow: 0 8px 20px rgba(12, 28, 52, 0.06);
            }

            .empty-state h3 {
                margin: 0;
                color: var(--brand);
                font-size: 1.04rem;
            }

            .empty-state p {
                margin: 0.42rem 0 0;
                color: var(--ink-500);
                font-size: 0.9rem;
            }

            [data-testid="stChatMessage"] {
                background: var(--surface);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 0.45rem 0.58rem;
                box-shadow: 0 7px 18px rgba(21, 41, 71, 0.05);
            }

            /* Chống chữ bị chìm trong dark mode khi card chat nền sáng */
            [data-testid="stChatMessage"] *,
            [data-testid="stChatMessage"] p,
            [data-testid="stChatMessage"] span,
            [data-testid="stChatMessage"] li,
            [data-testid="stChatMessage"] strong {
                color: #1f2d3f !important;
            }

            .citation-row {
                background: #fffdf6;
                border: 1px solid #f5dd91;
                border-left: 4px solid var(--accent);
                border-radius: 10px;
                padding: 0.5rem 0.65rem;
                margin-bottom: 0.45rem;
                color: #2f3b4a;
                font-size: 0.86rem;
            }

            .citation-row span {
                color: #4d5b6d;
            }

            [data-testid="stExpander"] {
                border-radius: 12px;
                border: 1px solid var(--line);
                background: #fafcff;
            }

            [data-testid="stExpander"] summary,
            [data-testid="stExpander"] summary span,
            [data-testid="stExpander"] summary p,
            [data-testid="stExpander"] details > summary > div {
                color: #1f2d3f !important;
            }

            [data-testid="stChatInput"] {
                padding-top: 0.25rem;
            }

            /* Ô nhập chat luôn đọc được ở cả 2 theme */
            [data-testid="stChatInput"] textarea,
            [data-testid="stChatInput"] input {
                background: #ffffff !important;
                color: #1f2d3f !important;
                border: 1px solid #c9d4e7 !important;
            }

            [data-testid="stChatInput"] textarea::placeholder,
            [data-testid="stChatInput"] input::placeholder {
                color: #6b778c !important;
            }

            [data-testid="stChatInput"] button {
                background: #0f67b6 !important;
                color: #ffffff !important;
                border: 1px solid #0f67b6 !important;
            }

            @media (max-width: 780px) {
                section[data-testid="stMain"] .block-container {
                    padding-left: 0.7rem;
                    padding-right: 0.7rem;
                }

                .hero-card {
                    border-radius: 14px;
                    padding: 0.95rem 0.9rem;
                }

                [data-testid="stSidebar"] {
                    border-right: none;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
