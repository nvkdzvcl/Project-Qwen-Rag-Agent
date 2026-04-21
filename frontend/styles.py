"""CSS toàn cục SmartDoc AI — một nơi chỉnh palette và layout."""


def inject_global_styles() -> None:
    import streamlit as st

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
                /* Kích thước cố định / giới hạn — responsive qua clamp & max-width */
                --content-max: 1120px;
                --workflow-step-width: clamp(140px, 18vw, 200px);
                --workflow-step-min-h: 128px;
                --upload-zone-min-h: 152px;
                --banner-min-h: 168px;
                --section-gap: 1rem;
            }

            .stApp {
                background-color: var(--bg-main);
                color: var(--text-main);
                overflow-x: hidden !important;
            }

            /* Vùng nội dung chính: không tràn ngang, căn giữa trên màn lớn */
            section[data-testid="stMain"] .block-container {
                max-width: min(var(--content-max), calc(100vw - 2rem)) !important;
                padding-left: max(0.75rem, env(safe-area-inset-left)) !important;
                padding-right: max(0.75rem, env(safe-area-inset-right)) !important;
                padding-top: 0.5rem !important;
                padding-bottom: 1.5rem !important;
            }

            html, body,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            [data-testid="stMainBlockContainer"] {
                overflow-x: hidden !important;
            }

            [data-testid="stSidebar"],
            [data-testid="stSidebarContent"] {
                background-color: var(--bg-sidebar) !important;
                overflow-x: hidden !important;
            }

            /* Chữ sidebar: trắng — trừ widget có nền sáng (select, …) */
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] .stMarkdown p,
            [data-testid="stSidebar"] .stMarkdown span,
            [data-testid="stSidebar"] [data-testid="stCaption"] {
                color: var(--text-light) !important;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h3 {
                color: var(--text-light) !important;
            }

            [data-testid="stSidebar"] h2 {
                color: var(--primary) !important;
                font-size: clamp(1.4rem, 3vw, 1.75rem) !important;
                font-weight: 700 !important;
            }

            /* Selectbox: nền trắng mặc định → chữ tối; nhãn widget vẫn trắng trên nền sidebar */
            [data-testid="stSidebar"] [data-testid="stSelectbox"] label {
                color: var(--text-light) !important;
            }

            /* Selectbox model — highlight nền xanh primary */
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] {
                background-color: #e8f4ff !important;
                border: 2px solid var(--primary) !important;
                border-radius: 8px !important;
            }

            [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *,
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] *::placeholder {
                color: var(--text-main) !important;
            }

            [data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
                fill: var(--text-main) !important;
            }

            /* Một số bản Streamlit dùng combobox / không có baseweb */
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"],
            [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="listbox"] {
                color: var(--text-main) !important;
                background-color: #ffffff !important;
            }

            /* Slider: nhãn & giá trị đọc được trên nền tối */
            [data-testid="stSidebar"] [data-testid="stSlider"] label,
            [data-testid="stSidebar"] [data-testid="stSlider"] .stMarkdown,
            [data-testid="stSidebar"] [data-testid="stSlider"] .stMarkdown p,
            [data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
            [data-testid="stSidebar"] [data-testid="stTickBarMax"] {
                color: var(--text-light) !important;
            }

            .chat-highlight {
                border-left: 3px solid var(--primary);
                background: #f0f7ff;
                border-radius: 8px;
                padding: 4px 8px;
                margin: 4px 0;
            }

            /* Sidebar history buttons */
            [data-testid="stSidebar"] .stButton > button {
                border: 1px solid rgba(255, 255, 255, 0.15);
                background-color: rgba(255, 255, 255, 0.05);
                color: var(--text-light) !important;
                text-align: left !important;
                font-size: 0.82rem !important;
                padding: 8px 12px !important;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                transition: all 0.2s ease;
            }
            
            [data-testid="stSidebar"] .stButton > button:hover {
                background-color: rgba(255, 255, 255, 0.1);
                border-color: rgba(255, 255, 255, 0.3);
            }
            
            /* Nút primary trong sidebar (+ Hội thoại mới) */
            [data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background-color: var(--primary) !important;
                border-color: var(--primary) !important;
                color: #fff !important;
                font-weight: 600 !important;
            }
            
            [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
                background-color: #0056b3 !important;
                border-color: #0056b3 !important;
            }
            
            /* Nút secondary trong sidebar (cuộc hội thoại active) */
            [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
                background-color: rgba(0, 123, 255, 0.15) !important;
                border: 1px solid rgba(0, 123, 255, 0.4) !important;
                color: #ffffff !important;
                font-weight: 500 !important;
            }
            
            [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
                background-color: rgba(0, 123, 255, 0.25) !important;
                border-color: rgba(0, 123, 255, 0.6) !important;
            }
            
            /* Nút tertiary trong sidebar (cuộc hội thoại không active) */
            [data-testid="stSidebar"] .stButton > button[kind="tertiary"] {
                background-color: rgba(255, 255, 255, 0.03) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                color: rgba(255, 255, 255, 0.85) !important;
            }
            
            [data-testid="stSidebar"] .stButton > button[kind="tertiary"]:hover {
                background-color: rgba(255, 255, 255, 0.08) !important;
                border-color: rgba(255, 255, 255, 0.2) !important;
                color: #ffffff !important;
            }
            
            /* Nút xóa (X) - màu đỏ nhẹ */
            [data-testid="stSidebar"] .stButton > button[key*="del_"] {
                background-color: rgba(220, 53, 69, 0.1) !important;
                border: 1px solid rgba(220, 53, 69, 0.3) !important;
                color: #ff6b6b !important;
                padding: 4px 8px !important;
                font-size: 0.9rem !important;
            }
            
            [data-testid="stSidebar"] .stButton > button[key*="del_"]:hover {
                background-color: rgba(220, 53, 69, 0.2) !important;
                border-color: rgba(220, 53, 69, 0.5) !important;
                color: #ff5252 !important;
            }

            /* Checkbox trong sidebar — label trắng, box sáng */
            [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
            [data-testid="stSidebar"] [data-testid="stCheckbox"] span,
            [data-testid="stSidebar"] [data-testid="stCheckbox"] p,
            [data-testid="stSidebar"] [data-testid="stCheckbox"] div,
            [data-testid="stSidebar"] [data-testid="stCheckbox"] label > div,
            [data-testid="stSidebar"] [data-testid="stCheckbox"] label > div > p {
                color: #ffffff !important;
                opacity: 1 !important;
            }

            [data-testid="stSidebar"] [data-testid="stCheckbox"] label span {
                color: #ffffff !important;
                opacity: 1 !important;
            }

            /* Ô vuông checkbox */
            [data-testid="stSidebar"] [data-testid="stCheckbox"] input[type="checkbox"] {
                accent-color: var(--primary) !important;
                width: 16px !important;
                height: 16px !important;
            }

            /* BaseWeb checkbox (Streamlit dùng baseweb) */
            [data-testid="stSidebar"] [data-baseweb="checkbox"] span {
                background-color: #ffffff !important;
                border-color: #aaaaaa !important;
            }

            [data-testid="stSidebar"] [data-baseweb="checkbox"] [data-checked="true"] span,
            [data-testid="stSidebar"] [data-baseweb="checkbox"] span[aria-checked="true"] {
                background-color: var(--primary) !important;
                border-color: var(--primary) !important;
            }

            [data-testid="stSidebar"] .stAlert {
                color: var(--text-main) !important;
            }

            .stMarkdown, .stCaption, .stText, .stAlert,
            .stTextInput, .stSelectbox, .stSlider {
                overflow-wrap: anywhere;
                word-break: break-word;
            }

            h1, h2, h3, h4, h5, h6 {
                color: var(--text-main);
            }

            .main-header {
                text-align: center;
                color: var(--primary);
                margin-top:3rem;
                margin-bottom: 1rem;
                border-bottom: 1px solid #e3e6ea;
                padding-bottom: 0.75rem;
            }

            .main-header h1 {
                font-size: clamp(1.35rem, 3.5vw, 1.85rem) !important;
                line-height: 1.2;
            }

            .feature-section {
                background: #ffffff;
                border: 1px solid #dfe4ea;
                border-radius: 18px;
                padding: clamp(12px, 2vw, 18px);
                box-shadow: 0 8px 24px rgba(15, 44, 84, 0.06);
                margin-bottom: var(--section-gap);
                width: 100%;
                max-width: 100%;
                box-sizing: border-box;
            }

            .workflow-flow {
                display: flex;
                align-items: stretch;
                justify-content: center;
                gap: clamp(6px, 1.5vw, 12px);
                flex-wrap: wrap;
                padding: 6px 0;
                width: 100%;
            }

            .workflow-step {
                background: #f8fbff;
                border: 1px solid #dfe7f2;
                border-radius: 14px;
                padding: 14px 12px;
                text-align: center;
                box-sizing: border-box;
                flex: 0 0 var(--workflow-step-width);
                width: var(--workflow-step-width);
                min-height: var(--workflow-step-min-h);
                max-width: 100%;
                display: flex;
                flex-direction: column;
                justify-content: center;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }

            .workflow-step:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0, 123, 255, 0.08);
            }

            .workflow-step-arrow {
                display: flex;
                align-items: center;
                color: var(--primary);
                font-size: 1.4rem;
                font-weight: 700;
                flex: 0 0 auto;
            }

            .workflow-step h4 {
                margin: 0;
                font-size: clamp(0.85rem, 2vw, 0.95rem);
                color: var(--primary);
            }

            .workflow-step p {
                margin-top: 6px;
                color: #5b6d7a;
                font-size: clamp(0.78rem, 1.8vw, 0.85rem);
                line-height: 1.45;
            }

            .upload-banner {
                text-align: center;
                padding: clamp(16px, 3vw, 22px);
                # background: linear-gradient(135deg, #fff8e1 0%, #fff3cd 100%);
                border: 1px solid #FFC107;
                border-radius: 16px;
                margin-bottom: 16px;
                min-height: var(--banner-min-h);
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }

            .upload-banner-icon {
                font-size: clamp(26px, 5vw, 32px);
                background: transparent;
                color: #FFC107;
                border: 2.5px solid #FFC107;
                width: clamp(52px, 12vw, 64px);
                height: clamp(52px, 12vw, 64px);
                flex-shrink: 0;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                margin-bottom: 12px;
                box-shadow: 0 4px 14px rgba(255, 193, 7, 0.3);
            }

            .upload-banner-title {
                font-size: clamp(1rem, 2.5vw, 1.1rem);
                font-weight: 700;
                color: #1f4068;
                margin-bottom: 6px;
            }

            .upload-banner-desc {
                font-size: clamp(0.82rem, 2vw, 0.9rem);
                color: #5b6d81;
                max-width: min(440px, 100%);
                margin: 0 auto;
                line-height: 1.5;
            }

            .upload-note {
                color: #6c757d;
                font-size: 0.88rem;
                margin-top: 8px;
            }

            div[data-testid="stFileUploader"] {
                background-color: #ffffff;
                border: 1px solid #e0e6ed;
                border-radius: 12px;
                padding: 16px;
                width: 100%;
                max-width: 100%;
                box-sizing: border-box;
            }

            div[data-testid="stFileUploader"] section > label {
                display: none;
            }

            div[data-testid="stFileUploaderDropzone"] {
                border: 2px dashed #FFC107 !important;
                background: #fff8e1 !important;
                border-radius: 10px !important;
                min-height: var(--upload-zone-min-h) !important;
                box-sizing: border-box !important;
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

            /* Number input trong sidebar */
            [data-testid="stSidebar"] [data-testid="stNumberInput"] label {
                color: var(--text-light) !important;
                font-size: 0.85rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
                background-color: #ffffff !important;
                color: var(--text-main) !important;
                border: 1px solid #ced4da !important;
                border-radius: 6px !important;
                text-align: center !important;
                font-weight: 600 !important;
            }

            [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
                background-color: #3a3f44 !important;
                color: var(--text-light) !important;
                border: 1px solid rgba(255,255,255,0.2) !important;
            }

            [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover {
                background-color: var(--primary) !important;
            }

            .stButton > button[kind="primary"] {
                background-color: var(--primary) !important;
                border-color: var(--primary) !important;
                color: #fff !important;
                font-weight: 600 !important;
                transition: all 0.2s ease;
            }
            
            .stButton > button[kind="primary"]:hover {
                background-color: #0056b3 !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 123, 255, 0.3);
            }

            .stButton > button[kind="secondary"] {
                background-color: var(--secondary) !important;
                border-color: var(--secondary) !important;
                color: #1f2d3d !important;
                font-weight: 600 !important;
                transition: all 0.2s ease;
            }
            
            .stButton > button[kind="secondary"]:hover {
                background-color: #ffb300 !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(255, 193, 7, 0.3);
            }

            /* Modal confirm overlay */
            .modal-overlay {
                background: rgba(0,0,0,0.18);
                border-radius: 14px;
                padding: 1.5rem 1.25rem 0.5rem;
                margin-bottom: 0.5rem;
                border: 1px solid #e0e0e0;
            }

            .modal-box {
                background: #fff;
                border-radius: 10px;
                padding: 1rem 1.25rem;
                box-shadow: 0 4px 18px rgba(0,0,0,0.08);
            }

            .modal-title {
                font-size: 1rem;
                font-weight: 600;
                color: #333;
                margin-bottom: 0.4rem;
            }

            .modal-body {
                font-size: 0.88rem;
                color: #666;
            }

            /* Dialog popup — màu nhạt, nền trắng */
            [data-testid="stDialog"] > div {
                background: #ffffff !important;
                border-radius: 16px !important;
                box-shadow: 0 8px 32px rgba(0,0,0,0.12) !important;
            }

            [data-testid="stDialog"] h2 {
                color: #444 !important;
                font-size: 1.05rem !important;
                font-weight: 600 !important;
            }

            [data-testid="stDialog"] p {
                color: #666 !important;
            }

            [data-testid="stDialog"] .stButton > button[kind="primary"] {
                background-color: #e05555 !important;
                border-color: #e05555 !important;
                color: #fff !important;
            }

            [data-testid="stDialog"] .stButton > button[kind="secondary"] {
                background-color: #f0f0f0 !important;
                border-color: #ddd !important;
                color: #555 !important;
            }

            .section-title {
                font-size: clamp(1rem, 2.4vw, 1.15rem);
                font-weight: 600;
                color: var(--text-main);
                margin: 1.25rem 0 0.5rem 0;
            }

            /* Sidebar: chiều rộng ổn định, không bị co quá hẹp */
            [data-testid="stSidebar"] {
                min-width: min(280px, 100vw) !important;
            }

            @media (max-width: 900px) {
                :root {
                    --workflow-step-width: clamp(140px, 38vw, 200px);
                }
                .workflow-flow {
                    justify-content: center;
                }
            }

            @media (max-width: 640px) {
                section[data-testid="stMain"] .block-container {
                    padding-left: 0.5rem !important;
                    padding-right: 0.5rem !important;
                }
                .workflow-step-arrow {
                    display: none;
                }
                .workflow-flow {
                    flex-direction: column;
                    align-items: stretch;
                }
                .workflow-step {
                    flex: 1 1 auto !important;
                    width: 100% !important;
                    min-height: auto;
                }
                .feature-section {
                    padding: 12px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
