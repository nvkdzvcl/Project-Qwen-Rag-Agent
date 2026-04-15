import os
import logging
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SmartDocLoader:
    def __init__(self):
        # Dictionary này chỉ chứa các phương thức load văn bản thông thường
        self._standard_handlers = {
            '.pdf': self._load_pdf_standard,
            '.docx': self._load_docx_standard,
        }

    # --- CÁC HÀM XỬ LÝ NỘI BỘ (PRIVATE) ---

    def _load_pdf_standard(self, file_path: str):
        logger.info(f"--- Đang nạp PDF (Standard): {os.path.basename(file_path)} ---")
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        for i, doc in enumerate(documents):
            doc.metadata.update({"page": i + 1, "source_type": "text_pdf"})
        return documents

    def _load_docx_standard(self, file_path: str):
        logger.info(f"--- Đang nạp Word (Standard): {os.path.basename(file_path)} ---")
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata.update({"source_type": "docx"})
        return documents

    # --- CÁC HÀM CÔNG KHAI (PUBLIC API) ---

    def load(self, file_path: str):
        """
        API PUBLIC: Chỉ dùng cho văn bản thông thường (PDF text, DOCX).
        Không hỗ trợ OCR tại đây để đảm bảo tốc độ và hiệu suất.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy tệp: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        handler = self._standard_handlers.get(file_ext)
        
        if not handler:
            raise ValueError(f"Định dạng {file_ext} không được hỗ trợ trong chế độ Standard.")

        return handler(file_path)

    def load_pdf_with_ocr(self, file_path: str):
        """
        API PUBLIC RIÊNG BIỆT: Chuyên dùng cho PDF dạng ảnh/scan.
        Chỉ hoạt động với file .pdf.
        """
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Hàm load_pdf_with_ocr chỉ hỗ trợ định dạng .pdf")
            
        logger.info(f"--- Đang nạp PDF (OCR Mode): {os.path.basename(file_path)} ---")
        
        try:
            import fitz
            from rapidocr_onnxruntime import RapidOCR
            import numpy as np
            from PIL import Image
            import io

            engine = RapidOCR()
            doc_pdf = fitz.open(file_path)
            ocr_documents = []

            for page_num, page in enumerate(doc_pdf):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes()))
                result, _ = engine(np.array(img))
                
                text = "\n".join([line[1] for line in result]) if result else ""
                
                ocr_documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "page": page_num + 1,
                        "source_type": "ocr_pdf"
                    }
                ))
            doc_pdf.close()
            return ocr_documents
        except Exception as e:
            logger.error(f"Lỗi khi thực hiện OCR: {e}")
            raise