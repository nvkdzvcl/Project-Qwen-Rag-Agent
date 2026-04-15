import os
import logging
from datetime import datetime
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SmartDocLoader:
    def __init__(self):
        self._standard_handlers = {
            '.pdf': self._load_pdf_standard,
            '.docx': self._load_docx_standard,
        }

    def _get_common_metadata(self, file_path, doc_type):
        """Hàm nội bộ để tạo khung metadata thống nhất"""
        return {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_type": doc_type
        }

    def _load_pdf_standard(self, file_path: str, doc_type: str):
        logger.info(f"--- Đang nạp PDF (Standard): {os.path.basename(file_path)} ---")
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        base_metadata = self._get_common_metadata(file_path, doc_type)
        for i, doc in enumerate(documents):
            # Cập nhật metadata chung và bổ sung thông tin riêng của trang
            new_meta = base_metadata.copy()
            new_meta.update({"page": i + 1, "source_type": "text_pdf"})
            doc.metadata = new_meta
        return documents

    def _load_docx_standard(self, file_path: str, doc_type: str):
        logger.info(f"--- Đang nạp Word (Standard): {os.path.basename(file_path)} ---")
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        
        base_metadata = self._get_common_metadata(file_path, doc_type)
        for doc in documents:
            new_meta = base_metadata.copy()
            new_meta.update({"page": 1, "source_type": "docx"})
            doc.metadata = new_meta
        return documents

    def load(self, file_path: str, doc_type: str = "general"):
        """API cho văn bản thường"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy tệp: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        handler = self._standard_handlers.get(file_ext)
        
        if not handler:
            raise ValueError(f"Định dạng {file_ext} không được hỗ trợ.")

        return handler(file_path, doc_type)

    def load_pdf_with_ocr(self, file_path: str, doc_type: str = "general"):
        """API cho PDF dạng ảnh (OCR)"""
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("Hàm này chỉ hỗ trợ .pdf")
            
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
            base_metadata = self._get_common_metadata(file_path, doc_type)

            for page_num, page in enumerate(doc_pdf):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes()))
                result, _ = engine(np.array(img))
                
                text = "\n".join([line[1] for line in result]) if result else ""
                
                new_meta = base_metadata.copy()
                new_meta.update({"page": page_num + 1, "source_type": "ocr_pdf"})
                
                ocr_documents.append(Document(
                    page_content=text,
                    metadata=new_meta
                ))
            doc_pdf.close()
            return ocr_documents
        except Exception as e:
            logger.error(f"Lỗi OCR: {e}")
            raise