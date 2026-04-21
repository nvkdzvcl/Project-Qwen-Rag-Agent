import os
import logging
import io
import fitz  # PyMuPDF
import numpy as np
import pdfplumber
from datetime import datetime
from PIL import Image
from langchain_core.documents import Document
from rapidocr_onnxruntime import RapidOCR
from docx import Document as DocxReader

logger = logging.getLogger(__name__)

class SmartDocLoader:
    def __init__(self):
        self.ocr_engine = RapidOCR()
        # Ngưỡng lọc ảnh rác
        self.MIN_IMG_SIZE = 40 

    def _get_common_metadata(self, file_path, doc_type):
        return {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            "upload_date": datetime.now().strftime("%Y-%m-%d"),
            "doc_type": doc_type,
            "file_extension": os.path.splitext(file_path)[1].lower()
        }

    def load(self, file_path: str, doc_type: str = "general"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy tệp: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf':
            return self.load_pdf(file_path, doc_type)
        elif file_ext == '.docx':
            return self.load_docx(file_path, doc_type)
        return []

    def load_pdf(self, file_path: str, doc_type: str = "general"):
        """Trích xuất Hybrid: Bảng (plumber) + Ảnh (fitz + OCR)"""
        logger.info(f"--- Đang nạp PDF Hybrid: {os.path.basename(file_path)} ---")
        final_documents = []
        base_metadata = self._get_common_metadata(file_path, doc_type)

        with pdfplumber.open(file_path) as pdf:
            doc_fitz = fitz.open(file_path)

            for page_num in range(len(pdf.pages)):
                page_content = []
                p_plumber = pdf.pages[page_num]
                p_fitz = doc_fitz[page_num]

                # 1. Trích xuất Bảng (Table) - Cực kỳ quan trọng cho SQL
                tables = p_plumber.extract_tables()
                if tables:
                    for table in tables:
                        rows = [" | ".join([str(cell).strip() if cell else "" for cell in row]) for row in table]
                        page_content.append(f"\n[TABLE_START]\n" + "\n".join(rows) + "\n[TABLE_END]\n")

                # 2. Trích xuất Text thường (đã lọc bỏ text trong bảng để tránh trùng)
                # layout=True giúp giữ cấu trúc cột cơ bản
                text = p_plumber.extract_text()
                if text:
                    page_content.append(text)

                # 3. Trích xuất và OCR ảnh từ PyMuPDF
                for img_idx, img_info in enumerate(p_fitz.get_images(full=True)):
                    try:
                        xref = img_info[0]
                        pix = fitz.Pixmap(doc_fitz, xref)
                        
                        if pix.width > self.MIN_IMG_SIZE and pix.height > self.MIN_IMG_SIZE:
                            img_data = pix.tobytes("png")
                            img = Image.open(io.BytesIO(img_data))
                            result, _ = self.ocr_engine(np.array(img))
                            if result:
                                ocr_text = "\n".join([line[1] for line in result])
                                page_content.append(f"\n[OCR_IMAGE_{img_idx}]:\n{ocr_text}\n")
                        pix = None # Giải phóng bộ nhớ
                    except Exception as e:
                        logger.error(f"Lỗi OCR ảnh trang {page_num+1}: {e}")

                new_meta = base_metadata.copy()
                new_meta.update({"page": page_num + 1, "source_type": "hybrid_pdf_v2"})
                
                final_documents.append(Document(
                    page_content="\n\n".join(page_content),
                    metadata=new_meta
                ))

            doc_fitz.close()
        return final_documents

    def load_docx(self, file_path: str, doc_type: str = "general"):
        """Trích xuất Word: Paragraph + Table chính xác"""
        logger.info(f"--- Đang nạp Word: {os.path.basename(file_path)} ---")
        doc = DocxReader(file_path)
        full_text = []
        
        for child in doc.element.body:
            if child.tag.endswith('p'):
                from docx.text.paragraph import Paragraph
                p = Paragraph(child, doc)
                if p.text.strip(): full_text.append(p.text)
            elif child.tag.endswith('tbl'):
                from docx.table import Table
                t = Table(child, doc)
                rows = [" | ".join([c.text.strip() for c in r.cells]) for r in t.rows]
                full_text.append(f"\n[TABLE_START]\n" + "\n".join(rows) + "\n[TABLE_END]\n")

        meta = self._get_common_metadata(file_path, doc_type)
        meta.update({"page": 1, "source_type": "docx"})
        return [Document(page_content="\n\n".join(full_text), metadata=meta)]