import os
import logging
import io
import fitz  # PyMuPDF
import numpy as np
import pdfplumber
import pytesseract
from datetime import datetime
from PIL import Image, ImageOps
from langchain_core.documents import Document
from docx import Document as DocxReader

logger = logging.getLogger(__name__)

class SmartDocLoaderTesseract:
    def __init__(self):
        # Kiểm tra xem tesseract đã được cài đặt trong hệ thống chưa
        try:
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR đã sẵn sàng.")
        except pytesseract.TesseractNotFoundError:
            logger.error("❌ Không tìm thấy Tesseract! Hãy chạy: sudo apt install tesseract-ocr")

    def _get_common_metadata(self, file_path, doc_type):
        return {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            "upload_date": datetime.now().strftime("%Y-%m-%d"),
            "doc_type": doc_type,
            "file_extension": os.path.splitext(file_path)[1].lower()
        }

    def load_pdf(self, file_path: str, doc_type: str = "general"):
        logger.info(f"--- Đang nạp PDF với Tesseract: {os.path.basename(file_path)} ---")
        final_documents = []
        base_metadata = self._get_common_metadata(file_path, doc_type)

        try:
            doc_fitz = fitz.open(file_path)
            with pdfplumber.open(file_path) as pdf:
                for page_num in range(len(pdf.pages)):
                    page_content = []
                    p_plumber = pdf.pages[page_num]
                    p_fitz = doc_fitz[page_num]

                    # 1. Trích xuất Bảng (pdfplumber)
                    tables = p_plumber.extract_tables()
                    if tables:
                        for table in tables:
                            rows = [" | ".join([str(cell).strip() if cell else "" for cell in row]) for row in table]
                            page_content.append(f"\n[TABLE_START]\n" + "\n".join(rows) + "\n[TABLE_END]\n")

                    # 2. Trích xuất Text kỹ thuật số
                    text = p_plumber.extract_text()
                    if text:
                        page_content.append(text)

                    # 3. OCR với Tesseract cho các ảnh trong file
                    image_list = p_fitz.get_images(full=True)
                    for img_idx, img_info in enumerate(image_list):
                        try:
                            xref = img_info[0]
                            # Lấy ảnh chất lượng cao bằng cách phóng to (Matrix 2x2)
                            pix = doc_fitz.extract_image(xref)
                            img = Image.open(io.BytesIO(pix["image"]))
                            
                            # Tiền xử lý: Chuyển ảnh xám để tăng độ tương phản cho dấu tiếng Việt
                            img = ImageOps.grayscale(img)
                            
                            # Cấu hình Tesseract: lang='vie' cho tiếng Việt, 'eng' cho tiếng Anh
                            # --psm 1: Tự động phân đoạn trang
                            ocr_text = pytesseract.image_to_string(img, lang='vie+eng', config='--psm 1')
                            
                            if ocr_text.strip():
                                page_content.append(f"\n[OCR_IMAGE_{img_idx}]:\n{ocr_text.strip()}\n")
                        except Exception as e:
                            logger.error(f"Lỗi Tesseract trang {page_num+1}: {e}")

                    # Đóng gói trang
                    combined_text = "\n\n".join(page_content)
                    if combined_text.strip():
                        new_meta = base_metadata.copy()
                        new_meta.update({"page": page_num + 1, "source_type": "tesseract_hybrid"})
                        final_documents.append(Document(page_content=combined_text, metadata=new_meta))

            doc_fitz.close()
        except Exception as e:
            logger.error(f"Lỗi: {e}")
            
        return final_documents

    def load(self, file_path: str, doc_type: str = "general"):
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.pdf': return self.load_pdf(file_path, doc_type)
        # (Thêm load_docx tương tự các bản trước nếu cần)
        return []