import os
import logging
import pandas as pd
from langchain_community.document_loaders import PDFPlumberLoader, PyMuPDFLoader, Docx2txtLoader
from langchain_core.documents import Document

# Cấu hình logging
logger = logging.getLogger(__name__)

class SmartDocLoader:
    def __init__(self):
        self._supported_methods = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.xlsx': self._load_excel,
        }


    # Load PDF với logic kiểm tra OCR
    def _load_pdf(self, file_path: str):
        """
        Xử lý PDF: Thử trích xuất văn bản trước, nếu là PDF ảnh sẽ chuyển sang OCR.
        """
        
        logger.info(f"--- Đang phân tích PDF: {os.path.basename(file_path)} ---")
        
        # 1. Dùng PDFPlumber để trích xuất văn bản
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        
        # Kiểm tra xem có nội dung chữ không
        # Nếu tổng số ký tự quá ít (ví dụ < 100 ký tự cho cả file), khả năng cao là file ảnh
        full_text_len = sum(len(doc.page_content.strip()) for doc in documents)
        
        if full_text_len < 100:
            logger.warning(f"Phát hiện PDF dạng ảnh hoặc thiếu text layer. Đang kích hoạt OCR...")
            return self._load_pdf_with_ocr(file_path)
        
        # Bổ sung metadata cho PDF thông thường
        for i, doc in enumerate(documents):
            doc.metadata.update({"page": i + 1, "source_type": "text_pdf"})
            
        return documents


    # Load PDF với OCR tích hợp của PyMuPDF
    def _load_pdf_with_ocr(self, file_path: str):
        """
        Sử dụng PyMuPDF kết hợp OCR tích hợp để xử lý PDF dạng ảnh.
        """
        
        try:
            # PyMuPDFLoader hỗ trợ tham số extract_images và tích hợp OCR nếu được cấu hình
            # Dùng cách tiếp cận 'extract_images' của LangChain PyMuPDF
            loader = PyMuPDFLoader(file_path, extract_images=True)
            documents = loader.load()
            
            for i, doc in enumerate(documents):
                doc.metadata.update({"page": i + 1, "source_type": "ocr_pdf"})
                
            logger.info(f"Hoàn tất OCR cho {len(documents)} trang.")
            return documents
        except Exception as e:
            logger.error(f"Lỗi trong quá trình OCR: {str(e)}")
            return [Document(page_content=f"Lỗi OCR: {str(e)}", metadata={"source": file_path})]

    
    
    # Load DOCX
    def _load_docx(self, file_path: str):
        """
        Phương thức trích xuất dữ liệu từ file Word (.docx).
        """
        try:
            logger.info(f"--- Đang trích xuất Word: {os.path.basename(file_path)} ---")
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            
            # Bổ sung metadata để đồng bộ với PDF
            for doc in documents:
                doc.metadata.update({
                    "file_name": os.path.basename(file_path),
                    "source_type": "docx"
                })
            
            return documents
        except Exception as e:
            logger.error(f"Lỗi khi đọc file Word {file_path}: {str(e)}")
            raise
        
    
    # Load Excel
    def _load_excel(self, file_path: str):
    """
    Trích xuất dữ liệu từ Excel, lưu tên Sheet vào Metadata.
    Biến mỗi hàng dữ liệu thành một Document để AI dễ truy vấn.
    """
    try:
        logger.info(f"--- Đang trích xuất Excel: {os.path.basename(file_path)} ---")
        
        # Đọc tất cả các sheets cùng lúc
        excel_file = pd.ExcelFile(file_path)
        documents = []

        for sheet_name in excel_file.sheet_names:
            # Đọc từng sheet thành DataFrame
            df = excel_file.parse(sheet_name)
            
            # Loại bỏ các hàng hoàn toàn trống
            df = df.dropna(how='all')
            
            # Chuyển đổi toàn bộ Sheet thành định dạng văn bản (Markdown) 
            # để AI hiểu cấu trúc hàng/cột tốt nhất
            sheet_content = df.to_markdown(index=False)
            
            # Tạo metadata chi tiết
            metadata = {
                "source_type": "excel",
                "file_name": os.path.basename(file_path),
                "sheet_name": sheet_name,  # Lưu tên Sheet ở đây
                "total_rows": len(df),
                "total_columns": len(df.columns)
            }
            
            # Đóng gói thành Document
            doc = Document(page_content=sheet_content, metadata=metadata)
            documents.append(doc)

        logger.info(f" Đã nạp {len(documents)} sheets từ file Excel.")
        return documents

    except Exception as e:
        logger.error(f" Lỗi khi xử lý file Excel: {str(e)}")
        raise
    
    # Phương thức Public API để nạp file
    def load(self, file_path: str):
        """Phương thức Public API"""
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Không tìm thấy tệp: {file_path}")

        # Cắt đuôi file chuyển đuôi thành chữ thường để tránh lỗi
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Lấy handler tương ứng với định dạng file
        handler = self._supported_methods.get(file_ext)
        
        if not handler:
            raise ValueError(f"Định dạng {file_ext} không được hỗ trợ.")

        return handler(file_path)