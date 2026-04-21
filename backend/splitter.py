import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)
class SmartDocSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Cấu hình separators thông minh hơn
        # Ưu tiên cắt ở đoạn văn (\n\n), sau đó mới đến các điểm đánh dấu bảng hoặc xuống dòng
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True, # Lưu vị trí bắt đầu của chunk trong file gốc
            separators=["\n\n", "\n", "[TABLE_END]", " ", ""] 
        )

    def split_documents(self, documents: list[Document]):
        if not documents:
            logger.warning("Danh sách Document đầu vào trống.")
            return []
            
        logger.info(f"--- Đang chia nhỏ {len(documents)} trang thành các chunks ---")
        
        # 1. Thực hiện chia nhỏ
        chunks = self.splitter.split_documents(documents)
        
        # 2. Xử lý Metadata nâng cao
        # Chúng ta sẽ đếm số chunk cho TỪNG file riêng biệt
        source_counts = {} 

        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            
            # Khởi tạo hoặc tăng biến đếm cho mỗi file nguồn
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
            
            # Gán index theo từng file để dễ quản lý trong RAG
            chunk.metadata["chunk_index_in_file"] = source_counts[source]
            
            # (Tùy chọn) Xử lý tiêu đề tài liệu vào chunk nếu cần (Context Enrichment)
            # chunk.page_content = f"Tài liệu: {source}\n\n" + chunk.page_content

        logger.info(f"Đã tạo tổng cộng {len(chunks)} chunks từ {len(source_counts)} tài liệu.")
        return chunks