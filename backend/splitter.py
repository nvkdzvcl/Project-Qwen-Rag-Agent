import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SmartDocSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Khởi tạo bộ chia văn bản.
        :param chunk_size: Kích thước tối đa của mỗi đoạn (1000 là mức chuẩn cho RAG).
        :param chunk_overlap: Độ gối đầu (200 giúp giữ ngữ cảnh giữa các đoạn).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Cấu hình ưu tiên cắt theo đoạn văn (\n\n) rồi mới đến dòng (\n)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: list[Document]):
        """
        Thực hiện chia nhỏ danh sách Document và kiểm tra Metadata.
        """
        if not documents:
            logger.warning("Danh sách Document đầu vào trống.")
            return []
            
        logger.info(f"--- Đang chia nhỏ {len(documents)} trang thành các chunks ---")
        
        # LangChain sẽ tự động copy metadata từ Document gốc sang từng chunk
        chunks = self.splitter.split_documents(documents)
        
        # Bổ sung logic kiểm tra để đảm bảo đáp ứng "Câu hỏi 8"
        if chunks:
            sample_meta = chunks[0].metadata
            logger.info(f"Đã tạo {len(chunks)} chunks thành công.")
            logger.info(f"Metadata kiểm tra: Source='{sample_meta.get('source')}', Type='{sample_meta.get('doc_type')}'")
            
            # (Tùy chọn) Bạn có thể thêm số thứ tự chunk vào metadata nếu muốn hiển thị chi tiết hơn
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                
        return chunks