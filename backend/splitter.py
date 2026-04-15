import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SmartDocSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Khởi tạo bộ chia văn bản với các tham số cụ thể.
        :param chunk_size: Kích thước tối đa của mỗi đoạn.
        :param chunk_overlap: Độ gối đầu giữa các đoạn.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = self._create_splitter()

    def _create_splitter(self):
        """Tạo đối tượng splitter từ LangChain"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split_documents(self, documents: list[Document]):
        """
        Thực hiện chia nhỏ danh sách Document.
        """
        if not documents:
            logger.warning("Danh sách Document đầu vào trống.")
            return []
            
        logger.info(f"--- Chia nhỏ với Size={self.chunk_size}, Overlap={self.chunk_overlap} ---")
        chunks = self.splitter.split_documents(documents)
        return chunks