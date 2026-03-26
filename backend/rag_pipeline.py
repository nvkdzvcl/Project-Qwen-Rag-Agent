import logging
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------
# ĐÁP ỨNG MỤC 7.2.5: THIẾT LẬP LOGGING HỆ THỐNG
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagPipeline:
    def __init__(self):
        """
        1. Tích hợp Embedding:
        Khởi tạo mô hình MPNet (768-dim) tối ưu cho tiếng Việt (Đáp ứng Mục 7.2.1).
        """
        logger.info("Đang tải Embedding Model (MPNet)...")
        try:
            self.embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", 
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True} 
            )
            self.vector_store = None
            logger.info("Tải Model thành công!")
        except Exception as e:
            logger.error(f"❌ Lỗi nghiêm trọng khi tải Embedding Model: {str(e)}")
            raise e

    def create_database(self, documents):
        """
        2. Quản lý Database Vector (FAISS):
        Lưu trữ vector vào bộ nhớ FAISS.
        """
        try:
            if not documents:
                raise ValueError("Không có dữ liệu để đưa vào database!")
            
        # Ghi log số lượng chunks đang được xử lý theo đúng format của thầy
            logger.info(f"Processing {len(documents)} chunks") 
            self.vector_store = FAISS.from_documents(documents, self.embedder) 
            logger.info("Khởi tạo FAISS Database thành công!")
            return True
        except Exception as e:
            logger.error(f"❌ Lỗi khi nạp dữ liệu vào FAISS: {str(e)}")
            raise e

    # ---------------------------------------------------------
    # ĐÁP ỨNG MỤC 7.2.4: MODIFY RETRIEVAL PARAMETERS (MMR)
    # ---------------------------------------------------------
    def get_retriever(self, search_type="mmr", k=5, fetch_k=30, lambda_mult=0.7):
        """
        3. Xây dựng luồng truy xuất:
        Hỗ trợ thuật toán MMR (Maximum Marginal Relevance) để tối ưu tính đa dạng.
        """
        try:
            if not self.vector_store:
                raise Exception("Database chưa được khởi tạo. Hãy gọi create_database() trước.")
            
            logger.info(f"Khởi tạo Retriever với chế độ: {search_type}")
        
            if search_type.lower() == "mmr":
                # Chế độ tìm kiếm nâng cao (MMR) theo cấu hình của thầy
                return self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": k,                 # Lấy top 5 kết quả
                        "fetch_k": fetch_k,     # Rảo qua 30 kết quả trước để chọn lọc
                        "lambda_mult": lambda_mult # Hệ số đa dạng hóa
                    }
                )
            else:
                # Fallback về chế độ tìm kiếm độ tương đồng (similarity) thông thường
                return self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )
        except Exception as e:
            logger.error(f"❌ Lỗi khi cấu hình Retriever: {str(e)}")
            raise e