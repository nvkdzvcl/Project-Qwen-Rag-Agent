import logging
from langchain_community.llms import Ollama
from backend.rag_pipeline import RagPipeline # Giữ nguyên đường dẫn import của bạn

logger = logging.getLogger(__name__)

class RAGController:
    def __init__(self):
        try:
            # 1. Khởi tạo Service Layer (Pipeline)
            self.pipeline = RagPipeline()
        
            # 2. Khởi tạo LLM
            logger.info("Đang kết nối với Ollama Qwen2.5:7b...")
            self.llm = Ollama(
                model="qwen2.5:7b",
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1
            )
            
            # QUAN TRỌNG: Bơm LLM vào Pipeline để nó có thể chạy Conversational RAG (Câu 6)
            self.pipeline.llm = self.llm

        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo hệ thống RAGController: {str(e)}")

    def process_new_document(self, documents):
        """Hàm này Role 1 sẽ gọi khi người dùng upload file PDF"""
        logger.info("Bắt đầu xử lý tài liệu mới...")
        try:
            # 1. Nạp dữ liệu vào FAISS (Vector Database)
            self.pipeline.create_database(documents)
            
            # 2. Nạp dữ liệu vào BM25 (Từ khóa) cho Hybrid Search (Câu 7)
            # Nếu thiếu dòng này, hàm get_retriever("hybrid") sẽ báo lỗi
            self.pipeline.build_hybrid_database(documents)
            
            logger.info("Hệ thống RAG đã sẵn sàng nhận câu hỏi!")
            return True, "Tài liệu đã được xử lý thành công!"
            
        except Exception as e:
            error_msg = f"Lỗi xử lý tài liệu. Đảm bảo file PDF không bị hỏng. Chi tiết: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def answer_question(self, question, session_id="default_session", filter_dict=None):
        """
        Hàm này Role 1 sẽ gọi khi người dùng đặt câu hỏi.
        Frontend chỉ cần định danh user bằng session_id.
        Ví dụ filter_dict: {"doc_type": "pdf", "source": "chuong1.pdf"}
        """
        try:
            if not self.pipeline.vector_store:
                return {"answer": "Lỗi: Vui lòng upload tài liệu trước khi đặt câu hỏi!", "sources": []}
            
            logger.info(f"Query: '{question}' | Filter: '{filter_dict}'")
        
            # Ủy quyền toàn bộ quy trình phức tạp (Reranking, Hybrid, Memory) cho Pipeline
            # Kết quả trả về là một Dictionary chứa cả 'answer' và 'sources'
            response = self.pipeline.ask_question(question, session_id=session_id, filter_dict=filter_dict)
            return response
        
        except ConnectionError as ce:
            logger.error(f"❌ Lỗi kết nối LLM: {str(ce)}")
            return {"answer": "⚠️ Lỗi: Không thể kết nối với Ollama. Vui lòng bật phần mềm Ollama!", "sources": []}
            
        except Exception as e:
            logger.error(f"❌ Lỗi không xác định khi truy vấn: {str(e)}", exc_info=True)
            return {"answer": f"⚠️ Lỗi xử lý câu hỏi. Chi tiết: {str(e)}", "sources": []}

    def clear_all_data(self):
        """Hàm API để Role 1 gọi khi bấm nút Xóa Database (Câu 3)"""
        return self.pipeline.clear_database()
    
    # ĐÃ BỔ SUNG: API để chạy Performance Benchmark (Câu 7)
    def run_performance_benchmark(self, question):
        """
        Chạy so sánh tốc độ giữa Vector Search và Hybrid Search.
        Có thể gắn vào một nút bấm ẩn trên UI dành cho giảng viên kiểm tra.
        """
        return self.pipeline.benchmark_hybrid_vs_vector(question)
    
    # ĐÃ BỔ SUNG: API để chạy Performance Benchmark Reranker (Câu 9)
    def run_reranker_benchmark(self, question, filter_dict=None):
        """
        Đo lường độ trễ (latency) giữa Bi-encoder và Cross-encoder.
        Đáp ứng tiêu chí so sánh hiệu năng của Câu 9.
        """
        return self.pipeline.benchmark_reranker_vs_bi_encoder(question, filter_dict)