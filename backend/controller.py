import logging
import gc
import time
from langchain_community.llms import Ollama
from backend.rag_pipeline import RagPipeline # Giữ nguyên đường dẫn import của bạn

# =========================================================
# BƯỚC 1: TẠO BỘ HỨNG LOG DÀNH CHO GIAO DIỆN
# =========================================================
class UILogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_queue = [] # Mảng lưu trữ log tạm thời

    def emit(self, record):
        # Bắt lấy từng dòng log, format nó và nhét vào mảng
        log_entry = self.format(record)
        self.log_queue.append(log_entry)

    def pull_logs(self):
        """Hàm này lấy log ra, đồng thời xóa log cũ đi để UI không bị in trùng lặp"""
        logs = self.log_queue.copy()
        self.log_queue.clear()
        return logs

# Khởi tạo bộ hứng và gắn nó vào Logger gốc (Root Logger) của toàn hệ thống
ui_handler = UILogHandler()
# Format log ngắn gọn, bỏ qua phần thời gian dài dòng để UI dễ nhìn
ui_handler.setFormatter(logging.Formatter('👉 %(message)s')) 
logging.getLogger().addHandler(ui_handler)

logger = logging.getLogger(__name__)

class RAGController:
    def __init__(self, default_model="qwen2.5:7b"):
        try:
            # 1. Khởi tạo Service Layer (Pipeline)
            self.pipeline = RagPipeline()
            self.llm = None
        
            # 2. Khởi tạo LLM
            logger.info(f"Đang kết nối với Model: {default_model}...")
            self.setup_llm(model_name=default_model)

        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo hệ thống RAGController: {str(e)}")

    def setup_llm(self, model_name: str, temperature: float = 0.7):
        """
        Hàm chuyên trách việc khởi tạo hoặc thay đổi LLM (Swap Model).
        Giúp linh hoạt đổi từ Qwen sang Llama, Phi... mà không load lại Pipeline.
        """
        logger.info(f"🔄 Đang kết nối/chuyển đổi sang model Ollama: {model_name}...")
        try:
            # Khởi tạo instance LLM mới
            new_llm = Ollama(
                model=model_name,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                num_ctx = 3072
            )
            
            # Cập nhật cho Controller
            self.llm = new_llm
            
            # QUAN TRỌNG: Cập nhật (Inject) ngay lập tức vào Pipeline để các chuỗi RAG dùng model mới
            self.pipeline.llm = self.llm
            
            logger.info(f"✅ Đã cấu hình thành công model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Lỗi khi thiết lập model {model_name}: {str(e)}")
            return False
        
    # ĐÃ SỬA LỖI: Thêm cờ clear_old=False để chống rò rỉ ngữ cảnh giữa các file
    def process_new_document(self, documents, clear_old=False):
        """
        Hàm này Role 1 sẽ gọi khi người dùng upload file PDF.
        Tham số clear_old=True cho phép ghi đè/xóa file cũ trước khi nạp file mới,
        giúp giải quyết triệt để lỗi "Retrieval Scope Leak".
        """
        logger.info(f"Bắt đầu xử lý tài liệu mới... (Chế độ xóa file cũ: {clear_old})")
        try:
            # Nếu giao diện truyền lệnh xóa tài liệu cũ, ta dọn sạch DB trước khi thêm
            if clear_old:
                logger.info("🧹 Đang dọn dẹp Database cũ để chuẩn bị nạp tài liệu độc lập...")
                self.clear_vector_store()
                
            # 1. Nạp dữ liệu vào FAISS (Vector Database)
            self.pipeline.create_database(documents)
            
            # 2. Nạp dữ liệu vào BM25 (Từ khóa) cho Hybrid Search (Câu 7)
            self.pipeline.build_hybrid_database(documents)
            
            logger.info("Hệ thống RAG đã sẵn sàng nhận câu hỏi!")
            return True, "Tài liệu đã được xử lý thành công!"
            
        except Exception as e:
            error_msg = f"Lỗi xử lý tài liệu. Đảm bảo file PDF không bị hỏng. Chi tiết: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def answer_question(self, question, session_id="default_session", filter_dict=None, advanced_mode=False):
        """
        Hàm này Role 1 sẽ gọi khi người dùng đặt câu hỏi.
        Frontend chỉ cần định danh user bằng session_id.
        Ví dụ filter_dict: {"doc_type": "pdf", "source": "chuong1.pdf"}
        """
        try:
            if not self.pipeline.vector_store:
                return {"answer": "Lỗi: Vui lòng upload tài liệu trước khi đặt câu hỏi!", "sources": []}
            
            logger.info(
                f"Query: '{question}' | Filter: '{filter_dict}' | Advanced: {advanced_mode}"
            )
        
            # Ủy quyền toàn bộ quy trình phức tạp (Reranking, Hybrid, Memory) cho Pipeline
            # Kết quả trả về là một Dictionary chứa cả 'answer' và 'sources'
            if advanced_mode:
                response = self.pipeline.ask_question_advanced(
                    question,
                    session_id=session_id,
                    filter_dict=filter_dict
                )
            else:
                response = self.pipeline.ask_question(
                    question,
                    session_id=session_id,
                    filter_dict=filter_dict
                )
            return response
        
        except ConnectionError as ce:
            logger.error(f"❌ Lỗi kết nối LLM: {str(ce)}")
            return {"answer": "⚠️ Lỗi: Không thể kết nối với Ollama. Vui lòng bật phần mềm Ollama!", "sources": []}
            
        except Exception as e:
            logger.error(f"❌ Lỗi không xác định khi truy vấn: {str(e)}", exc_info=True)
            return {"answer": f"⚠️ Lỗi xử lý câu hỏi. Chi tiết: {str(e)}", "sources": []}
        
    def answer_question_compare(self, question, session_id="compare_session", filter_dict=None):
        """
        Chạy so sánh song song giữa Standard RAG (Tốc độ) và Advanced RAG/Self-RAG (Chính xác).
        Hàm này trả về kết quả của cả 2 để UI vẽ thành 2 cột đối chiếu.
        """
        logger.info(f"⚖️ Bắt đầu chế độ COMPARE (So sánh 2 luồng RAG) cho câu hỏi: '{question}'")
        
        try:
            if not self.pipeline.vector_store:
                return {"answer": "Lỗi: Vui lòng upload tài liệu trước khi đặt câu hỏi!", "sources": []}
            
            logger.info(
                f"Query: '{question}' | Filter: '{filter_dict}'"
            )

            # -------------------------------------------------
            # LUỒNG 1: CHẠY RAG TIÊU CHUẨN (TỐC ĐỘ)
            # -------------------------------------------------
            logger.info("▶️ [COMPARE] Bắt đầu chạy luồng RAG Tiêu chuẩn...")
            start_standard = time.time()
            
            res_standard = self.pipeline.ask_question(
                question, session_id=f"{session_id}_std", filter_dict=filter_dict
            )
            
            res_standard['processing_time'] = round(time.time() - start_standard, 2)
            
            # -------------------------------------------------
            # XẢ RÁC RAM (CỨU CÁNH CHO MÁY 16GB)
            # -------------------------------------------------
            logger.info("🧹 Đang xả rác RAM/VRAM đệm để nhường tài nguyên cho luồng Advanced...")
            gc.collect()
            
            # -------------------------------------------------
            # LUỒNG 2: CHẠY ADVANCED RAG / SELF-RAG (CHẤT LƯỢNG)
            # -------------------------------------------------
            logger.info("▶️ [COMPARE] Bắt đầu chạy luồng Advanced RAG (Self-RAG)...")
            start_advanced = time.time()
            
            res_advanced = self.pipeline.ask_question_advanced(
                question, session_id=f"{session_id}_adv", filter_dict=filter_dict
            )
            
            res_advanced['processing_time'] = round(time.time() - start_advanced, 2)
            
            logger.info("🎉 Đã hoàn tất chế độ COMPARE!")
            
            # Trả về cục Data tổng để Frontend hiển thị
            return {
                "standard_rag": res_standard,
                "advanced_rag": res_advanced
            }
            
        except ConnectionError as ce:
            logger.error(f"❌ Lỗi kết nối LLM: {str(ce)}")
            return {"answer": "⚠️ Lỗi: Không thể kết nối với Ollama. Vui lòng bật phần mềm Ollama!", "sources": []}
        except Exception as e:
            logger.error(f"❌ Lỗi không xác định khi truy vấn: {str(e)}", exc_info=True)
            return {"answer": f"⚠️ Lỗi xử lý câu hỏi. Chi tiết: {str(e)}", "sources": []}

    # =========================================================
    # CÁC API HỖ TRỢ DỌN DẸP HỆ THỐNG (CÂU 3)
    # =========================================================
    
    def clear_vector_store(self):
        """
        Hàm API để Role 1 gọi khi user bấm nút 'Xóa Database / Xóa Tài liệu'.
        Hành động này ảnh hưởng đến toàn bộ hệ thống.
        """
        logger.info("Nhận API Request: Xóa toàn bộ Vector Database.")
        return self.pipeline.clear_vector_store()
        
    def clear_chat_history(self, session_id="default_session"):
        """
        Hàm API để Role 1 gọi khi user bấm nút 'Xóa Lịch sử Chat / New Chat'.
        Chỉ ảnh hưởng đến session của chính user đó.
        """
        logger.info(f"Nhận API Request: Xóa Lịch sử Chat cho session '{session_id}'.")
        return self.pipeline.clear_session_history(session_id=session_id)
    
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
    
    # =========================================================
    # API HỖ TRỢ GIAO DIỆN (UI) TRÍCH XUẤT LOG
    # =========================================================
    def get_system_logs(self) -> list[str]:
        """
        Role 1 (UI) gọi hàm này sau mỗi hành động (ví dụ sau khi user hỏi xong)
        để lấy toàn bộ quá trình suy nghĩ của Backend in lên màn hình.
        """
        return ui_handler.pull_logs()
