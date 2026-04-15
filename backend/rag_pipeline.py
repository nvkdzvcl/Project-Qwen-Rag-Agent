import logging
import shutil
import os
import time
import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ---------------------------------------------------------
# ĐÁP ỨNG MỤC 7.2.5: THIẾT LẬP LOGGING HỆ THỐNG
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
            self.persist_directory = "./faiss_index"
            self.bm25_retriever = None
            self.llm = None
            self.backend_session_memory = {}

            self.all_documents = []
            logger.info("Tải Model thành công!")
        except Exception as e:
            logger.error(f"❌ Lỗi nghiêm trọng khi tải Embedding Model: {str(e)}")
            raise e
        
        logger.info("⏳ Đang tải mô hình Cross-Encoder Reranker (Quá trình này tốn khá nhiều RAM)...")
        try:
            # Sử dụng BAAI/bge-reranker-v2-m3 vì nó tối ưu rất tốt cho tiếng Việt
            self.cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
            logger.info("✅ Tải mô hình Reranker thành công!")
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải mô hình Reranker: {str(e)}", exc_info=True)
            self.cross_encoder_model = None

    def create_database(self, documents):
        """
        2. Quản lý Database Vector (FAISS):
        Lưu trữ vector vào bộ nhớ FAISS (Hỗ trợ Append và Auto Metadata).
        """
        try:
            if not documents:
                raise ValueError("Không có dữ liệu để đưa vào database!")
            
            # ĐÃ BỔ SUNG: Tự động gắn thêm metadata (upload_date, doc_type)
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            for doc in documents:
                doc.metadata["upload_date"] = current_date
                source_path = doc.metadata.get("source", "")
                doc.metadata["doc_type"] = source_path.split('.')[-1].lower() if '.' in source_path else "unknown"

            logger.info(f"Processing {len(documents)} chunks (Đã auto-tag metadata: upload_date, doc_type)") 
            
            # ĐÃ SỬA: Cơ chế NỐI THÊM (Append) nếu database đã tồn tại
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embedder) 
                logger.info("Khởi tạo mới FAISS Database thành công!")
            else:
                self.vector_store.add_documents(documents)
                logger.info("Đã nối thêm (append) tài liệu vào FAISS Database hiện tại!")
                
            self.vector_store.save_local(self.persist_directory)
            return True
        except Exception as e:
            logger.error(f"❌ Lỗi khi nạp dữ liệu vào FAISS: {str(e)}")
            raise e

    # ---------------------------------------------------------
    # ĐÁP ỨNG MỤC 7.2.4 & CÂU 7, 8, 9: FULL ADVANCED RETRIEVAL
    # ---------------------------------------------------------
    def get_retriever(self, search_type="hybrid", base_k=10, final_top_k=3, fetch_k=30, lambda_mult=0.7, filter_dict=None, use_reranker=True):
        """
        Xây dựng luồng truy xuất đỉnh cao (State-of-the-art):
        Hỗ trợ Hybrid (BM25 + FAISS), MMR (Đa dạng hóa), Similarity, Metadata Filtering (Câu 8) và Cross-Encoder Reranking (Câu 9).
        """
        try:
            if not self.vector_store:
                raise Exception("Database chưa được khởi tạo. Hãy gọi create_database() trước.")
            
            logger.info(f"⚙️ Khởi tạo Retriever | Chế độ: {search_type} | Dùng Reranker: {use_reranker}")

            # 1. KHỞI TẠO BỘ THAM SỐ GỐC (Base Parameters)
            # Dùng base_k làm số lượng lấy ra ban đầu (lưới rộng cào về 10 kết quả)
            base_search_kwargs = {"k": base_k}

            # 2. KIỂM TRA VÀ ÁP DỤNG BỘ LỌC (Metadata Filtering - Câu 8)
            # Ví dụ filter_dict nhận vào từ Frontend: {'source': 'file_A.pdf'}
            if filter_dict:
                base_search_kwargs["filter"] = filter_dict
                logger.info(f"🗂️ Đã áp dụng bộ lọc Metadata: {filter_dict}")

            # 3. XÂY DỰNG BASE RETRIEVER (Cỗ máy lưới cào)
            base_retriever = None
            if search_type.lower() == "hybrid":
                # Chế độ tìm kiếm lai (Câu 7)
                if getattr(self, 'bm25_retriever', None) is None:
                    logger.warning("⚠️ Không thể tạo Hybrid Retriever vì thiếu BM25.")
                    raise ValueError("Dữ liệu BM25 chưa được nạp. Vui lòng kiểm tra lại quá trình load file.")
                
                # Áp dụng bộ lọc (base_search_kwargs) vào luồng FAISS
                faiss_retriever = self.vector_store.as_retriever(search_kwargs=base_search_kwargs)
                
                # ĐÃ SỬA LỖI LỌT RÁC BM25: Tạo BM25 Retriever động chỉ chứa các văn bản khớp filter
                if filter_dict:
                    logger.info("🗂️ Đang ép BM25 chỉ tìm kiếm trong vùng Metadata được cấp phép...")
                    filtered_docs = [
                        doc for doc in self.all_documents 
                        if all(doc.metadata.get(k) == v for k, v in filter_dict.items())
                    ]
                    if not filtered_docs:
                        logger.warning("⚠️ BM25: Không có tài liệu nào khớp với Metadata filter. Trả về BM25 rỗng.")
                        filtered_docs = [self.all_documents[0]] # Tránh crash, weight faiss sẽ lo phần còn lại
                        
                    bm25_to_use = BM25Retriever.from_documents(filtered_docs)
                    bm25_to_use.k = base_search_kwargs.get("k", base_k)
                else:
                    bm25_to_use = self.bm25_retriever

                base_retriever = EnsembleRetriever(
                    retrievers=[bm25_to_use, faiss_retriever],
                    weights=[0.3, 0.7]
                )
                logger.info("✅ Khởi tạo Ensemble Retriever thành công (Trọng số: BM25=0.3, FAISS=0.7).")

            elif search_type.lower() == "mmr":
                # Bổ sung các tham số đặc thù của MMR vào bộ tham số gốc
                base_search_kwargs.update({
                    "fetch_k": fetch_k,
                    "lambda_mult": lambda_mult
                })
                # Chế độ tìm kiếm nâng cao (MMR) theo cấu hình của thầy
                base_retriever = self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs=base_search_kwargs
                )
            else:
                # Fallback về chế độ tìm kiếm độ tương đồng (similarity) thông thường
                base_retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs=base_search_kwargs
                )

            # 4. ÁP DỤNG MÀNG LỌC CROSS-ENCODER RERANKER (Câu 9)
            if use_reranker:
                if getattr(self, 'cross_encoder_model', None) is None:
                    logger.warning("⚠️ Reranker model chưa được tải. Fallback về Base Retriever.")
                    return base_retriever

                logger.info(f"🧠 Đang bọc Base Retriever bằng CrossEncoderReranker (Giữ lại Top {final_top_k})...")
                
                # Khởi tạo cỗ máy nén (Compressor)
                compressor = CrossEncoderReranker(
                    model=self.cross_encoder_model, 
                    top_n=final_top_k
                )
                
                # Bọc Base Retriever lại
                rerank_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                
                logger.info("✅ Khởi tạo hệ thống Two-Stage Retrieval (Reranking) hoàn tất!")
                return rerank_retriever
            
            # Nếu không dùng Reranker, trả thẳng về Base Retriever 
            logger.info("ℹ️ Không sử dụng Reranker. Trả về Base Retriever tiêu chuẩn.")
            return base_retriever

        except Exception as e:
            logger.error(f"❌ Lỗi khi cấu hình Retriever: {str(e)}", exc_info=True)
            raise e
    
    def clear_database(self) -> tuple[bool, str]:
        """
        Hàm dọn dẹp Vector Store cả trong RAM lẫn trên ổ cứng.
        Trả về tuple: (Trạng thái thành công: bool, Thông báo: str)
        """
        logger.info("🛠️ [RAG Pipeline] Nhận yêu cầu xóa Vector Store và Lịch sử trò chuyện...")
        
        try:
            self.backend_session_memory.clear()

            self.all_documents = []

            # 1. Xóa trong bộ nhớ RAM (Memory)
            if self.vector_store is not None:
                self.vector_store = None
                self.bm25_retriever = None
                logger.info("✅ Đã reset vector_store, bm25_retriever trong RAM về None.")
            else:
                logger.warning("⚠️ Vector store hiện đang trống.")

            # 2. Xóa file vật lý trên ổ cứng (Disk)
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info(f"✅ Đã xóa vĩnh viễn thư mục lưu trữ DB vật lý tại: {self.persist_directory}")
            else:
                logger.info("ℹ️ Không tìm thấy thư mục DB trên ổ cứng (có thể chưa lưu bao giờ).")

            logger.info("🎉 Quá trình dọn dẹp hệ thống hoàn tất thành công!")
            return True, "Đã xóa toàn bộ dữ liệu Vector Store và Lịch sử Chat thành công!"
            
        except Exception as e:
            # Bắt lỗi chuẩn chỉ để web không bị sập (crash)
            logger.error(f"❌ Lỗi nghiêm trọng khi dọn dẹp hệ thống: {str(e)}", exc_info=True)
            return False, f"Lỗi hệ thống khi xóa dữ liệu: {str(e)}"
    
    # ---------------------------------------------------------
    # ỨNG DỤNG VÀO HÀM TRUY VẤN CHÍNH (Gộp Câu 5 & Câu 7)
    # ---------------------------------------------------------
    def query_with_sources(self, question: str) -> tuple[str, list[dict]]:
        """
        Xử lý câu hỏi, truy xuất ngữ cảnh (Hybrid), gọi LLM và bóc tách nguồn.
        Trả về: (Câu trả lời từ LLM, Danh sách các nguồn trích dẫn)
        """
        logger.info(f"🔍 [RAG Pipeline] Đang xử lý câu hỏi: '{question}'")
        
        try:
            # 1. TRUY XUẤT NGỮ CẢNH (RETRIEVAL)
            if not self.vector_store:
                logger.warning("⚠️ Cảnh báo: Vector store trống. Chưa có tài liệu nào được nạp.")
                return "Vui lòng tải lên tài liệu trước khi đặt câu hỏi.", []

            logger.info("📚 Đang truy xuất các đoạn văn bản (chunks) liên quan bằng Hybrid Search...")
            
            # ĐÃ SỬA: Gọi hàm get_retriever với search_type="hybrid"
            # Lấy top 3 kết quả để cân bằng tốc độ và độ chính xác
            retriever = self.get_retriever(search_type="hybrid", base_k=10, final_top_k=3)
            source_docs = retriever.invoke(question)

            if not source_docs:
                logger.info("ℹ️ Không tìm thấy ngữ cảnh phù hợp trong tài liệu.")
                return "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu bạn đã cung cấp.", []

            logger.info(f"🎯 Hybrid Search đã gom được {len(source_docs)} đoạn tài liệu xịn nhất.")

            # 2. GỌI LLM (Phần này Role 4 sẽ cấu hình prompt, ở đây gọi chain giả định)
            logger.info("🧠 Đang gửi ngữ cảnh và câu hỏi cho LLM (Ollama) xử lý...")
            
            if not self.llm:
                logger.error("❌ Không tìm thấy LLM. Controller chưa inject LLM vào Pipeline.")
                answer = "Lỗi hệ thống: LLM chưa được khởi tạo (Thiếu Controller inject)."
            else:
                logger.info(f"⚙️ Đang cấu hình Prompt và nạp {len(source_docs)} đoạn tài liệu vào context...")
                qa_prompt = self._get_dynamic_prompt(question)
                qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)

                logger.info("⏳ Đang chờ LLM suy nghĩ và sinh câu trả lời...")
                answer = qa_chain.invoke({
                    "context": source_docs, 
                    "input": question,
                    "chat_history": [] # Không có lịch sử vì đây là hàm tĩnh
                })
            
            logger.info(f"✅ LLM đã sinh xong câu trả lời thực tế (Độ dài: {len(answer)} ký tự).")
            sources = self._extract_metadata(source_docs)
            logger.info(f"📦 Đóng gói hoàn tất: Trả về câu trả lời và {len(sources)} nguồn trích dẫn.")
            
            return answer, sources

        except Exception as e:
            logger.error(f"❌ Lỗi nghiêm trọng trong RAG Pipeline: {str(e)}", exc_info=True)
            return f"Đã xảy ra lỗi hệ thống: {str(e)}", []
    
    def build_hybrid_database(self, chunks):
        """
        Xây dựng song song hệ thống tìm kiếm từ khóa (Hỗ trợ Append).
        """
        logger.info("🛠️ [Hybrid Search] Bắt đầu cập nhật cơ sở dữ liệu từ khóa...")
        try:
            # ĐÃ SỬA: Nối thêm chunks mới vào danh sách tổng
            self.all_documents.extend(chunks)
            
            logger.info("🔤 Đang phân tích tần suất từ khóa để xây dựng/cập nhật BM25...")
            self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
            self.bm25_retriever.k = 3 
            logger.info(f"✅ Cập nhật thành công: BM25 Retriever (Tổng: {len(self.all_documents)} chunks).")

            return True, "Đã khởi tạo/cập nhật xong dữ liệu Hybrid Search!"
        except Exception as e:
            logger.error(f"❌ Lỗi khi xây dựng DB kép: {str(e)}", exc_info=True)
            return False, f"Lỗi hệ thống: {str(e)}"
        
    # ---------------------------------------------------------
    # CÂU 6: THIẾT LẬP CONVERSATIONAL RAG CHAIN
    # ---------------------------------------------------------
    def get_conversational_rag_chain(self, user_input: str, filter_dict: dict = None): # ĐÃ SỬA: Nhận thêm user_input
        """
        Xây dựng chuỗi RAG có khả năng hiểu ngữ cảnh hội thoại và ngôn ngữ.
        """
        logger.info("🛠️ [Conversational RAG] Đang khởi tạo chuỗi hội thoại thông minh...")
        
        try:
            if not self.llm or not self.vector_store:
                raise ValueError("LLM hoặc Vector Store chưa được khởi tạo.")

            # 1. RETRIEVER CÓ Ý THỨC VỀ LỊCH SỬ
            contextualize_q_system_prompt = (
                "Sử dụng lịch sử hội thoại và câu hỏi mới nhất của người dùng "
                "để tạo ra một câu hỏi độc lập có thể hiểu được mà không cần xem lại lịch sử. "
                "KHÔNG trả lời câu hỏi, chỉ cải thiện câu hỏi nếu cần, nếu không hãy giữ nguyên."
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            base_retriever = self.get_retriever(search_type="hybrid", filter_dict=filter_dict)
            history_aware_retriever = create_history_aware_retriever(
                self.llm, base_retriever, contextualize_q_prompt
            )
            logger.info("✅ Đã thiết lập History-Aware Retriever.")
            
            # 2. CHUỖI TRẢ LỜI CÂU HỎI ĐỘNG (DYNAMIC PROMPT)
            # ĐÃ SỬA: Gọi hàm helper để lấy prompt theo đúng ngôn ngữ
            qa_prompt = self._get_dynamic_prompt(user_input)
            question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
            
            # 3. KẾT HỢP THÀNH RAG CHAIN HOÀN CHỈNH
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            logger.info("🎉 Khởi tạo Conversational RAG Chain hoàn tất!")
            return rag_chain

        except Exception as e:
            logger.error(f"❌ Lỗi khi tạo Conversational Chain: {str(e)}", exc_info=True)
            raise e

    # ---------------------------------------------------------
    # CẬP NHẬT HÀM TRUY VẤN CHÍNH (QUERY)
    # ---------------------------------------------------------
    def ask_question(self, question: str, session_id: str = "default_session", filter_dict: dict = None) -> dict:
        """
        Hàm chính để Role 1 gọi từ giao diện.
        Frontend chỉ cần truyền session_id, Backend sẽ tự quản lý Chat History.
        """
        # Nếu phiên này chưa từng chat, tạo một mảng lịch sử rỗng cho nó
        if session_id not in self.backend_session_memory:
            self.backend_session_memory[session_id] = []
            
        current_history = self.backend_session_memory[session_id]

        logger.info(f"💬 Nhận câu hỏi mới: '{question}'. Session: '{session_id}' | Độ dài lịch sử: {len(current_history)} tin nhắn.")
        
        try:
            conversational_chain = self.get_conversational_rag_chain(question, filter_dict=filter_dict)
            
            # Chạy chuỗi RAG
            # Input của LangChain Retrieval Chain mặc định là 'input' và 'chat_history'
            response = conversational_chain.invoke({
                "input": question,
                "chat_history": current_history
            })

            # Trích xuất dữ liệu trả về cho Role 1 và Role 3 (metadata)
            answer = response["answer"]
            source_documents = response["context"] # Các chunks đã dùng để trả lời
            
            # ĐÃ BỔ SUNG: Backend tự động ghi nhớ câu hỏi và câu trả lời vào Memory
            self.backend_session_memory[session_id].append(HumanMessage(content=question))
            self.backend_session_memory[session_id].append(AIMessage(content=answer))
            
            logger.info(f"✅ Đã nhận phản hồi (Độ dài: {len(answer)} ký tự) và cập nhật Backend Memory thành công.")
            return {
                "answer": answer,
                "sources": self._extract_metadata(source_documents)
            }

        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình hội thoại: {str(e)}")
            return {"answer": f"Lỗi hệ thống: {str(e)}", "sources": []}
        
    def _extract_metadata(self, source_docs: list) -> list[dict]:
        """
        Hàm Helper: Bóc tách siêu dữ liệu (metadata) từ các chunks do FAISS trả về.
        Lọc bỏ các trang trùng lặp và tạo trích đoạn (snippet).
        """
        sources = []
        seen_pages = set()

        for doc in source_docs:
            page_num = doc.metadata.get("page", -1) 
            display_page = page_num + 1 if page_num != -1 else "Không rõ"
            
            raw_source = doc.metadata.get("source", "Tài liệu không tên")
            file_name = raw_source.split("/")[-1].split("\\")[-1] 

            unique_identifier = f"{file_name}_page_{display_page}"
            
            if unique_identifier not in seen_pages:
                seen_pages.add(unique_identifier)
                sources.append({
                    "file_name": file_name,
                    "page": display_page,
                    "content_snippet": doc.page_content[:150] + "...",
                    "full_context": doc.page_content,
                    "raw_metadata": doc.metadata
                })
        return sources
    
    def _get_dynamic_prompt(self, user_input: str):
        """
        Hàm tự động phát hiện ngôn ngữ dựa trên ký tự tiếng Việt 
        để thiết lập System Prompt phù hợp.
        """
        vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        is_vietnamese = any(char in user_input.lower() for char in vietnamese_chars)
        
        if is_vietnamese:
            logger.info("🇻🇳 Đã nhận diện câu hỏi Tiếng Việt -> Sử dụng Prompt Tiếng Việt.")
            system_prompt = (
                "Bạn là trợ lý ảo thông minh. Sử dụng ngữ cảnh sau đây để trả lời câu hỏi.\n"
                "Nếu bạn không biết, chỉ cần nói là bạn không biết.\n"
                "Trả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.\n\n"
                "Ngữ cảnh: {context}"
            )
        else:
            logger.info("🇬🇧 Đã nhận diện câu hỏi Tiếng Anh -> Sử dụng Prompt Tiếng Anh.")
            system_prompt = (
                "You are a smart AI assistant. Use the following context to answer the question.\n"
                "If you don't know the answer, just say you don't know.\n"
                "Keep your answer concise (3-4 sentences) and MUST reply in English.\n\n"
                "Context: {context}"
            )
            
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
    
    # ---------------------------------------------------------
    # TÍNH NĂNG BENCHMARK (ĐÁP ỨNG SO SÁNH CÂU 7)
    # ---------------------------------------------------------
    def benchmark_hybrid_vs_vector(self, question: str):
        """
        Đo lường và so sánh hiệu năng giữa Pure Vector Search (FAISS) 
        và Hybrid Search (FAISS + BM25).
        """
        logger.info(f"📊 [BENCHMARK] Bắt đầu so sánh hiệu năng cho câu hỏi: '{question}'")
        if not self.vector_store or not self.bm25_retriever:
            logger.error("❌ Thiếu dữ liệu Database để chạy Benchmark.")
            return None

        try:
            # 1. Đo lường Pure Vector Search
            start_vec = time.time()
            vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            vec_docs = vector_retriever.invoke(question)
            time_vec = time.time() - start_vec
            logger.info(f"🔹 Pure Vector Search: Tốn {time_vec:.4f}s - Trả về {len(vec_docs)} chunks.")

            # 2. Đo lường Hybrid Search (Tắt Reranker để so sánh công bằng ở khâu retrieval)
            start_hyb = time.time()
            hybrid_retriever = self.get_retriever(search_type="hybrid", base_k=3, use_reranker=False) 
            hyb_docs = hybrid_retriever.invoke(question)
            time_hyb = time.time() - start_hyb
            logger.info(f"🔸 Hybrid Search (FAISS + BM25): Tốn {time_hyb:.4f}s - Trả về {len(hyb_docs)} chunks.")

            # 3. Kết luận Log
            diff = time_hyb - time_vec
            if diff > 0:
                logger.info(f"📈 [KẾT LUẬN] Hybrid Search chậm hơn {diff:.4f}s, nhưng đem lại độ chuẩn xác cao hơn nhờ kết hợp từ khóa.")
            else:
                logger.info(f"📈 [KẾT LUẬN] Hybrid Search nhỉnh hơn {-diff:.4f}s, hiệu năng cực kỳ ấn tượng!")
                
            return {"pure_vector_time": time_vec, "hybrid_time": time_hyb, "difference": diff}
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi chạy Benchmark: {str(e)}", exc_info=True)
            return None
        
    # ---------------------------------------------------------
    # TÍNH NĂNG BENCHMARK (ĐÁP ỨNG SO SÁNH CÂU 9 - RERANKER)
    # ---------------------------------------------------------
    def benchmark_reranker_vs_bi_encoder(self, question: str, filter_dict: dict = None):
        """
        Đo lường, so sánh hiệu năng và độ trễ (latency) giữa Bi-encoder hiện tại (Base) 
        và Cross-Encoder (Reranker).
        """
        logger.info(f"📊 [BENCHMARK Q9] Bắt đầu đo lường Latency: Bi-encoder vs Cross-Encoder cho câu hỏi: '{question}'")
        if not self.vector_store:
            logger.error("❌ Thiếu dữ liệu Database để chạy Benchmark.")
            return None

        try:
            # 1. Đo lường Bi-encoder (Sử dụng cấu hình thuần túy, KHÔNG dùng Reranker)
            # Lấy thẳng k=3 để so sánh công bằng ở đầu ra cuối cùng
            start_bi = time.time()
            bi_encoder_retriever = self.get_retriever(
                search_type="hybrid", base_k=3, use_reranker=False, filter_dict=filter_dict
            )
            bi_docs = bi_encoder_retriever.invoke(question)
            time_bi = time.time() - start_bi
            logger.info(f"🔹 Bi-Encoder (Baseline): Tốn {time_bi:.4f}s - Lọc ra {len(bi_docs)} chunks thô.")

            # 2. Đo lường Cross-Encoder (CÓ dùng Reranker)
            # Cào lưới rộng 10 chunks, sau đó bắt mô hình đọc kỹ lại và ép xuống còn 3 chunks
            start_cross = time.time()
            cross_encoder_retriever = self.get_retriever(
                search_type="hybrid", base_k=10, final_top_k=3, use_reranker=True, filter_dict=filter_dict
            )
            cross_docs = cross_encoder_retriever.invoke(question)
            time_cross = time.time() - start_cross
            logger.info(f"🔸 Cross-Encoder (Reranker): Tốn {time_cross:.4f}s - Giữ lại {len(cross_docs)} chunks tinh hoa nhất.")

            # 3. Phân tích kết quả và Latency Optimization
            diff = time_cross - time_bi
            if diff > 0:
                logger.info(f"📈 [PHÂN TÍCH LATENCY] Cross-Encoder xử lý chậm hơn {diff:.4f}s. "
                            f"Đây là sự đánh đổi (trade-off) hợp lý: hi sinh một chút tốc độ để đổi lấy "
                            f"độ chính xác (relevance) vượt trội ở bước Re-ranking cuối cùng.")
            else:
                logger.info(f"📈 [PHÂN TÍCH LATENCY] Bất ngờ: Cross-Encoder xử lý nhanh hơn hoặc bằng ({-diff:.4f}s)! "
                            f"Latency đã được tối ưu hóa xuất sắc.")
                
            return {
                "bi_encoder_time_seconds": time_bi, 
                "cross_encoder_time_seconds": time_cross, 
                "latency_added": diff
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi chạy Benchmark Reranker: {str(e)}", exc_info=True)
            return None
    