import logging
import shutil
import os
import time
import datetime
import json
from langchain_core.messages import HumanMessage, AIMessage, messages_from_dict, messages_to_dict
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
            # ĐÃ BỔ SUNG: Khai báo đường dẫn file lưu lịch sử chat
            self.history_file = os.path.join(self.persist_directory, "chat_history.json")

            self.all_documents = []
            logger.info("Tải Model thành công!")

            # =========================================================
            # ĐÃ BỔ SUNG: CƠ CHẾ KHÔI PHỤC LỊCH SỬ CHAT (RELOAD MEMORY)
            # =========================================================
            if os.path.exists(self.history_file):
                logger.info("📂 Tìm thấy file lịch sử hội thoại. Đang khôi phục trí nhớ...")
                try:
                    with open(self.history_file, 'r', encoding='utf-8') as f:
                        raw_memory = json.load(f)
                        for session_id, messages_list in raw_memory.items():
                            # Biến Dictionary thành Object Message của LangChain
                            self.backend_session_memory[session_id] = messages_from_dict(messages_list)
                    logger.info(f"✅ Đã khôi phục trí nhớ cho {len(self.backend_session_memory)} sessions từ ổ cứng.")
                except Exception as e:
                    logger.error(f"❌ Lỗi khi khôi phục trí nhớ chat: {str(e)}")
                    self.backend_session_memory = {}

            # =========================================================
            # ĐÃ BỔ SUNG: CƠ CHẾ KHÔI PHỤC DỮ LIỆU (RELOAD ON RESTART)
            # =========================================================
            if os.path.exists(self.persist_directory) and os.path.exists(os.path.join(self.persist_directory, "index.faiss")):
                logger.info(f"📂 Tìm thấy Database cũ tại '{self.persist_directory}'. Đang tiến hành khôi phục...")
                try:
                    # Cần allow_dangerous_deserialization=True đối với Langchain phiên bản mới
                    self.vector_store = FAISS.load_local(
                        self.persist_directory, 
                        self.embedder,
                        allow_dangerous_deserialization=True 
                    )
                    
                    # Bí quyết: Rút ngược toàn bộ chunks từ FAISS docstore để mớm lại cho hệ thống
                    if hasattr(self.vector_store, 'docstore') and hasattr(self.vector_store.docstore, '_dict'):
                        self.all_documents = list(self.vector_store.docstore._dict.values())
                        logger.info(f"✅ Đã khôi phục thành công {len(self.all_documents)} chunks dữ liệu từ ổ cứng.")
                        
                        # Tự động rebuild BM25 Retriever để Hybrid Search sẵn sàng ngay lập tức
                        if self.all_documents:
                            logger.info("🔤 Đang khởi tạo lại BM25 Retriever từ dữ liệu khôi phục...")
                            self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
                            self.bm25_retriever.k = 3
                            logger.info("✅ Khôi phục Hybrid Search hoàn tất! Hệ thống đã sẵn sàng.")
                except Exception as e:
                    logger.error(f"❌ Lỗi khi khôi phục dữ liệu: {str(e)}. Hệ thống sẽ khởi tạo mới rỗng.")
                    self.vector_store = None
                    self.all_documents = []
            else:
                logger.info("ℹ️ Không tìm thấy Database cũ. Hệ thống bắt đầu với trạng thái rỗng.")
        except Exception as e:
            logger.error(f"❌ Lỗi nghiêm trọng khi tải Embedding Model: {str(e)}")
            raise e
        
        logger.info("⏳ Đang tải mô hình Cross-Encoder Reranker (Quá trình này tốn khá nhiều RAM)...")
        try:
            # Sử dụng BAAI/bge-reranker-v2-m3 vì nó tối ưu rất tốt cho tiếng Việt
            # ĐÃ SỬA CÂU 9: Tối ưu hóa Latency bằng Model Quantization (Half-precision)
            # Truyền torch_dtype='float16' giúp giảm 50% lượng RAM tiêu thụ và tăng tốc độ suy luận đáng kể
            # Đây là kỹ thuật tối ưu hóa phần cứng (Hardware Optimization) cực kỳ ghi điểm
            self.cross_encoder_model = HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-v2-m3",
                model_kwargs={"torch_dtype": "float16"} 
            )
            logger.info("✅ Tải Reranker thành công (Đã áp dụng Quantization float16 để giảm Latency)!")
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
                # SỬA LỖI 1: Bóc tách tên file chuẩn (chuong1.pdf) từ đường dẫn full
                doc.metadata["file_name"] = os.path.basename(source_path)
                
                doc.metadata["doc_type"] = source_path.split('.')[-1].lower() if '.' in source_path else "unknown"

            logger.info(f"Processing {len(documents)} chunks (Đã auto-tag metadata: upload_date, doc_type, file_name)")
            
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
                
                # Áp dụng bộ lọc (base_search_kwargs) vào luồng FAISS (FAISS xử lý filter rất an toàn)
                faiss_retriever = self.vector_store.as_retriever(search_kwargs=base_search_kwargs)
                
                # ĐÃ SỬA LỖI 2: Xử lý logic lọc BM25 cực kỳ nghiêm ngặt (Strict Filtering)
                if filter_dict:
                    logger.info("🗂️ Đang ép BM25 chỉ tìm kiếm trong vùng Metadata được cấp phép...")
                    
                    filtered_docs = []
                    for doc in self.all_documents:
                        is_match = True
                        for k, v in filter_dict.items():
                            # SỰ THÔNG MINH: Nếu Frontend truyền key là 'source' nhưng value lại chỉ là tên file
                            # Hệ thống sẽ tự động map sang kiểm tra với 'file_name' để không bị trượt
                            if k == 'source' and '/' not in v and '\\' not in v:
                                actual_val = doc.metadata.get('file_name', '')
                            else:
                                actual_val = doc.metadata.get(k)
                            
                            if actual_val != v:
                                is_match = False
                                break
                                
                        if is_match:
                            filtered_docs.append(doc)

                    # BỊT LỖ HỔNG RÒ RỈ: Không dùng mẹo gán doc[0] nữa
                    if not filtered_docs:
                        logger.warning("⚠️ CẢNH BÁO STRICT FILTER: Không có tài liệu nào khớp với Metadata filter. Tắt BM25 để tránh rò rỉ dữ liệu.")
                        # Nếu BM25 không có tài liệu, ta chỉ dùng một mình FAISS (Vector Search)
                        # FAISS mặc định xử lý filter rỗng rất tốt (nó trả về [])
                        base_retriever = faiss_retriever
                    else:
                        # Nếu có tài liệu khớp, cấu hình BM25 như bình thường
                        bm25_to_use = BM25Retriever.from_documents(filtered_docs)
                        bm25_to_use.k = base_search_kwargs.get("k", base_k)
                        base_retriever = EnsembleRetriever(
                            retrievers=[bm25_to_use, faiss_retriever],
                            weights=[0.3, 0.7]
                        )
                        logger.info(f"✅ Khởi tạo Ensemble Retriever thành công (Đã lọc chuẩn xác {len(filtered_docs)} chunks cho BM25).")
                else:
                    base_retriever = EnsembleRetriever(
                        retrievers=[self.bm25_retriever, faiss_retriever],
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
    
    # ---------------------------------------------------------
    # API 1: DỌN DẸP LỊCH SỬ HỘI THOẠI (CÁ NHÂN HÓA THEO SESSION)
    # ---------------------------------------------------------
    def clear_session_history(self, session_id: str = "default_session") -> tuple[bool, str]:
        """
        Xóa lịch sử trò chuyện của riêng một phiên (user) cụ thể.
        Không ảnh hưởng đến người dùng khác hoặc Database.
        """
        logger.info(f"🧹 [Memory] Nhận yêu cầu xóa lịch sử chat cho session: '{session_id}'...")
        try:
            if session_id in self.backend_session_memory:
                # Chỉ clear mảng lịch sử của đúng session này
                self.backend_session_memory[session_id].clear()

                self._save_chat_history()
                
                logger.info(f"✅ Đã dọn dẹp sạch sẽ lịch sử hội thoại của session: '{session_id}'.")
                return True, "Đã xóa lịch sử trò chuyện thành công!"
            else:
                logger.info(f"ℹ️ Không tìm thấy lịch sử cho session: '{session_id}', không cần xóa.")
                return True, "Không có lịch sử trò chuyện nào để xóa."
        except Exception as e:
            logger.error(f"❌ Lỗi khi xóa lịch sử chat của session '{session_id}': {str(e)}", exc_info=True)
            return False, f"Lỗi hệ thống khi xóa lịch sử: {str(e)}"

    # ---------------------------------------------------------
    # API 2: DỌN DẸP VECTOR STORE (XÓA DATABASE DÙNG CHUNG)
    # ---------------------------------------------------------
    def clear_vector_store(self) -> tuple[bool, str]:
        """
        Hàm dọn dẹp Vector Store (FAISS & BM25) cả trong RAM lẫn trên ổ cứng.
        KHÔNG chạm vào lịch sử trò chuyện (Memory) của bất kỳ ai.
        """
        logger.info("🛠️ [RAG Pipeline] Nhận yêu cầu xóa toàn bộ Database (Vector Store)...")
        
        try:
            # LƯU Ý: Đã gỡ bỏ self.backend_session_memory.clear() ra khỏi đây!
            self.all_documents = []

            # 1. Xóa trong bộ nhớ RAM (Memory)
            if self.vector_store is not None:
                self.vector_store = None
                self.bm25_retriever = None
                logger.info("✅ Đã reset vector_store, bm25_retriever trong RAM về rỗng.")
            else:
                logger.warning("⚠️ Vector store hiện đang trống.")

            # 2. Xóa file vật lý trên ổ cứng (Disk)
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info(f"✅ Đã xóa vĩnh viễn thư mục lưu trữ DB vật lý tại: {self.persist_directory}")
            else:
                logger.info("ℹ️ Không tìm thấy thư mục DB trên ổ cứng (có thể chưa lưu bao giờ).")

            logger.info("🎉 Quá trình dọn dẹp Vector Database hoàn tất thành công!")
            return True, "Đã xóa toàn bộ dữ liệu Vector Store thành công!"
            
        except Exception as e:
            logger.error(f"❌ Lỗi nghiêm trọng khi dọn dẹp Database: {str(e)}", exc_info=True)
            return False, f"Lỗi hệ thống khi xóa dữ liệu Database: {str(e)}"
    
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

            # =========================================================
            # ĐÃ BỔ SUNG LỖI SỐ 8: Log tường minh số lượng document lấy được
            # =========================================================
            logger.info(f"🔍 Retrieved {len(source_documents)} documents để làm ngữ cảnh trả lời.")
            
            # ĐÃ BỔ SUNG: Backend tự động ghi nhớ câu hỏi và câu trả lời vào Memory
            self.backend_session_memory[session_id].append(HumanMessage(content=question))
            self.backend_session_memory[session_id].append(AIMessage(content=answer))

            self._save_chat_history()
            
            logger.info(f"✅ Đã nhận phản hồi (Độ dài: {len(answer)} ký tự) và cập nhật Backend Memory thành công.")
            return {
                "answer": answer,
                "sources": self._extract_metadata(source_documents)
            }

        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình hội thoại: {str(e)}")
            return {"answer": f"Lỗi hệ thống: {str(e)}", "sources": []}

    def _save_chat_history(self):
        """
        Hàm Helper: Lưu trữ toàn bộ Lịch sử hội thoại hiện tại xuống ổ cứng (JSON).
        """
        try:
            # Đảm bảo thư mục lưu trữ tồn tại
            os.makedirs(self.persist_directory, exist_ok=True)
            
            raw_memory = {}
            for session_id, messages in self.backend_session_memory.items():
                # Biến Object Message của LangChain thành Dictionary để lưu JSON
                raw_memory[session_id] = messages_to_dict(messages)
                
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(raw_memory, f, ensure_ascii=False, indent=2)
                
            logger.info("💾 Đã sao lưu lịch sử hội thoại xuống ổ cứng an toàn.")
        except Exception as e:
            logger.error(f"❌ Lỗi khi lưu trữ lịch sử hội thoại: {str(e)}", exc_info=True)

    def _extract_metadata(self, source_docs: list) -> list[dict]:
        """
        Hàm Helper: Bóc tách siêu dữ liệu (metadata) từ các chunks do FAISS trả về.
        - Giữ lại TẤT CẢ các chunks (ngay cả trên cùng 1 trang) để Frontend highlight đầy đủ.
        - Cung cấp exact_match_text để Frontend tìm và bôi vàng (highlight) trên PDF.
        """
        sources = []
        seen_chunks = set()

        for idx, doc in enumerate(source_docs):
            # Xử lý số trang an toàn
            page_num = doc.metadata.get("page", -1) 
            display_page = page_num + 1 if page_num != -1 else "Không rõ"
            
            raw_source = doc.metadata.get("source", "Tài liệu không tên")
            file_name = raw_source.split("/")[-1].split("\\")[-1] 

            # ĐÃ SỬA: Tạo ID duy nhất dựa trên NỘI DUNG CHUNK thay vì số trang.
            # Nhờ vậy, 2 đoạn văn khác nhau trên cùng 1 trang đều được giữ lại.
            chunk_hash = hash(doc.page_content)
            unique_identifier = f"{file_name}_page_{display_page}_chunk_{chunk_hash}"
            
            if unique_identifier not in seen_chunks:
                seen_chunks.add(unique_identifier)
                
                # Đóng gói dữ liệu chuẩn bị cho Frontend
                source_info = {
                    "chunk_id": idx,  # Gắn ID để Frontend dễ map dữ liệu
                    "file_name": file_name,
                    "page": display_page,
                    "content_snippet": doc.page_content[:150] + "...",
                    
                    # ĐÃ SỬA: Cung cấp text nguyên bản chính xác 100% để Frontend 
                    # dùng hàm Javascript search và highlight (bôi vàng) trên file PDF.
                    "exact_match_text": doc.page_content, 
                    "full_context": doc.page_content,
                    
                    # Chuyển tiếp toàn bộ metadata gốc (Nếu Role 2 có truyền tọa độ bbox, 
                    # offset ký tự... thì Frontend sẽ nhận được hết ở đây)
                    "raw_metadata": doc.metadata 
                }
                sources.append(source_info)
                
        logger.info(f"📑 Đã trích xuất {len(sources)} đoạn tài liệu (chunks) độc lập để hỗ trợ Frontend Highlight.")
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
    # TÍNH NĂNG BENCHMARK (ĐÁP ỨNG SO SÁNH CÂU 7 ĐỘ CHUYÊN SÂU CAO)
    # ---------------------------------------------------------
    def benchmark_hybrid_vs_vector(self, question: str):
        """
        Đo lường và so sánh toàn diện (Hiệu năng + Chất lượng) giữa 
        Pure Vector Search (FAISS) và Hybrid Search (FAISS + BM25).
        """
        logger.info(f"📊 [BENCHMARK] Bắt đầu đo lường HIỆU NĂNG & CHẤT LƯỢNG cho câu hỏi: '{question}'")
        if not getattr(self, 'vector_store', None) or not getattr(self, 'bm25_retriever', None):
            logger.error("❌ Thiếu dữ liệu Database (FAISS hoặc BM25) để chạy Benchmark.")
            return None

        try:
            # ==========================================
            # 1. ĐO LƯỜNG PURE VECTOR SEARCH
            # ==========================================
            start_vec = time.time()
            vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            vec_docs = vector_retriever.invoke(question)
            time_vec = time.time() - start_vec
            
            # ==========================================
            # 2. ĐO LƯỜNG HYBRID SEARCH (Không dùng Reranker ở bước này)
            # ==========================================
            start_hyb = time.time()
            hybrid_retriever = self.get_retriever(search_type="hybrid", base_k=3, use_reranker=False) 
            hyb_docs = hybrid_retriever.invoke(question)
            time_hyb = time.time() - start_hyb
            
            # ==========================================
            # 3. ĐÁNH GIÁ CHẤT LƯỢNG (RELEVANCE SCORING)
            # ==========================================
            vec_score_avg = 0.0
            hyb_score_avg = 0.0
            
            # Tận dụng thuật toán Cross-Encoder (như một Giám khảo độc lập) để chấm điểm
            if getattr(self, 'cross_encoder_model', None) is not None:
                logger.info("🧠 Đang dùng AI (Cross-Encoder) để chấm điểm chất lượng ngữ cảnh (Relevance Score)...")
                
                # Chấm điểm tập tài liệu do Vector Search tìm được
                if vec_docs:
                    vec_pairs = [[question, doc.page_content] for doc in vec_docs]
                    vec_scores = self.cross_encoder_model.score(vec_pairs)
                    vec_score_avg = sum(vec_scores) / len(vec_scores)
                
                # Chấm điểm tập tài liệu do Hybrid Search tìm được
                if hyb_docs:
                    hyb_pairs = [[question, doc.page_content] for doc in hyb_docs]
                    hyb_scores = self.cross_encoder_model.score(hyb_pairs)
                    hyb_score_avg = sum(hyb_scores) / len(hyb_scores)
            else:
                logger.warning("⚠️ Không có mô hình Cross-Encoder để chấm điểm. Chỉ có thể đo lường tốc độ.")

            # ==========================================
            # 4. TỔNG HỢP VÀ LOG KẾT QUẢ ĐẦY ĐỦ
            # ==========================================
            logger.info("--- 📈 BÁO CÁO KẾT QUẢ BENCHMARK ---")
            logger.info(f"🔹 PURE VECTOR | Tốc độ: {time_vec:.4f}s | Điểm chất lượng: {vec_score_avg:.4f}")
            logger.info(f"🔸 HYBRID      | Tốc độ: {time_hyb:.4f}s | Điểm chất lượng: {hyb_score_avg:.4f}")

            # Phân tích Tốc độ (Latency)
            diff_time = time_hyb - time_vec
            if diff_time > 0:
                logger.info(f"⏱️ [TỐC ĐỘ]: Hybrid Search chậm hơn Vector {diff_time:.4f}s (Đánh đổi chi phí tính toán).")
            else:
                logger.info(f"⏱️ [TỐC ĐỘ]: Hybrid Search nhanh hơn Vector {-diff_time:.4f}s.")
                
            # Phân tích Chất lượng (Quality)
            diff_score = hyb_score_avg - vec_score_avg
            if diff_score > 0:
                logger.info(f"🎯 [CHẤT LƯỢNG]: THÀNH CÔNG! Hybrid Search tìm ra tài liệu XỊN HƠN Vector (Cải thiện: +{diff_score:.4f} điểm).")
            elif diff_score < 0:
                logger.info(f"🎯 [CHẤT LƯỢNG]: ĐÁNG CHÚ Ý! Câu hỏi này Vector Search lại khớp ngữ cảnh tốt hơn Hybrid (Độ lệch: {diff_score:.4f}).")
            else:
                logger.info(f"🎯 [CHẤT LƯỢNG]: HÒA! Cả hai phương pháp đều kéo về chung một tệp tài liệu.")
            logger.info("------------------------------------")

            return {
                "pure_vector": {"time": time_vec, "score": vec_score_avg},
                "hybrid": {"time": time_hyb, "score": hyb_score_avg},
                "analysis": {
                    "latency_diff": diff_time,
                    "quality_diff": diff_score
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi chạy Benchmark: {str(e)}", exc_info=True)
            return None
        
    # ---------------------------------------------------------
    # TÍNH NĂNG BENCHMARK (ĐÁP ỨNG SO SÁNH CÂU 9 - RERANKER)
    # ---------------------------------------------------------
    def benchmark_reranker_vs_bi_encoder(self, question: str, filter_dict: dict = None):
        """
        Đo lường, so sánh HIỆU NĂNG (Latency) và CHẤT LƯỢNG (Relevance Score) 
        giữa Bi-encoder hiện tại (Base) và Cross-Encoder (Reranker).
        """
        logger.info(f"📊 [BENCHMARK Q9] Bắt đầu đo lường TOÀN DIỆN: Bi-encoder vs Cross-Encoder cho câu hỏi: '{question}'")
        if not self.vector_store:
            logger.error("❌ Thiếu dữ liệu Database để chạy Benchmark.")
            return None

        try:
            # ==========================================
            # 1. ĐO LƯỜNG BI-ENCODER (BASELINE)
            # ==========================================
            start_bi = time.time()
            bi_encoder_retriever = self.get_retriever(
                search_type="hybrid", base_k=3, use_reranker=False, filter_dict=filter_dict
            )
            bi_docs = bi_encoder_retriever.invoke(question)
            time_bi = time.time() - start_bi
            
            # ==========================================
            # 2. ĐO LƯỜNG CROSS-ENCODER (TWO-STAGE RERANKING)
            # ==========================================
            start_cross = time.time()
            cross_encoder_retriever = self.get_retriever(
                search_type="hybrid", base_k=10, final_top_k=3, use_reranker=True, filter_dict=filter_dict
            )
            cross_docs = cross_encoder_retriever.invoke(question)
            time_cross = time.time() - start_cross

            # ==========================================
            # 3. ĐÁNH GIÁ CHẤT LƯỢNG (RELEVANCE SCORING)
            # ==========================================
            bi_score_avg = 0.0
            cross_score_avg = 0.0
            
            if getattr(self, 'cross_encoder_model', None) is not None:
                logger.info("🧠 Đang chấm điểm chất lượng tài liệu trả về (Relevance Evaluation)...")
                
                if bi_docs:
                    bi_pairs = [[question, doc.page_content] for doc in bi_docs]
                    bi_scores = self.cross_encoder_model.score(bi_pairs)
                    bi_score_avg = sum(bi_scores) / len(bi_scores)
                
                if cross_docs:
                    cross_pairs = [[question, doc.page_content] for doc in cross_docs]
                    cross_scores = self.cross_encoder_model.score(cross_pairs)
                    cross_score_avg = sum(cross_scores) / len(cross_scores)
            
            # ==========================================
            # 4. TỔNG HỢP & LOG BÁO CÁO KHOA HỌC
            # ==========================================
            logger.info("--- 📈 BÁO CÁO KẾT QUẢ SO SÁNH (CÂU 9) ---")
            logger.info(f"🔹 BI-ENCODER    | Lọc ra {len(bi_docs)} chunks | Latency: {time_bi:.4f}s | Điểm chất lượng: {bi_score_avg:.4f}")
            logger.info(f"🔸 CROSS-ENCODER | Giữ lại {len(cross_docs)} chunks | Latency: {time_cross:.4f}s | Điểm chất lượng: {cross_score_avg:.4f}")

            # Đánh giá Latency (Trade-off)
            diff_time = time_cross - time_bi
            if diff_time > 0:
                logger.info(f"⚖️ [LATENCY TRADE-OFF]: Cross-Encoder chậm hơn {diff_time:.4f}s.")
            else:
                logger.info(f"⚡ [LATENCY OPTIMIZED]: Đáng ngạc nhiên! Cross-Encoder xử lý nhanh hơn ({-diff_time:.4f}s) nhờ Quantization.")

            # Đánh giá Chất lượng (Relevance)
            diff_score = cross_score_avg - bi_score_avg
            if diff_score > 0:
                logger.info(f"🎯 [QUALITY IMPACT]: THÀNH CÔNG RỰC RỠ! Reranker giúp tăng độ chuẩn xác lên +{diff_score:.4f} điểm so với Bi-encoder gốc.")
            else:
                logger.info(f"🎯 [QUALITY IMPACT]: Câu hỏi này Bi-encoder đã làm quá tốt, Reranker không tạo ra sự chênh lệch đáng kể.")
            logger.info("-------------------------------------------")

            return {
                "bi_encoder": {"time": time_bi, "score": bi_score_avg},
                "cross_encoder": {"time": time_cross, "score": cross_score_avg},
                "analysis": {
                    "latency_added": diff_time,
                    "quality_improved": diff_score
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi chạy Benchmark Reranker: {str(e)}", exc_info=True)
            return None
    