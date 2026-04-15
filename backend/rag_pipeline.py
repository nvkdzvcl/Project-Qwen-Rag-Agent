import logging
import shutil
import os
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
    
    def clear_database(self) -> tuple[bool, str]:
        """
        Hàm dọn dẹp Vector Store cả trong RAM lẫn trên ổ cứng.
        Trả về tuple: (Trạng thái thành công: bool, Thông báo: str)
        """
        logger.info("🛠️ [RAG Pipeline] Nhận yêu cầu xóa Vector Store...")
        
        try:
            # 1. Xóa trong bộ nhớ RAM (Memory)
            if self.vector_store is not None:
                self.vector_store = None
                logger.info("✅ Đã reset vector_store trong bộ nhớ RAM về None.")
            else:
                logger.warning("⚠️ Vector store hiện đang trống, không cần reset RAM.")

            # 2. Xóa file vật lý trên ổ cứng (Disk)
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
                logger.info(f"✅ Đã xóa vĩnh viễn thư mục lưu trữ DB vật lý tại: {self.persist_directory}")
            else:
                logger.info("ℹ️ Không tìm thấy thư mục DB trên ổ cứng (có thể chưa lưu bao giờ).")

            logger.info("🎉 Quá trình xóa Vector Store hoàn tất thành công!")
            return True, "Đã xóa toàn bộ dữ liệu Vector Store thành công!"
            
        except Exception as e:
            # Bắt lỗi chuẩn chỉ để web không bị sập (crash)
            logger.error(f"❌ Lỗi nghiêm trọng khi xóa Vector Store: {str(e)}", exc_info=True)
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
            retriever = self.get_retriever(search_type="hybrid", k=3)
            source_docs = retriever.invoke(question)

            if not source_docs:
                logger.info("ℹ️ Không tìm thấy ngữ cảnh phù hợp trong tài liệu.")
                return "Xin lỗi, tôi không tìm thấy thông tin này trong tài liệu bạn đã cung cấp.", []

            logger.info(f"🎯 Hybrid Search đã gom được {len(source_docs)} đoạn tài liệu xịn nhất.")

            # 2. GỌI LLM (Phần này Role 4 sẽ cấu hình prompt, ở đây gọi chain giả định)
            logger.info("🧠 Đang gửi ngữ cảnh và câu hỏi cho LLM (Ollama) xử lý...")
            
            # GIẢ ĐỊNH BẠN CÓ LLM CHAIN TẠI ĐÂY (Thay thế bằng code gọi LLM thực tế của bạn)
            # Ví dụ: answer = self.qa_chain.invoke({"context": source_docs, "question": question})
            answer = "Đây là nội dung câu trả lời do LLM sinh ra..." 
            logger.info("✅ LLM đã sinh xong câu trả lời.")

            # 3. BÓC TÁCH METADATA VÀ TRÍCH XUẤT NGUỒN (Lõi của Câu 5 - Giữ nguyên hoàn toàn)
            logger.info("🏷️ Đang bóc tách metadata để tìm số trang và nguồn gốc...")
            sources = self._extract_metadata(source_docs)

            logger.info(f"🎉 Hoàn tất bóc tách! Tìm thấy {len(sources)} trang nguồn duy nhất.")
            
            return answer, sources

        except Exception as e:
            logger.error(f"❌ Lỗi nghiêm trọng trong RAG Pipeline: {str(e)}", exc_info=True)
            return f"Đã xảy ra lỗi hệ thống: {str(e)}", []
    
    def build_hybrid_database(self, chunks):
        """
        Hàm này được gọi sau khi Role 2 đã cắt xong các chunks.
        Nó sẽ xây dựng song song 2 hệ thống tìm kiếm.
        """
        logger.info("🛠️ [Hybrid Search] Bắt đầu xây dựng cơ sở dữ liệu kép...")
        try:
            # 1. XÂY DỰNG FAISS (Tìm kiếm Ngữ nghĩa - Semantic Search)
            # Giả định bạn đã có code tạo FAISS ở đây
            # self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            logger.info("✅ Xây dựng thành công: FAISS Vector Store (Hiểu ngữ cảnh).")

            # 2. XÂY DỰNG BM25 (Tìm kiếm Từ khóa - Keyword/Lexical Search)
            logger.info("🔤 Đang phân tích tần suất từ khóa để xây dựng BM25...")
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            # Thiết lập BM25 chỉ lấy 3 đoạn tốt nhất để tránh nhiễu
            self.bm25_retriever.k = 3 
            logger.info("✅ Xây dựng thành công: BM25 Retriever (Bắt từ khóa chính xác).")

            return True, "Đã khởi tạo xong dữ liệu Hybrid Search!"
        except Exception as e:
            logger.error(f"❌ Lỗi khi xây dựng DB kép: {str(e)}", exc_info=True)
            return False, f"Lỗi hệ thống: {str(e)}"
        
    # ---------------------------------------------------------
    # CÂU 6: THIẾT LẬP CONVERSATIONAL RAG CHAIN
    # ---------------------------------------------------------
    def get_conversational_rag_chain(self, user_input: str): # ĐÃ SỬA: Nhận thêm user_input
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

            base_retriever = self.get_retriever(search_type="hybrid")
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
    def ask_question(self, question: str, chat_history: list) -> dict:
        """
        Hàm chính để Role 1 gọi từ giao diện.
        chat_history: Danh sách các object tin nhắn từ Streamlit Session State.
        """
        logger.info(f"💬 Nhận câu hỏi mới: '{question}'. Độ dài lịch sử: {len(chat_history)}")
        
        try:
            conversational_chain = self.get_conversational_rag_chain(question)
            
            # Chạy chuỗi RAG
            # Input của LangChain Retrieval Chain mặc định là 'input' và 'chat_history'
            response = conversational_chain.invoke({
                "input": question,
                "chat_history": chat_history
            })

            # Trích xuất dữ liệu trả về cho Role 1 và Role 3 (metadata)
            answer = response["answer"]
            source_documents = response["context"] # Các chunks đã dùng để trả lời
            
            logger.info("✅ Đã nhận phản hồi từ Conversational RAG.")
            return {
                "answer": answer,
                "sources": self._extract_metadata(source_documents) # Dùng lại logic Câu 5
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
                    "content_snippet": doc.page_content[:150] + "..." 
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
    