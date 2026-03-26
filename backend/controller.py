from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from backend.rag_pipeline import RagPipeline
import logging

logger = logging.getLogger(__name__)

class RAGController:
    def __init__(self):
        try:
            # 1. Khởi tạo Pipeline (Role 3)
            self.pipeline = RagPipeline()
        
            # 2. BỔ SUNG: Cấu hình chuẩn LLM với các thông số tối ưu
            logger.info("Đang kết nối với Ollama Qwen2.5:7b...")
            self.llm = Ollama(
                model="qwen2.5:7b",
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1
            )
        
            self.retriever = None

        except Exception as e:
            logger.error(f"❌ Lỗi khởi tạo hệ thống RAGController: {str(e)}")

    # 3. BỔ SUNG: Hàm tự động phát hiện ngôn ngữ để thiết lập Prompt (Dựa trên Listing 6)
    def _get_dynamic_prompt(self, user_input):
        # Tập hợp các ký tự có dấu đặc trưng của tiếng Việt
        vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        is_vietnamese = any(char in user_input.lower() for char in vietnamese_chars)
        
        if is_vietnamese:
            prompt_text = """Sử dụng ngữ cảnh sau đây để trả lời câu hỏi.
Nếu bạn không biết, chỉ cần nói là bạn không biết.
Trả lời ngắn gọn (3-4 câu) BẮT BUỘC bằng tiếng Việt.

Ngữ cảnh: {context}
Câu hỏi: {input}
Trả lời:"""
        else:
            prompt_text = """Use the following context to answer the question.
If you don't know the answer, just say you don't know.
Keep answer concise (3-4 sentences).

Context: {context}
Question: {input}
Answer:"""
            
        return PromptTemplate.from_template(prompt_text)

    def process_new_document(self, documents):
        """Hàm này Role 1 sẽ gọi khi người dùng upload file PDF"""
        logger.info("Bắt đầu xử lý tài liệu mới...")
        try:
            self.pipeline.create_database(documents)
            self.retriever = self.pipeline.get_retriever()
            logger.info("Hệ thống RAG đã sẵn sàng nhận câu hỏi!")
        except Exception as e:
            error_msg = f"Lỗi xử lý tài liệu. Đảm bảo file PDF không bị hỏng. Chi tiết: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def answer_question(self, question):
        try:
            """Hàm này Role 1 sẽ gọi khi người dùng đặt câu hỏi"""
            if not self.retriever:
                return "Lỗi: Vui lòng upload tài liệu trước khi đặt câu hỏi!"
            
            # 4. BỔ SUNG: Ghi log câu hỏi đúng định dạng của thầy
            logger.info(f"Query: {question}")
        
            # Nhận diện ngôn ngữ và tạo prompt tương ứng
            dynamic_prompt = self._get_dynamic_prompt(question)
        
            # Móc nối LLM, Prompt và Retriever
            document_chain = create_stuff_documents_chain(self.llm, dynamic_prompt)
            qa_chain = create_retrieval_chain(self.retriever, document_chain)
        
            # Thực thi truy vấn
            response = qa_chain.invoke({"input": question})
        
            # 5. BỔ SUNG: Ghi log số lượng văn bản được truy xuất (Mục 7.2.5)
            retrieved_docs = response.get("context", [])
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
            return response["answer"]
        
        except ConnectionError as ce:
            # Bắt lỗi khi người dùng quên bật Ollama (Rất hay xảy ra)
            logger.error(f"❌ Lỗi kết nối LLM: {str(ce)}")
            return "⚠️ Lỗi: Không thể kết nối với Ollama. Vui lòng kiểm tra xem phần mềm Ollama đã được bật chưa!"
            
        except Exception as e:
            # Bắt các lỗi vặt khác (hết RAM, timeout,...)
            logger.error(f"❌ Lỗi không xác định khi truy vấn: {str(e)}")
            return f"⚠️ Lỗi xử lý câu hỏi. Vui lòng thử lại. Chi tiết cho Developer: {str(e)}"