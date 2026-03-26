import time
from langchain_core.documents import Document
from backend.controller import RAGController

def run_comprehensive_test():
    print("🚀 BẮT ĐẦU KIỂM THỬ HỆ THỐNG THEO MỤC 6.2.1 (TEST CASES)...\n")
    
    # Khởi tạo hệ thống
    start_init = time.time()
    controller = RAGController()
    print(f"⏱ Thời gian khởi tạo LLM & Pipeline: {time.time() - start_init:.2f}s\n")
    
    # Giả lập dữ liệu phủ cả 3 Test Cases của thầy
    dummy_docs = [
        # Dữ liệu cho Test Case 1 (Technical manual)
        Document(page_content="Technical manual: To install the software, first download the package, extract it to the C: drive, and run setup.exe as Administrator."),
        # Dữ liệu cho Test Case 2 (Research paper)
        Document(page_content="Research findings: The integration of Agentic RAG reduces hallucination rates by 40%. The implication is that enterprise systems can now automate customer support with higher reliability without human oversight."),
        # Dữ liệu cho Test Case 3 (Cooking recipe)
        Document(page_content="Cooking recipe: To make a perfect omelette, beat 3 eggs with a pinch of salt, heat butter in a pan, and cook for 2 minutes on medium heat.")
    ]
    
    print("Mô phỏng nạp tài liệu PDF...")
    controller.process_new_document(dummy_docs)
    print("-" * 50)

    # ---------------------------------------------------------
    # TEST CASE 1: Simple Factual Question 
    # ---------------------------------------------------------
    print("\n▶ TEST CASE 1: Simple Factual Question (Expected: Step-by-step instructions)")
    tc1_query = "What is the installation procedure?"
    print(f"❓ Câu hỏi: {tc1_query}")
    start_q1 = time.time()
    ans1 = controller.answer_question(tc1_query)
    print(f"💡 Trả lời: {ans1}")
    print(f"⏱ Thời gian xử lý & sinh câu trả lời: {time.time() - start_q1:.2f}s")
    print("-" * 50)

    # ---------------------------------------------------------
    # TEST CASE 2: Complex Reasoning 
    # ---------------------------------------------------------
    print("\n▶ TEST CASE 2: Complex Reasoning (Expected: Summary với analysis)")
    tc2_query = "What are the main findings of the research and their implications?"
    print(f"❓ Câu hỏi: {tc2_query}")
    start_q2 = time.time()
    ans2 = controller.answer_question(tc2_query)
    print(f"💡 Trả lời: {ans2}")
    print(f"⏱ Thời gian xử lý & sinh câu trả lời: {time.time() - start_q2:.2f}s")
    print("-" * 50)

    # ---------------------------------------------------------
    # TEST CASE 3: Out-of-context Question 
    # ---------------------------------------------------------
    print("\n▶ TEST CASE 3: Out-of-context Question (Expected: 'I don't know' response)")
    tc3_query = "How to solve differential equations?"
    print(f"❓ Câu hỏi: {tc3_query}")
    start_q3 = time.time()
    ans3 = controller.answer_question(tc3_query)
    print(f"💡 Trả lời: {ans3}")
    print(f"⏱ Thời gian xử lý & sinh câu trả lời: {time.time() - start_q3:.2f}s")
    print("-" * 50)

if __name__ == "__main__":
    run_comprehensive_test()