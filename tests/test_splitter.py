import os
import sys

# Thiết lập đường dẫn để import module từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.loader import SmartDocLoader
from backend.splitter import SmartDocSplitter

def verify_splitter():
    # 1. Cấu hình tham số kiểm tra
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    # Thêm tham số giả định để test metadata filtering
    DOC_TYPE = "BÀI TẬP SQL"
    FILE_PATH = "test_data/TH2 - HQTCSDL - QL ban hang.pdf" 

    if not os.path.exists(FILE_PATH):
        print(f"Lỗi: Không tìm thấy file {FILE_PATH}")
        return

    print(f"BẮT ĐẦU KIỂM TRA SPLITTER (PHIÊN BẢN CÂU 8)")
    print("=" * 70)

    # 2. Nạp tài liệu với Metadata đầy đủ
    print(f"--- Bước 1: Nạp file với nhãn '{DOC_TYPE}' ---")
    loader = SmartDocLoader()
    # Truyền thêm DOC_TYPE vào hàm load
    raw_docs = loader.load_pdf_with_ocr(FILE_PATH, doc_type=DOC_TYPE)

    # 3. Chạy Splitter
    print(f"--- Bước 2: Chia nhỏ tài liệu (Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}) ---")
    splitter = SmartDocSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw_docs)

    print(f"\nKẾT QUẢ KIỂM TRA:")
    print(f"Tổng số trang gốc: {len(raw_docs)}")
    print(f"Tổng số chunk tạo ra: {len(chunks)}")

    # 4. Kiểm tra chi tiết 2 chunk đầu tiên để xác minh OVERLAP
    if len(chunks) >= 2:
        print("\nKIỂM TRA ĐỘ GỐI ĐẦU (OVERLAP):")
        end_of_chunk0 = chunks[0].page_content[-CHUNK_OVERLAP:].strip()
        start_of_chunk1 = chunks[1].page_content[:CHUNK_OVERLAP + 50].strip()

        print(f"--- Cuối Chunk 0:\n...{end_of_chunk0}")
        print(f"--- Đầu Chunk 1:\n{start_of_chunk1}...")

        if end_of_chunk0[:20] in start_of_chunk1:
            print("\n  XÁC NHẬN: Overlap hoạt động chính xác!")
        else:
            print("\n  Cảnh báo: Overlap có thể không khớp chính xác do ngắt theo dấu xuống dòng.")

    # 5. Kiểm tra Metadata TOÀN DIỆN (Đáp ứng Câu hỏi 8)
    print("\nKIỂM TRA BẢO TOÀN METADATA (MULTI-DOC RAG):")
    test_chunk = chunks[0] # Kiểm tra chunk đầu tiên
    
    metadata = test_chunk.metadata
    fields = {
        "Nguồn (source)": metadata.get('source'),
        "Trang (page)": metadata.get('page'),
        "Phân loại (doc_type)": metadata.get('doc_type'),
        "Ngày upload (upload_date)": metadata.get('upload_date'),
        "Số thứ tự chunk (chunk_index)": metadata.get('chunk_index')
    }

    all_ok = True
    for label, value in fields.items():
        if value is not None:
            print(f"  {label}: {value}")
        else:
            print(f"  {label}: BỊ MẤT DỮ LIỆU")
            all_ok = False
    
    if all_ok:
        print("\n XÁC NHẬN: Metadata đáp ứng hoàn hảo yêu cầu lọc (Filtering) và trích dẫn nguồn.")

    # 6. Kiểm tra độ dài thực tế
    max_actual_size = max(len(c.page_content) for c in chunks)
    print(f"\n KIỂM TRA ĐỘ DÀI: Chunk lớn nhất đạt {max_actual_size} ký tự")
    
    if max_actual_size <= CHUNK_SIZE:
        print("  XÁC NHẬN: Độ dài nằm trong giới hạn cho phép.")

if __name__ == "__main__":
    verify_splitter()