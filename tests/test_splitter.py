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
    FILE_PATH = "test_data/Chương 3.docx"

    if not os.path.exists(FILE_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {FILE_PATH}")
        return

    # 2. Nạp tài liệu (Sử dụng hàm thường để lấy text nhanh)
    print(f"--- Đang nạp file: {os.path.basename(FILE_PATH)} ---")
    loader = SmartDocLoader()
    raw_docs = loader.load(FILE_PATH)

    # 3. Chạy Splitter
    splitter = SmartDocSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw_docs)

    print(f"\n📊 KẾT QUẢ KIỂM TRA:")
    print(f"✅ Tổng số trang gốc: {len(raw_docs)}")
    print(f"✅ Tổng số chunk tạo ra: {len(chunks)}")

    # 4. Kiểm tra chi tiết 2 chunk đầu tiên để xác minh OVERLAP
    if len(chunks) >= 2:
        print("\n🔍 KIỂM TRA ĐỘ GỐI ĐẦU (OVERLAP):")
        
        # Lấy phần cuối của chunk 0 và phần đầu của chunk 1
        end_of_chunk0 = chunks[0].page_content[-CHUNK_OVERLAP:].strip()
        start_of_chunk1 = chunks[1].page_content[:CHUNK_OVERLAP + 50].strip() # Lấy dư một chút để so khớp

        print(f"--- Cuối Chunk 0 (độ dài {CHUNK_OVERLAP}):\n...{end_of_chunk0}")
        print(f"--- Đầu Chunk 1 (độ dài {CHUNK_OVERLAP}):\n{start_of_chunk1}...")

        if end_of_chunk0[:20] in start_of_chunk1:
            print("\n✔️  XÁC NHẬN: Overlap hoạt động chính xác! Ngữ cảnh đã được nối tiếp.")
        else:
            print("\n⚠️ Cảnh báo: Overlap có thể không khớp chính xác do Splitter ưu tiên cắt tại dấu xuống dòng.")

    # 5. Kiểm tra Metadata
    print("\n🔍 KIỂM TRA METADATA:")
    test_chunk = chunks[-1] # Kiểm tra chunk cuối cùng
    source = test_chunk.metadata.get('source')
    page = test_chunk.metadata.get('page')
    
    if source and page:
        print(f"✔️  XÁC NHẬN: Metadata được bảo toàn (Trang: {page}, Nguồn: {os.path.basename(source)})")
    else:
        print("❌ Lỗi: Metadata bị mất trong quá trình split.")

    # 6. Kiểm tra độ dài thực tế
    max_actual_size = max(len(c.page_content) for c in chunks)
    print(f"\n📏 KIỂM TRA ĐỘ DÀI: Chunk lớn nhất đạt {max_actual_size} ký tự (Giới hạn: {CHUNK_SIZE})")
    
    if max_actual_size <= CHUNK_SIZE:
        print("✔️  XÁC NHẬN: Không có chunk nào vượt quá giới hạn cho phép.")

if __name__ == "__main__":
    verify_splitter()