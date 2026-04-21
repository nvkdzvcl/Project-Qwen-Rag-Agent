import os
import sys
import logging
from colorama import Fore, Style, init

# Thêm thư mục gốc vào sys.path để import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from backend.loader import SmartDocLoader
    from backend.splitter import SmartDocSplitter
except ImportError:
    print(Fore.RED + "Lỗi: Không tìm thấy các module trong folder backend.")
    sys.exit(1)

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Cấu hình file test (Bạn có thể đổi sang file PDF nếu muốn)
TARGET_FILE = "TH2 - HQTCSDL - QL ban hang.pdf"

def test_splitter_logic():
    loader = SmartDocLoader()
    # Khởi tạo splitter với chunk size nhỏ (vd: 500) để dễ quan sát việc cắt đoạn
    splitter = SmartDocSplitter(chunk_size=600, chunk_overlap=100)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(base_dir, "test_data", TARGET_FILE)

    if not os.path.exists(file_path):
        print(Fore.RED + f"File {TARGET_FILE} không tồn tại trong data/ để test.")
        return

    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.CYAN}TESTING: SMARTDOC SPLITTER LOGIC")
    print(f"{Fore.CYAN}{'='*80}\n")

    # 1. Load tài liệu thành các trang
    print(f"{Fore.YELLOW}1. Đang nạp tài liệu...")
    pages = loader.load(file_path, doc_type="StudyMaterial")
    print(f"   -> Đã load {len(pages)} trang gốc.\n")

    # 2. Chia nhỏ tài liệu
    print(f"{Fore.YELLOW}2. Đang thực hiện chia nhỏ (Chunking)...")
    chunks = splitter.split_documents(pages)
    print(f"   -> Đã tạo ra {len(chunks)} chunks.\n")

    # 3. Kiểm tra chi tiết 3 chunk đầu tiên
    print(f"{Fore.YELLOW}3. Kiểm tra chi tiết cấu trúc Chunk:")
    print(f"{Fore.GREEN}{'─'*80}")

    for i, chunk in enumerate(chunks[:3]):  # Xem mẫu 3 chunk đầu
        content = chunk.page_content
        meta = chunk.metadata
        
        print(f"{Fore.BLUE}[CHUNK {i+1}] | Source: {meta.get('source')} | Trang: {meta.get('page')}")
        print(f"{Fore.BLUE}Index in file: {meta.get('chunk_index_in_file')} | Start Index: {meta.get('start_index')}")
        
        # Hiển thị nội dung có đánh dấu bắt đầu/kết thúc chunk
        print(f"{Fore.WHITE} Nội dung trích dẫn:")
        print(f"{Style.DIM}   \"...{content[:200]}...\"") # In 200 ký tự đầu
        print(f"{Style.DIM}   \"...{content[-200:]}...\"") # In 200 ký tự cuối
        
        # Kiểm tra Metadata Filtering (Câu hỏi 8)
        print(f"{Fore.MAGENTA} Kiểm tra Metadata Filtering:")
        print(f"   - doc_type: {meta.get('doc_type')} (Sẵn sàng để lọc)")
        print(f"   - upload_date: {meta.get('upload_date')} (Sẵn sàng để sắp xếp)")
        
        print(f"{Fore.GREEN}{'─'*80}")

    # 4. Kiểm tra độ gối đầu (Overlap) giữa Chunk 1 và Chunk 2
    if len(chunks) > 1:
        print(f"\n{Fore.YELLOW}4. Kiểm tra độ gối đầu (Overlap):")
        # Tìm phần giao nhau (đơn giản hóa bằng cách kiểm tra vài từ cuối chunk 1 có trong chunk 2 không)
        last_words = " ".join(chunks[0].page_content.split()[-10:])
        if last_words in chunks[1].page_content:
            print(f"   {Fore.GREEN}[PASS] Phát hiện phần gối đầu giữa Chunk 1 và Chunk 2.")
        else:
            print(f"   {Fore.CYAN}[INFO] Không thấy phần gối đầu rõ rệt (có thể do splitter cắt ở đoạn văn mới).")

    # 5. Tổng kết
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{Fore.MAGENTA}KẾT QUẢ TEST SPLITTER:")
    print(f" - Tổng số Chunk: {len(chunks)}")
    print(f" - Kích thước Chunk trung bình: ~{sum(len(c.page_content) for c in chunks)//len(chunks)} ký tự")
    print(f" - Metadata toàn vẹn: {'Đúng' if 'source' in chunks[0].metadata else 'Sai'}")
    print(f"{Fore.CYAN}{'='*80}")

if __name__ == "__main__":
    test_splitter_logic()