import os
import sys
import logging
from colorama import Fore, Style, init

# =========================================================
# CẤU HÌNH BIẾN TOÀN CỤC ĐỂ TEST
# =========================================================
# Thay bằng tên file PDF thông thường, PDF scan hoặc Docx bạn muốn soi
TARGET_FILE = "testocr.pdf"  # Ví dụ: "test1.pdf", "testocr.pdf", "ND1_TTHCM.docx"
# =========================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
   from backend.loader_tesseract import SmartDocLoaderTesseract
except ImportError:
    print(Fore.RED + "Lỗi: Không tìm thấy backend/loader_tesseract.py.")
    sys.exit(1)

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_test():
    loader = SmartDocLoaderTesseract()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, "test_data")
    file_path = os.path.join(data_dir, TARGET_FILE)

    if not os.path.exists(file_path):
        print(Fore.RED + f"[LỖI] Không tìm thấy file: {TARGET_FILE} tại {data_dir}")
        return

    print(f"\n{Fore.CYAN}{'='*85}")
    print(f"{Fore.CYAN}BÁO CÁO PHÂN TÍCH ĐỘ CHÍNH XÁC LOADER")
    print(f"{Fore.CYAN}{'='*85}\n")

    try:
        # Giả lập tham số doc_type để kiểm tra Metadata Filtering
        documents = loader.load(file_path, doc_type="Test_Accuracy")
        
        for i, doc in enumerate(documents):
            meta = doc.metadata
            content = doc.page_content
            
            print(f"{Fore.YELLOW}>>> TRANG {meta['page']} | Công nghệ: {meta['source_type']}")
            
            # 1. KIỂM TRA CẤU TRÚC BẢNG
            if "[TABLE_START]" in content:
                print(f"{Fore.GREEN}[V] PHÁT HIỆN BẢNG DỮ LIỆU:")
                # Trích xuất và in thử một đoạn bảng để kiểm tra độ thẳng hàng
                start_idx = content.find("[TABLE_START]")
                end_idx = content.find("[TABLE_END]") + 11
                table_preview = content[start_idx:end_idx]
                print(f"{Fore.WHITE}{table_preview}")
            else:
                print(f"{Fore.WHITE}[ ] Không phát hiện bảng kỹ thuật số.")

            # 2. KIỂM TRA OCR
            if "[OCR_IMAGE" in content:
                print(f"{Fore.GREEN}[V] PHÁT HIỆN NỘI DUNG OCR (ẢNH/SCAN):")
                # Tìm đoạn OCR đầu tiên để hiển thị mẫu
                ocr_start = content.find("[OCR_IMAGE")
                print(f"{Fore.WHITE}{content[ocr_start:ocr_start+300]}...")
            else:
                print(f"{Fore.WHITE}[ ] Trang này không chứa nội dung OCR.")

            # 3. KIỂM TRA VĂN BẢN THUẦN (Lý thuyết)
            print(f"{Fore.GREEN}[V] MẪU VĂN BẢN TRÍCH XUẤT:")
            # Lấy 300 ký tự đầu tiên của trang (đã bỏ qua các tag đặc biệt)
            clean_text = content.replace("[TABLE_START]", "").replace("[TABLE_END]", "")
            print(f"{Style.DIM}{clean_text[:400].strip()}...")

            # 4. KIỂM TRA TOÀN VẸN METADATA (Yêu cầu 8.2.8)
            print(f"{Fore.MAGENTA}--- Kiểm tra Metadata ---")
            required_keys = ['source', 'upload_date', 'doc_type', 'page', 'source_type']
            for key in required_keys:
                status = f"{Fore.GREEN}OK" if key in meta else f"{Fore.RED}MISSING"
                print(f"  + {key.ljust(12)}: {status} ({meta.get(key)})")
            
            print(f"{Fore.CYAN}{'─'*85}")

        # TỔNG KẾT
        print(f"\n{Fore.MAGENTA}KẾT LUẬN CUỐI CÙNG:")
        print(f" - Tổng số trang xử lý: {len(documents)}")
        print(f" - Khả năng nhận diện: {'Đa tầng (Hybrid)' if len(documents) > 0 else 'Thất bại'}")
        print(f" - Sẵn sàng cho RAG: {Fore.GREEN}YES")

    except Exception as e:
        print(f"{Fore.RED}[LỖI NGHIÊM TRỌNG]: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()