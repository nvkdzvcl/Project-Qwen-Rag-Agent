import os
import sys

# Thiết lập đường dẫn để import được module backend
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from backend.loader import SmartDocLoader

def print_docs(docs, mode_name):
    """Hàm phụ trợ để in dữ liệu Document và Metadata mới"""
    if not docs:
        print(f"   [!] Không có dữ liệu được trích xuất trong chế độ {mode_name}.")
        return

    print(f"   ✅ {mode_name} thành công: Lấy được {len(docs)} đoạn/trang.")
    for i, doc in enumerate(docs):
        # Lấy thông tin từ metadata (Cập nhật các trường mới theo yêu cầu Câu 8)
        page_info = doc.metadata.get('page', 'N/A')
        source_file = doc.metadata.get('source', 'N/A')
        source_type = doc.metadata.get('source_type', 'N/A')
        upload_date = doc.metadata.get('upload_date', 'N/A')
        doc_type = doc.metadata.get('doc_type', 'N/A')
        
        # In nội dung (giới hạn 100 ký tự đầu để dễ nhìn)
        content = doc.page_content.strip().replace('\n', ' ')
        preview = (content[:100] + '..') if len(content) > 100 else content
        
        print(f"      - [Trang {page_info}][{source_type}]")
        print(f"        📂 File: {source_file} | Phân loại: {doc_type}")
        print(f"        ⏰ Upload: {upload_date}")
        print(f"        📝 Nội dung: {preview}")
        print("        " + "-"*30)

def run_test():
    loader = SmartDocLoader()
    test_dir = os.path.join(project_root, "test_data")

    if not os.path.exists(test_dir):
        print(f"❌ Thư mục {test_dir} không tồn tại.")
        return

    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.pdf', '.docx')) and not f.startswith('.')]

    if not files:
        print(f"❌ Không tìm thấy file .pdf hoặc .docx nào trong {test_dir}")
        return

    print(f"🚀 BẮT ĐẦU KIỂM TRA HỆ THỐNG LOADER (PHIÊN BẢN MULTI-DOC)")
    print("=" * 70)

    for file_name in files:
        file_path = os.path.join(test_dir, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        
        # Giả định loại tài liệu để test metadata filtering
        sample_doc_type = "BÀI TẬP SQL" if "TH2" in file_name else "TÀI LIỆU CHUNG"
        
        print(f"📄 TẬP TIN: {file_name}")

        # --- LUỒNG 1: LOAD STANDARD ---
        print(f"   🔹 Đang thử nghiệm hàm: load(doc_type='{sample_doc_type}')...")
        try:
            # Truyền thêm tham số doc_type vào đây
            std_docs = loader.load(file_path, doc_type=sample_doc_type)
            print_docs(std_docs, "Standard Load")
        except Exception as e:
            print(f"   ❌ Lỗi Standard Load: {e}")

        # --- LUỒNG 2: LOAD OCR ---
        if ext == '.pdf':
            print(f"   🔸 Đang thử nghiệm hàm: load_pdf_with_ocr(doc_type='{sample_doc_type}')...")
            try:
                # Truyền thêm tham số doc_type vào đây
                ocr_docs = loader.load_pdf_with_ocr(file_path, doc_type=sample_doc_type)
                print_docs(ocr_docs, "OCR Load")
            except Exception as e:
                print(f"   ❌ Lỗi OCR Load: {e}")
        
        print("-" * 70)

if __name__ == "__main__":
    run_test()