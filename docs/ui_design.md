# Thiết kế Giao diện (UI Design)

Ứng dụng **Knowledge Graph Extractor** được thiết kế dưới dạng Single Page Application (SPA), hướng tới tính tiện dụng trong việc xem xét lượng thông tin lớn như văn bản và đồ thị mạng.

## 1. Phong cách thiết kế (Design System)
- **Màu sắc chủ đạo:** 
  - Nền hệ thống: Trắng/Xám nhạt (Sạch sẽ, chuẩn công cụ phân tích).
  - Màu nhấn (Accent color): Xanh dương (Blue) cho nút bấm chính và đường liên kết.
  - Bảng màu Thực thể (Entity Colors): Sử dụng palette phân biệt rõ ràng cho từng loại:
    - *PERSON:* Đỏ nhạt
    - *ORGANIZATION:* Xanh nước biển
    - *LOCATION:* Xanh lá cây
    - *DATE / TIME:* Tím
    - *MONEY:* Vàng / Cam
- **Font chữ:** Font San-serif hiện đại (Inter / Roboto) để đảm bảo độ đọc tốt cho văn bản dài.
- **Thư viện Icon:** `lucide-react` cho các biểu tượng giao diện.

## 2. Bố cục Màn hình chính (Dashboard Layout)

Màn hình chính được chia thành 2 khu vực dọc chính (Split Pane):

### 2.1 Khu vực trái: Bảng nhập liệu (Input Panel)
Chiếm khoảng **30-40%** chiều ngang.
- **Header:** Tiêu đề dự án và nút trạng thái kết nối backend.
- **Vùng Upload PDF:** Drag & drop khu vực thả file để trích xuất text.
- **Textarea:** Ô nhập liệu văn bản gốc có thanh scroll.
- **Nút hành động:** Nút "Trích xuất Thực thể & Quan hệ".
- **Ghi chú:** Có thanh điều chỉnh tỷ lệ chiều rộng (resize) giữa vùng trái và vùng phải.

### 2.2 Khu vực phải: Bảng Kết quả (Result/Visualization Panel)
Chiếm khoảng **60-70%** chiều ngang. Có một thanh Tab (Navigation bar) ở trên cùng để chuyển đổi các View.

#### Tab 1: Đồ thị (Graph View)
- Hiển thị không gian vẽ đồ thị bằng `react-force-graph-2d`.
- Các hạt (Nodes) đại diện cho thực thể, có tên ở giữa, màu tùy thuộc vào loại.
- Các đường thẳng (Edges) biểu thị quan hệ, có text chạy theo đường chỉ thị loại quan hệ.
- **Tương tác:** Kéo thả node, cuộn chuột zoom, hover hiện popup chi tiết.

#### Tab 2: Highlight Văn bản (Highlight View)
- Đoạn văn bản gốc được render lại.
- Những đoạn chữ là thực thể sẽ được bọc bởi một thẻ span có background màu theo loại nhãn (Ví dụ: [Vingroup](bg-blue) đầu tư...).

#### Tab 3: Thực thể (Entities) & Quan hệ (Relations)
- Dạng bảng (Table View).
- Hỗ trợ các cột: ID, Tên, Loại, Số lượng kết nối (Degree).
- Thanh search để lọc entity theo tên.

#### Tab 4: Thống kê & Phân tích (Metrics & Insight)
- Hiển thị báo cáo tự động Markdown (Insight).
- Hiển thị các chỉ số tính toán (PageRank, độ trung tâm) trên danh sách.

#### Tab 5: Chatbot (Trợ lý AI)
- Giao diện dạng bong bóng chat (Chat bubbles), giống Messenger/ChatGPT.
- User bên phải, Bot bên trái.
- Khu vực dưới cùng là ô nhập Text và Nút Gửi. Phía trên ô text là danh sách "Suggestion Chips" (Gợi ý câu hỏi) tự sinh từ đồ thị hiện tại.

## 3. Các thông báo phản hồi (Feedback/Modals)
- Sử dụng Toast notifications (góc trên bên phải) để báo hiệu các sự kiện: 
  - "Tải PDF thành công"
  - "Đang xử lý mô hình AI..." (kèm icon loading quay).
  - Lỗi kết nối Backend.
