# Kế hoạch công việc theo Sprint (Sprint Plan)

Dự án được chia thành **5 Sprint**, mỗi Sprint kéo dài **2 tuần** áp dụng phương pháp Agile/Scrum.

## Sprint 1: Khởi tạo dự án & Xây dựng NER Core
**Mục tiêu:** Cài đặt cơ sở hạ tầng, train và tích hợp mô hình PhoBERT NER thành công.

- `Task 1.1`: Khởi tạo Repository, cấu hình dự án React Vite và FastAPI cơ bản.
- `Task 1.2`: Thu thập và làm sạch bộ dữ liệu NER tiếng Việt.
- `Task 1.3`: Huấn luyện mô hình PhoBERT gán nhãn 9 loại thực thể.
- `Task 1.4`: Xuất mô hình (safetensors) và viết API `/extract` để gọi suy luận (inference).
- `Task 1.5`: Viết module trích xuất thực thể thô và kết nối quan hệ cơ bản.

## Sprint 2: Frontend Đồ thị & Xử lý File
**Mục tiêu:** Hoàn thiện giao diện dashboard chính và hiển thị được Knowledge Graph trực quan.

- `Task 2.1`: Cài đặt Tailwind CSS và xây dựng layout Dashboard (các tab).
- `Task 2.2`: Tích hợp `react-force-graph-2d` để vẽ đồ thị từ JSON trả về.
- `Task 2.3`: Xây dựng tính năng Highlight thực thể trên văn bản.
- `Task 2.4`: Phát triển API `/upload-pdf` xử lý bóc tách text với `pypdf`.
- `Task 2.5`: Tích hợp upload PDF và API trích xuất trên UI.

## Sprint 3: Tính năng phân tích & Link Prediction
**Mục tiêu:** Tính toán độ đo toán học mạng lưới và ứng dụng Machine Learning để dự đoán quan hệ.

- `Task 3.1`: Tích hợp thư viện `networkx` tính các graph metrics (PageRank, Degree, CC).
- `Task 3.2`: Xây dựng tính năng tạo Insight Report (Markdown format).
- `Task 3.3`: Khai phá đặc trưng (Feature Engineering) cho bài toán Link Prediction.
- `Task 3.4`: Train mô hình phân loại (Scikit-learn) và lưu file `.joblib`.
- `Task 3.5`: Tạo API `/predict-links` và hiển thị các kết nối dự đoán (nét đứt) trên UI đồ thị.

## Sprint 4: Chatbot RAG & Hệ thống Memory
**Mục tiêu:** Đưa chatbot thông minh vào hệ thống, lưu trữ ngữ cảnh hội thoại.

- `Task 4.1`: Khởi tạo CSDL PostgreSQL và viết scripts di chuyển cấu trúc bảng (schema migration).
- `Task 4.2`: Xây dựng module Chat Memory lưu lịch sử session.
- `Task 4.3`: Phát triển Rule-based Chat Engine với 17+ intents thông dụng.
- `Task 4.4`: Tích hợp thuật toán Rank-BM25 để truy xuất tài liệu (RAG pipeline).
- `Task 4.5`: Viết Adapter kết nối API OpenAI / Gemini LLM.
- `Task 4.6`: Xây dựng UI Chatbot trên React, hiển thị lịch sử và suggestion chips.

## Sprint 5: Tối ưu hoá, Testing & Triển khai
**Mục tiêu:** Bảo vệ hệ thống, fix bugs, đánh giá chất lượng và đóng gói dự án.

- `Task 5.1`: Áp dụng Rate Limiting bảo vệ API bằng thư viện `slowapi`.
- `Task 5.2`: Đánh giá hiệu suất mô hình NER trên tập Test (tạo báo cáo F1 score).
- `Task 5.3`: Viết tài liệu hệ thống (README, Backlog, Architecture...).
- `Task 5.4`: Xử lý lỗi UI/UX, responsive cho các màn hình.
- `Task 5.5`: Đóng gói ứng dụng để test Local (Viết hướng dẫn chạy cho môi trường Windows/Linux).
