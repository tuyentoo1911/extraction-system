# Danh sách Backlog chức năng (Product Backlog)

Dưới đây là danh sách các chức năng (User Stories/Tasks) của hệ thống **Knowledge Graph Extractor**, được phân loại theo các module chính.

## Epic 1: Xử lý văn bản & Khai phá dữ liệu (NLP & Data Extraction)
| ID | Tên chức năng | Mô tả | Trạng thái dự kiến |
|----|---|---|---|
| F1.1 | Trích xuất thực thể (NER) | Chạy mô hình PhoBERT fine-tuned để nhận diện 9 loại thực thể (PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PERCENT, PRODUCT, EVENT, INDUSTRY) từ văn bản tiếng Việt. | Hoàn thành |
| F1.2 | Trích xuất và xây dựng quan hệ | Tự động nhận diện và kết nối các thực thể để tạo thành đồ thị tri thức (Knowledge Graph). | Hoàn thành |
| F1.3 | Xử lý file PDF | Bóc tách text từ file PDF upload (hỗ trợ tối đa 20MB, đa trang). | Hoàn thành |
| F1.4 | Dự đoán liên kết (Link Prediction) | Gợi ý các mối quan hệ có khả năng tồn tại giữa các node dựa trên mô hình Machine Learning (`influence_predictor`). | Hoàn thành |
| F1.5 | Cải tiến OCR (Tương lai) | Tích hợp Tesseract/PaddleOCR để xử lý các PDF dạng ảnh scan thay vì chỉ bóc tách text thuần. | Backlog |

## Epic 2: Giao diện trực quan & Trải nghiệm người dùng (Frontend)
| ID | Tên chức năng | Mô tả | Trạng thái dự kiến |
|----|---|---|---|
| F2.1 | Giao diện hiển thị đồ thị | Hiển thị đồ thị dạng mạng lưới tương tác (zoom, pan, click xem chi tiết node) sử dụng `react-force-graph-2d`. | Hoàn thành |
| F2.2 | Quản lý dạng Tab | Chuyển đổi mượt mà giữa các tính năng: Graph, Entities, Relations, Metrics, Insight, Highlight, Chatbot. | Hoàn thành |
| F2.3 | Highlight văn bản | Đánh dấu màu sắc các thực thể trực tiếp trên đoạn văn bản gốc giúp người dùng đối chiếu. | Hoàn thành |
| F2.4 | Bảng dữ liệu thực thể/quan hệ | Liệt kê chi tiết, lọc, và sắp xếp các entity/relation đã trích xuất. | Hoàn thành |
| F2.5 | Export đồ thị | Thêm tính năng tải xuống đồ thị (GEXF, GraphML, PNG) trực tiếp từ giao diện. | Backlog |

## Epic 3: Phân tích đồ thị (Graph Analytics)
| ID | Tên chức năng | Mô tả | Trạng thái dự kiến |
|----|---|---|---|
| F3.1 | Tính toán Graph Metrics | Sử dụng NetworkX để tính toán PageRank, Degree, Betweenness, v.v. | Hoàn thành |
| F3.2 | Tạo Insight Report | Tự động sinh báo cáo tổng hợp bằng Markdown về cấu trúc cộng đồng, Hub Nodes. | Hoàn thành |

## Epic 4: Chatbot thông minh & RAG
| ID | Tên chức năng | Mô tả | Trạng thái dự kiến |
|----|---|---|---|
| F4.1 | Rule-based Chat Engine | Hỗ trợ hỏi đáp nội bộ offline không cần LLM với 17+ loại intents (Chào hỏi, tìm kiếm đường đi, đếm số lượng node...). | Hoàn thành |
| F4.2 | LLM & RAG Integration | Tích hợp thuật toán BM25 và LLM (OpenAI/Gemini) để sinh câu trả lời tự nhiên từ Knowledge Base. | Hoàn thành |
| F4.3 | Bộ nhớ hội thoại (Memory) | Lưu trữ ngữ cảnh qua nhiều lượt hội thoại sử dụng PostgreSQL. | Hoàn thành |
| F4.4 | Streaming Chat | Thay thế HTTP bằng Server-Sent Events (SSE) để stream câu trả lời chữ chạy. | Backlog |

## Epic 5: Hệ thống Backend & Cơ sở hạ tầng
| ID | Tên chức năng | Mô tả | Trạng thái dự kiến |
|----|---|---|---|
| F5.1 | Restful APIs | Tạo các endpoint FastAPI xử lý tác vụ đồng thời (async). | Hoàn thành |
| F5.2 | Rate Limiting | Hạn chế số lượng request chống spam API với SlowAPI. | Hoàn thành |
| F5.3 | Quản lý Cơ sở dữ liệu | Thiết lập PostgreSQL connection pool. | Hoàn thành |
| F5.4 | Triển khai Docker | Container hóa hệ thống với Docker & Docker Compose. | Backlog |
| F5.5 | Phân quyền người dùng | Thêm JWT Auth để quản lý data theo multi-user. | Backlog |
