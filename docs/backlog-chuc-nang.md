# Backlog Chuc Nang - Knowledge Graph Extractor

## 1) Muc tieu backlog

Backlog nay tong hop cac tinh nang uu tien cho he thong trich xuat do thi tri thuc, bao gom frontend, backend, chatbot, van hanh va chat luong.

## 2) Nguyen tac uu tien

- Muc do tac dong den gia tri nguoi dung (cao -> thap)
- Muc do bat buoc de van hanh on dinh
- Muc do phuc tap ky thuat va phu thuoc lien module
- Rui ro bao mat, hieu nang, kha nang mo rong

## 3) Danh sach backlog theo muc uu tien

| ID | Tinh nang | Mo ta ngan | Uu tien | Uoc luong | Phu thuoc | Trang thai |
|---|---|---|---|---|---|---|
| BL-01 | Upload va xu ly PDF lon | Toi uu xu ly file nhieu trang, giam loi timeout | P0 | 8 SP | API `/extract`, parser PDF | Todo |
| BL-02 | Batch extraction | Ho tro trich xuat nhieu tai lieu trong 1 lan | P0 | 13 SP | Queue job, luu tien trinh | Todo |
| BL-03 | Entity review workflow | Man hinh xac nhan/chinh sua entity truoc khi build graph | P0 | 8 SP | NER output schema | Todo |
| BL-04 | Relation validation | UI cho phep sua/xoa/them relation thu cong | P0 | 8 SP | Graph service | Todo |
| BL-05 | Export ket qua da dang | Export JSON, CSV, GEXF, GraphML co tuy chon | P1 | 5 SP | Graph builder | Todo |
| BL-06 | Version hoa do thi | Luu nhieu phien ban graph theo tai lieu/session | P1 | 13 SP | Database, storage | Todo |
| BL-07 | Search va filter nang cao | Loc theo entity type, do trung tam, thuoc tinh | P1 | 5 SP | Metrics API | Todo |
| BL-08 | Dashboard quan tri he thong | Theo doi API latency, loi, luu luong su dung | P1 | 8 SP | Logging + metrics | Todo |
| BL-09 | Chat memory governance | Chinh sach luu tru, xoa, anonymize hoi thoai | P1 | 5 SP | PostgreSQL schema | Todo |
| BL-10 | Prompt template management | Quan ly prompt theo nguc canh va version | P2 | 5 SP | Chat service | Todo |
| BL-11 | Authentication/Authorization | Login va phan quyen user (admin/editor/viewer) | P2 | 13 SP | User service + JWT | Todo |
| BL-12 | Multi-tenant du lieu | Tach du lieu theo to chuc/du an | P2 | 21 SP | DB design, middleware | Todo |
| BL-13 | Test automation E2E | Kich ban test tu upload -> graph -> chat | P2 | 8 SP | CI pipeline | Todo |
| BL-14 | i18n giao dien | Ho tro VN/EN cho dashboard va chatbot UI | P3 | 5 SP | Frontend labels | Todo |
| BL-15 | Plugin nguon du lieu | Ket noi docx/url/database lam input extraction | P3 | 13 SP | Ingestion layer | Todo |

## 4) Dinh nghia hoan thanh (Definition of Done)

- Co tai lieu yeu cau va acceptance criteria ro rang
- Co unit test hoac integration test phu hop
- Da review code va pass lint/test tren CI
- Da cap nhat tai lieu huong dan su dung/van hanh
- Da duoc Product Owner nghiem thu tren moi truong staging

## 5) Backlog refinement cadence

- Refinement hang tuan: 60-90 phut
- Chot pham vi sprint vao ngay dau sprint
- Danh gia lai uu tien backlog theo metric su dung thuc te va phan hoi nguoi dung
