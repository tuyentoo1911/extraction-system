# Thiet Ke Giao Dien - Knowledge Graph Extractor

## 1) Muc tieu thiet ke UI/UX

- Luong thao tac ro rang: Input -> Extraction -> Graph -> Analysis -> Chat
- Uu tien kha nang doc hieu ket qua NER/graph cho nguoi khong ky thuat
- Giam so buoc thao tac lap lai
- Dam bao responsive cho laptop va man hinh lon

## 2) Cau truc man hinh chinh

## 2.1 Dashboard tong

- Header: ten du an, session, trang thai he thong
- Left panel: khu vuc nhap text/upload file + nut chay extraction
- Main content: cac tab chuc nang
- Right panel (tuy chon): quick stats va action nhanh (export, reset, save)

## 2.2 Cac tab chinh

- `Graph`: hien thi node-edge interactive, zoom/pan, click node xem chi tiet
- `Entities`: bang entity, loc theo type, tim kiem nhanh
- `Relations`: bang quan he, highlight theo node duoc chon
- `Metrics`: bieu dien metric chinh va bang top node
- `Insight`: bao cao dien giai bang ngon ngu tu nhien
- `Chatbot`: hoi dap theo ngu canh graph va lich su phien

## 3) User flow de xuat

1. Nguoi dung nhap van ban hoac upload PDF.
2. Bam "Extract" de sinh entity va relation.
3. Kiem tra ket qua trong tab Entities/Relations.
4. Quan sat cau truc trong tab Graph.
5. Xem metrics va insight de rut ket luan.
6. Dat cau hoi bo sung tai tab Chatbot.

## 4) Component level design

- InputPanel:
  - TextArea + upload file + validate input
  - Nut Extract, loading state, error state
- GraphCanvas:
  - Force graph canvas
  - Node tooltip, edge tooltip, mini legend
- DataTable:
  - Sort/filter/search
  - Pagination cho tap du lieu lon
- ChatPanel:
  - Message list, suggestion chips, input box
  - Session info va trang thai truy xuat context

## 5) Nguyen tac UI

- Mau sac:
  - Entity type co mau nhat quan giua cac tab
  - Trang thai thanh cong/canh bao/loi theo semantic color
- Typography:
  - Uu tien de doc, co cap bac heading ro rang
- Feedback:
  - Moi action quan trong deu co loading/success/error
- Accessibility:
  - Contrast dat nguong co ban
  - Co keyboard focus va label ro rang cho input

## 6) Wireframe text (don gian)

```text
+--------------------------------------------------------------+
| Header: Project | Session | Status                           |
+------------------------+-------------------------------------+
| Input Panel            | Tabs: Graph | Entities | ...        |
| - Text / PDF upload    |                                     |
| - Extract button       | Main View Content                   |
| - Validation messages  |                                     |
+------------------------+-------------------------------------+
| Footer: logs / notifications                                  |
+--------------------------------------------------------------+
```

## 7) Tieu chi nghiem thu giao dien

- Khong vo layout o do phan giai laptop pho bien
- Tinh nang chinh thao tac duoc trong <= 3 click tu dashboard
- Loading state khong gay "dang" giao dien
- Cac bang du lieu lon van thao tac muot o muc chap nhan duoc
