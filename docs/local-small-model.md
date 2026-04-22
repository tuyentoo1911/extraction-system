# Chatbot voi model local nho

Muc tieu cua tai lieu nay la chay chatbot hien tai bang mot model local nho, khong phu thuoc OpenAI/Gemini. Backend da ho tro them provider `local_hf` trong `server/llm_client.py`.

## 1. Huong khuyen nghi

Cho project nay, nen di theo 2 giai doan:

1. Chay local inference truoc
2. Sau do moi fine-tune / instruction-tune

Ly do:
- Chatbot cua ban da co `RAG + graph context + rule fallback`
- Voi bai toan hoi dap theo KG/KB/context, chat luong prompt + retrieval thuong quan trong hon fine-tune som
- Fine-tune khong the thay du lieu context runtime; no chi giup model biet cach tra loi dung format va dung phong cach

## 2. Cau hinh local model

Them vao file `.env` o root project:

```env
LLM_PROVIDER=local_hf
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
LLM_DEVICE_MAP=auto
LLM_TORCH_DTYPE=auto
LLM_MAX_INPUT_TOKENS=3072
LLM_MAX_NEW_TOKENS=256
LLM_TEMPERATURE=0.2
LLM_TOP_P=0.9
```

Ban cung co the dat `LLM_MODEL` thanh duong dan local, vi du:

```env
LLM_MODEL=D:/models/qwen2.5-3b-instruct
```

## 3. Model nho nen thu

Neu uu tien tieng Viet + instruction following:

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`

Neu may yeu hon:

- `Qwen/Qwen2.5-0.5B-Instruct`

Luu y:
- 0.5B se rat nhe nhung chat luong suy luan KG va format tra loi se kem hon
- 1.5B hoac 3B thuong la diem can bang tot hon cho chatbot noi bo

## 4. Fine-tune dung bai toan

Neu ban muon fine-tune that su, nen fine-tune cho:

- format tra loi ngan gon
- uu tien relation 1-hop truoc 2-hop
- tu choi khi context khong du
- giu dung ngon ngu Viet/Anh theo user

Khong nen ky vong fine-tune de "nho" toan bo knowledge graph, vi KG/KB cua ban thay doi theo input. Phan do van nen dua vao context runtime.

Dataset SFT nen co dang:

```json
{
  "messages": [
    {"role": "system", "content": "Ban la AI Chatbot cua he thong Knowledge Graph Extractor..."},
    {"role": "user", "content": "Question: ...\n\nKnowledge Graph:\n...\n\nKnowledge Base:\n...\n\nInput text:\n...\n\nRAG context:\n..."},
    {"role": "assistant", "content": "[Thuc the A] --(quan he)--> [Thuc the B]"}
  ]
}
```

## 5. Cach gan model fine-tuned vao app

Sau khi fine-tune xong, ban khong can sua flow chatbot. Chi can doi:

```env
LLM_PROVIDER=local_hf
LLM_MODEL=D:/duong-dan/toi/checkpoint-hoac-model-da-fine-tune
```

Neu ban fine-tune bang LoRA/QLoRA, cach deploy on dinh nhat cho app nay la:

- merge adapter vao base model roi luu thanh mot thu muc model hoan chinh
- tro `LLM_MODEL` vao thu muc da merge do

## 6. Gioi han hien tai

- Moi truong Codex hien tai khong co Python runtime san sang, nen chua the test load model ngay tai day
- Backend da duong hoa san cho local inference, nhung ban van can cai model va chay tren may cua ban

## 7. Buoc tiep theo hop ly nhat

1. Thu `Qwen/Qwen2.5-1.5B-Instruct` hoac `Qwen/Qwen2.5-3B-Instruct`
2. Do chat luong tren bo cau hoi KG thuc te
3. Neu format chua deu, tao bo SFT nho de fine-tune
4. Sau do tro `LLM_MODEL` sang checkpoint fine-tuned
