// ============================================================
// AI INTEGRATION — Gọi Python backend NER server
// Backend: server/main.py (FastAPI chạy trên port 8000)
// ============================================================

import type { ChatMessage, Entity, GraphData, MetricsData, Relation } from '../types';

const API_BASE_CANDIDATES = ['http://localhost:8000', 'http://localhost:8001'];
let resolvedApiBase: string | null = null;

async function resolveApiBase(): Promise<string> {
  if (resolvedApiBase) return resolvedApiBase;

  for (const base of API_BASE_CANDIDATES) {
    try {
      const res = await fetch(`${base}/health`);
      if (res.ok) {
        resolvedApiBase = base;
        return base;
      }
    } catch {
      // thử candidate tiếp theo
    }
  }
  throw new Error('Không kết nối được server. Hãy chạy: npm run server');
}

export async function getApiBase(): Promise<string> {
  return resolveApiBase();
}

// ── Kiểm tra server ─────────────────────────────────────────
async function checkServer(): Promise<void> {
  const apiBase = await resolveApiBase();
  try {
    const res = await fetch(`${apiBase}/health`);
    const data = await res.json();
    if (!data.model_ready) {
      const msg = data.model_error
        ? `Model lỗi: ${data.model_error}`
        : 'Model đang khởi động, vui lòng thử lại sau vài giây...';
      throw new Error(msg);
    }
  } catch (e: any) {
    if (e.message?.includes('fetch')) {
      throw new Error('Không kết nối được server. Hãy chạy: npm run server');
    }
    throw e;
  }
}

// ── callExtract ──────────────────────────────────────────────
export async function callExtract(
  text: string,
  useDeepAnalysis: boolean
): Promise<GraphData> {
  await checkServer();
  const apiBase = await resolveApiBase();

  const res = await fetch(`${apiBase}/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, use_deep_analysis: useDeepAnalysis }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Lỗi trích xuất thực thể');
  }

  return res.json() as Promise<GraphData>;
}

// ── callPredictLinks ─────────────────────────────────────────
export async function callPredictLinks(
  entities: Entity[],
  relations: Relation[],
  useDeepAnalysis: boolean
): Promise<Relation[]> {
  await checkServer();
  const apiBase = await resolveApiBase();

  const res = await fetch(`${apiBase}/predict-links`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entities, relations, use_deep_analysis: useDeepAnalysis }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Lỗi dự đoán liên kết');
  }

  const data = await res.json();
  return data.predicted_relations as Relation[];
}

export async function callMetrics(data: GraphData): Promise<MetricsData> {
  await checkServer();
  const apiBase = await resolveApiBase();
  const res = await fetch(`${apiBase}/metrics`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entities: data.entities, relations: data.relations }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Lỗi tính graph metrics');
  }
  return res.json() as Promise<MetricsData>;
}

// ── callInsight ──────────────────────────────────────────────
// Tạo insight phân tích từ dữ liệu đồ thị (không cần AI model)
export async function callInsight(
  _inputText: string,
  data: GraphData
): Promise<string> {
  const { entities, relations } = data;

  // Tính degree centrality
  const degree: Record<string, number> = {};
  relations.forEach(r => {
    degree[r.source] = (degree[r.source] || 0) + 1;
    degree[r.target] = (degree[r.target] || 0) + 1;
  });

  const sorted = [...entities].sort((a, b) => (degree[b.id] || 0) - (degree[a.id] || 0));
  const central = sorted[0];
  const centralDegree = degree[central?.id] || 0;

  // Thống kê loại entity
  const typeCounts: Record<string, number> = {};
  entities.forEach(e => { typeCounts[e.type] = (typeCounts[e.type] || 0) + 1; });
  const typeStats = Object.entries(typeCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([t, c]) => `**${t}**: ${c}`)
    .join(', ');

  // Quan hệ dự đoán
  const predicted = relations.filter(r => r.isPredicted).length;
  const actual = relations.length - predicted;

  return `## Tổng quan đồ thị tri thức

| Chỉ số | Giá trị |
|--------|---------|
| Tổng thực thể | **${entities.length}** |
| Tổng quan hệ | **${actual}** |
| Quan hệ dự đoán | **${predicted}** |

## Phân loại thực thể
${typeStats}

## Thực thể trung tâm
${central ? `**${central.name}** (${central.type}) — Degree Centrality: **${centralDegree}**
> Đây là node có nhiều kết nối nhất, đóng vai trò trung tâm trong mạng lưới quan hệ.` : '_Chưa xác định_'}

## Top 5 thực thể ảnh hưởng cao
${sorted.slice(0, 5).map((e, i) =>
  `${i + 1}. **${e.name}** (${e.type}) — ${degree[e.id] || 0} kết nối`
).join('\n')}

## Nhận xét
- Mạng lưới gồm **${entities.length} thực thể** và **${actual} quan hệ** được trích xuất tự động bằng mô hình NER.
- Tỷ lệ kết nối trung bình: **${entities.length > 0 ? (actual * 2 / entities.length).toFixed(1) : 0}** quan hệ/thực thể.
${predicted > 0 ? `- **${predicted} liên kết dự đoán** được thêm vào dựa trên phân tích mẫu loại thực thể.` : ''}
`;
}

// ── callChat ─────────────────────────────────────────────────
export async function callChat(
  messages: ChatMessage[],
  userMessage: string,
  data: GraphData,
  _inputText: string
): Promise<string> {
  // Rule-based Q&A dựa trên dữ liệu đồ thị
  const q = userMessage.toLowerCase();
  const { entities, relations } = data;

  const getEntityName = (id: string) => entities.find(e => e.id === id)?.name || id;

  // Hỏi về entity cụ thể
  const matchedEntity = entities.find(e =>
    q.includes(e.name.toLowerCase())
  );

  if (matchedEntity) {
    const rels = relations.filter(r =>
      r.source === matchedEntity.id || r.target === matchedEntity.id
    );
    const propText = matchedEntity.properties?.length
      ? matchedEntity.properties.map(p => `- **${p.key}**: ${p.value}`).join('\n')
      : '_Không có thuộc tính_';
    const relText = rels.length
      ? rels.map(r => {
          const other = r.source === matchedEntity.id
            ? getEntityName(r.target)
            : getEntityName(r.source);
          return `- ${r.label} → **${other}**`;
        }).join('\n')
      : '_Chưa có quan hệ_';

    return `## ${matchedEntity.name} (${matchedEntity.type})

**Thuộc tính:**
${propText}

**Quan hệ (${rels.length}):**
${relText}`;
  }

  // Hỏi về số lượng
  if (q.includes('bao nhiêu') || q.includes('tổng') || q.includes('số lượng')) {
    return `Đồ thị hiện có:
- **${entities.length}** thực thể
- **${relations.length}** quan hệ
- Loại thực thể: ${[...new Set(entities.map(e => e.type))].join(', ')}`;
  }

  // Hỏi về loại entity
  const typeKeywords: Record<string, string[]> = {
    'Person':       ['person', 'người', 'nhân vật', 'cá nhân'],
    'Organization': ['organization', 'tổ chức', 'công ty', 'doanh nghiệp'],
    'Location':     ['location', 'địa điểm', 'địa danh', 'nơi'],
    'Product':      ['product', 'sản phẩm', 'hàng hóa'],
    'Event':        ['event', 'sự kiện', 'hội nghị'],
    'Money':        ['money', 'tiền', 'giá trị', 'doanh thu'],
    'Date':         ['date', 'ngày', 'thời gian', 'năm'],
    'Industry':     ['industry', 'ngành', 'lĩnh vực'],
    'Percent':      ['percent', 'phần trăm', '%', 'tỷ lệ'],
  };
  for (const [type, keywords] of Object.entries(typeKeywords)) {
    if (keywords.some(kw => q.includes(kw))) {
      const filtered = entities.filter(e => e.type === type);
      return `## Danh sách ${type} (${filtered.length})
${filtered.map(e => `- **${e.name}**`).join('\n') || '_Không có_'}`;
    }
  }

  // Hỏi về quan hệ
  if (q.includes('quan hệ') || q.includes('liên kết') || q.includes('kết nối')) {
    const top5 = relations.slice(0, 5);
    return `## Một số quan hệ trong đồ thị

${top5.map(r => `- **${getEntityName(r.source)}** → *${r.label}* → **${getEntityName(r.target)}**`).join('\n')}

Tổng cộng **${relations.length}** quan hệ.`;
  }

  // Mặc định
  return `Tôi có thể trả lời các câu hỏi về **${entities.length} thực thể** và **${relations.length} quan hệ** trong đồ thị.

Ví dụ:
- "Cho tôi biết về [tên thực thể]"
- "Có bao nhiêu tổ chức?"
- "Liệt kê các ngày/thời gian"
- "Giá trị tiền tệ nào được đề cập?"
- "Liệt kê các quan hệ"

_Lưu ý: Chức năng hỏi đáp hiện dùng rule-based. Bạn có thể tích hợp LLM vào hàm \`callChat()\` trong \`src/lib/ai.ts\`._`;
}
