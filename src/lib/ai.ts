// ============================================================
// AI integration - calls the Python backend server
// Backend: server/main.py (FastAPI on port 8000)
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
      // try next candidate
    }
  }
  throw new Error('Cannot connect to the server. Run: npm run server');
}

export async function getApiBase(): Promise<string> {
  return resolveApiBase();
}

async function checkServer(): Promise<void> {
  const apiBase = await resolveApiBase();
  try {
    const res = await fetch(`${apiBase}/health`);
    const data = await res.json();
    if (!data.model_ready) {
      const msg = data.model_error
        ? `Model error: ${data.model_error}`
        : 'Model is starting up. Please try again in a few seconds.';
      throw new Error(msg);
    }
  } catch (e: any) {
    if (e.message?.includes('fetch')) {
      throw new Error('Cannot connect to the server. Run: npm run server');
    }
    throw e;
  }
}

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
    throw new Error(err.detail || 'Entity extraction error');
  }

  return res.json() as Promise<GraphData>;
}

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
    throw new Error(err.detail || 'Link prediction error');
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
    throw new Error(err.detail || 'Graph metrics error');
  }
  return res.json() as Promise<MetricsData>;
}

export async function callInsight(
  inputText: string,
  data: GraphData
): Promise<string> {
  await checkServer();
  const apiBase = await resolveApiBase();
  const res = await fetch(`${apiBase}/insight`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      entities: data.entities,
      relations: data.relations,
      input_text: inputText,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Insight generation error');
  }

  const payload = await res.json();
  return payload.insight_markdown as string;
}

export async function callChat(
  messages: ChatMessage[],
  userMessage: string,
  data: GraphData,
  _inputText: string
): Promise<string> {
  const q = userMessage.toLowerCase();
  const { entities, relations } = data;

  const getEntityName = (id: string) => entities.find(e => e.id === id)?.name || id;

  const matchedEntity = entities.find(e => q.includes(e.name.toLowerCase()));

  if (matchedEntity) {
    const rels = relations.filter(r => r.source === matchedEntity.id || r.target === matchedEntity.id);
    const propText = matchedEntity.properties?.length
      ? matchedEntity.properties.map(p => `- **${p.key}**: ${p.value}`).join('\n')
      : '_No properties_';
    const relText = rels.length
      ? rels.map(r => {
          const other = r.source === matchedEntity.id ? getEntityName(r.target) : getEntityName(r.source);
          return `- ${r.label} -> **${other}**`;
        }).join('\n')
      : '_No relations yet_';

    return `## ${matchedEntity.name} (${matchedEntity.type})\n\n**Properties:**\n${propText}\n\n**Relations (${rels.length}):**\n${relText}`;
  }

  if (q.includes('how many') || q.includes('count') || q.includes('total')) {
    return `Current graph:\n- **${entities.length}** entities\n- **${relations.length}** relations\n- Entity types: ${[...new Set(entities.map(e => e.type))].join(', ')}`;
  }

  const typeKeywords: Record<string, string[]> = {
    Person: ['person', 'people', 'human'],
    Organization: ['organization', 'company', 'business'],
    Location: ['location', 'place', 'country', 'city'],
    Product: ['product'],
    Event: ['event'],
    Money: ['money', 'revenue', 'value'],
    Date: ['date', 'time', 'year'],
    Industry: ['industry', 'sector'],
    Percent: ['percent', 'ratio'],
  };

  for (const [type, keywords] of Object.entries(typeKeywords)) {
    if (keywords.some(kw => q.includes(kw))) {
      const filtered = entities.filter(e => e.type === type);
      return `## ${type} list (${filtered.length})\n${filtered.map(e => `- **${e.name}**`).join('\n') || '_None_'}`;
    }
  }

  if (q.includes('relation') || q.includes('link') || q.includes('connection')) {
    const top5 = relations.slice(0, 5);
    return `## Sample relations\n\n${top5.map(r => `- **${getEntityName(r.source)}** -> *${r.label}* -> **${getEntityName(r.target)}**`).join('\n')}\n\nTotal **${relations.length}** relations.`;
  }

  return `I can answer questions about **${entities.length} entities** and **${relations.length} relations** in the graph.\n\nExamples:\n- "Tell me about [entity name]"\n- "How many organizations are there?"\n- "List dates or times"\n- "What money values are mentioned?"\n- "List relations"\n\n_Current chat remains rule-based. You can later plug an LLM into callChat() in src/lib/ai.ts._`;
}
