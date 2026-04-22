// ============================================================

import type { ChatApiResponse, ChatMessage, Entity, GraphData, MetricsData, Relation } from '../types';

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
  sessionId: string | null,
  userMessage: string,
  data: GraphData,
  inputText: string,
): Promise<ChatApiResponse> {
  const apiBase = await resolveApiBase();

  const res = await fetch(`${apiBase}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message: userMessage,
      entities: data.entities,
      relations: data.relations,
      input_text: inputText,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Chat error');
  }

  return res.json() as Promise<ChatApiResponse>;
}
