// ============================================================

import type {
  ChatApiResponse,
  ChatMessage,
  Entity,
  GraphData,
  MetricsData,
  Relation,
  TabId,
  WorkspaceSessionDetail,
  WorkspaceSessionSummary,
} from '../types';

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
  const payload = JSON.stringify({
    entities: data.entities,
    relations: data.relations,
    input_text: inputText,
  });

  const postInsight = async (base: string) =>
    fetch(`${base}/insight`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload,
    });

  const apiBase = await resolveApiBase();
  let res = await postInsight(apiBase);

  // Fail over to the other candidate when current backend returns server error.
  if (!res.ok && res.status >= 500) {
    const fallbackBase = API_BASE_CANDIDATES.find((base) => base !== apiBase);
    if (fallbackBase) {
      try {
        const health = await fetch(`${fallbackBase}/health`);
        if (health.ok) {
          const fallbackRes = await postInsight(fallbackBase);
          if (fallbackRes.ok) {
            resolvedApiBase = fallbackBase;
            res = fallbackRes;
          }
        }
      } catch {
        // Ignore fallback errors and surface the original response below.
      }
    }
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Insight generation error');
  }

  const responsePayload = await res.json();
  return responsePayload.insight_markdown as string;
}

/** Compact metrics text for chat / RAG (keep under server max length). */
export function formatMetricsSummaryForChat(m: MetricsData): string {
  const g = m.global_metrics;
  const lines: string[] = [
    `Global: nodes=${g.node_count} edges=${g.edge_count} density=${g.density.toFixed(4)} avg_degree=${g.avg_degree.toFixed(2)} connected_components=${g.connected_components}`,
  ];
  const row = (n: MetricsData['node_metrics'][number]) =>
    `${n.name} (${n.type}): degree=${n.degree} deg_cent=${n.degree_centrality.toFixed(4)} betw=${n.betweenness_centrality.toFixed(4)} close=${n.closeness_centrality.toFixed(4)} pr=${n.pagerank.toFixed(4)}`;

  const pushTop = (title: string, arr: MetricsData['top_degree']) => {
    if (!arr?.length) return;
    lines.push(`${title}:`);
    arr.slice(0, 12).forEach((n, i) => lines.push(`  ${i + 1}. ${row(n)}`));
  };

  pushTop('Top degree', m.top_degree);
  pushTop('Top PageRank', m.top_pagerank);
  pushTop('Top betweenness', m.top_betweenness);

  return lines.join('\n');
}

export async function callChat(
  sessionId: string | null,
  userMessage: string,
  data: GraphData,
  inputText: string,
  options?: {
    insightMarkdown?: string | null;
    metricsData?: MetricsData | null;
  },
): Promise<ChatApiResponse> {
  const apiBase = await resolveApiBase();

  const insightMarkdown =
    typeof options?.insightMarkdown === 'string' ? options.insightMarkdown.slice(0, 200_000) : '';
  const metricsSummary =
    options?.metricsData != null ? formatMetricsSummaryForChat(options.metricsData).slice(0, 80_000) : '';

  const res = await fetch(`${apiBase}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message: userMessage,
      entities: data.entities,
      relations: data.relations,
      input_text: inputText,
      insight_markdown: insightMarkdown,
      metrics_summary: metricsSummary,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Chat error');
  }

  return res.json() as Promise<ChatApiResponse>;
}

export async function listWorkspaceSessions(limit = 50): Promise<WorkspaceSessionSummary[]> {
  const apiBase = await resolveApiBase();
  const res = await fetch(`${apiBase}/workspace/sessions?limit=${limit}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Cannot load workspace history');
  }
  return res.json() as Promise<WorkspaceSessionSummary[]>;
}

export async function getWorkspaceSession(sessionId: string): Promise<WorkspaceSessionDetail> {
  const apiBase = await resolveApiBase();
  const res = await fetch(`${apiBase}/workspace/sessions/${sessionId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Cannot load workspace session');
  }
  return res.json() as Promise<WorkspaceSessionDetail>;
}

export async function saveWorkspaceSession(params: {
  sessionId: string | null;
  title?: string;
  inputText: string;
  graphData: GraphData | null;
  metricsData: MetricsData | null;
  insightMarkdown?: string | null;
  chatSessionId?: string | null;
  chatEngine?: 'ollama' | 'local' | 'llm' | 'rule-based' | null;
  chatHistory?: ChatMessage[];
  activeTab: TabId;
}): Promise<string> {
  const safeInsight =
    typeof params.insightMarkdown === 'string'
      ? params.insightMarkdown.slice(0, 200_000)
      : null;
  const safeChatHistory = (params.chatHistory ?? []).slice(-50);

  const apiBase = await resolveApiBase();
  const res = await fetch(`${apiBase}/workspace/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: params.sessionId,
      title: params.title ?? null,
      input_text: params.inputText,
      graph_data: params.graphData,
      metrics_data: params.metricsData,
      insight_markdown: safeInsight,
      chat_session_id: params.chatSessionId ?? null,
      chat_engine: params.chatEngine ?? null,
      chat_history: safeChatHistory,
      active_tab: params.activeTab,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Cannot save workspace session');
  }
  const data = await res.json();
  return data.session_id as string;
}

export async function deleteWorkspaceSession(sessionId: string): Promise<void> {
  const apiBase = await resolveApiBase();
  const res = await fetch(`${apiBase}/workspace/sessions/${sessionId}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Cannot delete workspace session');
  }
}
