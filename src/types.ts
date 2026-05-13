export interface EntityProperty {
  key: string;
  value: string;
}

export interface Entity {
  id: string;
  name: string;
  type: string;
  properties?: EntityProperty[];
}

export interface Relation {
  source: string;
  target: string;
  label: string;
  isPredicted?: boolean;
  confidence?: number; // 0..1, only for predicted links
}

export interface GraphData {
  entities: Entity[];
  relations: Relation[];
}

export interface NodeMetrics {
  id: string;
  name: string;
  type: string;
  degree: number;
  degree_centrality: number;
  betweenness_centrality: number;
  closeness_centrality: number;
  pagerank: number;
}

export interface GlobalMetrics {
  node_count: number;
  edge_count: number;
  density: number;
  avg_degree: number;
  connected_components: number;
}

export interface MetricsData {
  global_metrics: GlobalMetrics;
  node_metrics: NodeMetrics[];
  top_degree: NodeMetrics[];
  top_pagerank: NodeMetrics[];
  top_betweenness: NodeMetrics[];
}

export type ChatMessage = {
  role: 'user' | 'model';
  content: string;
};

export interface ChatApiResponse {
  session_id: string;
  reply: string;
  engine: 'ollama' | 'local' | 'llm' | 'rule-based';
  history: ChatMessage[];
  /** Heuristic confidence score 0-1 based on answer grounding */
  confidence?: number;
  /** Number of context items that support the answer */
  evidence_count?: number;
  /** Parsed intent of the query */
  intent?: string;
}

export type TabId = 'graph' | 'entities' | 'relations' | 'highlight' | 'metrics' | 'insight' | 'chatbot' | 'json';

export interface WorkspaceSessionSummary {
  id: string;
  title: string;
  preview_text: string;
  entities_count: number;
  relations_count: number;
  created_at: string;
  updated_at: string;
}

export interface WorkspaceSessionDetail {
  id: string;
  title: string;
  input_text: string;
  graph_data: GraphData | null;
  metrics_data: MetricsData | null;
  insight_markdown: string | null;
  chat_session_id: string | null;
  chat_engine: 'ollama' | 'local' | 'llm' | 'rule-based' | null;
  chat_history: ChatMessage[];
  active_tab: TabId;
  created_at: string;
  updated_at: string;
}
