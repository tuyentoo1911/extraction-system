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

export type TabId = 'graph' | 'entities' | 'relations' | 'highlight' | 'metrics' | 'insight' | 'chatbot' | 'json';
