import React from 'react';
import type { MetricsData } from '../../types';

interface MetricsViewProps {
  data: MetricsData;
}

function MetricCard({ title, value }: { title: string; value: string | number }) {
  return (
    <div className="border border-black/10 bg-white p-4">
      <div className="font-mono text-[10px] uppercase tracking-widest text-black/50">{title}</div>
      <div className="mt-2 text-2xl font-bold">{value}</div>
    </div>
  );
}

function TopTable({ title, rows, metricKey }: { title: string; rows: MetricsData['top_degree']; metricKey: keyof MetricsData['top_degree'][number] }) {
  return (
    <div className="border border-black/10 bg-white">
      <div className="px-4 py-3 border-b border-black/10 font-mono text-[10px] uppercase tracking-widest">{title}</div>
      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead className="bg-[#f4f4f0]">
            <tr>
              <th className="text-left px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-black/60">Entity</th>
              <th className="text-left px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-black/60">Type</th>
              <th className="text-right px-4 py-2 font-mono text-[10px] uppercase tracking-widest text-black/60">Score</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((n) => (
              <tr key={`${title}-${n.id}`} className="border-t border-black/5">
                <td className="px-4 py-2 font-medium">{n.name}</td>
                <td className="px-4 py-2 text-black/70">{n.type}</td>
                <td className="px-4 py-2 text-right font-mono">
                  {typeof n[metricKey] === 'number' ? Number(n[metricKey]).toFixed(6) : String(n[metricKey])}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function MetricsView({ data }: MetricsViewProps) {
  const g = data.global_metrics;

  return (
    <div className="p-6 space-y-6 h-full overflow-y-auto">
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        <MetricCard title="Node count" value={g.node_count} />
        <MetricCard title="Edge count" value={g.edge_count} />
        <MetricCard title="Density" value={g.density.toFixed(6)} />
        <MetricCard title="Avg degree" value={g.avg_degree.toFixed(3)} />
        <MetricCard title="Components" value={g.connected_components} />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <TopTable title="Top Degree" rows={data.top_degree} metricKey="degree_centrality" />
        <TopTable title="Top PageRank" rows={data.top_pagerank} metricKey="pagerank" />
        <TopTable title="Top Betweenness" rows={data.top_betweenness} metricKey="betweenness_centrality" />
      </div>
    </div>
  );
}
