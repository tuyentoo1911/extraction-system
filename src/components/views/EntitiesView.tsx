import React, { useState } from 'react';
import { TYPE_BADGE_COLORS } from '../../constants/graph';
import type { GraphData } from '../../types';

interface EntitiesViewProps {
  data: GraphData;
}

export default function EntitiesView({ data }: EntitiesViewProps) {
  const [selectedType, setSelectedType] = useState<string | null>(null);

  const uniqueTypes = Array.from(new Set(data.entities.map(e => e.type))).sort();
  const filteredEntities = selectedType
    ? data.entities.filter(e => e.type === selectedType)
    : data.entities;

  let suggestedType: string | null = null;
  if (selectedType) {
    const relatedTypesCount: Record<string, number> = {};
    data.relations.forEach(rel => {
      const sourceEnt = data.entities.find(e => e.id === rel.source);
      const targetEnt = data.entities.find(e => e.id === rel.target);
      if (sourceEnt && targetEnt) {
        if (sourceEnt.type === selectedType && targetEnt.type !== selectedType) {
          relatedTypesCount[targetEnt.type] = (relatedTypesCount[targetEnt.type] || 0) + 1;
        } else if (targetEnt.type === selectedType && sourceEnt.type !== selectedType) {
          relatedTypesCount[sourceEnt.type] = (relatedTypesCount[sourceEnt.type] || 0) + 1;
        }
      }
    });
    let maxCount = 0;
    for (const [type, count] of Object.entries(relatedTypesCount)) {
      if (count > maxCount) { maxCount = count; suggestedType = type; }
    }
  }

  return (
    <div className="p-6 flex flex-col h-full overflow-y-auto">
      <div className="flex flex-wrap gap-2 mb-4">
        <button
          onClick={() => setSelectedType(null)}
          className={`px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider border transition-colors ${
            selectedType === null
              ? 'bg-black text-white border-black'
              : 'bg-white text-black/60 border-black/20 hover:border-black hover:text-black'
          }`}
        >
          Tất cả ({data.entities.length})
        </button>
        {uniqueTypes.map(type => (
          <button
            key={type}
            onClick={() => setSelectedType(type)}
            className={`px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider border transition-colors ${
              selectedType === type
                ? 'bg-black text-white border-black'
                : 'bg-white text-black/60 border-black/20 hover:border-black hover:text-black'
            }`}
          >
            {type} ({data.entities.filter(e => e.type === type).length})
          </button>
        ))}
      </div>

      {suggestedType && (
        <div className="mb-6 flex items-center gap-2 font-mono text-xs bg-black/5 p-3 border border-black/10">
          <span className="text-black/60">Gợi ý liên quan:</span>
          <button
            onClick={() => setSelectedType(suggestedType)}
            className="font-bold text-[#f25f22] hover:underline uppercase tracking-widest"
          >
            {suggestedType}
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 pb-6">
        {filteredEntities.map((ent) => (
          <div key={ent.id} className="p-4 border border-black bg-white flex flex-col gap-4">
            <div className="flex justify-between items-start">
              <span className="font-mono text-[10px] text-black/40">{ent.id}</span>
              <span className={`font-mono text-[10px] px-2 py-1 border uppercase tracking-wider ${TYPE_BADGE_COLORS[ent.type] || 'bg-gray-100 text-gray-800 border-gray-300'}`}>
                {ent.type}
              </span>
            </div>
            <div className="font-bold text-lg">{ent.name}</div>

            {ent.properties && ent.properties.length > 0 && (
              <div className="mt-auto pt-4 border-t border-black/10">
                <table className="w-full text-xs text-left border-collapse">
                  <thead>
                    <tr className="border-b border-black/10">
                      <th className="py-1 font-mono text-[9px] uppercase tracking-widest text-black/40 font-normal w-1/3">Thuộc tính</th>
                      <th className="py-1 font-mono text-[9px] uppercase tracking-widest text-black/40 font-normal">Giá trị</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ent.properties.map((prop, idx) => (
                      <tr key={idx} className="border-b border-black/5 last:border-0">
                        <td className="py-1.5 font-mono text-[10px] text-black/60">{prop.key}</td>
                        <td className="py-1.5 font-medium text-black/80">{prop.value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
