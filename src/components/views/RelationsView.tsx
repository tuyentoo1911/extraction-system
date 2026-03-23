import React, { useState } from 'react';
import { Search } from 'lucide-react';
import type { GraphData } from '../../types';

interface RelationsViewProps {
  data: GraphData;
}

export default function RelationsView({ data }: RelationsViewProps) {
  const [searchTerm, setSearchTerm] = useState('');

  const getEntityName = (id: string) => data.entities.find(e => e.id === id)?.name || id;

  const filteredRelations = data.relations.filter(rel =>
    rel.label.toLowerCase().includes(searchTerm.toLowerCase()) ||
    getEntityName(rel.source).toLowerCase().includes(searchTerm.toLowerCase()) ||
    getEntityName(rel.target).toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="p-6 flex flex-col h-full overflow-y-auto">
      <div className="mb-6 relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="w-4 h-4 text-black/40" />
        </div>
        <input
          type="text"
          placeholder="Tìm kiếm theo nhãn quan hệ hoặc tên thực thể..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          className="w-full pl-10 pr-4 py-3 border border-black bg-white font-mono text-sm outline-none focus:border-[#f25f22] transition-colors"
        />
      </div>

      <div className="flex flex-col gap-2 pb-6">
        {filteredRelations.length === 0 ? (
          <div className="text-center p-8 font-mono text-xs text-black/40 uppercase tracking-widest border border-dashed border-black/20">
            Không tìm thấy quan hệ nào
          </div>
        ) : (
          filteredRelations.map((rel, idx) => (
            <div
              key={idx}
              className={`flex items-center gap-4 p-4 border ${rel.isPredicted ? 'border-[#10b981] bg-[#ecfdf5]' : 'border-black bg-white'}`}
            >
              <div className="flex-1 text-right font-bold">{getEntityName(rel.source)}</div>
              <div className="flex flex-col items-center px-4">
                <span className={`font-mono text-[10px] uppercase tracking-widest mb-1 ${rel.isPredicted ? 'text-[#10b981]' : 'text-[#f25f22]'}`}>
                  {rel.label} {rel.isPredicted && '(Dự đoán)'}
                </span>
                <div className={`w-24 h-px relative ${rel.isPredicted ? 'bg-[#10b981]/50 border-t border-dashed border-[#10b981]' : 'bg-black/20'}`}>
                  <div className={`absolute right-0 top-1/2 -translate-y-1/2 w-2 h-2 border-t border-r rotate-45 ${rel.isPredicted ? 'border-[#10b981]' : 'border-black/40'}`} />
                </div>
              </div>
              <div className="flex-1 font-bold">{getEntityName(rel.target)}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
