import React, { useMemo } from 'react';
import { TYPE_COLORS } from '../../constants/graph';
import type { GraphData } from '../../types';

interface HighlightViewProps {
  data: GraphData;
  inputText: string;
}

type Segment = {
  text: string;
  type?: string;
};

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function isWordChar(ch: string | undefined): boolean {
  if (!ch) return false;
  return /[\p{L}\p{N}_]/u.test(ch);
}

export default function HighlightView({ data, inputText }: HighlightViewProps) {
  const segments = useMemo<Segment[]>(() => {
    if (!inputText.trim()) return [{ text: '' }];

    const entities = [...data.entities]
      .filter((e) => e.name.trim().length > 2)
      .sort((a, b) => b.name.length - a.name.length);

    if (entities.length === 0) return [{ text: inputText }];

    const ranges: Array<{ start: number; end: number; type: string }> = [];
    for (const ent of entities) {
      const regex = new RegExp(escapeRegExp(ent.name), 'gi');
      let match: RegExpExecArray | null;
      while ((match = regex.exec(inputText)) !== null) {
        const start = match.index;
        const end = start + match[0].length;
        const before = inputText[start - 1];
        const after = inputText[end];
        if (isWordChar(before) || isWordChar(after)) continue;
        const overlap = ranges.some((r) => !(end <= r.start || start >= r.end));
        if (!overlap) ranges.push({ start, end, type: ent.type });
      }
    }

    ranges.sort((a, b) => a.start - b.start);
    const result: Segment[] = [];
    let cursor = 0;

    for (const r of ranges) {
      if (cursor < r.start) result.push({ text: inputText.slice(cursor, r.start) });
      result.push({ text: inputText.slice(r.start, r.end), type: r.type });
      cursor = r.end;
    }
    if (cursor < inputText.length) result.push({ text: inputText.slice(cursor) });
    return result.length ? result : [{ text: inputText }];
  }, [data.entities, inputText]);

  return (
    <div className="p-6 h-full overflow-auto bg-[#f4f4f0]">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="font-mono text-[10px] uppercase tracking-widest text-black/50">
          Highlight thực thể trong văn bản
        </div>
        <div className="font-mono text-[10px] uppercase tracking-widest text-black/40 border border-black/15 px-2 py-1 bg-white">
          {data.entities.length} thực thể
        </div>
      </div>
      <div className="border border-black/15 bg-white p-5 leading-8 text-[15px] whitespace-pre-wrap shadow-[2px_2px_0_rgba(0,0,0,0.05)]">
        {segments.map((seg, idx) =>
          seg.type ? (
            <span
              key={idx}
              className="inline-flex items-center gap-1.5 px-2 py-0.5 mx-[1px] border border-black/25 rounded-sm font-medium bg-white text-black"
              style={{
                boxShadow: `inset 3px 0 0 ${TYPE_COLORS[seg.type] || '#9ca3af'}`,
              }}
            >
              {seg.text}
              <span
                className="font-mono text-[9px] uppercase tracking-widest border-l border-black/15 pl-1.5"
                style={{ color: TYPE_COLORS[seg.type] || '#111111' }}
              >
                {seg.type}
              </span>
            </span>
          ) : (
            <span key={idx}>{seg.text}</span>
          ),
        )}
      </div>
    </div>
  );
}
