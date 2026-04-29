import React, { useEffect, useState } from 'react';
import { Play, Loader2, Lightbulb } from 'lucide-react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { callInsight } from '../../lib/ai';
import type { GraphData } from '../../types';

interface InsightViewProps {
  data: GraphData;
  inputText: string;
  initialInsight?: string | null;
  onInsightChange?: (insight: string | null) => void;
}

export default function InsightView({ data, inputText, initialInsight = null, onInsightChange }: InsightViewProps) {
  const [insight, setInsight] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setInsight(initialInsight);
  }, [initialInsight]);

  const generateInsight = async () => {
    setLoading(true);
    try {
      const text = await callInsight(inputText, data);
      const next = text || 'Không có phân tích nào được tạo ra.';
      setInsight(next);
      onInsightChange?.(next);
    } catch (err) {
      console.error(err);
      const message =
        err instanceof Error && err.message
          ? err.message
          : 'Có lỗi xảy ra khi tạo phân tích.';
      setInsight(`Không thể tạo phân tích: ${message}`);
      onInsightChange?.(`Không thể tạo phân tích: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 h-full overflow-auto bg-white">
      {!insight && !loading && (
        <div className="flex flex-col items-center justify-center h-full gap-4">
          <Lightbulb className="w-12 h-12 text-black/20" />
          <p className="font-mono text-sm text-black/50 uppercase tracking-widest text-center max-w-md">
            AI sẽ phân tích đồ thị tri thức và văn bản gốc để tìm ra các mẫu, mối liên hệ ẩn và thông tin quan trọng.
          </p>
          <button
            onClick={generateInsight}
            className="mt-4 bg-black text-white px-6 py-3 font-mono text-xs tracking-widest uppercase hover:bg-[#f25f22] transition-colors flex items-center gap-2"
          >
            <Play className="w-4 h-4 fill-current" />
            Tạo phân tích (Insight)
          </button>
        </div>
      )}

      {loading && (
        <div className="flex flex-col items-center justify-center h-full gap-4">
          <Loader2 className="w-8 h-8 animate-spin text-[#f25f22]" />
          <div className="font-mono text-xs tracking-widest uppercase animate-pulse">
            Đang phân tích dữ liệu...
          </div>
        </div>
      )}

      {insight && !loading && (
        <div className="max-w-3xl mx-auto pb-12">
          <div className="flex justify-between items-center mb-8 border-b border-black/10 pb-4">
            <h2 className="font-mono text-lg font-bold uppercase tracking-widest flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-[#f25f22]" />
              AI Insight
            </h2>
            <button
              onClick={generateInsight}
              className="text-xs font-mono uppercase tracking-widest text-black/50 hover:text-black flex items-center gap-1"
            >
              <Play className="w-3 h-3" /> Tạo lại
            </button>
          </div>
          <div className="markdown-body prose prose-sm max-w-none font-sans leading-relaxed text-black/80">
            <Markdown remarkPlugins={[remarkGfm]}>{insight}</Markdown>
          </div>
        </div>
      )}
    </div>
  );
}
