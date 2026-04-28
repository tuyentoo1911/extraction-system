import React, { useState } from 'react';
import { ArrowLeft, CircleDot, Network, Database, Code, Lightbulb, MessageSquare, Sparkles, Tags, BarChart3, PanelBottomOpen } from 'lucide-react';
import { callExtract, callMetrics, callPredictLinks } from './lib/ai';
import InputPanel from './components/InputPanel';
import TabButton from './components/TabButton';
import GraphView from './components/views/GraphView';
import EntitiesView from './components/views/EntitiesView';
import RelationsView from './components/views/RelationsView';
import HighlightView from './components/views/HighlightView';
import MetricsView from './components/views/MetricsView';
import InsightView from './components/views/InsightView';
import ChatbotView from './components/views/ChatbotView';
import type { GraphData, MetricsData, Relation, TabId } from './types';

const SAMPLE_TEXT = "";

export default function Dashboard({ onBack }: { onBack: () => void }) {
  const [inputText, setInputText] = useState(SAMPLE_TEXT);
  const [isProcessing, setIsProcessing] = useState(false);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('graph');
  const [error, setError] = useState<string | null>(null);
  const [isInputOpen, setIsInputOpen] = useState(true);

  const [copied, setCopied] = useState(false);

  const handleCopyJson = () => {
    if (!graphData) return;
    navigator.clipboard.writeText(JSON.stringify(graphData, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleExtract = async () => {
    if (!inputText.trim()) return;
    setIsProcessing(true);
    setError(null);
    try {
      const data = await callExtract(inputText, false);
      setGraphData(data);
      setMetricsData(null);
      setActiveTab('graph');
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Có lỗi xảy ra trong quá trình trích xuất.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePredictLinks = async () => {
    if (!graphData || graphData.entities.length === 0) return;
    setIsProcessing(true);
    setError(null);
    try {
      const predicted = await callPredictLinks(graphData.entities, graphData.relations, false);
      if (predicted && predicted.length > 0) {
        const newRelations = predicted.map((r: Relation) => ({ ...r, isPredicted: true }));
        setGraphData(prev => {
          if (!prev) return prev;
          const updated = { ...prev, relations: [...prev.relations, ...newRelations] };
          setMetricsData(null);
          return updated;
        });
        setActiveTab('graph');
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Có lỗi xảy ra trong quá trình dự đoán.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleComputeMetrics = async () => {
    if (!graphData) return;
    setIsProcessing(true);
    setError(null);
    try {
      const metrics = await callMetrics(graphData);
      setMetricsData(metrics);
      setActiveTab('metrics');
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Có lỗi xảy ra trong quá trình tính metrics.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-[#f4f4f0] text-[#111111] font-sans overflow-hidden">
      <header className="w-full border-b border-black/10 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1600px] mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <button
              onClick={onBack}
              className="flex items-center gap-2 font-mono text-xs font-bold tracking-widest uppercase hover:text-[#f25f22] transition-colors"
            >
              <CircleDot className="w-4 h-4" />
              <span>KGE.SYS</span>
            </button>
          </div>
          <div className="font-mono text-xs tracking-widest uppercase text-black/40 hidden sm:block">
            Workspace / Trích xuất
          </div>
        </div>
      </header>

      <main className="flex-1 flex flex-col overflow-hidden max-w-[1600px] w-full mx-auto relative min-h-0">
        <div className="flex-1 flex flex-col bg-[#f4f4f0] min-w-0 min-h-0">
          <div className="flex border-b border-black/10 bg-white overflow-x-auto hide-scrollbar justify-between items-center pr-4">
            <div className="flex">
              <TabButton active={activeTab === 'graph'} onClick={() => setActiveTab('graph')} icon={<Network className="w-4 h-4" />} label="Đồ thị" />
              <TabButton active={activeTab === 'entities'} onClick={() => setActiveTab('entities')} icon={<Database className="w-4 h-4" />} label="Thực thể" />
              <TabButton active={activeTab === 'relations'} onClick={() => setActiveTab('relations')} icon={<ArrowLeft className="w-4 h-4 rotate-180" />} label="Quan hệ" />
              <TabButton active={activeTab === 'highlight'} onClick={() => setActiveTab('highlight')} icon={<Tags className="w-4 h-4" />} label="Highlight" />
              <TabButton active={activeTab === 'metrics'} onClick={() => setActiveTab('metrics')} icon={<BarChart3 className="w-4 h-4" />} label="Metrics" />
              <TabButton active={activeTab === 'insight'} onClick={() => setActiveTab('insight')} icon={<Lightbulb className="w-4 h-4" />} label="Insight" />
              <TabButton active={activeTab === 'chatbot'} onClick={() => setActiveTab('chatbot')} icon={<MessageSquare className="w-4 h-4" />} label="Hỏi đáp" />
              <TabButton active={activeTab === 'json'} onClick={() => setActiveTab('json')} icon={<Code className="w-4 h-4" />} label="JSON" />
            </div>
            {graphData && (
              <div className="flex items-center gap-2">
                <button
                  onClick={handleComputeMetrics}
                  disabled={isProcessing}
                  className="flex items-center gap-2 px-4 py-2 font-mono text-[10px] tracking-widest uppercase border border-black hover:bg-black hover:text-white transition-colors disabled:opacity-50 whitespace-nowrap"
                >
                  <BarChart3 className="w-3 h-3" />
                  Tính metrics
                </button>
                <button
                  onClick={handlePredictLinks}
                  disabled={isProcessing}
                  className="flex items-center gap-2 px-4 py-2 font-mono text-[10px] tracking-widest uppercase border border-black hover:bg-black hover:text-white transition-colors disabled:opacity-50 whitespace-nowrap"
                >
                  <Sparkles className="w-3 h-3" />
                  Dự đoán liên kết
                </button>
              </div>
            )}
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden relative">
            {!graphData && !isProcessing && (
              <div className="absolute inset-0 flex items-center justify-center text-black/30 font-mono text-sm uppercase tracking-widest">
                Chưa có dữ liệu. Hãy nhập văn bản và trích xuất.
              </div>
            )}
            {isProcessing && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/50 backdrop-blur-sm z-10">
                <div className="w-8 h-8 border-2 border-[#f25f22] border-t-transparent rounded-full animate-spin mb-4" />
                <div className="font-mono text-xs tracking-widest uppercase animate-pulse">Đang phân tích ngữ nghĩa...</div>
              </div>
            )}
            {graphData && !isProcessing && (
              <div className="h-full min-h-0">
                {activeTab === 'graph' && <GraphView data={graphData} />}
                {activeTab === 'entities' && <EntitiesView data={graphData} />}
                {activeTab === 'relations' && <RelationsView data={graphData} />}
                {activeTab === 'highlight' && <HighlightView data={graphData} inputText={inputText} />}
                {activeTab === 'metrics' && (
                  metricsData ? (
                    <MetricsView data={metricsData} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-black/40 font-mono text-xs uppercase tracking-widest">
                      Bấm "Tính metrics" để phân tích đồ thị.
                    </div>
                  )
                )}
                {activeTab === 'insight' && <InsightView data={graphData} inputText={inputText} />}
                {activeTab === 'chatbot' && <ChatbotView data={graphData} inputText={inputText} />}
                {activeTab === 'json' && (
                  <div className="relative h-full group">
                    <button
                      onClick={handleCopyJson}
                      className="absolute top-4 right-6 z-10 flex items-center gap-2 px-3 py-1.5 font-mono text-[10px] tracking-widest uppercase bg-white border border-black hover:bg-black hover:text-white transition-all shadow-sm"
                    >
                      {copied ? (
                        <>Đã sao chép!</>
                      ) : (
                        <>
                          <Code className="w-3 h-3" />
                          Copy JSON
                        </>
                      )}
                    </button>
                    <pre className="p-6 font-mono text-xs text-black/80 whitespace-pre-wrap h-full overflow-auto bg-white/50">
                      {JSON.stringify(graphData, null, 2)}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="absolute bottom-3 left-0 right-0 z-30 pointer-events-none">
          {isInputOpen ? (
            <InputPanel
              isOpen
              inputText={inputText}
              setInputText={setInputText}
              useDeepAnalysis={false}
              setUseDeepAnalysis={() => { }}
              isProcessing={isProcessing}
              error={error}
              onExtract={handleExtract}
              onClose={() => setIsInputOpen(false)}
            />
          ) : (
            <div className="flex-shrink-0 bg-transparent px-4 py-0">
              <div className="max-w-3xl mx-auto flex justify-center">
                <button
                  type="button"
                  onClick={() => setIsInputOpen(true)}
                  className="pointer-events-auto inline-flex items-center gap-2 px-4 py-2 rounded-full border border-black/20 bg-white shadow-[0_3px_12px_rgba(0,0,0,0.08)] hover:bg-black hover:text-white transition-colors font-mono text-[10px] uppercase tracking-widest"
                >
                  <PanelBottomOpen className="w-4 h-4" />
                  Mở ô nhập
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
