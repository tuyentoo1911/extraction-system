import React, { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowLeft, CircleDot, Network, Database, Code, Lightbulb, MessageSquare, Sparkles, Tags, BarChart3, PanelBottomOpen, History, Trash2, PanelLeftClose, PanelLeftOpen, Plus, MoreHorizontal } from 'lucide-react';
import { callExtract, callMetrics, callPredictLinks, deleteWorkspaceSession, getWorkspaceSession, listWorkspaceSessions, saveWorkspaceSession } from './lib/ai';
import InputPanel from './components/InputPanel';
import TabButton from './components/TabButton';
import GraphView from './components/views/GraphView';
import EntitiesView from './components/views/EntitiesView';
import RelationsView from './components/views/RelationsView';
import HighlightView from './components/views/HighlightView';
import MetricsView from './components/views/MetricsView';
import InsightView from './components/views/InsightView';
import ChatbotView from './components/views/ChatbotView';
import type { ChatMessage, GraphData, MetricsData, Relation, TabId, WorkspaceSessionSummary } from './types';

const SAMPLE_TEXT = "";
const LAST_WORKSPACE_KEY = 'kge_last_workspace_id';

export default function Dashboard({ onBack }: { onBack: () => void }) {
  const [inputText, setInputText] = useState(SAMPLE_TEXT);
  const [isProcessing, setIsProcessing] = useState(false);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('graph');
  const [error, setError] = useState<string | null>(null);
  const [isInputOpen, setIsInputOpen] = useState(true);
  const [workspaceId, setWorkspaceId] = useState<string | null>(null);
  const [historyItems, setHistoryItems] = useState<WorkspaceSessionSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(true);
  const [historyMenuId, setHistoryMenuId] = useState<string | null>(null);
  const [insightMarkdown, setInsightMarkdown] = useState<string | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatEngine, setChatEngine] = useState<'llm' | 'rule-based' | null>(null);
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const insightRef = useRef<string | null>(null);
  const chatMessagesRef = useRef<ChatMessage[]>([]);
  const chatEngineRef = useRef<'llm' | 'rule-based' | null>(null);
  const chatSessionIdRef = useRef<string | null>(null);

  const refreshHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const items = await listWorkspaceSessions(60);
      setHistoryItems(items);
    } catch (err) {
      console.error(err);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const persistWorkspace = useCallback(async (next: {
    inputText: string;
    graphData: GraphData | null;
    metricsData: MetricsData | null;
    activeTab: TabId;
    title?: string;
    insightMarkdown?: string | null;
    chatSessionId?: string | null;
    chatEngine?: 'llm' | 'rule-based' | null;
    chatHistory?: ChatMessage[];
  }) => {
    const id = await saveWorkspaceSession({
      sessionId: workspaceId,
      title: next.title,
      inputText: next.inputText,
      graphData: next.graphData,
      metricsData: next.metricsData,
      insightMarkdown: next.insightMarkdown ?? insightRef.current,
      chatSessionId: next.chatSessionId ?? chatSessionIdRef.current,
      chatEngine: next.chatEngine ?? chatEngineRef.current,
      chatHistory: next.chatHistory ?? chatMessagesRef.current,
      activeTab: next.activeTab,
    });
    setWorkspaceId(id);
    try {
      localStorage.setItem(LAST_WORKSPACE_KEY, id);
    } catch {
      // ignore storage errors
    }
    return id;
  }, [workspaceId]);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  useEffect(() => {
    const lastId = (() => {
      try {
        return localStorage.getItem(LAST_WORKSPACE_KEY);
      } catch {
        return null;
      }
    })();
    if (!lastId) return;

    void (async () => {
      try {
        const detail = await getWorkspaceSession(lastId);
        setWorkspaceId(detail.id);
        setInputText(detail.input_text || '');
        setGraphData(detail.graph_data ?? null);
        setMetricsData(detail.metrics_data ?? null);
        setInsightMarkdown(detail.insight_markdown ?? null);
        setChatMessages(detail.chat_history ?? []);
        setChatEngine(detail.chat_engine ?? null);
        setChatSessionId(detail.chat_session_id ?? null);
        setActiveTab(detail.active_tab || 'graph');
      } catch {
        // stale id or load error
      }
    })();
  }, []);

  const handleExtract = async () => {
    if (!inputText.trim()) return;
    setIsProcessing(true);
    setError(null);
    try {
      const data = await callExtract(inputText, false);
      setGraphData(data);
      setMetricsData(null);
      setInsightMarkdown(null);
      setChatMessages([]);
      setChatEngine(null);
      setChatSessionId(null);
      insightRef.current = null;
      chatMessagesRef.current = [];
      chatEngineRef.current = null;
      chatSessionIdRef.current = null;
      setActiveTab('graph');
      setIsInputOpen(false);
      await persistWorkspace({
        inputText,
        graphData: data,
        metricsData: null,
        activeTab: 'graph',
        insightMarkdown: null,
        chatSessionId: null,
        chatEngine: null,
        chatHistory: [],
      });
      await refreshHistory();
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
          void persistWorkspace({
            inputText,
            graphData: updated,
            metricsData: null,
            activeTab: 'graph',
          }).then(() => refreshHistory());
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
      await persistWorkspace({
        inputText,
        graphData,
        metricsData: metrics,
        activeTab: 'metrics',
      });
      await refreshHistory();
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Có lỗi xảy ra trong quá trình tính metrics.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleOpenMetricsTab = () => {
    setActiveTab('metrics');
    if (graphData && !isProcessing) {
      void handleComputeMetrics();
    }
  };

  const handleOpenHistory = async (sessionId: string) => {
    setIsProcessing(true);
    setError(null);
    try {
      const hasCurrentData =
        Boolean(workspaceId)
        || Boolean(inputText.trim())
        || Boolean(graphData)
        || Boolean(metricsData)
        || Boolean(insightMarkdown)
        || chatMessages.length > 0;

      // Persist current workspace state before switching to another session.
      if (hasCurrentData) {
        await persistWorkspace({
          inputText,
          graphData,
          metricsData,
          activeTab,
          insightMarkdown,
          chatSessionId,
          chatEngine,
          chatHistory: chatMessages,
        });
      }

      const detail = await getWorkspaceSession(sessionId);
      setWorkspaceId(detail.id);
      try {
        localStorage.setItem(LAST_WORKSPACE_KEY, detail.id);
      } catch {
        // ignore storage errors
      }
      setInputText(detail.input_text || '');
      setGraphData(detail.graph_data ?? null);
      setMetricsData(detail.metrics_data ?? null);
      setInsightMarkdown(detail.insight_markdown ?? null);
      setChatMessages(detail.chat_history ?? []);
      setChatEngine(detail.chat_engine ?? null);
      setChatSessionId(detail.chat_session_id ?? null);
      insightRef.current = detail.insight_markdown ?? null;
      chatMessagesRef.current = detail.chat_history ?? [];
      chatEngineRef.current = detail.chat_engine ?? null;
      chatSessionIdRef.current = detail.chat_session_id ?? null;
      setActiveTab(detail.active_tab || 'graph');
      await refreshHistory();
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Không tải được phiên lịch sử.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDeleteHistory = async (sessionId: string) => {
    try {
      await deleteWorkspaceSession(sessionId);
      if (workspaceId === sessionId) {
        setWorkspaceId(null);
        try {
          localStorage.removeItem(LAST_WORKSPACE_KEY);
        } catch {
          // ignore storage errors
        }
      }
      await refreshHistory();
    } catch (err) {
      console.error(err);
    }
  };

  const handleNewWorkspace = () => {
    setWorkspaceId(null);
    try {
      localStorage.removeItem(LAST_WORKSPACE_KEY);
    } catch {
      // ignore storage errors
    }
    setInputText('');
    setGraphData(null);
    setMetricsData(null);
    setInsightMarkdown(null);
    setChatMessages([]);
    setChatEngine(null);
    setChatSessionId(null);
    insightRef.current = null;
    chatMessagesRef.current = [];
    chatEngineRef.current = null;
    chatSessionIdRef.current = null;
    setActiveTab('graph');
    setError(null);
  };

  const handleInsightChange = useCallback((nextInsight: string | null) => {
    insightRef.current = nextInsight;
    setInsightMarkdown(nextInsight);
    void persistWorkspace({
      inputText,
      graphData,
      metricsData,
      activeTab: 'insight',
      insightMarkdown: nextInsight,
    }).then(() => refreshHistory());
  }, [persistWorkspace, inputText, graphData, metricsData, refreshHistory]);

  const handleChatStateChange = useCallback((next: {
    messages: ChatMessage[];
    engine: 'llm' | 'rule-based' | null;
    sessionId: string | null;
  }) => {
    chatMessagesRef.current = next.messages;
    chatEngineRef.current = next.engine;
    chatSessionIdRef.current = next.sessionId;
    setChatMessages(next.messages);
    setChatEngine(next.engine);
    setChatSessionId(next.sessionId);
    void persistWorkspace({
      inputText,
      graphData,
      metricsData,
      activeTab: 'chatbot',
      chatHistory: next.messages,
      chatEngine: next.engine,
      chatSessionId: next.sessionId,
    });
  }, [persistWorkspace, inputText, graphData, metricsData]);

  return (
    <div className="h-screen flex flex-col bg-[#f4f4f0] text-[#111111] font-sans overflow-hidden">
      <main className="flex-1 flex overflow-hidden w-full min-h-0 p-2 gap-2">
        {isHistoryOpen && (
        <aside className="w-[200px] bg-white rounded-xl flex flex-col min-h-0 shadow-[0_2px_8px_rgba(0,0,0,0.04)]">
          <div className="p-2.5">
            <button
              onClick={onBack}
              className="w-full h-8 mb-2 flex items-center justify-center gap-2 font-mono text-[10px] font-bold tracking-widest uppercase rounded-md hover:bg-[#f25f22] hover:text-white active:bg-black active:text-white transition-colors"
            >
              <CircleDot className="w-3.5 h-3.5" />
              <span>KGE.SYS</span>
            </button>
            <button
              onClick={handleNewWorkspace}
              className="w-full flex items-center justify-center gap-2 h-9 rounded-lg bg-[#f8f8f6] hover:bg-[#f25f22] hover:text-white active:bg-black active:text-white transition-colors font-mono text-[10px] uppercase tracking-widest"
            >
              <Plus className="w-3.5 h-3.5" />
              Phiên mới
            </button>
          </div>
          <div className="px-2.5 py-2 flex items-center justify-between font-mono text-[10px] uppercase tracking-widest text-black/60">
            <div className="flex items-center gap-2">
              <History className="w-3.5 h-3.5" />
              Lịch sử phiên
            </div>
            <button
              onClick={() => setIsHistoryOpen(false)}
              className="h-6 w-6 rounded-md flex items-center justify-center hover:bg-[#f25f22] hover:text-white active:bg-black active:text-white transition-colors"
              title="Ẩn lịch sử"
            >
              <PanelLeftClose className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="flex-1 min-h-0 overflow-y-auto p-1.5 space-y-1.5">
            {historyLoading && (
              <div className="px-3 py-2 text-[10px] font-mono uppercase tracking-widest text-black/40">Đang tải...</div>
            )}
            {!historyLoading && historyItems.length === 0 && (
              <div className="px-3 py-2 text-[10px] font-mono uppercase tracking-widest text-black/40">
                Chưa có lịch sử
              </div>
            )}
            {historyItems.map(item => (
              <div key={item.id} className={`group relative rounded-md overflow-visible border transition-colors ${item.id === workspaceId ? 'border-black bg-black text-white' : 'border-black/15 bg-[#f8f8f6] hover:border-black/35 text-black'}`}>
                <button
                  onClick={() => {
                    setHistoryMenuId(null);
                    handleOpenHistory(item.id);
                  }}
                  className="w-full text-left px-2 py-1.5 pr-8"
                >
                  <div className="font-mono text-[9px] uppercase tracking-widest truncate">{item.title}</div>
                  <div className={`text-[10px] mt-1 truncate ${item.id === workspaceId ? 'text-white/70' : 'text-black/55'}`}>
                    {item.preview_text || '(trống)'}
                  </div>
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setHistoryMenuId((prev) => (prev === item.id ? null : item.id));
                  }}
                  className={`absolute top-1.5 right-1.5 h-5 w-5 rounded flex items-center justify-center transition-opacity ${
                    item.id === workspaceId
                      ? 'hover:bg-white/15'
                      : 'hover:bg-black/10'
                  } opacity-0 group-hover:opacity-100`}
                  title="Tùy chọn"
                >
                  <MoreHorizontal className="w-3.5 h-3.5" />
                </button>
                {historyMenuId === item.id && (
                  <div className="absolute top-7 right-1 z-20 bg-white border border-black/15 rounded-md shadow-[0_4px_12px_rgba(0,0,0,0.12)] p-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setHistoryMenuId(null);
                        handleDeleteHistory(item.id);
                      }}
                      className="px-2 py-1 text-[9px] font-mono uppercase tracking-widest flex items-center gap-1.5 rounded hover:bg-[#f25f22] hover:text-white active:bg-black"
                    >
                      <Trash2 className="w-3 h-3" />
                      Xóa
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </aside>
        )}
        <div className="flex-1 flex flex-col bg-[#f4f4f0] min-w-0 min-h-0 rounded-xl overflow-visible relative">
          <div className="flex bg-[#f8f8f6] overflow-x-auto hide-scrollbar justify-between items-center px-2 py-1">
            <div className="flex items-center gap-1">
              {!isHistoryOpen && (
                <button
                  onClick={() => setIsHistoryOpen(true)}
                  className="h-7 w-7 border border-black/20 rounded-md bg-white hover:bg-[#f25f22] hover:text-white active:bg-black active:text-white transition-colors flex items-center justify-center mr-1"
                  title="Mở lịch sử"
                >
                  <PanelLeftOpen className="w-3.5 h-3.5" />
                </button>
              )}
              <TabButton active={activeTab === 'graph'}    onClick={() => setActiveTab('graph')}    icon={<Network className="w-4 h-4" />}      label="Đồ thị" />
              <TabButton active={activeTab === 'entities'} onClick={() => setActiveTab('entities')} icon={<Database className="w-4 h-4" />}     label="Thực thể" />
              <TabButton active={activeTab === 'relations'}onClick={() => setActiveTab('relations')}icon={<ArrowLeft className="w-4 h-4 rotate-180" />} label="Quan hệ" />
              <TabButton active={activeTab === 'highlight'}onClick={() => setActiveTab('highlight')}icon={<Tags className="w-4 h-4" />} label="Highlight" />
              <TabButton active={activeTab === 'metrics'}  onClick={handleOpenMetricsTab}  icon={<BarChart3 className="w-4 h-4" />} label="Metrics" />
              <TabButton active={activeTab === 'insight'}  onClick={() => setActiveTab('insight')}  icon={<Lightbulb className="w-4 h-4" />}    label="Insight" />
              <TabButton active={activeTab === 'chatbot'}  onClick={() => setActiveTab('chatbot')}  icon={<MessageSquare className="w-4 h-4" />} label="Hỏi đáp" />
              <TabButton active={activeTab === 'json'}     onClick={() => setActiveTab('json')}     icon={<Code className="w-4 h-4" />}         label="JSON" />
            </div>
            <div className="flex items-center gap-2">
              <div className="font-mono text-[10px] tracking-widest uppercase text-black/40 hidden lg:block mr-1">
                Workspace / Trích xuất
              </div>
            </div>
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
                {activeTab === 'graph'     && (
                  <div className="h-full min-h-0 relative">
                    <button
                      onClick={handlePredictLinks}
                      disabled={isProcessing}
                      className="absolute bottom-4 right-4 z-20 h-8 w-8 border border-black/20 rounded-md bg-white hover:bg-[#f25f22] hover:text-white active:bg-black active:text-white transition-colors disabled:opacity-50 flex items-center justify-center"
                      title="Dự đoán liên kết"
                    >
                      <Sparkles className="w-4 h-4" />
                    </button>
                    <GraphView data={graphData} />
                  </div>
                )}
                {activeTab === 'entities'  && <EntitiesView  data={graphData} />}
                {activeTab === 'relations' && <RelationsView data={graphData} />}
                {activeTab === 'highlight' && <HighlightView data={graphData} inputText={inputText} />}
                {activeTab === 'metrics'   && (
                  metricsData ? (
                    <MetricsView data={metricsData} />
                  ) : (
                    <div className="h-full flex items-center justify-center text-black/40 font-mono text-xs uppercase tracking-widest">
                      Bấm "Tính metrics" để phân tích đồ thị.
                    </div>
                  )
                )}
                {activeTab === 'insight'   && (
                  <InsightView
                    data={graphData}
                    inputText={inputText}
                    initialInsight={insightMarkdown}
                    onInsightChange={handleInsightChange}
                  />
                )}
                {activeTab === 'chatbot'   && (
                  <ChatbotView
                    data={graphData}
                    inputText={inputText}
                    initialMessages={chatMessages}
                    initialEngine={chatEngine}
                    initialSessionId={chatSessionId}
                    onChatStateChange={handleChatStateChange}
                  />
                )}
                {activeTab === 'json'      && (
                  <pre className="p-6 font-mono text-xs text-black/80 whitespace-pre-wrap">
                    {JSON.stringify(graphData, null, 2)}
                  </pre>
                )}
              </div>
            )}
          </div>

          {activeTab !== 'chatbot' && (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 w-full px-4 z-30 pointer-events-none">
              {isInputOpen ? (
                <InputPanel
                  isOpen
                  inputText={inputText}
                  setInputText={setInputText}
                  useDeepAnalysis={false}
                  setUseDeepAnalysis={() => {}}
                  isProcessing={isProcessing}
                  error={error}
                  onExtract={handleExtract}
                  onClose={() => setIsInputOpen(false)}
                />
              ) : (
                <div className="flex-shrink-0 bg-transparent py-1">
                  <div className="max-w-3xl mx-auto flex justify-center">
                    <button
                      type="button"
                      onClick={() => setIsInputOpen(true)}
                      className="pointer-events-auto inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-black/20 bg-white shadow-[0_3px_12px_rgba(0,0,0,0.08)] hover:bg-[#f25f22] hover:text-white active:bg-black active:text-white transition-colors font-mono text-[10px] uppercase tracking-widest"
                    >
                      <PanelBottomOpen className="w-4 h-4" />
                      Mở ô nhập
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
