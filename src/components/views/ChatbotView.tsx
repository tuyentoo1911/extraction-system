import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { Loader2, MessageSquare, RotateCcw, Send } from 'lucide-react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { callChat } from '../../lib/ai';
import type { ChatMessage, GraphData, MetricsData } from '../../types';

const SESSION_KEY = 'kge_chat_session_id';

function getStoredSessionId(): string | null {
  try {
    return localStorage.getItem(SESSION_KEY);
  } catch {
    return null;
  }
}

function storeSessionId(id: string) {
  try {
    localStorage.setItem(SESSION_KEY, id);
  } catch {  }
}

function clearStoredSession() {
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch {  }
}

type ChatEngine = 'ollama' | 'local' | 'llm' | 'rule-based' | null;

interface ChatbotViewProps {
  data: GraphData;
  inputText: string;
  /** Latest Insight markdown from Insight tab (sent with each chat request). */
  insightMarkdown?: string | null;
  /** Cached metrics from Metrics tab (compact summary sent to chat API). */
  metricsData?: MetricsData | null;
  initialMessages?: ChatMessage[];
  initialEngine?: ChatEngine;
  initialSessionId?: string | null;
  onChatStateChange?: (state: {
    messages: ChatMessage[];
    engine: ChatEngine;
    sessionId: string | null;
  }) => void;
}

function buildSuggestions(data: GraphData): string[] {
  const { entities, relations } = data;
  if (!entities.length) return ['Giúp'];

  const suggestions: string[] = [];

  suggestions.push('Tóm tắt đồ thị');
  suggestions.push('Thực thể quan trọng nhất');

  const topEntity = entities[0];
  if (topEntity) {
    suggestions.push(`Cho tôi biết về ${topEntity.name}`);
  }

  if (entities.length >= 2) {
    suggestions.push(`So sánh ${entities[0].name} và ${entities[1].name}`);
  }

  if (relations.length > 0) {
    suggestions.push('Liệt kê quan hệ');
  }

  const predicted = relations.filter(r => r.isPredicted);
  if (predicted.length > 0) {
    suggestions.push('Quan hệ dự đoán');
  }

  suggestions.push('Giúp');

  return suggestions;
}

export default function ChatbotView({
  data,
  inputText,
  insightMarkdown = null,
  metricsData = null,
  initialMessages = [],
  initialEngine = null,
  initialSessionId = null,
  onChatStateChange,
}: ChatbotViewProps) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [engine, setEngine] = useState<ChatEngine>(initialEngine);
  const [lastConfidence, setLastConfidence] = useState<number | null>(null);
  const [lastIntent, setLastIntent] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(initialSessionId ?? getStoredSessionId());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const suggestions = useMemo(() => buildSuggestions(data), [data]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    setMessages(initialMessages);
    setEngine(initialEngine);
    sessionIdRef.current = initialSessionId ?? null;
  }, [initialMessages, initialEngine, initialSessionId]);

  const sendMessage = useCallback(async (text: string) => {
    const userMessage = text.trim();
    if (!userMessage) return;

    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setInput('');
    setIsTyping(true);

    try {
      const resp = await callChat(sessionIdRef.current, userMessage, data, inputText, {
        insightMarkdown,
        metricsData,
      });

      if (resp.session_id) {
        sessionIdRef.current = resp.session_id;
        storeSessionId(resp.session_id);
      }
      const nextEngine = resp.engine as ChatEngine;
      setEngine(nextEngine);
      if (resp.confidence !== undefined) setLastConfidence(resp.confidence);
      if (resp.intent) setLastIntent(resp.intent);
      setMessages(prev => {
        const nextMessages: ChatMessage[] = [
        ...prev,
        { role: 'model' as const, content: resp.reply || 'Xin lỗi, tôi không thể trả lời câu hỏi này.' },
        ];
        onChatStateChange?.({
          messages: nextMessages,
          engine: nextEngine,
          sessionId: sessionIdRef.current,
        });
        return nextMessages;
      });
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => {
        const nextMessages: ChatMessage[] = [
        ...prev,
        { role: 'model' as const, content: 'Đã có lỗi xảy ra khi kết nối với AI. Vui lòng thử lại.' },
        ];
        onChatStateChange?.({
          messages: nextMessages,
          engine: engine ?? null,
          sessionId: sessionIdRef.current,
        });
        return nextMessages;
      });
    } finally {
      setIsTyping(false);
    }
  }, [data, inputText, insightMarkdown, metricsData, onChatStateChange, engine]);

  const handleSend = useCallback(() => {
    sendMessage(input);
  }, [input, sendMessage]);

  const handleReset = useCallback(() => {
    sessionIdRef.current = null;
    clearStoredSession();
    setMessages([]);
    setEngine(null);
    onChatStateChange?.({ messages: [], engine: null, sessionId: null });
  }, [onChatStateChange]);

  return (
    <div className="flex flex-col h-full bg-white">
      <div className="flex items-center justify-between px-6 py-2 border-b border-black/10 bg-[#f4f4f0]">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] uppercase tracking-widest text-black/50">
            {engine === 'ollama' ? 'Ollama'
              : engine === 'local' ? 'Local LLM'
              : engine === 'llm' ? 'LLM'
              : 'Rule-based'}
          </span>
          {lastIntent && (
            <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-black/5 text-black/40 uppercase tracking-wider">
              {lastIntent}
            </span>
          )}
          {lastConfidence !== null && engine !== 'rule-based' && (
            <span
              className={`font-mono text-[9px] px-1.5 py-0.5 rounded uppercase tracking-wider ${
                lastConfidence >= 0.7
                  ? 'bg-green-100 text-green-700'
                  : lastConfidence >= 0.4
                  ? 'bg-yellow-100 text-yellow-700'
                  : 'bg-red-100 text-red-600'
              }`}
              title={`Confidence: ${(lastConfidence * 100).toFixed(0)}%`}
            >
              {(lastConfidence * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 px-3 py-1.5 text-black/50 hover:text-[#f25f22] transition-colors font-mono text-[10px] uppercase tracking-widest"
          title="Bắt đầu cuộc hội thoại mới"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-black/40 font-mono text-sm text-center space-y-4">
            <MessageSquare className="w-12 h-12 mb-2 opacity-20" />
            <p className="uppercase tracking-widest">Hỏi đáp về Knowledge Graph</p>
            <p className="text-[10px] max-w-md normal-case tracking-normal">
              Bạn có thể hỏi về thực thể, quan hệ, so sánh, thống kê, tra cứu Knowledge Base và nhiều hơn nữa.
            </p>

            <div className="flex flex-wrap justify-center gap-2 mt-4 max-w-lg">
              {suggestions.map((s) => (
                <button
                  key={s}
                  onClick={() => sendMessage(s)}
                  className="px-3 py-1.5 border border-black/20 text-black/60 text-xs font-mono hover:border-[#f25f22] hover:text-[#f25f22] transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 ${msg.role === 'user' ? 'bg-black text-white' : 'bg-[#f4f4f0] text-black border border-black/10'}`}>
              <div className="font-mono text-[10px] uppercase tracking-widest mb-2 opacity-50">
                {msg.role === 'user' ? 'Bạn' : 'AI Assistant'}
              </div>
              <div className={`prose prose-sm max-w-none font-sans ${msg.role === 'user' ? 'prose-invert' : ''}`}>
                <Markdown remarkPlugins={[remarkGfm]}>{msg.content}</Markdown>
              </div>
            </div>
          </div>
        ))}

        {isTyping && (
          <div className="flex justify-start">
            <div className="bg-[#f4f4f0] text-black border border-black/10 p-4 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-[#f25f22]" />
              <span className="font-mono text-xs uppercase tracking-widest text-black/50">AI đang suy nghĩ...</span>
            </div>
          </div>
        )}

        {messages.length > 0 && !isTyping && (
          <div className="flex flex-wrap gap-2 pt-2">
            {suggestions.slice(0, 4).map((s) => (
              <button
                key={s}
                onClick={() => sendMessage(s)}
                className="px-2.5 py-1 border border-black/15 text-black/40 text-[10px] font-mono hover:border-[#f25f22] hover:text-[#f25f22] transition-colors"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-black/10 bg-[#f4f4f0]">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Nhập câu hỏi của bạn..."
            className="flex-1 px-4 py-3 border border-black bg-white font-mono text-sm outline-none focus:border-[#f25f22] transition-colors"
          />
          <button
            onClick={handleSend}
            disabled={isTyping || !input.trim()}
            className="bg-black text-white px-6 py-3 font-mono text-sm uppercase tracking-widest hover:bg-[#f25f22] transition-colors disabled:opacity-50 disabled:hover:bg-black flex items-center justify-center"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
