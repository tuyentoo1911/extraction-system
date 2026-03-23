import React, { useState, useRef, useEffect } from 'react';
import { Loader2, MessageSquare, Send } from 'lucide-react';
import Markdown from 'react-markdown';
import { callChat } from '../../lib/ai';
import type { ChatMessage, GraphData } from '../../types';

interface ChatbotViewProps {
  data: GraphData;
  inputText: string;
}

export default function ChatbotView({ data, inputText }: ChatbotViewProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setInput('');
    setIsTyping(true);

    try {
      const replyText = await callChat(messages, userMessage, data, inputText);
      setMessages(prev => [...prev, { role: 'model', content: replyText || 'Xin lỗi, tôi không thể trả lời câu hỏi này.' }]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, { role: 'model', content: 'Đã có lỗi xảy ra khi kết nối với AI. Vui lòng thử lại.' }]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white">
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-black/40 font-mono text-sm uppercase tracking-widest text-center space-y-4">
            <MessageSquare className="w-12 h-12 mb-2 opacity-20" />
            <p>Hỏi đáp về Knowledge Graph</p>
            <p className="text-[10px] max-w-md normal-case tracking-normal">
              Bạn có thể hỏi bất kỳ thông tin nào liên quan đến các thực thể và mối quan hệ đã được trích xuất.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-4 ${msg.role === 'user' ? 'bg-black text-white' : 'bg-[#f4f4f0] text-black border border-black/10'}`}>
              <div className="font-mono text-[10px] uppercase tracking-widest mb-2 opacity-50">
                {msg.role === 'user' ? 'Bạn' : 'AI Assistant'}
              </div>
              <div className={`prose prose-sm max-w-none ${msg.role === 'user' ? 'prose-invert' : ''}`}>
                <Markdown>{msg.content}</Markdown>
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
