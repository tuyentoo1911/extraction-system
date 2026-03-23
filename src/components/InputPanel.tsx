import React, { useRef, useState, useLayoutEffect } from 'react';
import { FileText, Play, Loader2, Upload, X, FileUp, Paperclip } from 'lucide-react';
import { getApiBase } from '../lib/ai';

interface InputPanelProps {
  isOpen: boolean;
  inputText: string;
  setInputText: (text: string) => void;
  useDeepAnalysis: boolean;
  setUseDeepAnalysis: (val: boolean) => void;
  isProcessing: boolean;
  error: string | null;
  onExtract: () => void;
  onClose?: () => void;
}

export default function InputPanel({
  isOpen: _isOpen,
  inputText,
  setInputText,
  useDeepAnalysis: _useDeepAnalysis,
  setUseDeepAnalysis: _setUseDeepAnalysis,
  isProcessing,
  error,
  onExtract,
  onClose,
}: InputPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const [pdfInfo, setPdfInfo] = useState<{ filename: string; pages: number } | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [pdfError, setPdfError] = useState<string | null>(null);

  const autosizeComposer = () => {
    const el = composerRef.current;
    if (!el) return;
    el.style.height = '0px';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  };

  useLayoutEffect(() => {
    autosizeComposer();
  }, [inputText]);

  async function handlePdfUpload(file: File) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setPdfError('Chỉ hỗ trợ file .pdf');
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setPdfError('File quá lớn (tối đa 20 MB)');
      return;
    }

    setPdfLoading(true);
    setPdfError(null);
    setPdfInfo(null);

    try {
      const form = new FormData();
      form.append('file', file);
      const apiBase = await getApiBase();
      const res = await fetch(`${apiBase}/upload-pdf`, {
        method: 'POST',
        body: form,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Lỗi đọc PDF');
      }
      const data = await res.json();
      setInputText(data.text);
      setPdfInfo({ filename: data.filename, pages: data.page_count });
    } catch (e: any) {
      setPdfError(e.message || 'Không đọc được file PDF');
    } finally {
      setPdfLoading(false);
    }
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handlePdfUpload(file);
    e.target.value = '';
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) handlePdfUpload(file);
  }

  function clearPdf() {
    setPdfInfo(null);
    setPdfError(null);
    setInputText('');
  }

  return (
    <div className="flex-shrink-0 bg-transparent px-4 py-0 pointer-events-none">
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className="pointer-events-auto max-w-3xl mx-auto rounded-full border border-black/15 bg-white px-4 py-2.5 flex items-center gap-2 shadow-[0_1px_4px_rgba(0,0,0,0.06)]"
      >
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="h-8 w-8 rounded-full border border-black/20 hover:bg-black hover:text-white transition-colors flex items-center justify-center shrink-0"
          title="Đính kèm PDF"
        >
          {pdfLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Paperclip className="w-4 h-4" />}
        </button>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="h-8 w-8 rounded-full border border-black/20 hover:bg-black hover:text-white transition-colors flex items-center justify-center shrink-0"
            title="Đóng ô nhập"
          >
            <X className="w-4 h-4" />
          </button>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={handleFileChange}
        />

        <textarea
          ref={composerRef}
          value={inputText}
          onChange={(e) => {
            setInputText(e.target.value);
            if (pdfInfo) setPdfInfo(null);
            autosizeComposer();
          }}
          placeholder="Hỏi bất kỳ điều gì hoặc dán văn bản để trích xuất..."
          className="flex-1 min-h-[28px] max-h-[180px] resize-none outline-none text-sm leading-6 bg-transparent placeholder:text-black/35 overflow-y-hidden"
          rows={1}
        />

        <button
          type="button"
          onClick={onExtract}
          disabled={isProcessing || !inputText.trim()}
          className="h-8 w-8 rounded-full bg-black text-white hover:bg-[#f25f22] disabled:opacity-50 disabled:hover:bg-black transition-colors flex items-center justify-center shrink-0"
          title="Gửi"
        >
          {isProcessing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-current ml-[1px]" />}
        </button>
      </div>

      {(pdfInfo || pdfError || error) && (
        <div className="pointer-events-auto max-w-3xl mx-auto mt-2 font-mono text-[10px] bg-white/95 rounded-lg border border-black/10 px-2 py-1 shadow-sm">
          {pdfInfo && (
            <div className="inline-flex items-center gap-2 px-2 py-1 border border-green-300 bg-green-50 text-green-800">
              <FileUp className="w-3 h-3" />
              <span className="max-w-[320px] truncate">{pdfInfo.filename} ({pdfInfo.pages} trang)</span>
              <button type="button" onClick={clearPdf} className="p-0.5 hover:bg-green-100">
                <X className="w-3 h-3" />
              </button>
            </div>
          )}
          {pdfError && <p className="text-red-500 mt-1">{pdfError}</p>}
          {error && <p className="text-red-500 mt-1">{error}</p>}
        </div>
      )}
    </div>
  );
}
