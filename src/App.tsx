import { useState } from 'react';
import { Search, ArrowRight, Asterisk, CircleDot, ChevronDown } from 'lucide-react';
import { motion } from 'motion/react';
import Dashboard from './Dashboard';

export default function App() {
  const [currentView, setCurrentView] = useState<'landing' | 'app'>('landing');

  if (currentView === 'app') {
    return <Dashboard onBack={() => setCurrentView('landing')} />;
  }

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Header */}
      <header className="w-full border-b border-black/10 bg-[#f4f4f0]/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-2 font-mono text-sm font-bold tracking-widest">
            <CircleDot className="w-5 h-5" />
            <span>KGE.SYS</span>
          </div>

          {/* Desktop Nav */}
          <nav className="hidden md:flex items-center gap-8 font-mono text-xs tracking-widest text-black/60 uppercase">
            <a href="#" className="hover:text-black transition-colors">Nền tảng</a>
            <a href="#" className="hover:text-black transition-colors">Giải pháp</a>
            <a href="#" className="hover:text-black transition-colors">Tài liệu</a>
            <a href="#" className="hover:text-black transition-colors">Công ty</a>
          </nav>

          {/* Actions */}
          <div className="flex items-center gap-6 font-mono text-xs tracking-widest uppercase">
            <button className="text-black/60 hover:text-black transition-colors">
              <Search className="w-4 h-4" />
            </button>
            <a href="#" className="hidden sm:block hover:text-black/60 transition-colors">Đăng nhập</a>
            <button className="bg-black text-white px-4 py-2 hover:bg-black/80 transition-colors">
              Yêu cầu Demo
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col items-center justify-center py-8 md:py-12 px-4 relative z-10">
        
        {/* Top Heading */}
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="display-text text-4xl sm:text-5xl md:text-7xl lg:text-8xl text-center uppercase mb-6 md:mb-10"
        >
          Trích Xuất Dữ Liệu.
        </motion.h1>

        {/* Diagram */}
        <div className="relative w-full max-w-4xl mx-auto my-4 md:my-6 overflow-x-auto pb-4">
          <div className="min-w-[600px] flex items-center justify-between relative px-4">
            {/* Connecting Lines (SVG) */}
            <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
              <svg className="w-full h-full absolute" style={{ zIndex: -1 }}>
                {/* Left lines */}
                <motion.line 
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
                  x1="50%" y1="50%" x2="20%" y2="20%" stroke="currentColor" strokeWidth="1" className="text-black/20" 
                />
                <motion.line 
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
                  x1="50%" y1="50%" x2="20%" y2="50%" stroke="currentColor" strokeWidth="1" className="text-black/20" 
                />
                <motion.line 
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.8, delay: 0.6, ease: "easeOut" }}
                  x1="50%" y1="50%" x2="20%" y2="80%" stroke="currentColor" strokeWidth="1" className="text-black/20" 
                />
                
                {/* Right lines */}
                <motion.line 
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
                  x1="50%" y1="50%" x2="80%" y2="20%" stroke="currentColor" strokeWidth="1" className="text-black/20" 
                />
                <motion.line 
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.8, delay: 0.5, ease: "easeOut" }}
                  x1="50%" y1="50%" x2="80%" y2="50%" stroke="currentColor" strokeWidth="1" className="text-black/20" 
                />
                <motion.line 
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.8, delay: 0.7, ease: "easeOut" }}
                  x1="50%" y1="50%" x2="80%" y2="80%" stroke="currentColor" strokeWidth="1" className="text-black/20" 
                />
              </svg>
            </div>

            {/* Left Nodes */}
            <div className="flex flex-col gap-4 md:gap-6 z-10">
              <Node label="Người" delay={0.2} />
              <Node label="Tổ chức" delay={0.4} />
              <Node label="Địa điểm" delay={0.6} />
            </div>

            {/* Center Hub */}
            <motion.div 
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 360 }}
              transition={{ 
                scale: { duration: 0.8, ease: "easeOut" },
                rotate: { duration: 20, repeat: Infinity, ease: "linear" }
              }}
              className="w-14 h-14 md:w-16 md:h-16 bg-[#e5e5e0] border border-black/10 flex items-center justify-center relative z-10 shrink-0 mx-8"
            >
              <Asterisk className="w-6 h-6 md:w-8 md:h-8 text-black/60" />
              {/* Red dot indicator */}
              <div className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 md:w-2.5 md:h-2.5 bg-[#f25f22] rounded-full" />
            </motion.div>

            {/* Right Nodes */}
            <div className="flex flex-col gap-4 md:gap-6 z-10">
              <Node label="Sản phẩm" delay={0.3} />
              <Node label="Sự kiện" delay={0.5} />
              <Node label="Đa ngôn ngữ" delay={0.7} />
            </div>
          </div>
        </div>

        {/* Bottom Heading */}
        <motion.h2 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="display-text text-4xl sm:text-5xl md:text-7xl lg:text-8xl text-center uppercase mt-6 md:mt-10 mb-4 md:mb-6"
        >
          Kiến Tạo Tri Thức.
        </motion.h2>

        {/* Subtext */}
        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="font-mono text-sm md:text-base text-center max-w-2xl text-black/70 leading-relaxed mb-8"
        >
          Hệ thống trích xuất thông tin từ tài liệu. Nhận diện thực thể đa ngôn ngữ, 
          xác định mối quan hệ và xây dựng đồ thị tri thức tự động với độ chính xác cao.
        </motion.p>

        {/* Main CTA */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="flex items-stretch group cursor-pointer mb-12"
          onClick={() => setCurrentView('app')}
        >
          <div className="bg-[#f25f22] w-12 flex items-center justify-center text-white group-hover:bg-[#d94f18] transition-colors">
            <ArrowRight className="w-5 h-5" />
          </div>
          <div className="bg-black text-white px-8 py-4 font-mono text-sm tracking-widest uppercase group-hover:bg-black/90 transition-colors">
            Dùng thử miễn phí
          </div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 1 }}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 cursor-pointer text-black/40 hover:text-black transition-colors"
          onClick={() => {
            document.getElementById('overview')?.scrollIntoView({ behavior: 'smooth' });
          }}
        >
          <span className="font-mono text-[10px] tracking-widest uppercase">Tìm hiểu thêm</span>
          <motion.div
            animate={{ y: [0, 5, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          >
            <ChevronDown className="w-4 h-4" />
          </motion.div>
        </motion.div>

      </main>

      {/* Overview Section */}
      <section id="overview" className="w-full max-w-7xl mx-auto px-6 py-12 md:py-16 border-t border-black/10 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 md:gap-12">
          {/* Left: Title */}
          <div className="lg:col-span-4">
            <div className="sticky top-24">
              <h3 className="font-mono text-sm tracking-widest uppercase text-black/50 mb-4 flex items-center gap-2">
                <CircleDot className="w-4 h-4" /> Tổng quan hệ thống
              </h3>
              <h2 className="text-4xl md:text-5xl font-bold tracking-tight uppercase leading-[0.9] mb-6">
                Mục Tiêu <br/>
                <span className="text-[#f25f22]">Cốt Lõi.</span>
              </h2>
              <div className="p-5 border border-black bg-white/80 backdrop-blur-sm shadow-[4px_4px_0px_0px_rgba(0,0,0,1)]">
                <p className="font-mono text-sm text-black/80 leading-relaxed">
                  <strong className="text-black uppercase block mb-2">Đầu vào:</strong>
                  Từ tài liệu thô (PDF, DOCX, web, email…), hệ thống tự động xử lý và chuyển đổi thành tri thức có cấu trúc.
                </p>
              </div>
            </div>
          </div>

          {/* Right: Features Grid */}
          <div className="lg:col-span-8 grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6">
            <FeatureCard 
              number="01"
              title="Trích xuất thực thể"
              subtitle="NER"
              desc="Nhận diện tự động người, tổ chức, địa điểm, sản phẩm từ văn bản phi cấu trúc."
            />
            <FeatureCard 
              number="02"
              title="Xác định quan hệ"
              subtitle="Relation Extraction"
              desc="Khám phá và phân loại các mối liên hệ ngữ nghĩa giữa các thực thể được tìm thấy."
            />
            <FeatureCard 
              number="03"
              title="Chuẩn hóa & Liên kết"
              subtitle="Entity Linking"
              desc="Đồng nhất các thực thể và liên kết chúng với các cơ sở tri thức hiện có."
            />
            <FeatureCard 
              number="04"
              title="Knowledge Graph"
              subtitle="Tự động xây dựng"
              desc="Tạo lập đồ thị tri thức trực quan, cho phép truy vấn và suy luận phức tạp."
            />
            <FeatureCard 
              number="05"
              title="Đa ngôn ngữ"
              subtitle="Cross-lingual"
              desc="Hỗ trợ xử lý đồng thời nhiều ngôn ngữ: Việt, Anh, Nhật và hơn thế nữa."
              className="sm:col-span-2"
            />
          </div>
        </div>
      </section>
    </div>
  );
}

function Node({ label, delay = 0 }: { label: string, delay?: number }) {
  return (
    <motion.div 
      initial={{ opacity: 0, x: label === "Người" || label === "Tổ chức" || label === "Địa điểm" ? -20 : 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay }}
      className="px-6 py-2 border border-black/30 rounded-full bg-[#f4f4f0] font-mono text-xs tracking-widest uppercase text-black/80 shadow-sm hover:border-black hover:bg-white transition-all cursor-default"
    >
      {label}
    </motion.div>
  );
}

function FeatureCard({ number, title, subtitle, desc, className = "" }: { number: string, title: string, subtitle: string, desc: string, className?: string }) {
  return (
    <div className={`border border-black p-8 bg-white hover:bg-black hover:text-white transition-colors group flex flex-col ${className}`}>
      <div className="flex justify-between items-start mb-12">
        <span className="font-mono text-4xl font-bold text-black/20 group-hover:text-white/20 transition-colors">{number}</span>
        <div className="w-3 h-3 bg-[#f25f22] rounded-none" />
      </div>
      <div className="mt-auto">
        <h4 className="font-mono text-xs tracking-widest uppercase text-[#f25f22] mb-3">{subtitle}</h4>
        <h3 className="text-2xl font-bold uppercase tracking-tight mb-4 leading-none">{title}</h3>
        <p className="font-mono text-sm text-black/60 group-hover:text-white/70 leading-relaxed transition-colors">
          {desc}
        </p>
      </div>
    </div>
  );
}
