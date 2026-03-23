import React from 'react';

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}

export default function TabButton({ active, onClick, icon, label }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-6 py-4 font-mono text-xs tracking-widest uppercase transition-colors border-r border-black/10
        ${active ? 'bg-[#f4f4f0] text-black font-bold' : 'bg-white text-black/50 hover:bg-black/5 hover:text-black'}
      `}
    >
      {icon}
      {label}
    </button>
  );
}
