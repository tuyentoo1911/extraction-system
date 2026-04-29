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
      className={`flex items-center gap-2 px-4 py-2.5 font-mono text-[10px] tracking-widest uppercase transition-colors border border-transparent rounded-md
        ${active ? 'bg-black text-white border-black' : 'text-black/55 hover:bg-[#f25f22] hover:text-white active:bg-black active:text-white'}
      `}
    >
      {icon}
      {label}
    </button>
  );
}
