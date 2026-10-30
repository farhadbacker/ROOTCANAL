
import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="bg-slate-900/80 backdrop-blur-md border-b border-slate-700/50 sticky top-0 z-50 transition-all duration-300">
      <div className="container mx-auto px-4 md:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3 group cursor-pointer">
          <div className="relative w-10 h-10 flex items-center justify-center bg-gradient-to-tr from-cyan-500 to-blue-600 rounded-xl shadow-lg shadow-cyan-500/20 group-hover:shadow-cyan-500/40 transition-all duration-300">
            <svg className="w-6 h-6 text-white" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17.5H11V16.5L13 14.5V11.5C13 10.12 11.88 9 10.5 9H10V7H14V10.5L12 12.5V13H13V17.5ZM10 14H12V15H10V14Z" fill="currentColor" />
            </svg>
            <div className="absolute inset-0 rounded-xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          </div>
          <div>
            <h1 className="text-lg md:text-xl font-bold tracking-tight text-slate-100 leading-tight">
              Root Canal <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">AI Diagnostics</span>
            </h1>
            <p className="text-[10px] text-slate-400 font-medium tracking-wider uppercase">Advanced Dental Imaging</p>
          </div>
        </div>

        <div className="hidden md:flex items-center gap-4">
          <div className="px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs font-medium text-slate-400">
            v1.2.0 Beta
          </div>
        </div>
      </div>
    </header>
  );
};
