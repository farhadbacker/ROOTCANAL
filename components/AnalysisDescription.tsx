
import React from 'react';

interface AnalysisDescriptionProps {
  description: string;
}

export const AnalysisDescription: React.FC<AnalysisDescriptionProps> = ({ description }) => {
  return (
    <div className="space-y-4 text-slate-300 leading-relaxed font-light">
      {description.split('\n').map((paragraph, index) => {
        const trimmed = paragraph.trim();

        if (trimmed.startsWith('### ')) {
          return (
            <div key={index} className="mt-6 mb-3 pb-2 border-b border-slate-700/50">
              <h3 className="text-lg font-bold text-cyan-300 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                {trimmed.substring(4)}
              </h3>
            </div>
          );
        }

        if (trimmed.startsWith('## ')) {
          return (
            <h2 key={index} className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400 mt-8 mb-4">
              {trimmed.substring(3)}
            </h2>
          );
        }

        if (trimmed.startsWith('- ')) {
          return (
            <div key={index} className="flex items-start gap-3 ml-2 mb-2 group">
              <span className="mt-2 w-1.5 h-1.5 rounded-full bg-slate-600 group-hover:bg-cyan-400 transition-colors duration-300 shrink-0"></span>
              <p className="text-slate-300 group-hover:text-slate-200 transition-colors">{trimmed.substring(2)}</p>
            </div>
          );
        }

        if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
          return (
            <div key={index} className="bg-slate-800/50 border-l-4 border-cyan-500 p-3 my-4 rounded-r-lg">
              <p className="font-bold text-cyan-100">{trimmed.substring(2, trimmed.length - 2)}</p>
            </div>
          );
        }

        if (trimmed === '') {
          return null;
        }

        return <p key={index} className="mb-2">{trimmed}</p>
      })}
    </div>
  );
};
