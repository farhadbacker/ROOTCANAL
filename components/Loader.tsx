
import React from 'react';

export const Loader: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center text-center">
      <div className="w-12 h-12 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin mb-4"></div>
      <p className="text-lg font-semibold text-gray-300">Analyzing Radiograph...</p>
      <p className="text-sm text-slate-400">This may take a few moments.</p>
    </div>
  );
};
