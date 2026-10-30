import React, { useState } from 'react';
import type { DetectionBox } from '../types';
import { AnalysisDisplay } from './AnalysisDisplay';

interface ComparisonResult {
    detections: DetectionBox[];
    details?: {
        name: string;
        description: string;
        expected_inference_time_ms: number;
        parameters: string;
    };
    error?: string;
}

interface ComparisonDisplayProps {
    imageUrl: string;
    results: Record<string, ComparisonResult>;
    imageSize?: { width: number; height: number; };
    onClose: () => void;
}

export const ComparisonDisplay: React.FC<ComparisonDisplayProps> = ({ imageUrl, results, imageSize, onClose }) => {
    const [activeTab, setActiveTab] = useState<string>(Object.keys(results)[0]);

    // Color palette for different classes
    const classColors: Record<string, string> = {
        'No Endodontic Treatment': '#94a3b8', // slate-400
        'Incomplete Endodontic Treatment': '#facc15', // yellow-400
        'Complete Endodontic Treatment': '#4ade80', // green-400
        'Total Endodontic Failure': '#f87171', // red-400
        'default': '#38bdf8' // sky-400
    };

    const currentResult: ComparisonResult | undefined = results[activeTab];

    if (!currentResult) return null;

    return (
        <div className="fixed inset-0 z-50 bg-slate-900/90 backdrop-blur-sm flex items-center justify-center p-4 md:p-8">
            <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col overflow-hidden animate-fade-in">

                {/* Header */}
                <div className="p-4 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center">
                    <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-teal-300">
                        Model Comparison Analysis
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-colors"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Content Layout */}
                <div className="flex-grow flex flex-col md:flex-row overflow-hidden">

                    {/* Sidebar / Tabs */}
                    <div className="w-full md:w-64 bg-slate-900/50 border-r border-slate-700 overflow-y-auto p-2 space-y-2">
                        <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-widest px-2 py-2">Select Model</h3>
                        {Object.entries(results as Record<string, ComparisonResult>).map(([modelKey, result]) => (
                            <button
                                key={modelKey}
                                onClick={() => setActiveTab(modelKey)}
                                className={`w-full text-left px-4 py-3 rounded-xl transition-all duration-200 border ${activeTab === modelKey
                                    ? 'bg-gradient-to-r from-cyan-500/20 to-teal-500/20 border-cyan-500/50 text-cyan-50'
                                    : 'bg-slate-800/30 border-transparent text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                                    }`}
                            >
                                <div className="font-medium text-sm truncate">{result.details?.name || modelKey}</div>
                                <div className="text-xs opacity-60 mt-1 flex justify-between">
                                    <span>{result.detections?.length || 0} detections</span>
                                    {result.error && <span className="text-red-400">Error</span>}
                                </div>
                            </button>
                        ))}
                    </div>

                    {/* Main Display Area */}
                    <div className="flex-grow flex flex-col h-full overflow-hidden bg-black/20 relative">

                        {/* Image Canvas */}
                        <div className="flex-grow relative p-4 flex items-center justify-center overflow-auto">
                            {currentResult?.error ? (
                                <div className="text-center text-red-400 p-8 border border-red-500/30 bg-red-500/10 rounded-xl">
                                    <p className="font-bold mb-2">Analysis Failed</p>
                                    <p className="text-sm">{currentResult.error}</p>
                                </div>
                            ) : (
                                <div className="relative w-full h-full flex items-center justify-center rounded-lg overflow-hidden border border-slate-700 bg-black/40">
                                    <AnalysisDisplay
                                        imageUrl={imageUrl}
                                        results={currentResult?.detections?.map(d => {
                                            const imgW = imageSize?.width || 640;
                                            const imgH = imageSize?.height || 640;

                                            return {
                                                box: {
                                                    x: d.bbox[0] / imgW,
                                                    y: d.bbox[1] / imgH,
                                                    width: (d.bbox[2] - d.bbox[0]) / imgW,
                                                    height: (d.bbox[3] - d.bbox[1]) / imgH
                                                },
                                                label: d.class,
                                                confidence: d.confidence
                                            };
                                        }) || []}
                                    />
                                </div>
                            )}
                        </div>

                        {/* Info Panel for Active Model */}
                        <div className="h-1/3 min-h-[200px] bg-slate-900 border-t border-slate-700 p-6 overflow-y-auto">
                            <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div>
                                    <h4 className="text-cyan-400 font-bold mb-2 flex items-center gap-2">
                                        {currentResult?.details?.name || activeTab}
                                        <span className="text-xs bg-slate-800 text-slate-400 px-2 py-1 rounded border border-slate-600">
                                            {Number(currentResult?.details?.expected_inference_time_ms || 0)}ms
                                        </span>
                                    </h4>
                                    <p className="text-slate-400 text-sm mb-4">{currentResult?.details?.description}</p>

                                    <div className="space-y-2">
                                        <div className="flex justify-between text-xs border-b border-slate-800 pb-1">
                                            <span className="text-slate-500">Parameters</span>
                                            <span className="text-slate-300 font-mono">{currentResult?.details?.parameters}</span>
                                        </div>
                                    </div>
                                </div>

                                <div>
                                    <h5 className="text-slate-200 font-semibold mb-3 text-sm uppercase tracking-wider">Detections ({currentResult?.detections?.length || 0})</h5>
                                    <div className="space-y-2 max-h-[120px] overflow-y-auto pr-2 custom-scrollbar">
                                        {currentResult?.detections?.map((det, i) => (
                                            <div key={i} className="flex justify-between items-center text-sm p-2 rounded bg-slate-800/50 border border-slate-700/50">
                                                <span className="flex items-center gap-2">
                                                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: classColors[det.class] || classColors['default'] }}></span>
                                                    <span className="text-slate-300">{det.class}</span>
                                                </span>
                                                <span className="text-cyan-400 font-mono text-xs">{(det.confidence * 100).toFixed(1)}%</span>
                                            </div>
                                        ))}
                                        {(!currentResult?.detections || currentResult.detections.length === 0) && (
                                            <p className="text-slate-500 italic text-sm">No pathologies detected.</p>
                                        )}
                                    </div>
                                </div>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>
    );
};
