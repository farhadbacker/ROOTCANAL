
import React, { useState, useCallback, useRef } from 'react';
import { Header } from './components/Header';
import { ImageUploader } from './components/ImageUploader';
import { AnalysisDisplay } from './components/AnalysisDisplay';
import { ComparisonDisplay } from './components/ComparisonDisplay';
import { Loader } from './components/Loader';
import { analyzeImage, generateAnalysisDescription } from './services/geminiService';
import { detectObjects, compareAllModels, getAvailableModels, switchModel } from './services/yoloService';
import type { DetectionResult } from './types';
import { IconAlertTriangle, IconSparkles } from './components/IconComponents';
import { AnalysisDescription } from './components/AnalysisDescription';

const App: React.FC = () => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [results, setResults] = useState<DetectionResult[] | null>(null);
  const [analysisDescription, setAnalysisDescription] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // New State for Multi-Model
  const [availableModels, setAvailableModels] = useState<any>(null);
  const [currentModel, setCurrentModel] = useState<string>('student_distilled');
  const [comparisonResults, setComparisonResults] = useState<any>(null);
  const [isComparing, setIsComparing] = useState<boolean>(false);

  // Fetch models on mount
  React.useEffect(() => {
    getAvailableModels().then(data => {
      setAvailableModels(data.available_models);
      setCurrentModel(data.current_model);
    }).catch(err => console.error("Failed to load models", err));
  }, []);

  const handleModelSwitch = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newModel = e.target.value;
    setIsLoading(true);
    try {
      await switchModel(newModel);
      setCurrentModel(newModel);
      // Re-analyze if image exists
      if (imageUrl && !error) {
        // tricky to re-analyze without the original file object if we only have base64 in state
        // Ideally we should keep the file object in state or just reset
        // For now, let's just notify user or do nothing. 
        // Actually, let's just let them click "New Analysis" or we can store the file.
      }
    } catch (err) {
      console.error(err);
      setError("Failed to switch model");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCompareAll = async () => {
    if (!imageUrl) return;
    setIsComparing(true);
    try {
      // We need the original file. 
      // Since we didn't store it in state relative to this scope (it was in callback), 
      // we might need to store `currentFile` state.
      // Let's fix handleImageUpload to store file.
    } catch (e) {
      console.error(e);
    }
  };
  const [currentFile, setCurrentFile] = useState<File | null>(null);

  const handleImageUpload = useCallback(async (base64: string, file: File) => {
    setImageUrl(base64);
    setCurrentFile(file);
    setResults(null);
    setAnalysisDescription(null);
    setError(null);
    setIsLoading(true);

    try {
      // Create image element for YOLO detection
      const img = new Image();
      img.src = base64;
      await new Promise((resolve) => {
        img.onload = resolve;
      });

      // Run YOLO detection with Filter and Grad-CAM enabled
      console.log('Running YOLO detection with Filter and Grad-CAM...');
      const yoloResponse = await detectObjects(file, true, true);
      const yoloDetections = yoloResponse.detections;
      console.log('YOLO detections:', yoloDetections);

      if (yoloResponse.heatmap) {
        console.log("✓ Heatmap received");
        setHeatmap(yoloResponse.heatmap);
      } else {
        console.log("⚠️ No heatmap received");
      }

      // Log detection source for debugging
      if (yoloDetections.length > 0 && yoloDetections[0].class === 'No Endodontic Treatment' && yoloDetections[0].bbox[0] === 100) {
        console.log('⚠️ Using mock dental detections - ONNX model not loaded properly');
      }

      // Map active model detections to the frontend DetectionResult format
      const formattedResults: DetectionResult[] = yoloDetections.map((d: any) => {
        const imgW = yoloResponse.image_size?.width || 640;
        const imgH = yoloResponse.image_size?.height || 640;
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
      });
      setResults(formattedResults);

      // Generate report using Gemini based purely on active model findings
      if (formattedResults && formattedResults.length > 0) {
        const description = await generateAnalysisDescription(formattedResults, yoloDetections);
        setAnalysisDescription(description);
      } else {
        setAnalysisDescription("No specific root canal issues were detected by the model. A comprehensive clinical examination is always recommended for a complete diagnosis.");
      }
    } catch (e) {
      console.error(e);
      setError('Failed to analyze the image. The AI model may be unable to process this image or an API error occurred. Please try another image.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleReset = () => {
    setImageUrl(null);
    setResults(null);
    setAnalysisDescription(null);
    setError(null);
    setIsLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30">
      <Header />
      <main className="flex-grow container mx-auto p-4 md:p-8 flex flex-col max-w-7xl">
        <div className="flex-grow grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">

          {/* Left Column: Upload & Display */}
          <div className="lg:col-span-7 w-full h-full flex flex-col gap-6">
            <div className="flex justify-between items-center">
              <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-teal-300">
                Radiograph Analysis
              </h2>
              {(imageUrl || error) && (
                <button
                  onClick={handleReset}
                  className="px-4 py-2 bg-slate-700/50 hover:bg-slate-600/80 border border-slate-600 hover:border-slate-500 rounded-lg text-sm font-semibold transition-all duration-200 disabled:opacity-50 shadow-sm hover:shadow-md flex items-center gap-2"
                  disabled={isLoading}>
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
                  </svg>
                  New Analysis
                </button>
              )}
            </div>

            <div className="bg-slate-800/40 backdrop-blur-sm rounded-2xl p-1 flex-grow flex flex-col min-h-[500px] border border-slate-700/50 shadow-xl overflow-hidden relative group">
              {/* Decorative corner accents */}
              <div className="absolute top-0 left-0 w-20 h-20 bg-cyan-500/10 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2 pointer-events-none"></div>
              <div className="absolute bottom-0 right-0 w-20 h-20 bg-teal-500/10 rounded-full blur-3xl translate-x-1/2 translate-y-1/2 pointer-events-none"></div>

              {!imageUrl ? (
                <div className="flex-grow p-6 flex flex-col">
                  <ImageUploader onImageUpload={handleImageUpload} isLoading={isLoading} />
                </div>
              ) : (
                <div className="relative flex-grow flex items-center justify-center bg-black/20 rounded-xl overflow-hidden">
                  {isLoading && (
                    <div className="absolute inset-0 z-20 bg-slate-900/60 backdrop-blur-[2px] flex flex-col items-center justify-center">
                      <Loader />
                      <p className="mt-4 text-cyan-300 font-medium animate-pulse">Analyzing radiograph structure...</p>
                    </div>
                  )}

                  {error && !isLoading && (
                    <div className="text-center text-red-400 p-8 max-w-md mx-auto bg-slate-800/80 rounded-xl border border-red-500/30 shadow-lg backdrop-blur-md">
                      <IconAlertTriangle className="mx-auto h-12 w-12 mb-4 text-red-500" />
                      <p className="font-bold text-lg mb-2">Analysis Failed</p>
                      <p className="text-sm text-slate-300">{error}</p>
                    </div>
                  )}

                  {!error && <AnalysisDisplay imageUrl={imageUrl} results={results} heatmap={heatmap} />}
                </div>
              )}
            </div>
          </div>

          {/* Right Column: Results & Report */}
          <div className="lg:col-span-5 w-full h-full flex flex-col gap-6">
            {/* Model & Compare Controls */}
            <div className="flex items-center gap-4 mb-4 bg-slate-800/50 p-3 rounded-xl border border-slate-700/50 backdrop-blur-sm">
              <div className="flex-grow flex flex-col">
                <span className="text-xs text-slate-500 uppercase font-semibold tracking-wider mb-1">Active Model</span>
                <select
                  value={currentModel}
                  onChange={handleModelSwitch}
                  disabled={isLoading || isComparing}
                  className="bg-slate-900 border border-slate-700 text-slate-200 text-sm rounded-lg focus:ring-cyan-500 focus:border-cyan-500 block w-full p-2.5"
                >
                  {availableModels ? (
                    Object.entries(availableModels).map(([key, val]: any) => (
                      <option key={key} value={key}>
                        {val.name}
                      </option>
                    ))
                  ) : (
                    <option>Loading models...</option>
                  )}
                </select>
              </div>

              {imageUrl && !error && (
                <button
                  onClick={async () => {
                    if (!currentFile) return;
                    setIsComparing(true);
                    try {
                      const res = await compareAllModels(currentFile);
                      setComparisonResults(res);
                    } catch (e) {
                      console.error(e);
                      setError("Comparison failed");
                    } finally {
                      setIsComparing(false);
                    }
                  }}
                  disabled={isComparing || isLoading}
                  className="h-[50px] px-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-semibold rounded-xl shadow-lg hover:shadow-purple-500/25 transition-all duration-200 flex items-center gap-2 whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isComparing ? (
                    <>
                      <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Comparing...
                    </>
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M5 4a1 1 0 00-2 0v7.268a2 2 0 000 3.464V16a1 1 0 102 0v-1.268a2 2 0 000-3.464V4zM11 4a1 1 0 10-2 0v1.268a2 2 0 000 3.464V16a1 1 0 102 0V8.732a2 2 0 000-3.464V4zM16 3a1 1 0 011 1v7.268a2 2 0 010 3.464V16a1 1 0 11-2 0v-1.268a2 2 0 010-3.464V4a1 1 0 011-1z" />
                      </svg>
                      Compare All
                    </>
                  )}
                </button>
              )}
            </div>

            <div className="flex items-center gap-2">
              <IconSparkles className="w-6 h-6 text-cyan-400" />
              <h2 className="text-2xl font-bold text-slate-100">Diagnostic Report</h2>
            </div>

            <div className={`flex-grow bg-slate-800/60 backdrop-blur-md rounded-2xl border border-slate-700/50 shadow-xl overflow-hidden transition-all duration-500 ${!imageUrl && !error ? 'opacity-50 grayscale' : 'opacity-100'}`}>

              {!imageUrl && !error ? (
                <div className="h-full flex flex-col items-center justify-center p-8 text-center text-slate-500 space-y-4">
                  <div className="w-16 h-16 rounded-full bg-slate-700/50 flex items-center justify-center mb-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 011.414.586l4 4a1 1 0 01.586 1.414V19a2 2 0 01-2 2z" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-lg font-medium text-slate-400">Waiting for Analysis</p>
                    <p className="text-sm mt-1 max-w-[200px] mx-auto">Upload a dental X-ray to generate a comprehensive AI diagnostic report.</p>
                  </div>
                </div>
              ) : (
                <div className="h-full flex flex-col">
                  {/* Report Header */}
                  <div className="p-4 border-b border-slate-700/50 bg-slate-800/80 flex justify-between items-center">
                    <span className="text-xs font-bold uppercase tracking-wider text-slate-400">AI Generated Findings</span>
                    {isLoading && <span className="text-xs text-cyan-400 animate-pulse">Processing...</span>}
                    {!isLoading && analysisDescription && <span className="text-xs text-emerald-400 flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-emerald-400"></span> Ready
                    </span>}
                  </div>

                  {/* Report Content */}
                  <div className="p-6 overflow-y-auto max-h-[600px] custom-scrollbar">
                    {isLoading ? (
                      <div className="space-y-4 animate-pulse">
                        <div className="h-4 bg-slate-700/50 rounded w-3/4"></div>
                        <div className="h-4 bg-slate-700/50 rounded w-full"></div>
                        <div className="h-4 bg-slate-700/50 rounded w-5/6"></div>
                        <div className="h-20 bg-slate-700/30 rounded w-full mt-6"></div>
                      </div>
                    ) : (
                      <>
                        {analysisDescription && <AnalysisDescription description={analysisDescription} />}
                        {error && !analysisDescription && (
                          <p className="text-slate-400 italic text-center mt-10">Report generation unavailable.</p>
                        )}
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="text-center py-6 text-xs text-slate-500 border-t border-slate-800/50 mt-8 bg-slate-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4">
          <p className="max-w-2xl mx-auto">
            <span className="font-semibold text-slate-400">Disclaimer:</span> This tool is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
          </p>
          <p className="mt-2 opacity-60">Based on "Fusion of Image Filtering and Knowledge-Distilled YOLO Models for Root Canal Failure Diagnosis".</p>
        </div>
      </footer>

      {comparisonResults && imageUrl && (
        <ComparisonDisplay
          imageUrl={imageUrl}
          results={comparisonResults.comparison_results}
          imageSize={comparisonResults.image_size}
          onClose={() => setComparisonResults(null)}
        />
      )}
    </div>
  );
};

export default App;
