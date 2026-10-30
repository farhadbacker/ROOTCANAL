
import React, { useState, useCallback, useRef } from 'react';
import { IconUpload } from './IconComponents';

interface ImageUploaderProps {
  onImageUpload: (base64: string, file: File) => void;
  isLoading: boolean;
}

export const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageUpload, isLoading }) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File | null) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        if (typeof e.target?.result === 'string') {
          onImageUpload(e.target.result, file);
        }
      };
      reader.readAsDataURL(file);
    }
  }, [onImageUpload]);

  const onDragEnter = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault(); // Necessary to allow drop
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div
      className={`relative w-full h-full min-h-[400px] p-8 border-2 border-dashed rounded-xl flex flex-col items-center justify-center text-center transition-all duration-500 group overflow-hidden ${isDragging
          ? 'border-cyan-400 bg-cyan-900/20 scale-[1.02] shadow-2xl shadow-cyan-500/20'
          : 'border-slate-700 hover:border-slate-500 bg-slate-800/50 hover:bg-slate-800/80'
        }`}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      {/* Background decoration */}
      <div className={`absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-blue-500/5 transition-opacity duration-500 ${isDragging ? 'opacity-100' : 'opacity-0'}`}></div>

      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        onChange={onFileChange}
        accept="image/*"
        disabled={isLoading}
      />

      <div className={`relative z-10 p-6 rounded-full bg-slate-800/80 mb-6 transition-all duration-500 ${isDragging ? 'scale-110 shadow-lg shadow-cyan-500/30' : 'shadow-md'}`}>
        <IconUpload className={`w-12 h-12 transition-colors duration-300 ${isDragging ? 'text-cyan-400' : 'text-slate-400 group-hover:text-cyan-300'}`} />
      </div>

      <h3 className="relative z-10 text-xl font-bold text-slate-200 mb-2 group-hover:text-white transition-colors">
        {isDragging ? 'Drop Image Here' : 'Upload Radiograph'}
      </h3>

      <p className="relative z-10 text-slate-400 mb-8 max-w-xs mx-auto leading-relaxed">
        Drag & drop your dental X-ray here, or click to browse files
      </p>

      <button
        onClick={onButtonClick}
        disabled={isLoading}
        className="relative z-10 px-8 py-3 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-bold rounded-lg transition-all duration-300 shadow-lg shadow-cyan-900/50 hover:shadow-cyan-500/30 disabled:opacity-50 disabled:cursor-not-allowed transform hover:-translate-y-0.5 active:translate-y-0"
      >
        Select Image
      </button>

      <div className="relative z-10 mt-8 flex items-center gap-4 text-xs text-slate-500 font-medium uppercase tracking-wider">
        <span className="px-2 py-1 bg-slate-800 rounded border border-slate-700">JPEG</span>
        <span className="px-2 py-1 bg-slate-800 rounded border border-slate-700">PNG</span>
        <span className="px-2 py-1 bg-slate-800 rounded border border-slate-700">WEBP</span>
      </div>
    </div>
  );
};
