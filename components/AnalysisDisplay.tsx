import React, { useState, useRef, useEffect } from 'react';
import type { DetectionResult } from '../types';
import { CLASS_COLORS } from '../constants';

interface AnalysisDisplayProps {
  imageUrl: string;
  results: DetectionResult[] | null;
  heatmap?: string | null;
}

export const AnalysisDisplay: React.FC<AnalysisDisplayProps> = ({ imageUrl, results, heatmap }) => {
  const imageRef = useRef<HTMLImageElement>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  useEffect(() => {
    const updateSize = () => {
      if (imageRef.current) {
        setImageSize({
          width: imageRef.current.offsetWidth,
          height: imageRef.current.offsetHeight,
        });
      }
    };

    const imgElement = imageRef.current;
    if (imgElement) {
      imgElement.addEventListener('load', updateSize);
      window.addEventListener('resize', updateSize);

      // If image is already loaded (e.g., from cache)
      if (imgElement.complete) {
        updateSize();
      }
    }

    return () => {
      if (imgElement) {
        imgElement.removeEventListener('load', updateSize);
      }
      window.removeEventListener('resize', updateSize);
    };
  }, [imageUrl]);

  return (
    <div className="relative w-full h-full flex items-center justify-center bg-black/40">
      <div className="relative inline-block max-w-full max-h-full">
        <img
          ref={imageRef}
          src={imageUrl}
          alt="Dental radiograph"
          className="max-w-full max-h-full object-contain shadow-2xl"
        />
        {/* Heatmap Overlay */}
        {heatmap && imageSize.width > 0 && (
          <img
            src={`data:image/jpeg;base64,${heatmap}`}
            alt="AI Attention Heatmap"
            className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-500"
            style={{
              maxWidth: '100%',
              maxHeight: '100%',
              opacity: 0.5,
              mixBlendMode: 'screen',
            }}
          />
        )}

        {results && imageSize.width > 0 && (
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              width: imageSize.width,
              height: imageSize.height,
            }}
          >
            {results.map((result, index) => {
              const { x, y, width, height } = result.box;
              const colorConfig = CLASS_COLORS[result.label] || CLASS_COLORS['default'];
              const confidencePercent = (result.confidence * 100).toFixed(1);
              const isHovered = hoveredIndex === index;

              return (
                <div
                  key={index}
                  className={`absolute transition-all duration-300 pointer-events-auto cursor-help group
                  ${colorConfig.border} border-2 
                  ${isHovered ? 'bg-white/10 z-20 scale-[1.02]' : 'z-10'}
                `}
                  style={{
                    left: `${x * 100}%`,
                    top: `${y * 100}%`,
                    width: `${width * 100}%`,
                    height: `${height * 100}%`,
                    boxShadow: isHovered ? '0 0 20px rgba(0,0,0,0.5)' : 'none'
                  }}
                  onMouseEnter={() => setHoveredIndex(index)}
                  onMouseLeave={() => setHoveredIndex(null)}
                >
                  {/* Label Tag */}
                  <div
                    className={`
                    absolute -top-8 left-0 px-3 py-1.5 rounded-md text-xs font-bold text-white shadow-lg flex items-center gap-2 transition-all duration-300
                    ${colorConfig.bg}
                    ${isHovered ? 'scale-110 -translate-y-1' : ''}
                  `}
                    style={{ whiteSpace: 'nowrap' }}
                  >
                    <span>{result.label}</span>
                    <span className="bg-black/20 px-1.5 py-0.5 rounded text-[10px] font-mono">
                      {confidencePercent}%
                    </span>
                  </div>

                  {/* Corner Accents */}
                  <div className={`absolute -top-1 -left-1 w-3 h-3 border-t-4 border-l-4 ${colorConfig.border} opacity-80`}></div>
                  <div className={`absolute -top-1 -right-1 w-3 h-3 border-t-4 border-r-4 ${colorConfig.border} opacity-80`}></div>
                  <div className={`absolute -bottom-1 -left-1 w-3 h-3 border-b-4 border-l-4 ${colorConfig.border} opacity-80`}></div>
                  <div className={`absolute -bottom-1 -right-1 w-3 h-3 border-b-4 border-r-4 ${colorConfig.border} opacity-80`}></div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
