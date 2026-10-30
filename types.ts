export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionResult {
  box: BoundingBox;
  label: string;
  confidence: number;
}

export interface DetectionBox {
  bbox: [number, number, number, number];
  class: string;
  confidence: number;
}

export interface YoloResponse {
  detections: DetectionBox[];
  image_size?: {
    width: number;
    height: number;
  };
  heatmap?: string;
}
