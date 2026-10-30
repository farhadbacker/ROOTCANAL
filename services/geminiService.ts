
import { GoogleGenAI, Type } from "@google/genai";
import type { DetectionResult } from '../types';

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  // This is a fallback for development. In a real environment, the key is expected to be set.
  console.warn("API_KEY is not set. Using a placeholder. This will likely fail.");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

const model = 'gemini-2.5-flash';

const objectDetectionPrompt = `
You are an expert AI dental radiologist. Your task is to analyze a dental X-ray image and identify teeth with potential root canal issues. 
You must act as an object detection model. For each tooth you identify, provide its root canal status based on the following categories:
- 'No Endodontic Treatment': A healthy tooth or a tooth with a cavity that has not reached the pulp. No signs of root canal procedure.
- 'Incomplete Endodontic Treatment': A root canal procedure has been started but is not complete. Canals may be partially filled or unfilled.
- 'Complete Endodontic Treatment': A successful root canal. The canals are fully filled to the apex, and there are no signs of infection.
- 'Total Endodontic Failure': A root canal has been performed, but there are clear signs of failure, such as periapical lesions (dark areas at the root tip), voids in the filling, or new decay.

For each finding, you must return:
1. A 'label' from the categories above.
2. A 'confidence' score from 0.0 to 1.0.
3. A normalized 'box' object with x, y, width, and height, where (x,y) is the top-left corner. All values must be between 0.0 and 1.0, relative to the image dimensions.

Return ONLY a JSON object that adheres to the provided schema. Do not return markdown or any other text. If no issues are found, return an empty array.
`;

const responseSchema = {
  type: Type.ARRAY,
  items: {
    type: Type.OBJECT,
    properties: {
      label: { type: Type.STRING },
      confidence: { type: Type.NUMBER },
      box: {
        type: Type.OBJECT,
        properties: {
          x: { type: Type.NUMBER },
          y: { type: Type.NUMBER },
          width: { type: Type.NUMBER },
          height: { type: Type.NUMBER },
        },
        required: ["x", "y", "width", "height"],
      },
    },
    required: ["label", "confidence", "box"],
  },
};

export const analyzeImage = async (base64Image: string, mimeType: string): Promise<DetectionResult[]> => {
  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: {
        parts: [
          { inlineData: { data: base64Image, mimeType: mimeType } },
          { text: objectDetectionPrompt },
        ],
      },
      config: {
        responseMimeType: "application/json",
        responseSchema: responseSchema,
      },
    });

    const jsonText = response.text.trim();
    const parsedResponse = JSON.parse(jsonText);
    const filtered = (parsedResponse as DetectionResult[]).filter(
      (result) => result.label !== 'No Endodontic Treatment'
    );
    return filtered;

  } catch (error) {
    console.error("Error calling Gemini API for image analysis:", error);
    throw new Error("Gemini API request failed during image analysis.");
  }
};

export const generateAnalysisDescription = async (results: DetectionResult[], yoloDetections?: any[]): Promise<string> => {
  const descriptionPrompt = `
You are an expert AI dental radiologist providing a consultation. Based on the following detection results from a dental radiograph, generate a concise and informative report for a dental professional.

First, provide a brief "Overall Summary" of the findings.

Then, for each identified issue, create a section with a markdown heading (e.g., "### Finding 1: Total Endodontic Failure"). In each section, briefly explain the likely visual indicators on the radiograph that led to the diagnosis and suggest potential next steps or treatment considerations.

Maintain a professional and objective tone.

The detected root canal issues are:
${JSON.stringify(results, null, 2)}

${yoloDetections ? `Additional object detections from YOLO model: ${JSON.stringify(yoloDetections, null, 2)}` : ''}
`;

  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: descriptionPrompt,
    });
    return response.text;
  } catch (error) {
    console.error("Error calling Gemini API for description generation:", error);
    throw new Error("Gemini API request failed during description generation.");
  }
};
