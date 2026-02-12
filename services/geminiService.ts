
import { GoogleGenAI } from "@google/genai";

export class GeminiService {
  private ai: GoogleGenAI;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
  }

  async analyzeMicroplastics(imageDescription: string, detections: any[]) {
    try {
      const prompt = `
        Analyze the following microplastic detection data for environmental impact:
        Image Description: ${imageDescription}
        Detected Objects: ${JSON.stringify(detections)}
        
        Provide a professional summary including:
        1. Estimated environmental impact.
        2. Potential sources of this contamination.
        3. Recommendations for targeted cleanup based on the data.
        Keep the tone professional and scientific.
      `;

      const response = await this.ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: prompt,
      });

      return response.text;
    } catch (error) {
      console.error("Gemini Analysis Error:", error);
      return "Unable to generate AI analysis at this time. Please check your connectivity and API configuration.";
    }
  }

  async getDatasetInsights(stats: any) {
    try {
      const prompt = `
        You are an environmental data scientist. Analyze these microplastic dataset statistics:
        ${JSON.stringify(stats)}
        
        Provide high-level strategic insights on:
        - Most common contaminants.
        - Geographic trends (if any).
        - Priority areas for ocean cleanup operations.
      `;

      const response = await this.ai.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: prompt,
      });

      return response.text;
    } catch (error) {
      console.error("Gemini Insights Error:", error);
      return "Insights currently unavailable.";
    }
  }
}

export const geminiService = new GeminiService();
