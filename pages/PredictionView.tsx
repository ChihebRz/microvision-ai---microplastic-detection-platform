
import React, { useState, useRef } from 'react';
import Card from '../components/Card';
import { geminiService } from '../services/geminiService';

const PredictionView: React.FC = () => {
  const [image, setImage] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [aiInsight, setAiInsight] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setImage(reader.result as string);
        setResults([]);
        setAiInsight(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const runInference = async () => {
    if (!image) return;
    setIsPredicting(true);
    
    // Simulate model inference delay
    setTimeout(async () => {
      const mockDetections = [
        { id: 1, class: 'Fiber', bbox: [0.2, 0.3, 0.1, 0.4], conf: 0.94 },
        { id: 2, class: 'Fragment', bbox: [0.6, 0.5, 0.2, 0.15], conf: 0.88 },
        { id: 3, class: 'Film', bbox: [0.4, 0.7, 0.1, 0.1], conf: 0.91 },
      ];
      setResults(mockDetections);
      setIsPredicting(false);

      // Trigger Gemini Analysis
      const insight = await geminiService.analyzeMicroplastics(
        "Ocean water sample from coastal Florida region, likely containing high concentrations of synthetic fibers.",
        mockDetections
      );
      setAiInsight(insight);
    }, 2000);
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      <header>
        <h2 className="text-3xl font-bold text-white tracking-tight">Inference Engine</h2>
        <p className="text-slate-400 mt-1">Upload environmental samples to perform high-precision microplastic detection.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <div 
            className={`relative aspect-video rounded-xl border-2 border-dashed transition-all flex items-center justify-center overflow-hidden bg-slate-900/50 ${
              !image ? 'border-slate-700 hover:border-sky-500' : 'border-slate-800'
            }`}
          >
            {image ? (
              <div className="relative w-full h-full">
                <img src={image} alt="Upload" className="w-full h-full object-contain" />
                {/* Overlay detections if available */}
                {results.map(res => (
                  <div 
                    key={res.id}
                    className="absolute border-2 border-sky-400 group cursor-help"
                    style={{
                      left: `${res.bbox[0] * 100}%`,
                      top: `${res.bbox[1] * 100}%`,
                      width: `${res.bbox[2] * 100}%`,
                      height: `${res.bbox[3] * 100}%`,
                    }}
                  >
                    <span className="absolute -top-6 left-0 bg-sky-400 text-white text-[10px] font-bold px-1.5 py-0.5 rounded flex items-center gap-1">
                      {res.class} {(res.conf * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center p-8">
                <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                </div>
                <p className="text-slate-300 font-medium">Drag and drop or click to upload</p>
                <p className="text-slate-500 text-sm mt-1">Supports JPEG, PNG up to 10MB</p>
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  className="absolute inset-0 opacity-0 cursor-pointer"
                  accept="image/*"
                />
              </div>
            )}
            {isPredicting && (
              <div className="absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex flex-col items-center justify-center z-10">
                <div className="w-12 h-12 border-4 border-sky-500/20 border-t-sky-500 rounded-full animate-spin mb-4" />
                <p className="text-sky-400 font-bold animate-pulse uppercase tracking-widest text-sm">Running Neural Network...</p>
              </div>
            )}
          </div>
          
          <div className="mt-6 flex gap-4">
            <button 
              onClick={() => fileInputRef.current?.click()}
              className="flex-1 py-3 bg-slate-800 text-slate-200 rounded-xl font-bold hover:bg-slate-700 border border-slate-700 transition-all"
            >
              CHOOSE ANOTHER FILE
            </button>
            <button 
              onClick={runInference}
              disabled={!image || isPredicting}
              className={`flex-1 py-3 rounded-xl font-bold transition-all ${
                !image || isPredicting 
                ? 'bg-slate-700 text-slate-500 cursor-not-allowed' 
                : 'bg-sky-500 text-white hover:bg-sky-400 shadow-lg shadow-sky-500/30'
              }`}
            >
              RUN PREDICTION
            </button>
          </div>
        </Card>

        <div className="space-y-6">
          <Card title="Detections Summary">
            {results.length > 0 ? (
              <div className="space-y-3">
                {results.map(res => (
                  <div key={res.id} className="flex justify-between items-center p-3 bg-slate-900/50 rounded-lg border border-slate-700/50">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 rounded-full bg-sky-400" />
                      <span className="text-sm font-semibold text-slate-200">{res.class}</span>
                    </div>
                    <span className="text-xs font-mono text-emerald-400">{(res.conf * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-500 italic">No results yet. Run inference to see data.</p>
            )}
          </Card>

          <Card title="AI Environmental Insight" className="min-h-[200px]">
            {aiInsight ? (
              <div className="text-sm text-slate-300 leading-relaxed space-y-4">
                <div className="p-3 bg-sky-500/10 border-l-4 border-sky-500 rounded-r-lg mb-4 italic text-xs">
                  Generated by Gemini 3 Flash
                </div>
                {aiInsight.split('\n').map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center text-center p-8 opacity-50">
                 <div className="w-10 h-10 mb-2 bg-slate-900 rounded-full flex items-center justify-center">
                    <svg className="w-6 h-6 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                 </div>
                 <p className="text-xs text-slate-400">Awaiting detection results for analysis...</p>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default PredictionView;
