
import React, { useState, useEffect } from 'react';
import Card from '../components/Card';
import { geminiService } from '../services/geminiService';
import { MOCK_DISTRIBUTION } from '../constants';

const InsightsPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [insight, setInsight] = useState<string>('');

  useEffect(() => {
    const fetchInsights = async () => {
      setLoading(true);
      const text = await geminiService.getDatasetInsights({
        totalSamples: 14282,
        distribution: MOCK_DISTRIBUTION,
        geography: 'Global Coastal Monitoring',
        topSource: 'Synthetic Textile Runoff (Fibers)'
      });
      setInsight(text || '');
      setLoading(false);
    };
    fetchInsights();
  }, []);

  return (
    <div className="space-y-6 animate-fadeIn max-w-4xl mx-auto">
      <header className="text-center mb-12">
        <h2 className="text-4xl font-extrabold text-white tracking-tight">AI Strategy & Insights</h2>
        <p className="text-slate-400 mt-2 text-lg">High-level synthesis of your environmental data through the lens of Gemini Intelligence.</p>
      </header>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="w-16 h-16 border-4 border-sky-500/20 border-t-sky-500 rounded-full animate-spin mb-6" />
          <p className="text-slate-400 font-medium">Synthesizing global dataset trends...</p>
        </div>
      ) : (
        <Card className="prose prose-invert max-w-none">
          <div className="flex items-center gap-4 mb-8 p-4 bg-sky-500/5 rounded-2xl border border-sky-500/20">
            <div className="w-12 h-12 bg-sky-500 rounded-xl flex items-center justify-center shrink-0">
               <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                 <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
               </svg>
            </div>
            <div>
              <h3 className="text-white font-bold text-xl m-0">Gemini Strategic Synthesis</h3>
              <p className="text-slate-400 text-sm m-0">Analysis of 14,282 labeled microplastic samples.</p>
            </div>
          </div>
          
          <div className="text-slate-300 leading-relaxed text-lg whitespace-pre-wrap">
            {insight}
          </div>
          
          <div className="mt-12 pt-8 border-t border-slate-700 grid grid-cols-1 md:grid-cols-2 gap-6 not-prose">
             <div className="p-4 bg-slate-900 rounded-xl">
               <h4 className="text-sky-400 font-bold mb-2">Priority Level: CRITICAL</h4>
               <p className="text-xs text-slate-500">Fiber distribution in coastal waters suggests a 22% year-over-year increase in household laundry discharge residues.</p>
             </div>
             <div className="p-4 bg-slate-900 rounded-xl">
               <h4 className="text-emerald-400 font-bold mb-2">Confidence Score: 98.4%</h4>
               <p className="text-xs text-slate-500">Model precision is sufficient for autonomous sorting facility deployment in North American pilot sites.</p>
             </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default InsightsPage;
