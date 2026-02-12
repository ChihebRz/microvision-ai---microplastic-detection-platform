
import React, { useState, useEffect } from 'react';
import Card from '../components/Card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const TrainingView: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [currentEpoch, setCurrentEpoch] = useState(0);

  useEffect(() => {
    let interval: any;
    if (isTraining && currentEpoch < 100) {
      interval = setInterval(() => {
        setCurrentEpoch(prev => prev + 1);
        setMetrics(prev => [
          ...prev,
          {
            epoch: prev.length + 1,
            loss: 1.2 * Math.exp(-(prev.length / 30)) + Math.random() * 0.05,
            mAP: 0.2 + (0.75 * (1 - Math.exp(-(prev.length / 40)))) + Math.random() * 0.02
          }
        ]);
      }, 800);
    } else if (currentEpoch >= 100) {
      setIsTraining(false);
    }
    return () => clearInterval(interval);
  }, [isTraining, currentEpoch]);

  const toggleTraining = () => {
    if (!isTraining) {
      setMetrics([]);
      setCurrentEpoch(0);
    }
    setIsTraining(!isTraining);
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      <header className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight">Training Hub</h2>
          <p className="text-slate-400 mt-1">Monitor YOLOv8 training telemetry and experiment logs.</p>
        </div>
        <div className="flex items-center gap-4">
           {isTraining && (
             <div className="flex items-center gap-2 px-3 py-1.5 bg-sky-500/10 border border-sky-500/30 rounded-lg">
               <div className="w-2 h-2 rounded-full bg-sky-500 animate-ping" />
               <span className="text-xs font-bold text-sky-400">EPOCH {currentEpoch}/100</span>
             </div>
           )}
           <button 
             onClick={toggleTraining}
             className={`px-6 py-2 rounded-xl text-sm font-bold transition-all ${
               isTraining 
                ? 'bg-rose-500 text-white hover:bg-rose-600 shadow-lg shadow-rose-500/20' 
                : 'bg-emerald-500 text-white hover:bg-emerald-600 shadow-lg shadow-emerald-500/20'
             }`}
           >
             {isTraining ? 'STOP TRAINING' : 'INITIATE TRAINING'}
           </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <Card className="lg:col-span-1 space-y-6">
          <div>
            <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Configuration</h4>
            <div className="space-y-4">
               <div>
                 <label className="text-[10px] text-slate-400 uppercase font-bold">Base Model</label>
                 <select className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-300 mt-1">
                   <option>YOLOv8n (Nano)</option>
                   <option>YOLOv8s (Small)</option>
                   <option>YOLOv8m (Medium)</option>
                 </select>
               </div>
               <div>
                 <label className="text-[10px] text-slate-400 uppercase font-bold">Image Size</label>
                 <select className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-300 mt-1">
                   <option>640px</option>
                   <option>1024px</option>
                 </select>
               </div>
               <div>
                 <label className="text-[10px] text-slate-400 uppercase font-bold">Learning Rate</label>
                 <input type="text" className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-sm text-slate-300 mt-1" defaultValue="0.001" />
               </div>
            </div>
          </div>
          
          <div className="pt-6 border-t border-slate-700">
             <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Latest Logs</h4>
             <div className="space-y-2 font-mono text-[10px] text-slate-400 h-48 overflow-y-auto">
                {metrics.slice(-5).reverse().map((m, i) => (
                  <div key={i} className="flex justify-between border-b border-slate-700 pb-1">
                    <span>EPOCH {m.epoch}</span>
                    <span className="text-emerald-400">mAP: {m.mAP.toFixed(4)}</span>
                  </div>
                ))}
                {!isTraining && <p className="italic">Waiting for session...</p>}
             </div>
          </div>
        </Card>

        <div className="lg:col-span-3 space-y-6">
          <Card title="Training Progress Metrics">
            <div className="h-[400px]">
               <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="epoch" stroke="#64748b" />
                    <YAxis yId="left" stroke="#64748b" />
                    <YAxis yId="right" orientation="right" stroke="#64748b" />
                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }} />
                    <Legend verticalAlign="top" height={36}/>
                    <Line yId="left" type="monotone" dataKey="loss" stroke="#38bdf8" strokeWidth={2} dot={false} name="Total Loss" />
                    <Line yId="right" type="monotone" dataKey="mAP" stroke="#2dd4bf" strokeWidth={2} dot={false} name="mAP@.5" />
                  </LineChart>
               </ResponsiveContainer>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default TrainingView;
