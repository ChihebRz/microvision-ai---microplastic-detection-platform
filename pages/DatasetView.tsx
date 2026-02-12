
import React, { useState } from 'react';
import Card from '../components/Card';
import { CLASSES } from '../constants';

const DatasetView: React.FC = () => {
  const [filter, setFilter] = useState('All');
  
  // Simulated dataset items
  const items = Array.from({ length: 8 }).map((_, i) => ({
    id: `img-${i}`,
    url: `https://picsum.photos/seed/${i + 50}/400/300`,
    class: CLASSES[i % CLASSES.length],
    boxes: 3 + (i % 5)
  }));

  return (
    <div className="space-y-6 animate-fadeIn">
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight">Dataset Explorer</h2>
          <p className="text-slate-400 mt-1">Audit labels, visualize bounding boxes, and manage subsets.</p>
        </div>
        <div className="flex bg-slate-800 p-1 rounded-xl border border-slate-700">
          {['All', 'Train', 'Val', 'Test'].map(tab => (
            <button
              key={tab}
              onClick={() => setFilter(tab)}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                filter === tab ? 'bg-sky-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </header>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
        {items.map((item) => (
          <div key={item.id} className="group relative bg-slate-800 rounded-2xl overflow-hidden border border-slate-700/50 hover:border-sky-500/50 transition-all cursor-pointer">
            <div className="aspect-[4/3] overflow-hidden relative">
              <img src={item.url} alt="Sample" className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110" />
              {/* Simulated Bounding Boxes */}
              <div className="absolute inset-0 p-4 pointer-events-none">
                 <div className="w-12 h-12 border-2 border-sky-400 rounded absolute top-10 left-10 flex items-start">
                    <span className="bg-sky-400 text-[8px] text-white px-1 leading-tight uppercase font-bold">{item.class}</span>
                 </div>
                 <div className="w-16 h-8 border-2 border-teal-400 rounded absolute bottom-20 right-12 flex items-start">
                    <span className="bg-teal-400 text-[8px] text-white px-1 leading-tight uppercase font-bold">Fiber</span>
                 </div>
              </div>
              <div className="absolute top-2 right-2 flex gap-1">
                <span className="bg-black/60 backdrop-blur-md px-2 py-0.5 rounded text-[10px] font-bold text-white uppercase tracking-tighter">
                  {item.boxes} OBJECTS
                </span>
              </div>
            </div>
            <div className="p-3">
              <div className="flex justify-between items-center">
                <p className="text-xs font-mono text-slate-400 uppercase">{item.id}.jpg</p>
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-300 font-bold uppercase">{item.class}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex justify-center mt-8">
        <button className="px-6 py-2 border border-slate-700 rounded-xl text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-all">
          Load More Samples
        </button>
      </div>
    </div>
  );
};

export default DatasetView;
