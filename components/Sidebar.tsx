
import React from 'react';
import { Page } from '../types';

interface SidebarProps {
  currentPage: Page;
  setCurrentPage: (page: Page) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ currentPage, setCurrentPage }) => {
  const navItems = [
    { id: Page.Dashboard, label: 'Dashboard', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6' },
    { id: Page.Dataset, label: 'Dataset Explorer', icon: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4' },
    { id: Page.Training, label: 'Training Hub', icon: 'M13 10V3L4 14h7v7l9-11h-7z' },
    { id: Page.Prediction, label: 'Inference Engine', icon: 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z' },
    { id: Page.Insights, label: 'AI Analysis', icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z' },
  ];

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-full sticky top-0">
      <div className="p-6">
        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 bg-sky-500 rounded-lg flex items-center justify-center shadow-lg shadow-sky-500/20">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-1.996 0l-2.387.477a2 2 0 00-1.022.547l-3.859 3.859a2 2 0 01-2.828 0l-.641-.641a2 2 0 010-2.828l3.859-3.859a2 2 0 00.547-1.022l.477-2.387a6 6 0 000-1.996l-.477-2.387a2 2 0 00-.547-1.022L2.121 2.121a2 2 0 012.828 0l.641.641a2 2 0 010 2.828l-3.859 3.859a2 2 0 00-.547 1.022l-.477 2.387a6 6 0 000 1.996l.477 2.387a2 2 0 00.547 1.022l3.859 3.859a2 2 0 010 2.828l-.641.641a2 2 0 01-2.828 0l-3.859-3.859z" />
            </svg>
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight text-white">MicroVision</h1>
            <p className="text-xs text-slate-400 font-medium tracking-wider">AI PLATFORM</p>
          </div>
        </div>

        <nav className="space-y-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setCurrentPage(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
                currentPage === item.id
                  ? 'bg-sky-500/10 text-sky-400'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              <svg
                className={`w-5 h-5 transition-colors ${
                  currentPage === item.id ? 'text-sky-400' : 'text-slate-500 group-hover:text-slate-300'
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.icon} />
              </svg>
              <span className="font-medium text-sm">{item.label}</span>
              {currentPage === item.id && (
                <div className="ml-auto w-1.5 h-1.5 rounded-full bg-sky-400" />
              )}
            </button>
          ))}
        </nav>
      </div>

      <div className="mt-auto p-6 border-t border-slate-800">
        <div className="bg-slate-800/50 rounded-2xl p-4">
          <p className="text-xs font-semibold text-slate-500 mb-2">SYSTEM STATUS</p>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-xs text-slate-300">GPU Clusters Active</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
