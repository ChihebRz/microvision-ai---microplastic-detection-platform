
import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import DatasetView from './pages/DatasetView';
import TrainingView from './pages/TrainingView';
import PredictionView from './pages/PredictionView';
import InsightsPage from './pages/InsightsPage';
import { Page } from './types';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<Page>(Page.Dashboard);

  const renderPage = () => {
    switch (currentPage) {
      case Page.Dashboard:
        return <Dashboard />;
      case Page.Dataset:
        return <DatasetView />;
      case Page.Training:
        return <TrainingView />;
      case Page.Prediction:
        return <PredictionView />;
      case Page.Insights:
        return <InsightsPage />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="flex min-h-screen bg-slate-900 text-slate-100 selection:bg-sky-500/30">
      <Sidebar currentPage={currentPage} setCurrentPage={setCurrentPage} />
      
      <main className="flex-1 p-6 md:p-10 overflow-y-auto">
        <div className="max-w-[1440px] mx-auto">
          {renderPage()}
        </div>
      </main>
      
      {/* Background Decorative Gradients */}
      <div className="fixed top-0 right-0 -z-10 w-[500px] h-[500px] bg-sky-500/10 blur-[120px] rounded-full pointer-events-none" />
      <div className="fixed bottom-0 left-64 -z-10 w-[400px] h-[400px] bg-indigo-500/10 blur-[100px] rounded-full pointer-events-none" />
    </div>
  );
};

export default App;
