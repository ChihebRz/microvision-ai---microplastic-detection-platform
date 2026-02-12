
import React from 'react';
import Card from '../components/Card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { MOCK_CHART_DATA, MOCK_DISTRIBUTION, COLORS } from '../constants';

const Dashboard: React.FC = () => {
  const stats = [
    { label: 'Total Samples', value: '14,282', change: '+12%', color: 'sky' },
    { label: 'Avg mAP@.5', value: '0.942', change: '+2.1%', color: 'teal' },
    { label: 'Classes Found', value: '5 Types', change: 'Stable', color: 'indigo' },
    { label: 'Training Runs', value: '42 Active', change: '8.4h avg', color: 'purple' },
  ];

  return (
    <div className="space-y-6 animate-fadeIn">
      <header className="flex justify-between items-center mb-8">
        <div>
          <h2 className="text-3xl font-bold text-white tracking-tight">Mission Control</h2>
          <p className="text-slate-400 mt-1">Real-time overview of your microplastic computer vision pipeline.</p>
        </div>
        <div className="flex gap-3">
          <button className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-xl text-sm font-medium text-slate-300 hover:bg-slate-700 transition-colors">
            Download Report
          </button>
          <button className="px-4 py-2 bg-sky-500 rounded-xl text-sm font-semibold text-white hover:bg-sky-400 shadow-lg shadow-sky-500/20 transition-all">
            New Experiment
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, idx) => (
          <Card key={idx} className="relative overflow-hidden group">
            <div className={`absolute top-0 right-0 w-24 h-24 -mr-8 -mt-8 bg-${stat.color}-500/5 rounded-full transition-transform group-hover:scale-110`} />
            <p className="text-sm font-medium text-slate-400">{stat.label}</p>
            <div className="flex items-baseline gap-3 mt-2">
              <h3 className="text-3xl font-bold text-white">{stat.value}</h3>
              <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${stat.change.startsWith('+') ? 'bg-green-500/10 text-green-400' : 'bg-slate-700 text-slate-400'}`}>
                {stat.change}
              </span>
            </div>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card title="Detection Volume (24h)" className="lg:col-span-2">
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={MOCK_CHART_DATA}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="name" stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                  cursor={{ fill: '#334155' }}
                />
                <Bar dataKey="count" fill={COLORS.primary} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>

        <Card title="Class Distribution">
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={MOCK_DISTRIBUTION}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {MOCK_DISTRIBUTION.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={Object.values(COLORS)[index % Object.values(COLORS).length]} />
                  ))}
                </Pie>
                <Tooltip 
                   contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#f8fafc' }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-4 grid grid-cols-2 gap-2">
              {MOCK_DISTRIBUTION.map((item, i) => (
                <div key={i} className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: Object.values(COLORS)[i % Object.values(COLORS).length] }} />
                  <span className="text-xs text-slate-400">{item.name} ({item.value}%)</span>
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>

      <Card title="Use Cases & Impact Tracking">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="p-4 border border-slate-700/50 rounded-xl bg-slate-800/30">
            <h4 className="text-sky-400 font-bold mb-2">Ocean Cleanup</h4>
            <p className="text-sm text-slate-400">Targeting high-density zones in the North Atlantic. 1.2 tons identified this month.</p>
          </div>
          <div className="p-4 border border-slate-700/50 rounded-xl bg-slate-800/30">
            <h4 className="text-teal-400 font-bold mb-2">Facility Sortation</h4>
            <p className="text-sm text-slate-400">Integrated into 12 recycling plants. Sorting efficiency increased by 14%.</p>
          </div>
          <div className="p-4 border border-slate-700/50 rounded-xl bg-slate-800/30">
            <h4 className="text-indigo-400 font-bold mb-2">Supply Chain</h4>
            <p className="text-sm text-slate-400">Monitoring micro-residue in food packaging production lines.</p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default Dashboard;
