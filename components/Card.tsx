
import React from 'react';

interface CardProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
}

const Card: React.FC<CardProps> = ({ children, title, className = "" }) => {
  return (
    <div className={`bg-slate-800 border border-slate-700/50 rounded-2xl shadow-xl overflow-hidden ${className}`}>
      {title && (
        <div className="px-6 py-4 border-b border-slate-700/50 bg-slate-800/50">
          <h3 className="font-semibold text-slate-200">{title}</h3>
        </div>
      )}
      <div className="p-6">
        {children}
      </div>
    </div>
  );
};

export default Card;
