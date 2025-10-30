import React from 'react';

export const EmptyState: React.FC<{ title: string; hint?: React.ReactNode; icon?: React.ReactNode }> = ({ title, hint, icon }) => (
  <div className="flex flex-col items-center justify-center h-40 bg-gray-50 rounded-lg border border-dashed">
    {icon}
    <p className="text-gray-500 mt-3">{title}</p>
    {hint && <div className="text-gray-500 text-sm">{hint}</div>}
  </div>
);


