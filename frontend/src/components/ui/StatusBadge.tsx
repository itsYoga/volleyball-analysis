import React from 'react';

export const StatusBadge: React.FC<{ status?: string }> = ({ status = '' }) => (
  <span
    className={`text-xs font-medium px-2.5 py-0.5 rounded-full border ${
      status === 'completed'
        ? 'bg-green-50 text-green-700 border-green-200'
        : status === 'processing'
        ? 'bg-blue-50 text-blue-700 border-blue-200'
        : status === 'failed'
        ? 'bg-red-50 text-red-700 border-red-200'
        : 'bg-gray-50 text-gray-700 border-gray-200'
    }`}
  >
    {status ? status.charAt(0).toUpperCase() + status.slice(1) : 'Unknown'}
  </span>
);


