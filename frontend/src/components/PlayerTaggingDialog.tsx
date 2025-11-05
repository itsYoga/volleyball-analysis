import React, { useState } from 'react';
import { X, Check } from 'lucide-react';

interface PlayerTaggingDialogProps {
  player: any;
  currentFrame: number;
  onClose: () => void;
  onConfirm: (jerseyNumber: number) => void;
}

export const PlayerTaggingDialog: React.FC<PlayerTaggingDialogProps> = ({
  player,
  currentFrame,
  onClose,
  onConfirm
}) => {
  const [jerseyNumber, setJerseyNumber] = useState<string>('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const num = parseInt(jerseyNumber);
    if (num >= 1 && num <= 99) {
      onConfirm(num);
      onClose();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full mx-4" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">標記球衣號碼</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <div className="mb-4">
          <p className="text-sm text-gray-600 mb-2">
            追蹤 ID: <span className="font-semibold">{player.id || player.stable_id}</span>
          </p>
          <p className="text-sm text-gray-600">
            當前幀: <span className="font-semibold">{currentFrame}</span>
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              球衣號碼 (1-99)
            </label>
            <input
              type="number"
              min="1"
              max="99"
              value={jerseyNumber}
              onChange={(e) => setJerseyNumber(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              placeholder="輸入球衣號碼"
              autoFocus
            />
          </div>

          <div className="flex gap-3 justify-end">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors font-medium"
            >
              取消
            </button>
            <button
              type="submit"
              disabled={!jerseyNumber || parseInt(jerseyNumber) < 1 || parseInt(jerseyNumber) > 99}
              className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Check className="w-4 h-4" />
              確認
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

