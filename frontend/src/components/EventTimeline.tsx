import React from 'react';

interface Props {
  actions: any[];
  scores: any[];
  gameStates: any[];
  duration: number; // total frames
  currentFrame?: number;
  onSeek: (sec: number) => void;
}

const actionIcon = (action?: string) => {
  switch (action) {
    case 'spike': return '💥';
    case 'set': return '👐';
    case 'receive': return '🛡️';
    case 'serve': return '🎯';
    case 'block': return '🧱';
    default: return '•';
  }
}

export const EventTimeline: React.FC<Props> = ({ actions = [], scores = [], gameStates = [], duration, currentFrame = 0, onSeek }) => (
  <div className="relative w-full h-24 flex flex-col justify-around py-2">

    {/* Track 1: Scores */}
    <div className="relative w-full h-6">
      {scores?.map((e: any, i: number) => (
        <button
          key={`score-${i}`}
          style={{ left: `${(e.frame / duration) * 100}%` }}
          className="absolute top-0 -translate-x-1/2 rounded-full bg-white border border-gray-300 shadow px-2.5 py-0.5 text-sm hover:shadow-md hover:border-blue-500 z-10"
          onClick={() => onSeek(e.timestamp)}
          title={`Score by #${e.player_id} @ ${e.timestamp?.toFixed(1)}s`}
        >
          🏐
        </button>
      ))}
    </div>

    {/* Track 2: Game State Bar */}
    <div className="relative w-full h-3 rounded-full bg-gray-200 overflow-hidden">
      {gameStates?.map((s: any, i: number) => (
        <div
          key={i}
          style={{
            left: `${(s.start_frame / duration) * 100}%`,
            width: `${((s.end_frame - s.start_frame) / duration) * 100}%`,
            background: s.state === 'Play' ? '#86efac' : s.state === 'No-Play' ? '#fecaca' : '#fde68a'
          }}
          className="absolute h-full"
          title={`${s.state}`}
        />
      ))}
    </div>

    {/* Track 3: Actions */}
    <div className="relative w-full h-6">
      {actions?.map((a: any, i: number) => (
        <button
          key={`act-${i}`}
          style={{ left: `${(a.frame / duration) * 100}%` }}
          className="absolute top-0 -translate-x-1/2 text-lg text-blue-700 hover:text-blue-900 hover:scale-125 transition-transform"
          onClick={() => onSeek(a.timestamp)}
          title={`#${a.player_id} ${a.action} @ ${a.timestamp?.toFixed(1)}s`}
        >
          {actionIcon(a.action)}
        </button>
      ))}
    </div>

    {/* Playhead */}
    <div
      className="absolute top-0 bottom-0 w-px bg-blue-600/70"
      style={{ left: `${(currentFrame / Math.max(1, duration)) * 100}%` }}
      aria-hidden
    />
  </div>
);
