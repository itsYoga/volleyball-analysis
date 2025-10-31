import React from 'react';
import { Zap, Hand, Shield, Target, Box, Trophy } from 'lucide-react';

interface Props {
  actions: any[];
  scores: any[];
  gameStates: any[];
  duration: number; // total frames
  currentFrame?: number;
  onSeek: (sec: number) => void;
  fps?: number; // frames per second for time conversion
  playerNames?: Record<number, string>;
}

const getActionIcon = (action?: string) => {
  const iconClass = "w-3.5 h-3.5";
  switch (action?.toLowerCase()) {
    case 'spike':
      return <Zap className={iconClass} />;
    case 'set':
      return <Hand className={iconClass} />;
    case 'receive':
      return <Shield className={iconClass} />;
    case 'serve':
      return <Target className={iconClass} />;
    case 'block':
      return <Box className={iconClass} />;
    default:
      return <div className="w-2 h-2 rounded-full bg-gray-400" />;
  }
};

const getActionColor = (action?: string) => {
  switch (action?.toLowerCase()) {
    case 'spike':
      return 'bg-red-500 border-red-600 text-white';
    case 'set':
      return 'bg-blue-500 border-blue-600 text-white';
    case 'receive':
      return 'bg-green-500 border-green-600 text-white';
    case 'serve':
      return 'bg-amber-500 border-amber-600 text-white';
    case 'block':
      return 'bg-purple-500 border-purple-600 text-white';
    default:
      return 'bg-gray-500 border-gray-600 text-white';
  }
};

export const EventTimeline: React.FC<Props> = ({ 
  actions = [], 
  scores = [], 
  gameStates = [], 
  duration, 
  currentFrame = 0, 
  onSeek,
  fps = 30,
  playerNames = {}
}) => {
  const containerRef = React.useRef<HTMLDivElement>(null);
  const [isDragging, setIsDragging] = React.useState(false);

  const getPlayerName = (playerId: number): string => {
    return playerNames[playerId] || `Player #${playerId}`;
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return; // Only left mouse button
    e.preventDefault();
    setIsDragging(true);
    handleSeek(e);
  };

  const handleSeek = React.useCallback((e: React.MouseEvent | MouseEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, x / rect.width));
    const targetFrame = Math.round(percentage * duration);
    const targetTime = targetFrame / fps;
    onSeek(targetTime);
  }, [duration, fps, onSeek]);

  const handleMouseMove = React.useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    handleSeek(e);
  }, [isDragging, handleSeek]);

  const handleMouseUp = React.useCallback(() => {
    setIsDragging(false);
  }, []);

  React.useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  return (
    <div 
      ref={containerRef}
      className="relative w-full space-y-3"
      onMouseDown={handleMouseDown}
      style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
    >
      {/* Scores Track */}
      {scores.length > 0 && (
        <div className="relative w-full h-8">
          <div className="text-xs text-gray-500 mb-1 font-medium">Scores</div>
          <div className="relative w-full h-6 bg-gray-100 rounded-md overflow-hidden">
            {scores.map((e: any, i: number) => (
              <button
                key={`score-${i}`}
                style={{ left: `${(e.frame / duration) * 100}%` }}
                className="absolute top-0 -translate-x-1/2 z-10 group"
                onClick={(evt) => {
                  evt.stopPropagation();
                  onSeek(e.timestamp);
                }}
                onMouseDown={(evt) => evt.stopPropagation()}
                title={`Score by ${getPlayerName(e.player_id)} @ ${e.timestamp?.toFixed(1)}s`}
              >
                <div className="bg-gradient-to-br from-yellow-400 to-yellow-600 border-2 border-yellow-700 rounded-full p-1.5 shadow-lg group-hover:scale-110 transition-transform">
                  <Trophy className="w-3 h-3 text-white" />
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Game State Track */}
      {gameStates.length > 0 && (
        <div className="relative w-full h-6">
          <div className="text-xs text-gray-500 mb-1 font-medium">Game State</div>
          <div className="relative w-full h-4 bg-gray-200 rounded-full overflow-hidden">
            {gameStates.map((s: any, i: number) => (
              <div
                key={i}
                style={{
                  left: `${(s.start_frame / duration) * 100}%`,
                  width: `${((s.end_frame - s.start_frame) / duration) * 100}%`,
                }}
                className={`absolute h-full ${
                  s.state === 'Play' 
                    ? 'bg-green-400' 
                    : s.state === 'No-Play' 
                    ? 'bg-gray-300' 
                    : 'bg-yellow-400'
                }`}
                title={`${s.state}`}
              />
            ))}
          </div>
        </div>
      )}

      {/* Actions Track */}
      {actions.length > 0 && (
        <div className="relative w-full h-10">
          <div className="text-xs text-gray-500 mb-1 font-medium">Actions</div>
          <div className="relative w-full h-8 bg-gray-50 rounded-lg border border-gray-200 overflow-visible">
            {actions.map((a: any, i: number) => {
              const position = (a.frame / duration) * 100;
              return (
                <button
                  key={`act-${i}`}
                  style={{ left: `${position}%` }}
                  className={`absolute top-1 -translate-x-1/2 z-10 border-2 rounded px-2 py-0.5 flex items-center gap-1 shadow-sm hover:shadow-md transition-all ${getActionColor(a.action)}`}
                  onClick={(evt) => {
                    evt.stopPropagation();
                    onSeek(a.timestamp);
                  }}
                  onMouseDown={(evt) => evt.stopPropagation()}
                  title={`${a.action.charAt(0).toUpperCase() + a.action.slice(1)}${a.player_id ? ` by ${getPlayerName(a.player_id)}` : ''} @ ${a.timestamp?.toFixed(1)}s`}
                >
                  {getActionIcon(a.action)}
                  <span className="text-xs font-semibold">{a.action}</span>
                  {a.player_id && (
                    <span className="text-xs opacity-90">{getPlayerName(a.player_id)}</span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Playhead Indicator */}
      <div
        className="absolute top-0 bottom-0 w-0.5 bg-blue-600 shadow-lg z-20 pointer-events-none"
        style={{ left: `${(currentFrame / Math.max(1, duration)) * 100}%` }}
      >
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3 h-3 bg-blue-600 rounded-full border-2 border-white shadow-md" />
      </div>
    </div>
  );
};
