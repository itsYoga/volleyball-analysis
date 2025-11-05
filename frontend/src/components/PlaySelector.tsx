import React from 'react';
import { PlayCircle, Clock, Target, Zap } from 'lucide-react';

interface Play {
  play_id: number;
  start_frame: number;
  start_timestamp: number;
  end_frame: number;
  end_timestamp: number;
  duration: number;
  actions: any[];
  scores: any[];
}

interface PlaySelectorProps {
  plays: Play[];
  currentTime: number;
  fps: number;
  onSeek: (timestamp: number) => void;
}

export const PlaySelector: React.FC<PlaySelectorProps> = ({
  plays,
  currentTime,
  fps,
  onSeek
}) => {
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getCurrentPlay = (): number | null => {
    const currentFrame = Math.round(currentTime * fps);
    for (let i = 0; i < plays.length; i++) {
      const play = plays[i];
      if (play.start_frame <= currentFrame && currentFrame <= play.end_frame) {
        return i;
      }
    }
    return null;
  };

  const currentPlayIndex = getCurrentPlay();

  if (plays.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-md p-6 border border-gray-200">
        <p className="text-gray-500 text-center">No plays detected in this video</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
      <div className="bg-gradient-to-r from-blue-50 to-blue-100 px-6 py-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <PlayCircle className="w-5 h-5 text-blue-600" />
          Plays ({plays.length})
        </h3>
        <p className="text-sm text-gray-600 mt-1">Select a play to jump to its start</p>
      </div>
      
      <div className="max-h-96 overflow-y-auto">
        {plays.map((play, index) => {
          const isCurrent = currentPlayIndex === index;
          const hasScore = play.scores && play.scores.length > 0;
          
          return (
            <button
              key={play.play_id}
              onClick={() => onSeek(play.start_timestamp)}
              className={`
                w-full px-6 py-4 text-left border-b border-gray-100 
                transition-all hover:bg-blue-50
                ${isCurrent ? 'bg-blue-50 border-l-4 border-l-blue-600' : ''}
              `}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-semibold text-gray-900">
                      Play #{play.play_id}
                    </span>
                    {isCurrent && (
                      <span className="px-2 py-0.5 bg-blue-600 text-white text-xs rounded-full">
                        Playing
                      </span>
                    )}
                    {hasScore && (
                      <span className="px-2 py-0.5 bg-yellow-500 text-white text-xs rounded-full flex items-center gap-1">
                        <Target className="w-3 h-3" />
                        Score
                      </span>
                    )}
                  </div>
                  
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <div className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {formatTime(play.start_timestamp)} - {formatTime(play.end_timestamp)}
                    </div>
                    <span className="text-gray-500">
                      ({play.duration?.toFixed(1)}s)
                    </span>
                  </div>
                  
                  {play.actions && play.actions.length > 0 && (
                    <div className="mt-2 flex items-center gap-2 flex-wrap">
                      <Zap className="w-3 h-3 text-gray-400" />
                      <span className="text-xs text-gray-500">
                        {play.actions.length} action{play.actions.length !== 1 ? 's' : ''}
                      </span>
                      {play.actions.slice(0, 3).map((action, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-0.5 bg-gray-100 text-gray-700 text-xs rounded"
                        >
                          {action.action}
                        </span>
                      ))}
                      {play.actions.length > 3 && (
                        <span className="text-xs text-gray-400">
                          +{play.actions.length - 3} more
                        </span>
                      )}
                    </div>
                  )}
                </div>
                
                <PlayCircle className={`w-5 h-5 flex-shrink-0 ${isCurrent ? 'text-blue-600' : 'text-gray-400'}`} />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
