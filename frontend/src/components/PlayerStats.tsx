import React, { useState, useMemo } from 'react';
import { Clock, User, Edit2, Check, X, Zap, Hand, Shield, Target, Box } from 'lucide-react';

interface PlayerStatsProps {
  actions: any[];
  playerTracks: any[];
  videoId: string;
  fps: number;
  onSeek: (sec: number) => void;
  onPlayerNameChange?: (playerId: number, name: string) => void;
  playerNames?: Record<number, string>;
}

const getActionIcon = (action: string) => {
  const iconClass = "w-4 h-4";
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

const getActionColor = (action: string): string => {
  switch (action?.toLowerCase()) {
    case 'spike':
      return 'bg-red-100 text-red-700 border-red-200';
    case 'set':
      return 'bg-blue-100 text-blue-700 border-blue-200';
    case 'receive':
      return 'bg-green-100 text-green-700 border-green-200';
    case 'serve':
      return 'bg-amber-100 text-amber-700 border-amber-200';
    case 'block':
      return 'bg-purple-100 text-purple-700 border-purple-200';
    default:
      return 'bg-gray-100 text-gray-700 border-gray-200';
  }
};

export const PlayerStats: React.FC<PlayerStatsProps> = ({
  actions,
  playerTracks,
  videoId,
  fps,
  onSeek,
  onPlayerNameChange,
  playerNames = {}
}) => {
  const [editingPlayerId, setEditingPlayerId] = useState<number | null>(null);
  const [editName, setEditName] = useState<string>('');

  // Get unique player IDs from tracks and actions
  const playerIds = useMemo(() => {
    const ids = new Set<number>();
    playerTracks.forEach((track: any) => {
      if (track.players) {
        track.players.forEach((player: any) => {
          if (player.id !== undefined && player.id !== null) {
            ids.add(player.id);
          }
        });
      }
    });
    actions.forEach((action: any) => {
      if (action.player_id !== undefined && action.player_id !== null) {
        ids.add(action.player_id);
      }
    });
    return Array.from(ids).sort((a, b) => a - b);
  }, [playerTracks, actions]);

  // Count unassigned actions
  const unassignedActions = useMemo(() => {
    return actions.filter(action => 
      action.player_id === undefined || action.player_id === null
    );
  }, [actions]);

  // Group actions by player
  const playerActions = useMemo(() => {
    const grouped: Record<number, any[]> = {};
    playerIds.forEach(id => {
      grouped[id] = [];
    });
    
    actions.forEach(action => {
      if (action.player_id !== undefined && action.player_id !== null) {
        if (!grouped[action.player_id]) {
          grouped[action.player_id] = [];
        }
        grouped[action.player_id].push(action);
      }
    });
    
    return grouped;
  }, [actions, playerIds]);

  // Get player statistics
  const playerStats = useMemo(() => {
    const stats: Record<number, { total: number; actions: Record<string, number> }> = {};
    
    playerIds.forEach(id => {
      const playerActionsList = playerActions[id] || [];
      const actionCounts: Record<string, number> = {};
      
      playerActionsList.forEach(action => {
        const actionType = action.action || 'unknown';
        actionCounts[actionType] = (actionCounts[actionType] || 0) + 1;
      });
      
      stats[id] = {
        total: playerActionsList.length,
        actions: actionCounts
      };
    });
    
    return stats;
  }, [playerIds, playerActions]);

  // Filter out players with no actions (must be before any early returns)
  const playersWithActions = useMemo(() => {
    return playerIds.filter(playerId => {
      const actionsList = playerActions[playerId] || [];
      return actionsList.length > 0;
    });
  }, [playerIds, playerActions]);

  const getPlayerName = (playerId: number): string => {
    return playerNames[playerId] || `Player #${playerId}`;
  };

  const handleStartEdit = (playerId: number) => {
    setEditingPlayerId(playerId);
    setEditName(playerNames[playerId] || `Player #${playerId}`);
  };

  const handleSaveEdit = (playerId: number) => {
    if (onPlayerNameChange) {
      onPlayerNameChange(playerId, editName.trim() || `Player #${playerId}`);
    }
    setEditingPlayerId(null);
    setEditName('');
  };

  const handleCancelEdit = () => {
    setEditingPlayerId(null);
    setEditName('');
  };

  if (playerIds.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-md p-8 border border-gray-200">
        <p className="text-gray-500 text-center">No players detected in this video</p>
      </div>
    );
  }

  if (playersWithActions.length === 0 && unassignedActions.length === 0) {
    return (
      <div className="bg-white rounded-xl shadow-md p-8 border border-gray-200">
        <p className="text-gray-500 text-center">No actions detected in this video</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Unassigned Actions Card */}
      {unassignedActions.length > 0 && (
        <div className="bg-yellow-50 rounded-xl shadow-md border border-yellow-200 overflow-hidden">
          <div className="bg-gradient-to-r from-yellow-50 to-amber-50 px-6 py-4 border-b border-yellow-200">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="font-semibold text-yellow-900">Unassigned Actions</h3>
                <p className="text-sm text-yellow-700 mt-1">Actions not matched to any player</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-yellow-900">{unassignedActions.length}</div>
                <div className="text-xs text-yellow-600">Total Actions</div>
              </div>
            </div>
          </div>
          <div className="px-6 py-4">
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {unassignedActions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => onSeek(action.timestamp)}
                  className="w-full text-left p-3 rounded-lg border border-yellow-200 hover:border-yellow-400 hover:bg-yellow-100 transition-all group"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${getActionColor(action.action)}`}>
                        {getActionIcon(action.action)}
                      </div>
                      <div>
                        <div className="font-medium text-gray-900">
                          {action.action.charAt(0).toUpperCase() + action.action.slice(1)}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-gray-500 mt-1">
                          <Clock className="w-3 h-3" />
                          {action.timestamp?.toFixed(1)}s
                          {action.confidence && (
                            <span className="ml-2">
                              ({Math.round(action.confidence * 100)}%)
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                    <div className="text-xs text-gray-400 group-hover:text-yellow-600 transition-colors">
                      Go to →
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {playersWithActions.map((playerId) => {
        const playerActionsList = playerActions[playerId] || [];
        const stats = playerStats[playerId];
        const isEditing = editingPlayerId === playerId;
        const displayName = getPlayerName(playerId);

        return (
          <div
            key={playerId}
            className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden"
          >
            {/* Player Header */}
            <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-6 py-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-blue-600 flex items-center justify-center text-white font-semibold">
                    <User className="w-5 h-5" />
                  </div>
                  {isEditing ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={editName}
                        onChange={(e) => setEditName(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveEdit(playerId);
                          if (e.key === 'Escape') handleCancelEdit();
                        }}
                        className="px-3 py-1 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        autoFocus
                      />
                      <button
                        onClick={() => handleSaveEdit(playerId)}
                        className="p-1 text-green-600 hover:bg-green-50 rounded transition-colors"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                      <button
                        onClick={handleCancelEdit}
                        className="p-1 text-red-600 hover:bg-red-50 rounded transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <div>
                        <h3 className="font-semibold text-gray-900">{displayName}</h3>
                        <p className="text-sm text-gray-500">ID: {playerId}</p>
                      </div>
                      <button
                        onClick={() => handleStartEdit(playerId)}
                        className="ml-2 p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded transition-colors"
                        title="Rename player"
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                    </>
                  )}
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
                  <div className="text-xs text-gray-500">Total Actions</div>
                </div>
              </div>
            </div>

            {/* Action Statistics */}
            {Object.keys(stats.actions).length > 0 && (
              <div className="px-6 py-3 bg-gray-50 border-b border-gray-200">
                <div className="flex flex-wrap gap-2">
                  {Object.entries(stats.actions).map(([actionType, count]) => (
                    <div
                      key={actionType}
                      className={`px-3 py-1 rounded-lg border text-sm font-medium ${getActionColor(actionType)}`}
                    >
                      {actionType.toUpperCase()}: {count}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions List */}
            <div className="px-6 py-4">
              {playerActionsList.length === 0 ? (
                <p className="text-gray-500 text-sm text-center py-4">No actions recorded for this player</p>
              ) : (
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Action Timeline</h4>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {playerActionsList.map((action, index) => (
                      <button
                        key={index}
                        onClick={() => onSeek(action.timestamp)}
                        className="w-full text-left p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-all group"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className={`p-2 rounded-lg ${getActionColor(action.action)}`}>
                              {getActionIcon(action.action)}
                            </div>
                            <div>
                              <div className="font-medium text-gray-900">
                                {action.action.charAt(0).toUpperCase() + action.action.slice(1)}
                              </div>
                              <div className="flex items-center gap-2 text-xs text-gray-500 mt-1">
                                <Clock className="w-3 h-3" />
                                {action.timestamp?.toFixed(1)}s
                                {action.confidence && (
                                  <span className="ml-2">
                                    ({Math.round(action.confidence * 100)}%)
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                          <div className="text-xs text-gray-400 group-hover:text-blue-600 transition-colors">
                            Go to →
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};

