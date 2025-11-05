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
  jerseyMappings?: Record<string, any>;  // 手動標記的球衣號碼映射
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
  playerNames = {},
  jerseyMappings = {}
}) => {
  const [editingPlayerId, setEditingPlayerId] = useState<number | null>(null);
  const [editName, setEditName] = useState<string>('');

  // Get unique player IDs from tracks and actions
  // 使用 stable_id 或 jersey_number 來識別玩家，而不是 track_id
  // 但只有當 jersey_number 真正存在時才使用它作為ID
  const playerIds = useMemo(() => {
    const ids = new Set<number>();
    const idToInfo: Record<number, { stableId?: number; jerseyNumber?: number; trackId?: number }> = {};
    
    // 首先創建 track_id 到 jersey_number 的映射（用於處理 actions）
    // 使用最頻繁出現的 jersey_number
    const trackIdToJerseyCounts: Record<number, Record<number, number>> = {};
    
    playerTracks.forEach((track: any) => {
      if (track.players) {
        track.players.forEach((player: any) => {
          const trackId = player.id;
          const jerseyNumber = player.jersey_number;
          
          // 只有當 jersey_number 存在且不等於 track_id 時，才是真正的 OCR 檢測
          // 檢查 null 和 undefined
          const isRealOCR = jerseyNumber !== undefined && 
                          jerseyNumber !== null && 
                          jerseyNumber !== trackId;
          
          if (isRealOCR) {
            // 記錄 track_id 到 jersey_number 的映射（計數）
            if (!trackIdToJerseyCounts[trackId]) {
              trackIdToJerseyCounts[trackId] = {};
            }
            trackIdToJerseyCounts[trackId][jerseyNumber] = (trackIdToJerseyCounts[trackId][jerseyNumber] || 0) + 1;
            
            // 使用球衣號碼作為 ID（添加到 playerIds）
            const playerId = jerseyNumber;
            ids.add(playerId);
            if (!idToInfo[playerId]) {
              idToInfo[playerId] = {};
            }
            idToInfo[playerId].trackId = trackId;
            idToInfo[playerId].jerseyNumber = jerseyNumber;
          } else if (player.stable_id !== undefined && player.stable_id !== null) {
            // 使用 stable_id
            const playerId = player.stable_id;
            ids.add(playerId);
            if (!idToInfo[playerId]) {
              idToInfo[playerId] = {};
            }
            idToInfo[playerId].trackId = trackId;
            idToInfo[playerId].stableId = player.stable_id;
          } else {
            // 最後使用 track_id
            const playerId = player.id;
            ids.add(playerId);
            if (!idToInfo[playerId]) {
              idToInfo[playerId] = {};
            }
            idToInfo[playerId].trackId = trackId;
          }
        });
      }
    });
    
    // 為每個 track_id 選擇最頻繁的 jersey_number
    const trackIdToJersey: Record<number, number> = {};
    Object.keys(trackIdToJerseyCounts).forEach(trackIdStr => {
      const trackId = Number(trackIdStr);
      const counts = trackIdToJerseyCounts[trackId];
      
      let maxCount = 0;
      let mostCommonJersey: number | null = null;
      
      Object.keys(counts).forEach(jerseyStr => {
        const jersey = Number(jerseyStr);
        const count = counts[jersey];
        if (count > maxCount) {
          maxCount = count;
          mostCommonJersey = jersey;
        }
      });
      
      if (mostCommonJersey !== null) {
        trackIdToJersey[trackId] = mostCommonJersey;
      }
    });
    
    // 處理 actions 中的 player_id（track_id），將其映射到 jersey_number
    // 這樣可以確保所有有 actions 的玩家都被包含，即使他們的 track_id 沒有在 playerTracks 中
    actions.forEach((action: any) => {
      if (action.player_id !== undefined && action.player_id !== null) {
        const trackId = action.player_id;
        
        // 如果這個 track_id 有對應的 jersey_number，使用 jersey_number
        if (trackIdToJersey[trackId]) {
          const jerseyNumber = trackIdToJersey[trackId];
          ids.add(jerseyNumber);
        } else {
          // 否則使用 track_id（確保它在 ids 中）
          ids.add(trackId);
        }
      }
    });
    
    // 合併手動標記的球衣號碼映射（確保所有手動標記的 track_id 都被包含）
    Object.keys(jerseyMappings).forEach(trackIdStr => {
      const trackId = Number(trackIdStr);
      const mapping = jerseyMappings[trackIdStr];
      if (mapping && mapping.jersey_number !== undefined && mapping.jersey_number !== null) {
        const jerseyNumber = mapping.jersey_number;
        ids.add(jerseyNumber);
        if (!idToInfo[jerseyNumber]) {
          idToInfo[jerseyNumber] = {};
        }
        idToInfo[jerseyNumber].trackId = trackId;
        idToInfo[jerseyNumber].jerseyNumber = jerseyNumber;
      }
    });
    
    return Array.from(ids).sort((a, b) => a - b);
  }, [playerTracks, actions, jerseyMappings]);

  // Count unassigned actions
  const unassignedActions = useMemo(() => {
    return actions.filter(action => 
      action.player_id === undefined || action.player_id === null
    );
  }, [actions]);

  // 創建 track_id 到 jersey_number 的映射（用於將 actions 中的 player_id 映射到球衣號碼）
  // 使用最頻繁出現的 jersey_number 作為映射，並合併手動標記的映射
  const trackIdToJerseyMap = useMemo(() => {
    // 使用 Map 來追蹤每個 track_id 對應的所有 jersey_number 及其出現次數
    const jerseyCounts: Record<number, Record<number, number>> = {};
    
    playerTracks.forEach((track: any) => {
      if (track.players) {
        track.players.forEach((player: any) => {
          const trackId = player.id;
          const jerseyNumber = player.jersey_number;
          
          // 只有當 jersey_number 存在且不等於 track_id 時，才是真正的 OCR 檢測
          // 檢查 null 和 undefined
          if (jerseyNumber !== undefined && jerseyNumber !== null && jerseyNumber !== trackId) {
            if (!jerseyCounts[trackId]) {
              jerseyCounts[trackId] = {};
            }
            jerseyCounts[trackId][jerseyNumber] = (jerseyCounts[trackId][jerseyNumber] || 0) + 1;
          }
        });
      }
    });
    
    // 為每個 track_id 選擇出現次數最多的 jersey_number
    const map: Record<number, number> = {};
    Object.keys(jerseyCounts).forEach(trackIdStr => {
      const trackId = Number(trackIdStr);
      const counts = jerseyCounts[trackId];
      
      // 找到出現次數最多的 jersey_number
      let maxCount = 0;
      let mostCommonJersey: number | null = null;
      
      Object.keys(counts).forEach(jerseyStr => {
        const jersey = Number(jerseyStr);
        const count = counts[jersey];
        if (count > maxCount) {
          maxCount = count;
          mostCommonJersey = jersey;
        }
      });
      
      if (mostCommonJersey !== null) {
        map[trackId] = mostCommonJersey;
      }
    });
    
    // 合併手動標記的球衣號碼映射（優先級最高）
    Object.keys(jerseyMappings).forEach(trackIdStr => {
      const trackId = Number(trackIdStr);
      const mapping = jerseyMappings[trackIdStr];
      if (mapping && mapping.jersey_number !== undefined && mapping.jersey_number !== null) {
        map[trackId] = mapping.jersey_number;
      }
    });
    
    return map;
  }, [playerTracks, jerseyMappings]);

  // Group actions by player (將 track_id 映射到 jersey_number)
  const playerActions = useMemo(() => {
    const grouped: Record<number, any[]> = {};
    
    // 初始化所有可能的 player IDs（包括 jersey_numbers 和 track_ids）
    playerIds.forEach(id => {
      grouped[id] = [];
    });
    
    actions.forEach(action => {
      if (action.player_id !== undefined && action.player_id !== null) {
        const trackId = action.player_id;
        
        // 嘗試將 track_id 映射到 jersey_number
        let targetPlayerId: number = trackId;
        
        // 如果這個 track_id 有對應的 jersey_number，使用 jersey_number
        if (trackIdToJerseyMap[trackId]) {
          targetPlayerId = trackIdToJerseyMap[trackId];
          
          // 確保 jersey_number 在 playerIds 中
          if (!playerIds.includes(targetPlayerId)) {
            // 如果 jersey_number 不在 playerIds 中，添加到 grouped
            if (!grouped[targetPlayerId]) {
              grouped[targetPlayerId] = [];
            }
          }
        }
        
        // 確保 targetPlayerId 在 grouped 中
        if (!grouped[targetPlayerId]) {
          grouped[targetPlayerId] = [];
        }
        
        grouped[targetPlayerId].push(action);
      }
    });
    
    return grouped;
  }, [actions, playerIds, trackIdToJerseyMap]);

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

  // 獲取玩家的顯示信息（包括球衣號碼）
  // 這個函數需要訪問 trackIdToJerseyMap，所以定義在它之後
  const getPlayerDisplayInfo = (playerId: number) => {
    // 先檢查手動標記的映射（優先級最高）
    for (const trackIdStr of Object.keys(jerseyMappings)) {
      const trackId = Number(trackIdStr);
      const mapping = jerseyMappings[trackIdStr];
      if (mapping && mapping.jersey_number === playerId) {
        return {
          trackId: trackId,
          stableId: undefined,
          jerseyNumber: playerId
        };
      }
    }
    
    // 檢查 playerId 是否在 trackIdToJerseyMap 的值中（是 jersey_number）
    let isJerseyNumber = false;
    let correspondingTrackIds: number[] = [];
    
    // 檢查 trackIdToJerseyMap 中是否有映射到這個 playerId 的 track_id
    Object.keys(trackIdToJerseyMap).forEach(trackIdStr => {
      const trackId = Number(trackIdStr);
      const jersey = trackIdToJerseyMap[trackId];
      if (jersey === playerId) {
        isJerseyNumber = true;
        correspondingTrackIds.push(trackId);
      }
    });
    
    // 如果 playerId 是 jersey_number（通過映射得到的），直接返回
    if (isJerseyNumber && correspondingTrackIds.length > 0) {
      // 嘗試從 playerTracks 中找到對應的玩家信息
      for (const track of playerTracks) {
        if (track.players) {
          for (const player of track.players) {
            const trackId = player.id;
            const jerseyNumber = player.jersey_number;
            
            // 如果找到匹配的 track_id 和 jersey_number
            if (correspondingTrackIds.includes(trackId) && jerseyNumber === playerId) {
              return {
                trackId: trackId,
                stableId: player.stable_id,
                jerseyNumber: jerseyNumber
              };
            }
          }
        }
      }
      
      // 如果沒找到，但 playerId 是 jersey_number，返回基本信息
      return {
        trackId: correspondingTrackIds[0], // 使用第一個對應的 track_id
        stableId: undefined,
        jerseyNumber: playerId  // 重要：這裡應該返回 playerId（jersey_number）
      };
    }
    
    // 從 playerTracks 中查找該玩家的信息（用於 track_id 或其他 ID）
    for (const track of playerTracks) {
      if (track.players) {
        for (const player of track.players) {
          const trackId = player.id;
          const jerseyNumber = player.jersey_number;
          
          // 確定該玩家的識別ID（與 playerIds 中的邏輯一致）
          let id: number;
          const isRealOCR = jerseyNumber !== undefined && 
                          jerseyNumber !== null && 
                          jerseyNumber !== trackId;
          
          if (isRealOCR) {
            id = jerseyNumber;
          } else if (player.stable_id !== undefined && player.stable_id !== null) {
            id = player.stable_id;
          } else {
            id = trackId;
          }
          
          // 如果 playerId 是 track_id 或其他 ID，直接匹配
          if (id === playerId) {
            return {
              trackId: trackId,
              stableId: player.stable_id,
              jerseyNumber: isRealOCR ? jerseyNumber : undefined
            };
          }
        }
      }
    }
    
    // 最後的備選方案
    return {
      trackId: playerId,
      stableId: undefined,
      jerseyNumber: undefined
    };
  };

  const getPlayerName = (playerId: number): string => {
    const info = getPlayerDisplayInfo(playerId);
    
    // 檢查 playerId 是否在 trackIdToJerseyMap 的值中（是 jersey_number）
    const isJerseyNumber = Object.values(trackIdToJerseyMap).includes(playerId);
    
    // 只有當 jersey_number 真正存在且不等於 track_id 時，才是 OCR 檢測到的球衣號碼
    // 或者當 playerId 在 trackIdToJerseyMap 的值中時，也是 jersey_number
    if (isJerseyNumber || (info.jerseyNumber !== undefined && info.jerseyNumber !== null)) {
      const jerseyNum = isJerseyNumber ? playerId : info.jerseyNumber;
      return `#${jerseyNum}`;
    }
    // 否則顯示 Player #X 或 Track ID: X（這是 track_id，不是 OCR 檢測）
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
                        {(() => {
                          const info = getPlayerDisplayInfo(playerId);
                          if (info.jerseyNumber !== undefined && info.jerseyNumber !== null) {
                            return (
                              <p className="text-sm text-gray-500">
                                球衣號碼: <span className="font-semibold text-green-600">#{info.jerseyNumber}</span>
                                {info.trackId !== playerId && ` (Track ID: ${info.trackId})`}
                              </p>
                            );
                          }
                          return <p className="text-sm text-gray-500">Track ID: {playerId}</p>;
                        })()}
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

