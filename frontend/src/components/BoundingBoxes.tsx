import React, { useRef, useEffect } from 'react';

interface BoundingBoxesProps {
  playerTracks: any[];
  actions: any[];
  currentTime: number;
  fps: number;
  videoSize: { width: number; height: number };
  showPlayers?: boolean;
  showActions?: boolean;
  playerNames?: Record<number, string>;
  onPlayerClick?: (player: any, event: React.MouseEvent) => void;  // 新增：點擊回調
  jerseyMappings?: Record<string, any>;  // 新增：球衣號碼映射
}

const getPlayerColor = (playerId: number): string => {
  const colors = [
    '#3B82F6', // blue
    '#10B981', // green
    '#F59E0B', // amber
    '#EF4444', // red
    '#8B5CF6', // purple
    '#EC4899', // pink
    '#06B6D4', // cyan
    '#F97316', // orange
  ];
  return colors[playerId % colors.length];
};

const getActionColor = (action: string): string => {
  const colorMap: Record<string, string> = {
    spike: '#EF4444',    // red
    set: '#3B82F6',      // blue
    receive: '#10B981',  // green
    serve: '#F59E0B',    // amber
    block: '#8B5CF6',    // purple
  };
  return colorMap[action] || '#6B7280';
};

export const BoundingBoxes: React.FC<BoundingBoxesProps> = ({
  playerTracks,
  actions,
  currentTime,
  fps,
  videoSize,
  showPlayers = true,
  showActions = true,
  playerNames = {},
  onPlayerClick,
  jerseyMappings = {}
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !videoSize?.width || !videoSize?.height) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

    // Helper function to get player name
    const getPlayerName = (playerId: number): string => {
      return playerNames[playerId] || `Player #${playerId}`;
    };

    // Ensure canvas size matches video size
    const canvas = canvasRef.current;
    if (canvas.width !== videoSize.width || canvas.height !== videoSize.height) {
      canvas.width = videoSize.width;
      canvas.height = videoSize.height;
    }

    // Clear canvas
    ctx.clearRect(0, 0, videoSize.width, videoSize.height);

    const currentFrame = Math.round(currentTime * fps);

    // Draw player bounding boxes
    if (showPlayers && playerTracks.length > 0) {
      // Find the track entry closest to current frame
      // Use linear interpolation for smoother rendering between frames
      let currentTrack = playerTracks.find(
        (track) => track.frame === currentFrame
      );
      
      // If exact match not found, find closest one within 15 frames (increased for better matching)
      if (!currentTrack) {
        currentTrack = playerTracks.reduce((closest, track) => {
          if (!closest) return track;
          const closestDiff = Math.abs(closest.frame - currentFrame);
          const trackDiff = Math.abs(track.frame - currentFrame);
          return trackDiff < closestDiff ? track : closest;
        }, null as any);
      }

      // Only draw if we found a track within reasonable range (increased to 15 frames for better matching)
      if (currentTrack && currentTrack.players && Math.abs(currentTrack.frame - currentFrame) <= 15) {
        // 移除置信度過濾，顯示所有檢測到的球員（因為後端已經處理了）
        currentTrack.players.forEach((player: any) => {
          if (!player.bbox || !Array.isArray(player.bbox) || player.bbox.length < 4) return;
          
          const [x1, y1, x2, y2] = player.bbox;
          const playerId = player.id || player.stable_id || 0;
          const color = getPlayerColor(playerId);
          
          // 檢查是否有球衣號碼映射
          const trackIdStr = String(player.id || playerId);
          const jerseyMapping = jerseyMappings[trackIdStr];
          const jerseyNumber = jerseyMapping?.jersey_number || player.jersey_number || null;
          
          // Draw bounding box（如果已標記，使用不同的樣式）
          ctx.strokeStyle = jerseyNumber ? '#10B981' : color;  // 已標記用綠色
          ctx.lineWidth = jerseyNumber ? 3 : 2;
          ctx.setLineDash(jerseyNumber ? [] : [5, 5]);  // 未標記用虛線
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          // Draw label background
          const label = jerseyNumber 
            ? `#${jerseyNumber}` 
            : (getPlayerName(playerId) || `ID: ${playerId}`);
          ctx.font = 'bold 12px system-ui, -apple-system, sans-serif';
          ctx.textBaseline = 'top';
          const textMetrics = ctx.measureText(label);
          const textWidth = textMetrics.width;
          const textHeight = 18;

          ctx.fillStyle = jerseyNumber ? '#10B981' : color;
          ctx.fillRect(x1, y1 - textHeight - 2, textWidth + 8, textHeight);

          // Draw label text
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(label, x1 + 4, y1 - textHeight + 2);
        });
      }
    }

    // Draw action bounding boxes
    if (showActions && actions.length > 0) {
      // 顯示當前幀的動作檢測（每一幀都顯示）
      const currentActions = actions.filter(
        (action) => 
          action.frame !== undefined && 
          action.frame === currentFrame  // 只顯示當前幀的動作檢測
      );

      currentActions.forEach((action) => {
        if (!action.bbox || !Array.isArray(action.bbox) || action.bbox.length < 4) return;
        
        const [x1, y1, x2, y2] = action.bbox;
        const color = getActionColor(action.action);
        
        // Draw bounding box with dashed line
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Draw action label with player name if available
        const playerName = action.player_id !== undefined && action.player_id !== null 
          ? getPlayerName(action.player_id) 
          : '';
        const label = `${action.action.toUpperCase()}${playerName ? ` ${playerName}` : action.player_id ? ` #${action.player_id}` : ''}`;
        ctx.font = 'bold 11px system-ui, -apple-system, sans-serif';
        ctx.textBaseline = 'top';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 16;

        // Label background
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - textHeight - 2, textWidth + 8, textHeight);

        // Label text
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x1 + 4, y1 - textHeight + 2);
      });
    }
  }, [playerTracks, actions, currentTime, fps, videoSize, showPlayers, showActions, playerNames, jerseyMappings]);

  return (
    <>
      <canvas
        ref={canvasRef}
        width={videoSize.width || 640}
        height={videoSize.height || 360}
        className="absolute left-0 top-0 pointer-events-none"
        style={{ 
          imageRendering: 'crisp-edges',
          width: '100%',
          height: 'auto',
          zIndex: 10,
          pointerEvents: 'none'  // 確保 canvas 不會阻擋點擊
        }}
      />
      {/* 移除 overlay div，改為在父組件中處理點擊事件 */}
    </>
  );
};

