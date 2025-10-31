import React, { useRef, useEffect } from 'react';

interface BoundingBoxesProps {
  playerTracks: any[];
  actions: any[];
  currentTime: number;
  fps: number;
  videoSize: { width: number; height: number };
  showPlayers?: boolean;
  showActions?: boolean;
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
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !videoSize?.width || !videoSize?.height) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;

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
      let currentTrack = playerTracks.find(
        (track) => track.frame === currentFrame
      );
      
      // If exact match not found, find closest one within 3 frames
      if (!currentTrack) {
        currentTrack = playerTracks.reduce((closest, track) => {
          if (!closest) return track;
          const closestDiff = Math.abs(closest.frame - currentFrame);
          const trackDiff = Math.abs(track.frame - currentFrame);
          return trackDiff < closestDiff ? track : closest;
        }, null as any);
      }

      // Only draw if we found a track within reasonable range (increased to 10 frames for better matching)
      if (currentTrack && currentTrack.players && Math.abs(currentTrack.frame - currentFrame) <= 10) {
        // Filter players by confidence threshold (0.5 = 50%)
        const filteredPlayers = currentTrack.players.filter((player: any) => 
          player.confidence === undefined || player.confidence >= 0.5
        );
        
        filteredPlayers.forEach((player: any) => {
          if (!player.bbox || !Array.isArray(player.bbox) || player.bbox.length < 4) return;
          
          const [x1, y1, x2, y2] = player.bbox;
          const color = getPlayerColor(player.id || 0);
          
          // Draw bounding box
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.setLineDash([]);
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

          // Draw label background
          const label = `Player #${player.id}`;
          ctx.font = 'bold 12px system-ui, -apple-system, sans-serif';
          ctx.textBaseline = 'top';
          const textMetrics = ctx.measureText(label);
          const textWidth = textMetrics.width;
          const textHeight = 18;

          ctx.fillStyle = color;
          ctx.fillRect(x1, y1 - textHeight - 2, textWidth + 8, textHeight);

          // Draw label text
          ctx.fillStyle = '#FFFFFF';
          ctx.fillText(label, x1 + 4, y1 - textHeight + 2);
        });
      }
    }

    // Draw action bounding boxes
    if (showActions && actions.length > 0) {
      // Filter actions within 2 frames of current time and by confidence threshold (0.6 = 60%)
      const currentActions = actions.filter(
        (action) => 
          action.frame !== undefined && 
          Math.abs(action.frame - currentFrame) <= 2 &&
          (action.confidence === undefined || action.confidence >= 0.6)
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

        // Draw action label
        const label = `${action.action.toUpperCase()} ${action.player_id ? `#${action.player_id}` : ''}`;
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
  }, [playerTracks, actions, currentTime, fps, videoSize, showPlayers, showActions]);

  return (
    <canvas
      ref={canvasRef}
      width={videoSize.width || 640}
      height={videoSize.height || 360}
      className="absolute left-0 top-0 pointer-events-none"
      style={{ 
        imageRendering: 'crisp-edges',
        width: '100%',
        height: 'auto'
      }}
    />
  );
};

