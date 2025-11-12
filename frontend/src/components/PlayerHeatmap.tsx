import React, { useRef, useEffect } from 'react';

interface PlayerHeatmapProps {
  playerTracks: any[];
  videoSize: { width: number; height: number } | any;
  playerFilter?: any;
  enabled?: boolean;
  currentTime?: number;
  fps?: number;
}

export const PlayerHeatmap: React.FC<PlayerHeatmapProps> = ({ 
  playerTracks, 
  videoSize, 
  playerFilter,
  enabled = false, // 默認關閉熱區圖
  currentTime = 0,
  fps = 30
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!enabled || !videoSize?.width || !videoSize?.height || !canvasRef.current) {
      if (enabled) {
        console.log('PlayerHeatmap: Disabled or missing requirements', {
          enabled,
          hasCanvas: !!canvasRef.current,
          videoSize,
          tracksCount: playerTracks?.length || 0
        });
      }
      return;
    }
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    // Ensure canvas size matches video size
    const canvas = canvasRef.current;
    if (canvas.width !== videoSize.width || canvas.height !== videoSize.height) {
      canvas.width = videoSize.width;
      canvas.height = videoSize.height;
    }
    
    ctx.clearRect(0, 0, videoSize.width, videoSize.height);
    
    const currentFrame = Math.round(currentTime * fps);
    
    // Only show heatmap for tracks near current time (within 10 frames for cumulative effect)
    const relevantTracks = playerTracks.filter((track: any) => 
      track.frame !== undefined && Math.abs(track.frame - currentFrame) <= 10
    );
    
    // 使用更淡的顏色和更小的覆蓋區域
    relevantTracks.forEach((track: any) => {
      if (!track.players || !Array.isArray(track.players)) return;
      
      track.players
        .filter((p: any) => !playerFilter || playerFilter === p.id)
        .forEach((p: any) => {
          if (!p.bbox || !Array.isArray(p.bbox) || p.bbox.length < 4) return;
          
          // 只繪製中心點附近的小區域，而不是整個bbox
          const [x1, y1, x2, y2] = p.bbox;
          const centerX = (x1 + x2) / 2;
          const centerY = (y1 + y2) / 2;
          const radius = Math.min((x2 - x1), (y2 - y1)) * 0.3; // 只覆蓋30%的區域
          
          // 根據距離當前時間的遠近調整透明度
          const frameDiff = Math.abs((track.frame || 0) - currentFrame);
          const alpha = Math.max(0.05, 0.15 * (1 - frameDiff / 10));
          
          // 使用更淡的顏色
          const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
          gradient.addColorStop(0, `rgba(255, 0, 0, ${alpha})`);
          gradient.addColorStop(0.5, `rgba(255, 0, 0, ${alpha * 0.5})`);
          gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
          
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
          ctx.fill();
        });
    });
  }, [playerTracks, videoSize, playerFilter, enabled, currentTime, fps]);
  
  if (!enabled) return null;
  
  return (
    <canvas
      ref={canvasRef}
      width={videoSize.width || 640}
      height={videoSize.height || 360}
      className="absolute left-0 top-0 pointer-events-none z-5"
      style={{
        width: '100%',
        height: '100%',
        objectFit: 'contain'
      }}
    />
  );
};
