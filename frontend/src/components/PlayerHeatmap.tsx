import React, { useRef, useEffect } from 'react';

interface PlayerHeatmapProps {
  playerTracks: any[];
  videoSize: { width: number; height: number } | any;
  playerFilter?: any;
  enabled?: boolean;
}

export const PlayerHeatmap: React.FC<PlayerHeatmapProps> = ({ 
  playerTracks, 
  videoSize, 
  playerFilter,
  enabled = false // 默認關閉熱區圖
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    if (!enabled || !videoSize?.width || !videoSize?.height || !canvasRef.current) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    ctx.clearRect(0, 0, videoSize.width, videoSize.height);
    
    // 使用更淡的顏色和更小的覆蓋區域
    playerTracks.forEach(({ players }) => {
      players
        .filter((p: any) => !playerFilter || playerFilter === p.id)
        .forEach((p: any) => {
          // 只繪製中心點附近的小區域，而不是整個bbox
          const [x1, y1, x2, y2] = p.bbox;
          const centerX = (x1 + x2) / 2;
          const centerY = (y1 + y2) / 2;
          const radius = Math.min((x2 - x1), (y2 - y1)) * 0.3; // 只覆蓋30%的區域
          
          // 使用更淡的顏色
          const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
          gradient.addColorStop(0, 'rgba(255, 0, 0, 0.1)');
          gradient.addColorStop(0.5, 'rgba(255, 0, 0, 0.05)');
          gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
          
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
          ctx.fill();
        });
    });
  }, [playerTracks, videoSize, playerFilter, enabled]);
  
  if (!enabled) return null;
  
  return (
    <canvas
      ref={canvasRef}
      width={videoSize.width || 640}
      height={videoSize.height || 360}
      className="pointer-events-none"
    />
  );
};
