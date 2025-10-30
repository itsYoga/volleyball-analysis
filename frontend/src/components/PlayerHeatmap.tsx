import React, { useRef, useEffect } from 'react';

interface PlayerHeatmapProps {
  playerTracks: any[];
  videoSize: { width: number; height: number } | any;
  playerFilter?: any;
}

export const PlayerHeatmap: React.FC<PlayerHeatmapProps> = ({ playerTracks, videoSize, playerFilter }) => {
  const canvasRef = useRef(null);
  useEffect(() => {
    if (!videoSize?.width || !videoSize?.height) return;
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0,0,videoSize.width,videoSize.height);
    playerTracks.forEach(({players}) => {
      players.filter(p => !playerFilter || playerFilter === p.id).forEach(p => {
        ctx.fillStyle = "rgba(255,0,0,0.07)";
        const [x1, y1, x2, y2] = p.bbox;
        ctx.fillRect(x1, y1, x2-x1, y2-y1);
      });
    });
  }, [playerTracks, videoSize, playerFilter]);
  return <canvas ref={canvasRef} width={videoSize.width||640} height={videoSize.height||360} className="pointer-events-none"/>;
};
