import React, { useRef, useEffect } from 'react';

interface BallTrackingProps {
  ballTrajectory: any[];
  currentTime: number;
  fps: number;
  videoSize: { width: number; height: number };
  enabled?: boolean;
}

export const BallTracking: React.FC<BallTrackingProps> = ({
  ballTrajectory,
  currentTime,
  fps,
  videoSize,
  enabled = false
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!enabled || !canvasRef.current || !videoSize?.width || !videoSize?.height) return;
    
    const ctx = canvasRef.current.getContext('2d');
    if (!ctx) return;
    
    const canvas = canvasRef.current;
    if (canvas.width !== videoSize.width || canvas.height !== videoSize.height) {
      canvas.width = videoSize.width;
      canvas.height = videoSize.height;
    }
    
    ctx.clearRect(0, 0, videoSize.width, videoSize.height);
    
    if (!ballTrajectory || ballTrajectory.length === 0) {
      // Debug: log when no trajectory data
      if (enabled) {
        console.log('BallTracking: No trajectory data available', { 
          trajectoryLength: ballTrajectory?.length || 0,
          enabled,
          currentTime 
        });
      }
      return;
    }
    
    const currentFrame = Math.round(currentTime * fps);
    
    // Filter ball positions within a window around current time (show recent trajectory)
    const timeWindow = 2.0; // Show last 2 seconds of trajectory
    const minTime = currentTime - timeWindow;
    const maxTime = currentTime;
    
    // Try multiple ways to match ball positions
    const relevantPositions = ballTrajectory.filter((pos: any) => {
      // Try timestamp first
      if (pos.timestamp !== undefined) {
        return pos.timestamp >= minTime && pos.timestamp <= maxTime;
      }
      // Fallback to frame-based matching
      if (pos.frame !== undefined) {
        const posFrame = pos.frame;
        const minFrame = currentFrame - (timeWindow * fps);
        const maxFrame = currentFrame;
        return posFrame >= minFrame && posFrame <= maxFrame;
      }
      return false;
    });
    
    if (relevantPositions.length === 0) {
      // Debug: log when no relevant positions found
      if (enabled) {
        console.log('BallTracking: No relevant positions found', {
          currentFrame,
          currentTime,
          totalTrajectory: ballTrajectory.length,
          firstPos: ballTrajectory[0],
          lastPos: ballTrajectory[ballTrajectory.length - 1]
        });
      }
      return;
    }
    
    // Draw trajectory path
    ctx.strokeStyle = '#FFD700'; // Gold color for ball trail
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Draw the path
    ctx.beginPath();
    relevantPositions.forEach((pos: any, index: number) => {
      const center = pos.center || [0, 0];
      const x = center[0] || 0;
      const y = center[1] || 0;
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    
    // Draw ball positions as circles with fading opacity
    relevantPositions.forEach((pos: any, index: number) => {
      const center = pos.center || [0, 0];
      const x = center[0] || 0;
      const y = center[1] || 0;
      const confidence = pos.confidence || 0;
      
      // Calculate opacity based on recency (more recent = more opaque)
      const posTime = pos.timestamp || (pos.frame || 0) / fps;
      const timeDiff = currentTime - posTime;
      const opacity = Math.max(0.3, 1 - (timeDiff / timeWindow));
      
      // Draw ball circle
      ctx.fillStyle = `rgba(255, 215, 0, ${opacity})`; // Gold with opacity
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw outline
      ctx.strokeStyle = `rgba(255, 255, 255, ${opacity})`;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw current ball position with highlight
      if (index === relevantPositions.length - 1) {
        // Current position - draw larger circle
        ctx.fillStyle = '#FFD700';
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // Draw confidence label if available
        if (confidence > 0) {
          ctx.fillStyle = '#FFFFFF';
          ctx.font = 'bold 10px system-ui, -apple-system, sans-serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'bottom';
          ctx.fillText(`${Math.round(confidence * 100)}%`, x, y - 16);
        }
      }
    });
  }, [ballTrajectory, currentTime, fps, videoSize, enabled]);
  
  if (!enabled) return null;
  
  return (
    <canvas
      ref={canvasRef}
      width={videoSize.width || 640}
      height={videoSize.height || 360}
      className="absolute left-0 top-0 pointer-events-none"
      style={{
        width: '100%',
        height: 'auto'
      }}
    />
  );
};

