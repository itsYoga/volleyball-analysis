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
    
    // Filter ball positions and remove outliers based on trajectory continuity
    // Sort by frame/timestamp to ensure correct order
    const sortedPositions = relevantPositions.sort((a: any, b: any) => {
      const frameA = a.frame || (a.timestamp || 0) * fps;
      const frameB = b.frame || (b.timestamp || 0) * fps;
      return frameA - frameB;
    });
    
    // Filter outliers: remove points that are too far from the trajectory
    const filteredPositions: any[] = [];
    if (sortedPositions.length > 0) {
      filteredPositions.push(sortedPositions[0]); // Always keep first point
      
      for (let i = 1; i < sortedPositions.length; i++) {
        const prev = sortedPositions[i - 1];
        const curr = sortedPositions[i];
        
        const prevCenter = prev.center || [0, 0];
        const currCenter = curr.center || [0, 0];
        
        // Calculate distance between consecutive points
        const distance = Math.sqrt(
          Math.pow(currCenter[0] - prevCenter[0], 2) + 
          Math.pow(currCenter[1] - prevCenter[1], 2)
        );
        
        // Calculate time difference
        const prevTime = prev.timestamp || (prev.frame || 0) / fps;
        const currTime = curr.timestamp || (curr.frame || 0) / fps;
        const timeDiff = Math.abs(currTime - prevTime);
        
        // Calculate velocity (pixels per second)
        const velocity = timeDiff > 0 ? distance / timeDiff : Infinity;
        
        // Filter conditions:
        // 1. Distance should be reasonable (max 200 pixels per frame)
        // 2. Velocity should be reasonable (max 1000 pixels/second)
        // 3. Confidence should be reasonable (>= 0.2)
        const maxDistance = 200.0;
        const maxVelocity = 1000.0;
        const minConfidence = 0.2;
        
        if (distance <= maxDistance && 
            velocity <= maxVelocity && 
            (curr.confidence || 0) >= minConfidence) {
          filteredPositions.push(curr);
        }
        // Otherwise skip this point (outlier)
      }
    }
    
    if (filteredPositions.length === 0) {
      return;
    }
    
    // Draw ball positions as circles with fading opacity (NO CONNECTING LINES)
    filteredPositions.forEach((pos: any, index: number) => {
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
      if (index === filteredPositions.length - 1) {
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

