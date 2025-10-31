import React, { useEffect, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getVideoUrl, getAnalysisResults, getVideo, getAnalysisStatus } from '../services/api';
import { EventTimeline } from './EventTimeline';
import { PlayerHeatmap } from './PlayerHeatmap';
import { BoundingBoxes } from './BoundingBoxes';
import { BallTracking } from './BallTracking';
import { PlayerStats } from './PlayerStats';
import { Loader2, AlertCircle, Clock, RefreshCw, ArrowLeft, PlayCircle, Users, Maximize2, Minimize2 } from 'lucide-react';

export const VideoPlayer: React.FC<{ videoId?: string }> = ({ videoId }) => {
  const params = useParams();
  const effectiveId = videoId && videoId.length > 0 ? videoId : (params.videoId as string);
  const [result, setResult] = useState<any>(null);
  const [status, setStatus] = useState<'idle'|'loading'|'processing'|'completed'|'failed'|'error'>('idle');
  const [error, setError] = useState<string>('');
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(false);
  const [showPlayerBoxes, setShowPlayerBoxes] = useState<boolean>(true);
  const [showActionBoxes, setShowActionBoxes] = useState<boolean>(true);
  const [showBallTracking, setShowBallTracking] = useState<boolean>(false);
  const [showPlayerStats, setShowPlayerStats] = useState<boolean>(false);
  const [playerNames, setPlayerNames] = useState<Record<number, string>>({});
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [progress, setProgress] = useState<number>(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const videoContainerRef = useRef<HTMLDivElement>(null);

  // Load player names from localStorage
  useEffect(() => {
    if (!effectiveId) return;
    const storedNames = localStorage.getItem(`playerNames_${effectiveId}`);
    if (storedNames) {
      try {
        setPlayerNames(JSON.parse(storedNames));
      } catch (e) {
        console.error('Failed to load player names:', e);
      }
    }
  }, [effectiveId]);

  // Save player names to localStorage
  const handlePlayerNameChange = (playerId: number, name: string) => {
    const newNames = { ...playerNames, [playerId]: name };
    setPlayerNames(newNames);
    if (effectiveId) {
      localStorage.setItem(`playerNames_${effectiveId}`, JSON.stringify(newNames));
    }
  };

  useEffect(() => {
    if (!effectiveId) return;
    let isMounted = true;
    let pollInterval: NodeJS.Timeout | null = null;
    
    const load = async () => {
      setStatus('loading');
      setError('');
      try {
        const meta = await getVideo(effectiveId);
        if (!isMounted) return;
        if (meta.status !== 'completed') {
          setStatus(meta.status as any);
          setResult(null);
          
          // 如果狀態是 processing，開始輪詢進度
          if (meta.status === 'processing' && meta.task_id) {
            setProgress(5); // 初始進度
            pollInterval = setInterval(async () => {
              try {
                const taskStatus = await getAnalysisStatus(meta.task_id);
                if (!isMounted) return;
                
                setProgress(taskStatus.progress || 0);
                
                // 如果任務完成，重新加載視頻信息
                if (taskStatus.status === 'completed') {
                  if (pollInterval) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                  }
                  // 重新加載結果
                  const res = await getAnalysisResults(effectiveId);
                  if (isMounted) {
                    setResult(res);
                    setStatus('completed');
                    setProgress(100);
                  }
                } else if (taskStatus.status === 'failed') {
                  if (pollInterval) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                  }
                  if (isMounted) {
                    setStatus('failed');
                    setError(taskStatus.error || 'Analysis failed');
                  }
                }
              } catch (e) {
                console.error('Failed to poll progress:', e);
              }
            }, 1000); // 每秒輪詢一次
          }
          return;
        }
        const res = await getAnalysisResults(effectiveId);
        if (!isMounted) return;
        setResult(res);
        setStatus('completed');
        setProgress(100);
      } catch (e: any) {
        setError(e?.message || 'Failed to load analysis results');
        setStatus('error');
      }
    };
    load();
    return () => { 
      isMounted = false;
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [effectiveId]);

  const handleSeek = (sec: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = sec;
      // Force update currentTime immediately
      setCurrentTime(sec);
    }
  };

  // Update currentTime more frequently for smoother rendering
  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    
    let animationFrameId: number | null = null;
    let lastTime = 0;
    
    const onTimeUpdate = () => {
      const newTime = el.currentTime || 0;
      // Update more frequently for smoother rendering
      if (Math.abs(newTime - lastTime) > 0.033) { // ~30fps minimum update rate
        setCurrentTime(newTime);
        lastTime = newTime;
      }
    };
    
    const onSeeked = () => {
      const newTime = el.currentTime || 0;
      setCurrentTime(newTime);
      lastTime = newTime;
    };
    
    // Use requestAnimationFrame for smoother updates during playback
    const updateLoop = () => {
      if (el && !el.paused) {
        const newTime = el.currentTime || 0;
        if (Math.abs(newTime - lastTime) > 0.016) { // ~60fps update rate
          setCurrentTime(newTime);
          lastTime = newTime;
        }
      }
      animationFrameId = requestAnimationFrame(updateLoop);
    };
    
    el.addEventListener('timeupdate', onTimeUpdate);
    el.addEventListener('seeked', onSeeked);
    
    const handlePlay = () => {
      if (animationFrameId === null) {
        animationFrameId = requestAnimationFrame(updateLoop);
      }
    };
    
    const handlePause = () => {
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
      onTimeUpdate(); // Update one last time on pause
    };
    
    el.addEventListener('play', handlePlay);
    el.addEventListener('pause', handlePause);
    
    // Start update loop if video is already playing
    if (!el.paused) {
      animationFrameId = requestAnimationFrame(updateLoop);
    }
    
    return () => {
      el.removeEventListener('timeupdate', onTimeUpdate);
      el.removeEventListener('seeked', onSeeked);
      el.removeEventListener('play', handlePlay);
      el.removeEventListener('pause', handlePause);
      if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [result]); // Re-setup when result changes (video loads)

  // Debug: Log ball tracking data when enabled (must be before any early returns)
  useEffect(() => {
    if (result && showBallTracking && result.ball_tracking) {
      console.log('Ball tracking data:', {
        hasTrajectory: !!result.ball_tracking.trajectory,
        trajectoryLength: result.ball_tracking.trajectory?.length || 0,
        ballTracking: result.ball_tracking
      });
    }
  }, [showBallTracking, result]);

  // Handle fullscreen mode - ensure UI stays visible
  useEffect(() => {
    const handleFullscreenChange = () => {
      const isFullscreenNow = !!(
        document.fullscreenElement ||
        (document as any).webkitFullscreenElement ||
        (document as any).mozFullScreenElement ||
        (document as any).msFullscreenElement
      );
      setIsFullscreen(isFullscreenNow);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
    document.addEventListener('mozfullscreenchange', handleFullscreenChange);
    document.addEventListener('MSFullscreenChange', handleFullscreenChange);

    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
      document.removeEventListener('webkitfullscreenchange', handleFullscreenChange);
      document.removeEventListener('mozfullscreenchange', handleFullscreenChange);
      document.removeEventListener('MSFullscreenChange', handleFullscreenChange);
    };
  }, []);

  const toggleFullscreen = async () => {
    const container = videoContainerRef.current;
    if (!container) return;

    try {
      if (!document.fullscreenElement && !(document as any).webkitFullscreenElement && 
          !(document as any).mozFullScreenElement && !(document as any).msFullscreenElement) {
        // Enter fullscreen
        if (container.requestFullscreen) {
          await container.requestFullscreen();
        } else if ((container as any).webkitRequestFullscreen) {
          await (container as any).webkitRequestFullscreen();
        } else if ((container as any).mozRequestFullScreen) {
          await (container as any).mozRequestFullScreen();
        } else if ((container as any).msRequestFullscreen) {
          await (container as any).msRequestFullscreen();
        }
      } else {
        // Exit fullscreen
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        } else if ((document as any).webkitExitFullscreen) {
          await (document as any).webkitExitFullscreen();
        } else if ((document as any).mozCancelFullScreen) {
          await (document as any).mozCancelFullScreen();
        } else if ((document as any).msExitFullscreen) {
          await (document as any).msExitFullscreen();
        }
      }
    } catch (error) {
      console.error('Fullscreen error:', error);
    }
  };

  // All early returns must come after all Hooks
  if (status === 'loading') {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-xl p-12 border border-gray-100">
          <div className="flex flex-col items-center justify-center gap-4">
            <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
            <p className="text-gray-600 font-medium">Loading video analysis...</p>
          </div>
        </div>
      </div>
    );
  }

  if (status === 'processing') {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-gradient-to-r from-yellow-50 to-amber-50 border-2 border-yellow-200 rounded-2xl shadow-lg p-8">
          <div className="flex items-start gap-4">
            <Clock className="w-8 h-8 text-yellow-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-yellow-900 font-semibold text-lg mb-2">Analysis in Progress</h3>
              <p className="text-yellow-800 mb-4">Your video is being analyzed. This may take a few minutes. Please check back later.</p>
              
              {/* Progress Bar */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-yellow-900">Progress</span>
                  <span className="text-sm font-semibold text-yellow-900">{Math.round(progress)}%</span>
                </div>
                <div className="w-full bg-yellow-200 rounded-full h-3 overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-yellow-500 to-amber-600 h-full rounded-full transition-all duration-300 ease-out"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
              
              <Link to="/library" className="inline-flex items-center gap-2 px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-colors font-medium">
                <ArrowLeft className="w-4 h-4" /> Back to Library
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (status === 'failed') {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-gradient-to-r from-red-50 to-rose-50 border-2 border-red-200 rounded-2xl shadow-lg p-8">
          <div className="flex items-start gap-4">
            <AlertCircle className="w-8 h-8 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-red-900 font-semibold text-lg mb-2">Analysis Failed</h3>
              <p className="text-red-800 mb-4">The analysis could not be completed. Please try re-running the analysis from your library.</p>
              <Link to="/library" className="inline-flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium">
                <ArrowLeft className="w-4 h-4" /> Back to Library
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (status === 'error') {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-gradient-to-r from-red-50 to-rose-50 border-2 border-red-200 rounded-2xl shadow-lg p-8">
          <div className="flex items-start gap-4">
            <AlertCircle className="w-8 h-8 text-red-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-red-900 font-semibold text-lg mb-2">Failed to Load Results</h3>
              <p className="text-red-800 mb-4 text-sm">{error}</p>
              <div className="flex gap-3">
                <button
                  onClick={() => { setStatus('idle'); setError(''); setResult(null); }}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors font-medium"
                >
                  <RefreshCw className="w-4 h-4" /> Retry
                </button>
                <Link to="/library" className="inline-flex items-center gap-2 px-4 py-2 border border-red-300 text-red-700 rounded-lg hover:bg-red-50 transition-colors font-medium">
                  <ArrowLeft className="w-4 h-4" /> Back to Library
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!result) return null;

  const { action_recognition, scores, players_tracking, game_states, video_info, ball_tracking } = result;
  const fps = video_info?.fps || 30;
  const totalFrames = video_info?.total_frames || Math.round((video_info?.duration || 0) * fps) || 1000;
  const currentFrame = Math.max(0, Math.min(totalFrames, Math.round(currentTime * fps)));

  return (
    <div className={`space-y-6 ${isFullscreen ? 'fixed inset-0 z-[9999] bg-black overflow-auto' : ''}`}>
      {/* Back Button - Hide in fullscreen */}
      {!isFullscreen && (
        <Link
          to="/library"
          className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 font-medium transition-colors"
        >
          <ArrowLeft className="w-4 h-4" /> Back to Library
        </Link>
      )}

      {/* Video Player Section */}
      <div className={`bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden ${isFullscreen ? 'rounded-none shadow-none border-0 h-full flex flex-col' : ''}`}>
        {/* Header - Always visible, even in fullscreen */}
        <div className={`p-6 bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200 ${isFullscreen ? 'bg-black/90 border-gray-700' : ''}`}>
          <h2 className={`text-xl font-semibold ${isFullscreen ? 'text-white' : 'text-gray-900'}`}>Video Analysis</h2>
          <p className={`text-sm mt-1 ${isFullscreen ? 'text-gray-300' : 'text-gray-600'}`}>Interactive timeline with event markers and player tracking</p>
        </div>
        
        <div className={`p-6 ${isFullscreen ? 'flex-1 overflow-auto' : ''}`}>
          {/* Controls - Always visible */}
          <div className={`flex items-center gap-4 mb-4 pb-4 border-b border-gray-200 flex-wrap ${isFullscreen ? 'border-gray-700' : ''}`}>
            <label className={`flex items-center gap-2 cursor-pointer ${isFullscreen ? 'text-white' : ''}`}>
              <input
                type="checkbox"
                checked={showPlayerBoxes}
                onChange={(e) => setShowPlayerBoxes(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <span className={`text-sm font-medium ${isFullscreen ? 'text-white' : 'text-gray-700'}`}>Show Player Boxes</span>
            </label>
            <label className={`flex items-center gap-2 cursor-pointer ${isFullscreen ? 'text-white' : ''}`}>
              <input
                type="checkbox"
                checked={showActionBoxes}
                onChange={(e) => setShowActionBoxes(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <span className={`text-sm font-medium ${isFullscreen ? 'text-white' : 'text-gray-700'}`}>Show Action Boxes</span>
            </label>
            <label className={`flex items-center gap-2 cursor-pointer ${isFullscreen ? 'text-white' : ''}`}>
              <input
                type="checkbox"
                checked={showBallTracking}
                onChange={(e) => setShowBallTracking(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <span className={`text-sm font-medium ${isFullscreen ? 'text-white' : 'text-gray-700'}`}>Show Ball Tracking</span>
            </label>
            <label className={`flex items-center gap-2 cursor-pointer ${isFullscreen ? 'text-white' : ''}`}>
              <input
                type="checkbox"
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <span className={`text-sm font-medium ${isFullscreen ? 'text-white' : 'text-gray-700'}`}>Show Heatmap</span>
            </label>
            <label className={`flex items-center gap-2 cursor-pointer ${isFullscreen ? 'text-white' : ''}`}>
              <input
                type="checkbox"
                checked={showPlayerStats}
                onChange={(e) => setShowPlayerStats(e.target.checked)}
                className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
              />
              <span className={`text-sm font-medium ${isFullscreen ? 'text-white' : 'text-gray-700'}`}>Show Player Stats</span>
            </label>
          </div>

          <div className="relative w-full bg-black rounded-xl overflow-hidden shadow-2xl" ref={videoContainerRef}>
            <div className="relative w-full" style={{ position: 'relative' }}>
              <video
                ref={videoRef}
                src={getVideoUrl(effectiveId)}
                controls
                className="w-full h-auto"
                style={{ display: 'block' }}
                onLoadedMetadata={(e) => {
                  // Update currentTime when video metadata loads
                  const video = e.target as HTMLVideoElement;
                  setCurrentTime(video.currentTime || 0);
                }}
              />
              {/* Fullscreen Toggle Button */}
              <button
                onClick={toggleFullscreen}
                className="absolute top-4 right-4 z-50 bg-black/70 hover:bg-black/90 text-white p-2 rounded-lg transition-all shadow-lg"
                title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
              >
                {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
              </button>
              {/* Bounding Boxes Overlay */}
              {(showPlayerBoxes || showActionBoxes) && (
                <BoundingBoxes
                  playerTracks={players_tracking || []}
                  actions={action_recognition?.action_detections || []}  // 使用 action_detections 來顯示每一幀的動態框
                  currentTime={currentTime}
                  fps={fps}
                  videoSize={{ width: video_info?.width || 640, height: video_info?.height || 360 }}
                  showPlayers={showPlayerBoxes}
                  showActions={showActionBoxes}
                  playerNames={playerNames}
                />
              )}
              {/* Heatmap Overlay (optional, less intrusive) */}
              {showHeatmap && (
                <PlayerHeatmap 
                  playerTracks={players_tracking || []} 
                  videoSize={{ width: video_info?.width || 640, height: video_info?.height || 360 }} 
                  enabled={showHeatmap}
                  currentTime={currentTime}
                  fps={fps}
                />
              )}
              {/* Ball Tracking Overlay */}
              {showBallTracking && (
                <BallTracking
                  ballTrajectory={ball_tracking?.trajectory || []}
                  currentTime={currentTime}
                  fps={fps}
                  videoSize={{ width: video_info?.width || 640, height: video_info?.height || 360 }}
                  enabled={showBallTracking}
                />
              )}
            </div>
          </div>
        </div>

        {/* Player Statistics Section */}
        {showPlayerStats && (
          <div className={`px-6 pb-6 border-t border-gray-200 pt-6 ${isFullscreen ? 'border-gray-700' : ''}`}>
            <div className={`rounded-xl p-4 border ${isFullscreen ? 'bg-black/80 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
              <h3 className={`text-sm font-semibold mb-4 flex items-center gap-2 ${isFullscreen ? 'text-white' : 'text-gray-700'}`}>
                <Users className="w-4 h-4" /> Player Statistics
              </h3>
              <PlayerStats
                actions={action_recognition?.actions || []}
                playerTracks={players_tracking || []}
                videoId={effectiveId}
                fps={fps}
                onSeek={handleSeek}
                onPlayerNameChange={handlePlayerNameChange}
                playerNames={playerNames}
              />
            </div>
          </div>
        )}

        {/* Event Timeline */}
        <div className={`px-6 pb-6 ${isFullscreen ? '' : ''}`}>
          <div className={`rounded-xl p-4 border ${isFullscreen ? 'bg-black/80 border-gray-700' : 'bg-gray-50 border-gray-200'}`}>
            <h3 className={`text-sm font-semibold mb-3 flex items-center gap-2 ${isFullscreen ? 'text-white' : 'text-gray-700'}`}>
              <PlayCircle className="w-4 h-4" /> Event Timeline
            </h3>
            <EventTimeline
              actions={action_recognition?.actions || []}
              scores={scores || []}
              gameStates={game_states || []}
              duration={totalFrames}
              currentFrame={currentFrame}
              onSeek={handleSeek}
              fps={fps}
              playerNames={playerNames}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
