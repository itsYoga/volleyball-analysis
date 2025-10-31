import React, { useEffect, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { getVideoUrl, getAnalysisResults, getVideo } from '../services/api';
import { EventTimeline } from './EventTimeline';
import { PlayerHeatmap } from './PlayerHeatmap';
import { Loader2, AlertCircle, Clock, RefreshCw, ArrowLeft, PlayCircle } from 'lucide-react';

export const VideoPlayer: React.FC<{ videoId?: string }> = ({ videoId }) => {
  const params = useParams();
  const effectiveId = videoId && videoId.length > 0 ? videoId : (params.videoId as string);
  const [result, setResult] = useState<any>(null);
  const [status, setStatus] = useState<'idle'|'loading'|'processing'|'completed'|'failed'|'error'>('idle');
  const [error, setError] = useState<string>('');
  const [currentTime, setCurrentTime] = useState<number>(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!effectiveId) return;
    let isMounted = true;
    const load = async () => {
      setStatus('loading');
      setError('');
      try {
        const meta = await getVideo(effectiveId);
        if (!isMounted) return;
        if (meta.status !== 'completed') {
          setStatus(meta.status as any);
          setResult(null);
          return;
        }
        const res = await getAnalysisResults(effectiveId);
        if (!isMounted) return;
        setResult(res);
        setStatus('completed');
      } catch (e: any) {
        setError(e?.message || 'Failed to load analysis results');
        setStatus('error');
      }
    };
    load();
    return () => { isMounted = false; };
  }, [effectiveId]);

  const handleSeek = (sec: number) => {
    if (videoRef.current) videoRef.current.currentTime = sec;
  };

  useEffect(() => {
    const el = videoRef.current;
    if (!el) return;
    const onTime = () => setCurrentTime(el.currentTime || 0);
    el.addEventListener('timeupdate', onTime);
    return () => el.removeEventListener('timeupdate', onTime);
  }, []);

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

  const { action_recognition, scores, players_tracking, game_states, video_info } = result;
  const fps = video_info?.fps || 30;
  const totalFrames = video_info?.total_frames || Math.round((video_info?.duration || 0) * fps) || 1000;
  const currentFrame = Math.max(0, Math.min(totalFrames, Math.round(currentTime * fps)));

  return (
    <div className="space-y-6">
      {/* Back Button */}
      <Link
        to="/library"
        className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 font-medium transition-colors"
      >
        <ArrowLeft className="w-4 h-4" /> Back to Library
      </Link>

      {/* Video Player Section */}
      <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
        <div className="p-6 bg-gradient-to-r from-gray-50 to-gray-100 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Video Analysis</h2>
          <p className="text-sm text-gray-600 mt-1">Interactive timeline with event markers and player tracking</p>
        </div>
        
        <div className="p-6">
          <div className="relative w-full bg-black rounded-xl overflow-hidden shadow-2xl">
            <video
              ref={videoRef}
              src={getVideoUrl(effectiveId)}
              controls
              width={video_info?.width || 640}
              height={video_info?.height || 360}
              className="w-full h-auto"
            />
            <div className="absolute left-0 top-0 pointer-events-none w-full h-full">
              <PlayerHeatmap playerTracks={players_tracking || []} videoSize={{ width: video_info?.width, height: video_info?.height }} />
            </div>
          </div>
        </div>

        {/* Event Timeline */}
        <div className="px-6 pb-6">
          <div className="bg-gray-50 rounded-xl p-4 border border-gray-200">
            <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
              <PlayCircle className="w-4 h-4" /> Event Timeline
            </h3>
            <EventTimeline
              actions={action_recognition?.actions || []}
              scores={scores || []}
              gameStates={game_states || []}
              duration={totalFrames}
              currentFrame={currentFrame}
              onSeek={handleSeek}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
