import React, { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getVideoUrl, getAnalysisResults, getVideo } from '../services/api';
import { EventTimeline } from './EventTimeline';
import { PlayerHeatmap } from './PlayerHeatmap';

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

  if (status === 'loading') return <div className="max-w-3xl mx-auto bg-white rounded shadow p-6">Loading...</div>;
  if (status === 'processing') return <div className="max-w-3xl mx-auto bg-yellow-50 border border-yellow-200 text-yellow-800 rounded p-6">Analysis in progress. Please check back later from Library.</div>;
  if (status === 'failed') return <div className="max-w-3xl mx-auto bg-red-50 border border-red-200 text-red-800 rounded p-6">Analysis failed. Please re-run analysis from Library.</div>;
  if (status === 'error') return (
    <div className="max-w-3xl mx-auto bg-red-50 border border-red-200 text-red-800 rounded p-6">
      <div className="font-semibold mb-2">Failed to load results</div>
      <div className="text-sm">{error}</div>
      <button onClick={() => { setStatus('idle'); setError(''); setResult(null); }} className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">Retry</button>
    </div>
  );
  if (!result) return null;

  const { action_recognition, scores, players_tracking, game_states, video_info } = result;
  const fps = video_info?.fps || 30;
  const totalFrames = video_info?.total_frames || Math.round((video_info?.duration || 0) * fps) || 1000;
  const currentFrame = Math.max(0, Math.min(totalFrames, Math.round(currentTime * fps)));

  return (
    <div className="relative w-full max-w-3xl mx-auto">
      <video ref={videoRef} src={getVideoUrl(effectiveId)} controls width={video_info?.width || 640} height={video_info?.height || 360} className="rounded shadow w-full" />
      <EventTimeline
        actions={action_recognition?.actions || []}
        scores={scores || []}
        gameStates={game_states || []}
        duration={totalFrames}
        currentFrame={currentFrame}
        onSeek={handleSeek}
      />
      <div className="absolute left-0 top-0 pointer-events-none">
        <PlayerHeatmap playerTracks={players_tracking || []} videoSize={{ width: video_info?.width, height: video_info?.height }} />
      </div>
    </div>
  );
};
