import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getVideos } from '../services/api';
import { EmptyState } from './ui/EmptyState';
import { Info } from 'lucide-react';

export const VideoLibrary: React.FC = () => {
  const [videos, setVideos] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getVideos()
      .then((res) => setVideos(res.videos || []))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <h1 className="text-2xl md:text-3xl font-bold">影片庫</h1>
      <p className="text-gray-600 mt-2">瀏覽並進入每支影片的詳細分析</p>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mt-8">
        {loading && <div className="text-gray-500">Loading...</div>}
        {!loading && videos.length === 0 && (
          <EmptyState title="No videos yet." icon={<Info className="w-8 h-8 text-gray-400" />} />
        )}
        {videos.map((v) => (
          <div key={v.id} className="bg-white rounded-xl shadow p-5 flex flex-col gap-3">
            <div className="font-semibold truncate" title={v.filename}>{v.filename}</div>
            <div className="text-sm text-gray-500">上傳時間：{new Date(v.upload_time).toLocaleString()}</div>
            <div className="flex items-center justify-between mt-2">
              <span className={`text-xs px-2 py-1 rounded-full border ${
                v.status === 'completed' ? 'bg-green-50 text-green-700 border-green-200' :
                v.status === 'processing' ? 'bg-blue-50 text-blue-700 border-blue-200' :
                v.status === 'failed' ? 'bg-red-50 text-red-700 border-red-200' : 'bg-gray-50 text-gray-700 border-gray-200'
              }`}>
                {v.status}
              </span>
              <Link to={`/player/${v.id}`} className="px-3 py-2 rounded bg-blue-600 text-white text-sm hover:bg-blue-700 transition">前往播放</Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
