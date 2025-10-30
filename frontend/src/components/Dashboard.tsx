import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getVideos } from '../services/api';
import { Loader2, PlayCircle, Library, Info } from 'lucide-react';
import { StatusBadge } from './ui/StatusBadge';
import { EmptyState } from './ui/EmptyState';

export const Dashboard: React.FC = () => {
  const [videos, setVideos] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getVideos()
      .then((res) => setVideos(res.videos || []))
      .finally(() => setLoading(false));
  }, []);

  const renderStatusBadge = (status: string) => <StatusBadge status={status} />;

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl md:text-3xl font-bold">Dashboard</h1>
        <p className="text-gray-600 mt-2">
          A quick overview of your uploaded videos and their analysis status.
        </p>
      </section>

      <section>
        {loading && (
          <div className="flex justify-center items-center h-40">
            <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
          </div>
        )}

        {!loading && videos.length === 0 && (
          <EmptyState
            title="No videos found."
            hint={<span>Go to the <Link to="/upload" className="text-blue-600 hover:underline font-medium">Upload</Link> page to start an analysis.</span>}
            icon={<Info className="w-8 h-8 text-gray-400" />}
          />
        )}

        {!loading && videos.length > 0 && (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {videos.map((v) => (
              <div key={v.id} className="bg-white rounded-xl shadow p-5 flex flex-col">
                <div className="flex items-center justify-between">
                  <div className="font-semibold truncate text-gray-800" title={v.filename}>
                    {v.filename}
                  </div>
                  {renderStatusBadge(v.status)}
                </div>
                <div className="text-sm text-gray-500 mt-2">
                  Uploaded: {new Date(v.upload_time).toLocaleString()}
                </div>
                <div className="flex gap-3 mt-4 pt-4 border-t border-gray-100">
                  <Link
                    to={`/player/${v.id}`}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 transition-colors"
                  >
                    <PlayCircle className="w-4 h-4" />
                    Play / View
                  </Link>
                  <Link
                    to="/library"
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
                  >
                    <Library className="w-4 h-4" />
                    Library
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
