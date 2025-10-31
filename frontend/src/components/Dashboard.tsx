import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getVideos } from '../services/api';
import { Loader2, PlayCircle, Library, Info, Upload, Video, Clock, CheckCircle2, AlertCircle } from 'lucide-react';
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

  const stats = {
    total: videos.length,
    completed: videos.filter(v => v.status === 'completed').length,
    processing: videos.filter(v => v.status === 'processing').length,
    failed: videos.filter(v => v.status === 'failed').length,
  };

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <section>
        <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 bg-clip-text text-transparent">
          Dashboard
        </h1>
        <p className="text-gray-600 mt-2 text-lg">
          Overview of your video analysis projects and insights
        </p>
      </section>

      {/* Stats Cards */}
      {!loading && videos.length > 0 && (
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl shadow-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-100 text-sm font-medium">Total Videos</p>
                <p className="text-3xl font-bold mt-1">{stats.total}</p>
              </div>
              <Video className="w-10 h-10 text-blue-200" />
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-green-500 to-green-600 rounded-xl shadow-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-100 text-sm font-medium">Completed</p>
                <p className="text-3xl font-bold mt-1">{stats.completed}</p>
              </div>
              <CheckCircle2 className="w-10 h-10 text-green-200" />
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-yellow-500 to-yellow-600 rounded-xl shadow-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-yellow-100 text-sm font-medium">Processing</p>
                <p className="text-3xl font-bold mt-1">{stats.processing}</p>
              </div>
              <Clock className="w-10 h-10 text-yellow-200" />
            </div>
          </div>
          
          <div className="bg-gradient-to-br from-red-500 to-red-600 rounded-xl shadow-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-red-100 text-sm font-medium">Failed</p>
                <p className="text-3xl font-bold mt-1">{stats.failed}</p>
              </div>
              <AlertCircle className="w-10 h-10 text-red-200" />
            </div>
          </div>
        </section>
      )}

      {/* Videos Section */}
      <section>
        {loading && (
          <div className="flex justify-center items-center h-64">
            <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
          </div>
        )}

        {!loading && videos.length === 0 && (
          <div className="bg-white rounded-2xl shadow-xl p-12 border border-gray-100">
            <EmptyState
              title="No videos found"
              hint={
                <span>
                  Get started by uploading your first video. Go to the{' '}
                  <Link to="/upload" className="text-blue-600 hover:text-blue-700 font-semibold underline decoration-2 underline-offset-2">
                    Upload
                  </Link>{' '}
                  page to begin analysis.
                </span>
              }
              icon={<Upload className="w-16 h-16 text-gray-300" />}
            />
          </div>
        )}

        {!loading && videos.length > 0 && (
          <>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Recent Videos</h2>
              <Link
                to="/library"
                className="text-sm text-blue-600 hover:text-blue-700 font-medium flex items-center gap-1"
              >
                View all <Library className="w-4 h-4" />
              </Link>
            </div>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {videos.slice(0, 6).map((v) => (
                <div
                  key={v.id}
                  className="group bg-white rounded-xl shadow-md hover:shadow-xl transition-all duration-300 border border-gray-100 overflow-hidden"
                >
                  <div className="p-6 flex flex-col h-full">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-gray-900 truncate group-hover:text-blue-600 transition-colors" title={v.filename}>
                          {v.filename}
                        </h3>
                        <p className="text-sm text-gray-500 mt-1 flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {new Date(v.upload_time).toLocaleDateString()}
                        </p>
                      </div>
                      <StatusBadge status={v.status} />
                    </div>
                    
                    <div className="mt-auto pt-4 border-t border-gray-100">
                      <div className="flex gap-2">
                        <Link
                          to={`/player/${v.id}`}
                          className="flex-1 inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-gradient-to-r from-blue-600 to-blue-700 text-white text-sm font-medium hover:from-blue-700 hover:to-blue-800 transition-all shadow-md hover:shadow-lg"
                        >
                          <PlayCircle className="w-4 h-4" />
                          Play
                        </Link>
                        <Link
                          to="/library"
                          className="px-4 py-2.5 rounded-lg border border-gray-300 text-gray-700 text-sm font-medium hover:bg-gray-50 transition-colors"
                        >
                          <Library className="w-4 h-4" />
                        </Link>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  );
}
