import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getVideos } from '../services/api';
import { EmptyState } from './ui/EmptyState';
import { StatusBadge } from './ui/StatusBadge';
import { Info, PlayCircle, Calendar, Search, Filter, Video as VideoIcon, Loader2 } from 'lucide-react';

export const VideoLibrary: React.FC = () => {
  const [videos, setVideos] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  useEffect(() => {
    getVideos()
      .then((res) => setVideos(res.videos || []))
      .finally(() => setLoading(false));
  }, []);

  const filteredVideos = videos.filter(v => {
    const matchesSearch = v.filename.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesStatus = statusFilter === 'all' || v.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const statusCounts = {
    all: videos.length,
    completed: videos.filter(v => v.status === 'completed').length,
    processing: videos.filter(v => v.status === 'processing').length,
    failed: videos.filter(v => v.status === 'failed').length,
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <section>
        <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 bg-clip-text text-transparent">
          Video Library
        </h1>
        <p className="text-gray-600 mt-2 text-lg">
          Browse and manage all your analyzed videos
        </p>
      </section>

      {/* Filters and Search */}
      <section className="bg-white rounded-xl shadow-md p-4 border border-gray-100">
        <div className="flex flex-col md:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search videos..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
            />
          </div>
          
          {/* Status Filter */}
          <div className="flex items-center gap-2">
            <Filter className="text-gray-400 w-5 h-5" />
            <div className="flex gap-2">
              {(['all', 'completed', 'processing', 'failed'] as const).map((status) => (
                <button
                  key={status}
                  onClick={() => setStatusFilter(status)}
                  className={`
                    px-4 py-2 rounded-lg text-sm font-medium transition-all
                    ${statusFilter === status
                      ? 'bg-blue-600 text-white shadow-md'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }
                  `}
                >
                  {status.charAt(0).toUpperCase() + status.slice(1)} ({statusCounts[status]})
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Videos Grid */}
      <section>
        {loading && (
          <div className="flex justify-center items-center h-64">
            <Loader2 className="w-10 h-10 text-blue-600 animate-spin" />
          </div>
        )}

        {!loading && filteredVideos.length === 0 && (
          <div className="bg-white rounded-2xl shadow-xl p-12 border border-gray-100">
            <EmptyState
              title={searchQuery || statusFilter !== 'all' ? "No videos match your filters" : "No videos yet"}
              hint={
                searchQuery || statusFilter !== 'all' ? (
                  <span>Try adjusting your search or filter criteria.</span>
                ) : (
                  <span>
                    Get started by uploading your first video. Go to the{' '}
                    <Link to="/upload" className="text-blue-600 hover:text-blue-700 font-semibold underline decoration-2 underline-offset-2">
                      Upload
                    </Link>{' '}
                    page to begin analysis.
                  </span>
                )
              }
              icon={<VideoIcon className="w-16 h-16 text-gray-300" />}
            />
          </div>
        )}

        {!loading && filteredVideos.length > 0 && (
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredVideos.map((v) => (
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
                      <p className="text-sm text-gray-500 mt-2 flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        {new Date(v.upload_time).toLocaleString()}
                      </p>
                    </div>
                    <StatusBadge status={v.status} />
                  </div>
                  
                  <div className="mt-auto pt-4 border-t border-gray-100">
                    <Link
                      to={`/player/${v.id}`}
                      className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg bg-gradient-to-r from-blue-600 to-blue-700 text-white text-sm font-medium hover:from-blue-700 hover:to-blue-800 transition-all shadow-md hover:shadow-lg"
                    >
                      <PlayCircle className="w-4 h-4" />
                      View Analysis
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
