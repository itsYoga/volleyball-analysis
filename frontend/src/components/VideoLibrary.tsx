import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getVideos, updateVideoName } from '../services/api';
import { EmptyState } from './ui/EmptyState';
import { StatusBadge } from './ui/StatusBadge';
import { PlayCircle, Calendar, Search, Filter, Video as VideoIcon, Loader2, Edit2, Check, X } from 'lucide-react';

export const VideoLibrary: React.FC = () => {
  const [videos, setVideos] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [editingVideoId, setEditingVideoId] = useState<string | null>(null);
  const [editName, setEditName] = useState<string>('');

  useEffect(() => {
    let isMounted = true;
    const loadVideos = async () => {
      try {
        const res = await getVideos();
        if (isMounted) {
          setVideos(res.videos || []);
        }
      } catch (error: any) {
        console.error('Failed to load videos:', error);
        // 如果請求失敗，設為空陣列而不是保持loading狀態
        if (isMounted) {
          setVideos([]);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };
    
    loadVideos();
    
    return () => {
      isMounted = false;
    };
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

  const handleStartEdit = (videoId: string, currentName: string) => {
    setEditingVideoId(videoId);
    setEditName(currentName);
  };

  const handleSaveEdit = async (videoId: string) => {
    if (!editName.trim()) {
      setEditingVideoId(null);
      return;
    }

    try {
      await updateVideoName(videoId, editName.trim());
      // 更新本地狀態
      setVideos(videos.map(v => v.id === videoId ? { ...v, filename: editName.trim() } : v));
      setEditingVideoId(null);
      setEditName('');
    } catch (error) {
      console.error('Failed to update video name:', error);
      alert('更新視頻名稱失敗，請重試');
    }
  };

  const handleCancelEdit = () => {
    setEditingVideoId(null);
    setEditName('');
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
                      {editingVideoId === v.id ? (
                        <div className="flex items-center gap-2">
                          <input
                            type="text"
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            className="flex-1 px-2 py-1 border border-blue-500 rounded text-sm font-semibold text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') {
                                handleSaveEdit(v.id);
                              } else if (e.key === 'Escape') {
                                handleCancelEdit();
                              }
                            }}
                          />
                          <button
                            onClick={() => handleSaveEdit(v.id)}
                            className="p-1 text-green-600 hover:text-green-700 transition-colors"
                            title="Save"
                          >
                            <Check className="w-4 h-4" />
                          </button>
                          <button
                            onClick={handleCancelEdit}
                            className="p-1 text-red-600 hover:text-red-700 transition-colors"
                            title="Cancel"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 group">
                          <h3 className="font-semibold text-gray-900 truncate group-hover:text-blue-600 transition-colors flex-1" title={v.filename}>
                            {v.filename}
                          </h3>
                          <button
                            onClick={() => handleStartEdit(v.id, v.filename)}
                            className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-blue-600 transition-all"
                            title="Rename video"
                          >
                            <Edit2 className="w-4 h-4" />
                          </button>
                        </div>
                      )}
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
