import React, { useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Upload, CheckCircle, AlertCircle, Loader2, Library, Play } from 'lucide-react';
import { uploadVideo, startAnalysis } from '../services/api';

interface UploadStatus {
  status: 'idle' | 'uploading' | 'analyzing' | 'completed' | 'error';
  message: string;
  progress: number;
  videoId?: string;
}

export const VideoUpload: React.FC = () => {
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({
    status: 'idle',
    message: 'Select a video file to begin',
    progress: 0
  });
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (file: File) => {
    // Validate file type
    if (!file.type.startsWith('video/')) {
      setUploadStatus({ status: 'error', message: 'Please select a valid video file.', progress: 0 });
      return;
    }

    // Max size 2GB
    const maxSize = 2 * 1024 * 1024 * 1024; // 2GB
    if (file.size > maxSize) {
      setUploadStatus({ status: 'error', message: 'File size cannot exceed 2GB.', progress: 0 });
      return;
    }

    try {
      // Start upload
      setUploadStatus({
        status: 'uploading',
        message: 'Uploading video...',
        progress: 0
      });

      const formData = new FormData();
      formData.append('file', file);

      const uploadResponse = await uploadVideo(formData);
      
      setUploadStatus({ status: 'analyzing', message: 'Video uploaded. Starting analysis...', progress: 50, videoId: uploadResponse.video_id });

      // 開始分析
      await startAnalysis(uploadResponse.video_id);
      
      setUploadStatus({ status: 'completed', message: 'Analysis has started. Check progress in your library.', progress: 100, videoId: uploadResponse.video_id });

    } catch (error: any) {
      setUploadStatus({ status: 'error', message: `Upload failed: ${error.message}`, progress: 0 });
    }
  };

  const getStatusIcon = () => {
    switch (uploadStatus.status) {
      case 'uploading':
      case 'analyzing':
        return <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />;
      case 'completed':
        return <CheckCircle className="h-8 w-8 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-8 w-8 text-red-600" />;
      default:
        return <Upload className="h-8 w-8 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (uploadStatus.status) {
      case 'uploading':
      case 'analyzing':
        return 'border-blue-300 bg-blue-50';
      case 'completed':
        return 'border-green-300 bg-green-50';
      case 'error':
        return 'border-red-300 bg-red-50';
      default:
        return 'border-gray-300 bg-white';
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-8">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            上傳排球影片
          </h1>
          <p className="text-gray-600">
            上傳您的排球比賽影片，系統將自動進行球軌跡追蹤和動作識別分析
          </p>
        </div>

        {/* 上傳區域 */}
        <div
          className={`relative border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
            dragActive ? 'border-blue-400 bg-blue-50' : getStatusColor()
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="video/*"
            onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            disabled={uploadStatus.status === 'uploading' || uploadStatus.status === 'analyzing'}
          />
          
          <div className="flex flex-col items-center">
            {getStatusIcon()}
            
            <div className="mt-4">
              <p className="text-lg font-medium text-gray-900">
                {uploadStatus.message}
              </p>
              
              {uploadStatus.status === 'uploading' || uploadStatus.status === 'analyzing' ? (
                <div className="mt-4">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${uploadStatus.progress}%` }}
                    ></div>
                  </div>
                  <p className="text-sm text-gray-500 mt-2">
                    {uploadStatus.progress}%
                  </p>
                </div>
              ) : uploadStatus.status === 'idle' ? (
                <p className="text-sm text-gray-500 mt-2">
                  點擊選擇文件或拖拽到此處
                </p>
              ) : null}
            </div>
          </div>
        </div>

        {/* File info */}
        <div className="mt-6 text-center">
          <p className="text-sm text-gray-500">Supported formats: MP4, AVI, MOV, WMV, FLV</p>
          <p className="text-sm text-gray-500">Maximum file size: 2GB</p>
        </div>

        {/* Completion actions */}
        {uploadStatus.status === 'completed' && uploadStatus.videoId && (
          <div className="mt-8 text-center">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <p className="text-green-800 font-medium">
                Video uploaded and analysis started successfully!
              </p>
              <p className="text-green-600 text-sm mt-1">
                Video ID: {uploadStatus.videoId}
              </p>
              <div className="mt-4 space-x-4">
                <Link to={`/library`} className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium">
                  <Library className="w-4 h-4" /> View Library
                </Link>
                <Link to={`/player/${uploadStatus.videoId}`} className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors font-medium">
                  <Play className="w-4 h-4" /> Play Now
                </Link>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
