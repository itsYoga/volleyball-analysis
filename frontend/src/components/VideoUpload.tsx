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
      <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-8 text-white">
          <h1 className="text-3xl md:text-4xl font-bold mb-2">
            Upload Video
          </h1>
          <p className="text-blue-100 text-lg">
            Upload your volleyball match video for automatic ball tracking and action recognition analysis
          </p>
        </div>

        <div className="p-8">
          {/* Upload Area */}
          <div
            className={`
              relative border-2 border-dashed rounded-xl p-16 text-center transition-all duration-300
              ${dragActive 
                ? 'border-blue-500 bg-gradient-to-br from-blue-50 to-purple-50 scale-[1.02]' 
                : getStatusColor()
              }
              ${uploadStatus.status === 'idle' ? 'hover:border-blue-400 hover:bg-gray-50 cursor-pointer' : ''}
            `}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="video/*"
              onChange={handleFileInput}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              disabled={uploadStatus.status === 'uploading' || uploadStatus.status === 'analyzing'}
            />
            
            <div className="flex flex-col items-center">
              <div className="mb-6">
                {getStatusIcon()}
              </div>
              
              <div className="space-y-3">
                <p className="text-xl font-semibold text-gray-900">
                  {uploadStatus.message}
                </p>
                
                {uploadStatus.status === 'uploading' || uploadStatus.status === 'analyzing' ? (
                  <div className="mt-6 space-y-3">
                    <div className="w-64 mx-auto bg-gray-200 rounded-full h-3 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-blue-600 to-purple-600 h-full rounded-full transition-all duration-500 shadow-lg"
                        style={{ width: `${uploadStatus.progress}%` }}
                      ></div>
                    </div>
                    <p className="text-sm font-medium text-gray-600">
                      {uploadStatus.progress}% complete
                    </p>
                  </div>
                ) : uploadStatus.status === 'idle' ? (
                  <div className="space-y-2">
                    <p className="text-gray-500">
                      Drag and drop your video here, or click to browse
                    </p>
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium mt-4">
                      <Upload className="w-4 h-4" />
                      Select File
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          {/* File Info */}
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-gray-600">
              <div className="flex items-center gap-2">
                <span className="font-medium">Formats:</span>
                <span>MP4, AVI, MOV, WMV, FLV</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="font-medium">Max Size:</span>
                <span>2GB</span>
              </div>
            </div>
          </div>

          {/* Success Message */}
          {uploadStatus.status === 'completed' && uploadStatus.videoId && (
            <div className="mt-8">
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-xl p-6">
                <div className="flex items-start gap-4">
                  <CheckCircle className="w-8 h-8 text-green-600 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-green-900 font-semibold text-lg mb-1">
                      Upload successful!
                    </p>
                    <p className="text-green-700 text-sm mb-4">
                      Your video has been uploaded and analysis has started. You can track progress in your library.
                    </p>
                    <div className="flex flex-wrap gap-3">
                      <Link 
                        to={`/library`} 
                        className="inline-flex items-center gap-2 px-5 py-2.5 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-all shadow-md hover:shadow-lg font-medium"
                      >
                        <Library className="w-4 h-4" /> View Library
                      </Link>
                      <Link 
                        to={`/player/${uploadStatus.videoId}`} 
                        className="inline-flex items-center gap-2 px-5 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-all shadow-md hover:shadow-lg font-medium"
                      >
                        <Play className="w-4 h-4" /> Watch Now
                      </Link>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
