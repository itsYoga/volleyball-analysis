import axios from 'axios';

// API基礎URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 創建axios實例
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30秒超時
});

// 請求攔截器
api.interceptors.request.use(
  (config) => {
    console.log(`API請求: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 響應攔截器
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API錯誤:', error);
    return Promise.reject(error);
  }
);

// API接口定義
export interface Video {
  id: string;
  filename: string;
  file_path: string;
  upload_time: string;
  status: 'uploaded' | 'processing' | 'completed' | 'failed';
  file_size: number;
  task_id?: string;
  analysis_time?: string;
}

export interface AnalysisTask {
  video_id: string;
  status: 'processing' | 'completed' | 'failed';
  start_time: string;
  progress: number;
  end_time?: string;
  error?: string;
}

export interface AnalysisResults {
  video_info: {
    width: number;
    height: number;
    fps: number;
    total_frames: number;
    duration: number;
  };
  ball_tracking: {
    trajectory: Array<{
      frame: number;
      timestamp: number;
      center: [number, number];
      bbox: [number, number, number, number];
      confidence: number;
    }>;
    detected_frames: number;
    total_frames: number;
  };
  action_recognition: {
    actions: Array<{
      frame: number;
      timestamp: number;
      bbox: [number, number, number, number];
      confidence: number;
      action: string;
    }>;
    action_counts: Record<string, number>;
    total_actions: number;
  };
  analysis_time: number;
}

// API函數
export const apiService = {
  // 健康檢查
  async healthCheck() {
    const response = await api.get('/health');
    return response.data;
  },

  // 上傳影片
  async uploadVideo(file: FormData) {
    const response = await api.post('/upload', file, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  // 開始分析
  async startAnalysis(videoId: string) {
    const response = await api.post(`/analyze/${videoId}`);
    return response.data;
  },

  // 獲取所有影片
  async getVideos() {
    const response = await api.get('/videos');
    return response.data;
  },

  // 獲取特定影片
  async getVideo(videoId: string) {
    const response = await api.get(`/videos/${videoId}`);
    return response.data;
  },

  // 獲取分析任務狀態
  async getAnalysisStatus(taskId: string) {
    const response = await api.get(`/analysis/${taskId}`);
    return response.data;
  },

  // 獲取分析結果
  async getAnalysisResults(videoId: string) {
    const response = await api.get(`/results/${videoId}`);
    return response.data;
  },

  // 播放影片
  getVideoUrl(videoId: string) {
    return `${API_BASE_URL}/play/${videoId}`;
  },
};

// 導出便捷函數
export const uploadVideo = apiService.uploadVideo;
export const startAnalysis = apiService.startAnalysis;
export const getVideos = apiService.getVideos;
export const getVideo = apiService.getVideo;
export const getAnalysisStatus = apiService.getAnalysisStatus;
export const getAnalysisResults = apiService.getAnalysisResults;
export const getVideoUrl = apiService.getVideoUrl;

export default apiService;
