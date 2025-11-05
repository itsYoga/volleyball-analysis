import axios from 'axios';

// API基礎URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 創建axios實例 - 不同端點使用不同超時時間
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30秒超時（適用於一般查詢，即使後端正在處理其他任務）
});

// 創建長時間運行的請求實例（用於上傳和分析）
const longRunningApi = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5分鐘超時（用於長時間操作）
});

// 請求攔截器（應用於兩個實例）
const requestInterceptor = (config: any) => {
  console.log(`API請求: ${config.method?.toUpperCase()} ${config.url}`);
  return config;
};

const requestErrorInterceptor = (error: any) => {
  return Promise.reject(error);
};

// 響應攔截器（應用於兩個實例）
const responseInterceptor = (response: any) => {
  return response;
};

const responseErrorInterceptor = (error: any) => {
  console.error('API錯誤:', error);
  return Promise.reject(error);
};

// 為兩個實例設置攔截器
api.interceptors.request.use(requestInterceptor, requestErrorInterceptor);
api.interceptors.response.use(responseInterceptor, responseErrorInterceptor);

longRunningApi.interceptors.request.use(requestInterceptor, requestErrorInterceptor);
longRunningApi.interceptors.response.use(responseInterceptor, responseErrorInterceptor);

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
      end_frame?: number;
      end_timestamp?: number;
      bbox: [number, number, number, number];
      confidence: number;
      action: string;
      player_id?: number;
      duration?: number;
    }>;
    action_detections: Array<{
      frame: number;
      timestamp: number;
      bbox: [number, number, number, number];
      confidence: number;
      action: string;
      player_id?: number;
    }>;
    action_counts: Record<string, number>;
    total_actions: number;
  };
  plays?: Array<{
    play_id: number;
    start_frame: number;
    start_timestamp: number;
    end_frame: number;
    end_timestamp: number;
    duration: number;
    actions: any[];
    scores: any[];
  }>;
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
    const response = await longRunningApi.post('/upload', file, {
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

  // 更新視頻名稱
  async updateVideoName(videoId: string, newFilename: string) {
    const response = await api.put(`/videos/${videoId}`, {
      new_filename: newFilename
    });
    return response.data;
  },

  // 刪除視頻
  async deleteVideo(videoId: string) {
    const response = await api.delete(`/videos/${videoId}`);
    return response.data;
  },

  // 設置球衣號碼映射
  async setJerseyMapping(videoId: string, trackId: number, jerseyNumber: number, frame: number, bbox: number[]) {
    const response = await api.post(`/videos/${videoId}/jersey-mapping`, {
      video_id: videoId,
      track_id: trackId,
      jersey_number: jerseyNumber,
      frame: frame,
      bbox: bbox
    });
    return response.data;
  },

  // 獲取球衣號碼映射
  async getJerseyMappings(videoId: string) {
    const response = await api.get(`/videos/${videoId}/jersey-mappings`);
    return response.data;
  },

  // 刪除球衣號碼映射
  async deleteJerseyMapping(videoId: string, trackId: string) {
    const response = await api.delete(`/videos/${videoId}/jersey-mapping/${trackId}`);
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
export const updateVideoName = apiService.updateVideoName;
export const deleteVideo = apiService.deleteVideo;
export const setJerseyMapping = apiService.setJerseyMapping;
export const getJerseyMappings = apiService.getJerseyMappings;
export const deleteJerseyMapping = apiService.deleteJerseyMapping;

export default apiService;
