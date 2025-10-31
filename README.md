# 🏐 Volleyball AI Analysis System

A modern, professional web application for volleyball video analysis using AI-powered ball tracking, player detection, and action recognition.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![React](https://img.shields.io/badge/react-18.2-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- 🎥 **Video Upload & Processing**: Drag-and-drop video upload with progress tracking
- ⚽ **Ball Tracking**: Real-time ball trajectory detection using VballNet
- 🏐 **Action Recognition**: Player action classification (spike, set, receive, serve, block)
- 📊 **Interactive Player**: Click-to-seek timeline with event markers
- 🔍 **Smart Filtering**: Search and filter videos by status, date, and metadata
- 📈 **Analytics Dashboard**: Statistics cards and visual insights
- 🎨 **Modern UI**: Professional, responsive design with smooth animations

## 🏗️ Project Structure

```
volleyball_analysis_webapp/
├── backend/                 # FastAPI backend service
│   ├── main.py             # Main API application
│   └── data/               # Backend data storage
│       ├── uploads/        # Uploaded videos
│       └── results/        # Analysis results
│
├── frontend/               # React + TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── Dashboard.tsx
│   │   │   ├── VideoUpload.tsx
│   │   │   ├── VideoLibrary.tsx
│   │   │   ├── VideoPlayer.tsx
│   │   │   ├── EventTimeline.tsx
│   │   │   ├── PlayerHeatmap.tsx
│   │   │   └── ui/         # Reusable UI components
│   │   ├── services/       # API service layer
│   │   └── index.tsx        # Entry point
│   └── public/             # Static assets
│
├── ai_core/                # AI processing core
│   ├── processor.py        # Main analyzer
│   └── worker.py           # Background worker
│
├── models/                 # Pre-trained AI models
│   ├── action_recognition_yv11m.pt
│   ├── player_detection_yv8.pt
│   └── VballNetV1_seq9_grayscale_148_h288_w512.onnx
│
├── data/                   # Data storage
│   ├── uploads/            # User uploaded videos
│   └── results/            # Analysis results JSON
│
├── tools/                  # Utility scripts
│   └── utils/             # Helper utilities
│
├── scripts/               # Development scripts
│   └── offline_test.py    # Testing utilities
│
├── requirements.txt       # Python dependencies
├── docker-compose.yml     # Docker configuration
├── start.sh              # Quick start script
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 16+
- Redis (for task queue)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/itsYoga/volleyball-analysis.git
   cd volleyball-analysis
   ```

2. **Set up Python environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up frontend**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Create necessary directories**
   ```bash
   mkdir -p data/uploads data/results backend/data/uploads backend/data/results static
   ```

### Running the Application

#### Option 1: Using the start script
```bash
chmod +x start.sh
./start.sh
```

#### Option 2: Manual startup

**Terminal 1 - Backend API:**
```bash
source venv/bin/activate
cd backend
uvicorn main:app --reload
```

**Terminal 2 - AI Worker:**
```bash
source venv/bin/activate
cd ai_core
python worker.py
```

**Terminal 3 - Frontend:**
```bash
cd frontend
npm start
```

### Access Points

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Celery** - Async task processing
- **Redis** - Task queue and caching

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **React Router** - Client-side routing
- **Lucide React** - Icon library
- **Axios** - HTTP client

### AI/ML
- **PyTorch** - Deep learning framework
- **Ultralytics** - YOLO models
- **OpenCV** - Computer vision
- **ONNX Runtime** - Optimized inference

## 📖 Usage Guide

### Uploading a Video

1. Navigate to the **Upload** page
2. Drag and drop a video file or click to browse
3. Supported formats: MP4, AVI, MOV, WMV, FLV (max 2GB)
4. Wait for upload and analysis to complete

### Viewing Analysis

1. Go to **Library** to see all videos
2. Use search and filters to find specific videos
3. Click **View Analysis** to open the interactive player
4. Use the timeline to jump to specific events (spikes, sets, receives)

### Dashboard Overview

The dashboard provides:
- **Statistics Cards**: Total videos, completed, processing, failed
- **Recent Videos**: Quick access to latest uploads
- **Status Tracking**: Real-time analysis progress

## 🔧 Development

### Project Structure Guidelines

- **Backend**: All API routes and business logic in `backend/`
- **Frontend**: Components organized by feature in `frontend/src/components/`
- **AI Core**: Model processing logic in `ai_core/`
- **Utilities**: Shared tools in `tools/`
- **Scripts**: Development/testing scripts in `scripts/`

### Code Style

- **Python**: Follow PEP 8, use type hints
- **TypeScript**: Use strict mode, proper typing
- **React**: Functional components with hooks

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|------------|
| GET | `/videos` | List all videos |
| GET | `/videos/{id}` | Get video metadata |
| POST | `/upload` | Upload video file |
| POST | `/videos/{id}/analyze` | Start analysis |
| GET | `/videos/{id}/results` | Get analysis results |
| GET | `/static/videos/{id}` | Stream video file |

See full API documentation at http://localhost:8000/docs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with modern web technologies
- Powered by PyTorch and Ultralytics
- Inspired by professional sports analysis tools

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Made with ❤️ for volleyball analysis**
