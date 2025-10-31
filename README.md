# Volleyball AI Analysis System

A modern, professional web application for volleyball video analysis using AI-powered ball tracking, player detection, and action recognition.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![React](https://img.shields.io/badge/react-18.2-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Video Upload & Processing**: Drag-and-drop video upload with progress tracking (supports up to 2GB)
- **Ball Tracking**: Real-time ball trajectory detection using VballNet ONNX model
- **Action Recognition**: Player action classification (spike, set, receive, serve, block) using YOLOv11 with confidence filtering (≥60%)
- **Player Detection & Tracking**: YOLOv8 + Norfair for player tracking across frames with confidence filtering (≥50%)
- **Interactive Player**: Drag-to-seek timeline with event markers and real-time bounding boxes
- **Smart Confidence Filtering**: Automatic filtering of low-confidence detections to reduce false positives
- **Player Management**: Rename players and view individual player statistics
- **Action Consolidation**: Continuous actions across multiple frames are consolidated into single action events
- **Smart Filtering**: Search and filter videos by status, date, and metadata
- **Analytics Dashboard**: Statistics cards and visual insights
- **Modern UI**: Professional, responsive design with Tailwind CSS
- **Interactive Timeline**: Drag to seek, click event markers to jump to specific moments
- **Real-time Visualizations**: Player and action bounding boxes, ball tracking trail, and player movement heatmap

## Project Structure

```
volleyball_analysis_webapp/
├── backend/                 # FastAPI backend service
│   ├── main.py             # Main API application
│   └── data/               # Backend data storage (gitignored)
│       ├── uploads/        # Uploaded videos
│       └── results/        # Analysis results JSON
│
├── frontend/               # React + TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── Dashboard.tsx      # Main dashboard with statistics
│   │   │   ├── VideoUpload.tsx    # Drag-and-drop upload interface
│   │   │   ├── VideoLibrary.tsx   # Video library with search/filter
│   │   │   ├── VideoPlayer.tsx    # Video player with timeline
│   │   │   ├── EventTimeline.tsx  # Interactive event timeline
│   │   │   ├── PlayerHeatmap.tsx  # Player movement heatmap
│   │   │   ├── BallTracking.tsx   # Ball trajectory visualization
│   │   │   ├── BoundingBoxes.tsx  # Player and action bounding boxes
│   │   │   ├── PlayerStats.tsx     # Player statistics and action lists
│   │   │   └── ui/                # Reusable UI components
│   │   ├── services/       # API service layer
│   │   │   └── api.ts      # Axios-based API client
│   │   ├── index.tsx       # Entry point
│   │   ├── App.tsx         # Main app with routing
│   │   └── index.css       # Tailwind CSS imports
│   └── public/             # Static assets
│
├── ai_core/                # AI processing core
│   ├── processor.py        # Main analyzer (VolleyballAnalyzer)
│   └── worker.py           # Celery worker for background processing
│
├── models/                 # Pre-trained AI models (gitignored)
│   ├── action_recognition_yv11m.pt
│   ├── player_detection_yv8.pt
│   └── VballNetV1_seq9_grayscale_148_h288_w512.onnx
│
├── data/                   # Data storage (gitignored)
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

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 16+
- Redis (optional, for Celery task queue)

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

5. **Place AI models** (required for analysis)
   - Download or obtain the model files
   - Place them in the `models/` directory:
     - `VballNetV1_seq9_grayscale_148_h288_w512.onnx`
     - `action_recognition_yv11m.pt`
     - `player_detection_yv8.pt`

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

**Terminal 2 - AI Worker (Optional - if using Celery):**
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

## Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Celery** - Async task processing (optional)
- **Redis** - Task queue and caching (optional)

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
- **Norfair** - Multi-object tracking

## Usage Guide

### Uploading a Video

1. Navigate to the **Upload** page
2. Drag and drop a video file or click to browse
3. Supported formats: MP4, AVI, MOV, WMV, FLV (max 2GB)
4. Wait for upload and analysis to complete

### Viewing Analysis

1. Go to **Library** to see all videos
2. Use search and filters to find specific videos
3. Click **View Analysis** to open the interactive player
4. **Timeline Controls**:
   - Drag the timeline to seek to any position in the video
   - Click event markers (spikes, sets, receives) to jump to specific moments
   - Watch the playhead indicator show current playback position
5. **Visualization Options**:
   - Toggle **Player Boxes** to show/hide player bounding boxes
   - Toggle **Action Boxes** to show/hide action detection boxes
   - Toggle **Ball Tracking** to visualize ball trajectory
   - Toggle **Heatmap** to visualize player movement patterns
   - Visualizations update in real-time as the video plays
6. **Confidence Filtering**: Only high-confidence detections (≥50% for players, ≥60% for actions) are displayed
7. **Player Statistics**:
   - Toggle **Show Player Stats** to view detailed player statistics
   - Rename players by clicking the edit icon next to their name
   - View each player's action timeline and statistics
   - Click on any action to jump to that moment in the video

### Dashboard Overview

The dashboard provides:
- **Statistics Cards**: Total videos, completed, processing, failed
- **Recent Videos**: Quick access to latest uploads
- **Status Tracking**: Real-time analysis progress

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|------------|
| GET | `/` | API root |
| GET | `/health` | Health check |
| GET | `/videos` | List all videos |
| GET | `/videos/{id}` | Get video metadata |
| POST | `/upload` | Upload video file |
| POST | `/analyze/{video_id}` | Start analysis |
| GET | `/analysis/{task_id}` | Get analysis task status |
| GET | `/results/{video_id}` | Get analysis results |
| GET | `/play/{video_id}` | Stream video file |

See full interactive API documentation at http://localhost:8000/docs

## Data Flow

1. **Upload**: User uploads video → `backend/data/uploads/`
2. **Process**: AI worker processes video → generates results
3. **Store**: Results saved to `backend/data/results/` as JSON
4. **Display**: Frontend fetches and displays results with interactive timeline

## Feature Status

### Completed Features
- Video upload with streaming support
- Ball tracking (VballNet)
- Action recognition (YOLOv11) with confidence filtering
- Player detection & tracking (YOLOv8 + Norfair) with confidence filtering
- Action-to-player association
- Action consolidation (continuous actions merged into single events)
- Score event detection
- Game state detection (Play/No-Play)
- Interactive timeline with drag-to-seek functionality
- Real-time bounding boxes synchronized with video playback
- Ball trajectory visualization
- Player heatmap visualization with time-based filtering
- Player statistics and action lists
- Player renaming functionality
- Modern responsive UI
- Search and filter functionality
- Non-blocking video analysis (thread pool execution)
- Robust error handling and timeout management

### Known Limitations

1. **Database**: Currently uses in-memory storage (`videos_db`), data lost on restart
   - Future: Integrate PostgreSQL

2. **Task Queue**: Uses FastAPI BackgroundTasks, not persistent
   - Future: Use Celery + Redis (worker.py already prepared)

3. **Game State**: Simplified logic for Play/No-Play detection
   - Can be enhanced based on actual requirements

4. **Ball Detection**: Ball tracking may require model tuning for optimal performance
   - Detection sensitivity can be adjusted via confidence thresholds

## Development

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

### Building for Production

```bash
# Frontend
cd frontend
npm run build

# Backend
# Use production ASGI server like gunicorn with uvicorn workers
```

## Docker Deployment

The project includes `docker-compose.yml` for containerized deployment:

```bash
docker-compose up -d
```

This will start:
- Redis service
- PostgreSQL database
- Backend API
- AI Worker
- Frontend service

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with modern web technologies
- Powered by PyTorch and Ultralytics
- Inspired by professional sports analysis tools

## Support

For issues and questions, please open an issue on GitHub: https://github.com/itsYoga/volleyball-analysis

## Contact

- **Author**: Yu-Jia Liang
- **Email**: ch993115@gmail.com
- **LinkedIn**: [https://www.linkedin.com/feed/](https://www.linkedin.com/feed/)

---

**Made with ❤️ for volleyball analysis**
