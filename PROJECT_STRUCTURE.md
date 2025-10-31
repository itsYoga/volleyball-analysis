# Project Structure Documentation

This document describes the organization and purpose of each directory and file in the project.

## 📁 Directory Structure

### `/backend`
FastAPI backend service providing REST API endpoints.

- **`main.py`**: Main FastAPI application with routes and middleware
- **`data/`**: Backend-specific data storage
  - `uploads/`: Uploaded video files (gitignored)
  - `results/`: Analysis result JSON files (gitignored)

### `/frontend`
React + TypeScript frontend application.

- **`src/`**: Source code
  - `components/`: React components
    - `Dashboard.tsx`: Main dashboard with statistics
    - `VideoUpload.tsx`: Video upload interface
    - `VideoLibrary.tsx`: Video library with search/filter
    - `VideoPlayer.tsx`: Video player with timeline
    - `EventTimeline.tsx`: Interactive event timeline component
    - `PlayerHeatmap.tsx`: Player movement heatmap overlay
    - `ui/`: Reusable UI components (StatusBadge, EmptyState)
  - `services/`: API service layer
    - `api.ts`: Axios-based API client
  - `index.tsx`: Application entry point
  - `App.tsx`: Main app component with routing
  - `index.css`: Global styles and Tailwind imports
- **`public/`**: Static assets
- **`package.json`**: Node.js dependencies

### `/ai_core`
AI processing core for video analysis.

- **`processor.py`**: Main analyzer class (`VolleyballAnalyzer`)
- **`worker.py`**: Celery worker for background processing

### `/models`
Pre-trained AI model files (gitignored - too large for Git).

- `action_recognition_yv11m.pt`: Action recognition model
- `player_detection_yv8.pt`: Player detection model
- `VballNetV1_seq9_grayscale_148_h288_w512.onnx`: Ball tracking model

### `/data`
Root-level data storage (may be consolidated in future).

- `uploads/`: User uploaded videos
- `results/`: Analysis results JSON

### `/tools`
Utility scripts and helper functions.

- `utils/`: Utility modules
  - `utils.py`: Annotation and data processing utilities

### `/scripts`
Development and testing scripts.

- `offline_test.py`: Offline testing utilities

### `/static`
Static file storage (served by FastAPI).

### Root Files

- **`requirements.txt`**: Python dependencies
- **`docker-compose.yml`**: Docker container configuration
- **`start.sh`**: Quick start script for all services
- **`README.md`**: Main project documentation
- **`DEVELOPMENT.md`**: Development guide
- **`.gitignore`**: Git ignore rules
- **`PROJECT_STRUCTURE.md`**: This file

## 🔄 Data Flow

1. **Upload**: User uploads video → `backend/data/uploads/`
2. **Process**: AI worker processes video → generates results
3. **Store**: Results saved to `backend/data/results/`
4. **Display**: Frontend fetches and displays results

## 📦 Model Files

Model files are stored locally but not in Git (too large). They should be:
- Downloaded separately
- Placed in `/models` directory
- Referenced in code via relative paths

## 🗂️ Future Reorganization

Consider consolidating:
- `data/` and `backend/data/` into single `data/` directory
- All scripts into `/scripts` directory
- Documentation into `/docs` directory

