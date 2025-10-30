import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { VideoUpload } from './components/VideoUpload';
import { VideoLibrary } from './components/VideoLibrary';
import { Dashboard } from './components/Dashboard';
import { VideoPlayer } from './components/VideoPlayer';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50 text-gray-900">
        {/* 導覽列 */}
        <header className="border-b bg-white/80 backdrop-blur">
          <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
            <NavLink to="/" className="text-lg font-bold">排球分析系統</NavLink>
            <nav className="hidden md:flex items-center gap-6 text-sm">
              <NavLink to="/" className={({isActive})=>`hover:text-blue-600 ${isActive? 'text-blue-600':''}`}>儀表板</NavLink>
              <NavLink to="/upload" className={({isActive})=>`hover:text-blue-600 ${isActive? 'text-blue-600':''}`}>上傳</NavLink>
              <NavLink to="/library" className={({isActive})=>`hover:text-blue-600 ${isActive? 'text-blue-600':''}`}>影片庫</NavLink>
            </nav>
          </div>
        </header>

        {/* 內容區域 */}
        <main className="mx-auto max-w-7xl px-6 py-10">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<div className="bg-white rounded-2xl shadow p-6 md:p-10"><VideoUpload /></div>} />
            <Route path="/library" element={<VideoLibrary />} />
            <Route path="/player/:videoId" element={<VideoPlayer />} />
          </Routes>
        </main>

        {/* 頁尾 */}
        <footer className="border-t py-8 text-center text-sm text-gray-500">
          <p>© {new Date().getFullYear()} 排球分析系統</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
