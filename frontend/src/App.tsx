import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import { VideoUpload } from './components/VideoUpload';
import { VideoLibrary } from './components/VideoLibrary';
import { Dashboard } from './components/Dashboard';
import { VideoPlayer } from './components/VideoPlayer';
import { LayoutDashboard, Upload, Library, Activity } from 'lucide-react';

function Navigation() {
  const location = useLocation();
  
  const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/library', label: 'Library', icon: Library },
  ];

  return (
    <nav className="hidden md:flex items-center gap-1">
      {navItems.map(({ path, label, icon: Icon }) => {
        const isActive = location.pathname === path;
        return (
          <NavLink
            key={path}
            to={path}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200
              ${isActive 
                ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-lg shadow-blue-500/30' 
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
              }
            `}
          >
            <Icon className="w-4 h-4" />
            {label}
          </NavLink>
        );
      })}
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-50">
        {/* Modern Header */}
        <header className="sticky top-0 z-50 border-b border-gray-200/80 bg-white/80 backdrop-blur-xl shadow-sm">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <NavLink to="/" className="flex items-center gap-3 group">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg blur opacity-75 group-hover:opacity-100 transition-opacity"></div>
                  <div className="relative bg-gradient-to-r from-blue-600 to-purple-600 p-2 rounded-lg">
                    <Activity className="w-5 h-5 text-white" />
                  </div>
                </div>
                <div>
                  <div className="text-lg font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                    Volleyball AI
                  </div>
                  <div className="text-xs text-gray-500 -mt-0.5">Analysis System</div>
                </div>
              </NavLink>
              <Navigation />
            </div>
          </div>
        </header>

        {/* Content Area */}
        <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<VideoUpload />} />
            <Route path="/library" element={<VideoLibrary />} />
            <Route path="/player/:videoId" element={<VideoPlayer />} />
          </Routes>
        </main>

        {/* Modern Footer */}
        <footer className="border-t border-gray-200 bg-white/50 backdrop-blur-sm mt-20">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
            <div className="flex flex-col md:flex-row items-center justify-between gap-4">
              <p className="text-sm text-gray-600">
                © {new Date().getFullYear()} Volleyball AI Analysis System. Powered by AI.
              </p>
              <div className="flex items-center gap-6 text-sm text-gray-500">
                <button className="hover:text-gray-900 transition-colors cursor-pointer">Privacy</button>
                <button className="hover:text-gray-900 transition-colors cursor-pointer">Terms</button>
                <button className="hover:text-gray-900 transition-colors cursor-pointer">Support</button>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
