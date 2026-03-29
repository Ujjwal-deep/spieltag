import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="fixed top-0 left-0 right-0 h-16 bg-dark-900/80 backdrop-blur-lg border-b border-dark-700/50 z-50">
      <div className="container mx-auto px-4 h-full flex items-center justify-between max-w-7xl">
        <Link to="/" className="flex items-center gap-2 group">
          <Activity className="w-6 h-6 text-neon-blue group-hover:neon-text-blue transition-all" />
          <span className="text-xl font-bold bg-gradient-to-r from-neon-blue to-neon-cyan bg-clip-text text-transparent">
            Spieltag
          </span>
        </Link>
        <div className="flex gap-6">
          <Link 
            to="/" 
            className={`text-sm font-medium transition-colors hover:text-white ${location.pathname === '/' ? 'text-white' : 'text-gray-400'}`}
          >
            Upcoming Matchday
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
