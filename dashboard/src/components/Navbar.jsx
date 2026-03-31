import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="fixed top-0 left-0 right-0 h-16 bg-[#0d0f0f] border-b border-[#1a1d1e] z-50">
      <div className="px-6 h-full flex items-center">
        <Link to="/" className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-[#2fd8a0]" />
          <span className="text-[17px] font-bold text-white tracking-tight">
            Spieltag
          </span>
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
