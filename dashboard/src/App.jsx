import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import MatchDetail from './pages/MatchDetail';

function App() {
  return (
    <Router>
      <div className="min-h-screen flex flex-col pt-16">
        <Navbar />
        <main className="flex-1 px-4 py-8 w-full max-w-6xl mx-auto">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/match/:matchId" element={<MatchDetail />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
