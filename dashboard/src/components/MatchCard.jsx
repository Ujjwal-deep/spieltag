import React from 'react';
import { Link } from 'react-router-dom';
import { ChevronRight, Calendar } from 'lucide-react';
import ProbabilityBar from './ProbabilityBar';

const MatchCard = ({ match, prediction }) => {
  const d = new Date(match.date);
  const formattedDate = d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  const formattedTime = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

  // Safety check, although we expect this data
  const probH = parseFloat(prediction?.prob_home || 0.33);
  const probD = parseFloat(prediction?.prob_draw || 0.33);
  const probA = parseFloat(prediction?.prob_away || 0.34);

  const getWinnerHighlight = () => {
    if (probH > probA && probH > probD) return 'home';
    if (probA > probH && probA > probD) return 'away';
    return 'draw';
  };
  
  const winner = getWinnerHighlight();

  return (
    <div className="glass-card flex flex-col p-6 group relative overflow-hidden">
      {/* Decorative gradient blob */}
      <div className="absolute -top-10 -right-10 w-32 h-32 bg-neon-blue/10 rounded-full blur-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      
      <div className="flex justify-between items-center mb-4 border-b border-dark-700/50 pb-3">
        <div className="flex items-center text-sm text-gray-400 gap-2">
          <Calendar className="w-4 h-4 text-neon-blue" />
          <span>{formattedDate} • {formattedTime}</span>
        </div>
      </div>

      <div className="flex justify-between items-center py-2 relative z-10">
        <div className={`text-xl font-bold ${winner === 'home' ? 'text-white' : 'text-gray-400'}`}>
          {match.home_team}
        </div>
        <div className="text-gray-500 font-medium px-4">vs</div>
        <div className={`text-xl font-bold text-right ${winner === 'away' ? 'text-white' : 'text-gray-400'}`}>
          {match.away_team}
        </div>
      </div>

      <div className="mt-4">
         <div className="flex justify-between text-xs font-semibold text-gray-500 mb-1 px-1">
           <span className={winner === 'home' ? 'text-neon-green' : ''}>H: {(probH * 100).toFixed(1)}%</span>
           <span className={winner === 'draw' ? 'text-yellow-500' : ''}>D: {(probD * 100).toFixed(1)}%</span>
           <span className={winner === 'away' ? 'text-neon-red' : ''}>A: {(probA * 100).toFixed(1)}%</span>
         </div>
         <ProbabilityBar homeProb={probH} drawProb={probD} awayProb={probA} />
      </div>

      <div className="mt-6 flex justify-between items-center pt-4 border-t border-dark-700/50">
        <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold">
          Confidence: <span className="text-gray-300">{(parseFloat(prediction?.confidence || 0) * 100).toFixed(1)}%</span>
        </div>
        <Link to={`/match/${match.match_id}`} className="flex items-center text-sm text-neon-cyan hover:neon-text-blue transition-all gap-1 font-medium group-hover:gap-2">
          Deep Dive <ChevronRight className="w-4 h-4" />
        </Link>
      </div>
    </div>
  );
};

export default MatchCard;
