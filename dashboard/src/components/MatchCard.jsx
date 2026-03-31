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
    <div className="match-card">
      <div className="match-card-bg" />
      <div className="match-card-inner">
        <div className="match-card-meta">
          <Calendar className="w-3 h-3" />
          <span>{formattedDate} &bull; {formattedTime}</span>
        </div>

        <div className="flex justify-between items-center mb-4">
          <div className="match-card-team">
            {match.home_team}
          </div>
          <div className="match-card-vs">
            vs
          </div>
          <div className="match-card-team text-right">
            {match.away_team}
          </div>
        </div>

        <div className="match-card-prob-row">
          <span
            className={`match-card-prob-h ${winner === 'home' ? 'match-card-winner' : ''}`}
          >
            H: {(probH * 100).toFixed(1)}%
          </span>
          <span className="match-card-prob-d">
            D: {(probD * 100).toFixed(1)}%
          </span>
          <span
            className={`match-card-prob-a ${winner === 'away' ? 'match-card-winner' : ''}`}
          >
            A: {(probA * 100).toFixed(1)}%
          </span>
        </div>

        <div className="match-card-bar-track">
          <div
            className="match-card-bar-seg match-card-bar-seg-h"
            style={{ width: `${probH * 100}%` }}
          />
          <div
            className="match-card-bar-seg match-card-bar-seg-d"
            style={{ width: `${probD * 100}%` }}
          />
          <div
            className="match-card-bar-seg match-card-bar-seg-a"
            style={{ width: `${probA * 100}%` }}
          />
        </div>

        <div className="match-card-footer">
          <div className="match-card-confidence">
            Confidence:{' '}
            <span>
              {(parseFloat(prediction?.confidence || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <Link
            to={`/match/${match.match_id}`}
            className="match-card-deep-dive"
            onClick={(e) => e.stopPropagation()}
          >
            Deep Dive <ChevronRight className="w-4 h-4" />
          </Link>
        </div>
      </div>
    </div>
  );
};

export default MatchCard;
