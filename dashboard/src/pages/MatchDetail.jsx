import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { supabase } from '../lib/supabaseClient';
import { Loader2, ArrowLeft, Sparkles } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const MatchDetail = () => {
  const { matchId } = useParams();
  const [match, setMatch] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generatingInsight, setGeneratingInsight] = useState(false);
  const [insight, setInsight] = useState(null);
  const [insightError, setInsightError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Fetch match data
        const { data: matchData } = await supabase
          .from('matches')
          .select('*')
          .eq('match_id', matchId)
          .single();
          
        if (matchData) {
          setMatch(matchData);
          if (matchData.ai_insight === 'GENERATING...') {
            setGeneratingInsight(true);
          } else if (matchData.ai_insight) {
            setInsight(matchData.ai_insight);
          }
        }

        // Fetch predictions
        const { data: predsData } = await supabase
          .from('predictions')
          .select('*')
          .eq('match_id', matchId);

        if (predsData) {
          // Format for recharts
          const formatted = predsData.map(p => ({
            name: p.model_name,
            Home: (parseFloat(p.prob_home) * 100).toFixed(1),
            Draw: (parseFloat(p.prob_draw) * 100).toFixed(1),
            Away: (parseFloat(p.prob_away) * 100).toFixed(1)
          }));
          
          // Put ensemble last
          formatted.sort((a,b) => a.name === 'Ensemble' ? 1 : b.name === 'Ensemble' ? -1 : 0);
          setPredictions(formatted);
        }
      } catch (err) {
        console.error('Error fetching deep dive:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [matchId]);

  const handleGenerateInsight = async () => {
    try {
      setGeneratingInsight(true);
      setInsightError(null);
      
      const res = await fetch('/api/generate-insights', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ matchId })
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.error || 'Failed to generate insight');
      }
      
      if (data.status === 'success' || data.status === 'ready') {
        setInsight(data.insight);
      }
      // If it's still processing by someone else, we just leave it as generating.
    } catch (err) {
      console.error(err);
      setInsightError(err.message);
    } finally {
      setGeneratingInsight(false);
    }
  };

  if (loading) return (
    <div className="flex justify-center items-center h-[60vh]">
      <Loader2 className="w-12 h-12 animate-spin text-neon-blue" />
    </div>
  );

  if (!match) return <div className="text-center text-white mt-20">Match not found.</div>;

  return (
    <div className="max-w-5xl mx-auto pb-20">
      <Link to="/" className="mb-8 dd-back-link">
        <ArrowLeft className="w-5 h-5" />
        Back to Matchday
      </Link>

      <div className="dd-hero">
        <div className="dd-eyebrow">Deep Dive Analysis</div>
        <div className="dd-teams">
          <div className="dd-team-name">{match.home_team}</div>
          <div className="dd-vs">vs</div>
          <div className="dd-team-name">{match.away_team}</div>
        </div>
        <div className="dd-date">
          {new Date(match.date).toLocaleString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          })}
        </div>
      </div>

      {/* Probability tiles */}
      {predictions.length > 0 && (
        <div className="dd-prob-strip">
          {(() => {
            const ensemble = predictions.find(p => p.name === 'Ensemble') || predictions[0];
            const homeVal = parseFloat(ensemble.Home);
            const drawVal = parseFloat(ensemble.Draw);
            const awayVal = parseFloat(ensemble.Away);
            return (
              <>
                <div className="dd-stat">
                  <div className="dd-stat-label">Home win</div>
                  <div className="dd-stat-val val-h">{homeVal.toFixed(1)}%</div>
                  <div className="dd-stat-sub">{match.home_team}</div>
                </div>
                <div className="dd-stat">
                  <div className="dd-stat-label">Draw</div>
                  <div className="dd-stat-val val-d">{drawVal.toFixed(1)}%</div>
                  <div className="dd-stat-sub">Either side</div>
                </div>
                <div className="dd-stat">
                  <div className="dd-stat-label">Away win</div>
                  <div className="dd-stat-val val-a">{awayVal.toFixed(1)}%</div>
                  <div className="dd-stat-sub">{match.away_team}</div>
                </div>
              </>
            );
          })()}
        </div>
      )}

      <div className="dd-section">
        <div className="dd-section-title">
          Model Consensus Comparison
        </div>
        <div className="bg-[#111314] border border-[#1f2224] rounded-[14px] p-5">
        <div className="h-96 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={predictions}
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <XAxis dataKey="name" stroke="#6b7280" tick={{fill: '#9ca3af', fontWeight: 600}} />
              <YAxis stroke="#4b5563" />
              <Tooltip 
                contentStyle={{ backgroundColor: '#18181b', borderColor: '#3f3f46', color: '#fff' }}
                itemStyle={{ fontWeight: 600 }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              <Bar
                dataKey="Home"
                fill="#10b981"
                radius={[4, 4, 0, 0]}
                isAnimationActive
                animationDuration={600}
                animationEasing="ease-out"
                animationBegin={0}
              />
              <Bar
                dataKey="Draw"
                fill="#eab308"
                radius={[4, 4, 0, 0]}
                isAnimationActive
                animationDuration={600}
                animationEasing="ease-out"
                animationBegin={80}
              />
              <Bar
                dataKey="Away"
                fill="#ef4444"
                radius={[4, 4, 0, 0]}
                isAnimationActive
                animationDuration={600}
                animationEasing="ease-out"
                animationBegin={160}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      </div>

      {/* AI Insight Section */}
      <div className="dd-section">
        <div className="dd-section-title">
          AI Match Context
        </div>
        
        {insight ? (
          <div className="dd-context-box">
            <div className="dd-context-header">
              <Sparkles className="w-4 h-4 text-neon-cyan" />
              <span>Analysis</span>
            </div>
            {insight}
          </div>
        ) : generatingInsight ? (
          <div className="flex items-center gap-4 text-neon-cyan py-6 animate-pulse">
            <Loader2 className="w-6 h-6 animate-spin" />
            <span className="font-medium">Synthesizing match data and generating analysis...</span>
          </div>
        ) : (
          <div>
            <p className="text-gray-400 mb-6">
              Generate a quick LLM summary based on the ensemble predictions and team forms.
            </p>
            <button 
              onClick={handleGenerateInsight}
              className="dd-ai-button"
            >
              <span className="dd-ai-button-bg" />
              <span className="dd-ai-button-inner">
                <Sparkles className="w-4 h-4" />
                Generate AI Analysis
              </span>
            </button>
            {insightError && (
              <p className="text-red-400 mt-4 text-sm">{insightError}</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MatchDetail;
