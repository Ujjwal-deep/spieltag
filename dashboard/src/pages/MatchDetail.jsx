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
      <Link to="/" className="inline-flex items-center text-gray-400 hover:text-white mb-8 transition-colors group">
        <ArrowLeft className="w-5 h-5 mr-2 group-hover:-translate-x-1 transition-transform" />
        Back to Dashboard
      </Link>
      
      <div className="glass-card p-10 mb-10 text-center bg-gradient-to-b from-dark-800 to-dark-900 border-t border-neon-blue/30">
        <div className="text-neon-cyan mb-3 font-semibold tracking-widest text-sm uppercase">Deep Dive Analysis</div>
        <h1 className="text-5xl font-black text-white mb-4 tracking-tight">
          {match.home_team} <span className="text-gray-600 font-light mx-4 text-4xl">vs</span> {match.away_team}
        </h1>
        <p className="text-gray-400 text-lg">
          {new Date(match.date).toLocaleString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>

      <div className="glass-card p-8 bg-dark-800/50">
        <h3 className="text-2xl font-bold text-white mb-8 flex items-center gap-3">
          <div className="w-2 h-8 bg-neon-green rounded-full"></div>
          Model Consensus Comparison
        </h3>
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
              <Bar dataKey="Home" fill="#10b981" radius={[4, 4, 0, 0]} animationDuration={1500} />
              <Bar dataKey="Draw" fill="#eab308" radius={[4, 4, 0, 0]} animationDuration={1500} />
              <Bar dataKey="Away" fill="#ef4444" radius={[4, 4, 0, 0]} animationDuration={1500} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* AI Insight Section */}
      <div className="glass-card p-8 mt-10 bg-dark-800/50 relative overflow-hidden group">
        <div className="absolute -top-10 -right-10 w-40 h-40 bg-neon-cyan/5 rounded-full blur-3xl" />
        <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <Sparkles className="w-6 h-6 text-neon-cyan" />
          AI Match Context
        </h3>
        
        {insight ? (
          <div className="text-gray-300 leading-relaxed text-lg border-l-2 border-neon-cyan/50 pl-5">
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
              className="inline-flex items-center gap-2 px-6 py-3 bg-neon-blue/10 hover:bg-neon-blue/20 text-neon-cyan border border-neon-blue/30 rounded-lg transition-all font-medium group-hover:shadow-[0_0_15px_rgba(56,189,248,0.2)]"
            >
              <Sparkles className="w-4 h-4" />
              Generate AI Analysis
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
