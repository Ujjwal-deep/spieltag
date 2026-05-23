import React, { useEffect, useState } from 'react';
import { supabase } from '../lib/supabaseClient';
import MatchCard from '../components/MatchCard';
import { Loader2, AlertCircle } from 'lucide-react';

const Home = () => {
  const [matches, setMatches] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const today = new Date();
        today.setHours(0, 0, 0, 0);

        // Fetch upcoming scheduled matches
        const { data: matchesData, error: matchesError } = await supabase
          .from('matches')
          .select('*')
          .eq('status', 'SCHEDULED')
          .gte('date', today.toISOString())
          .order('date', { ascending: true })
          .limit(9);

        if (matchesError) throw matchesError;

        if (matchesData && matchesData.length > 0) {
          const matchIds = matchesData.map(m => m.match_id);
          
          // Fetch ensemble predictions for these matches
          const { data: predsData, error: predsError } = await supabase
            .from('predictions')
            .select('*')
            .eq('model_name', 'Ensemble')
            .in('match_id', matchIds);

          if (predsError) throw predsError;

          const predsMap = {};
          if (predsData) {
            predsData.forEach(p => {
              predsMap[p.match_id] = p;
            });
          }
          
          setMatches(matchesData);
          setPredictions(predsMap);
        } else {
          setMatches([]);
        }
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return (
    <div className="flex justify-center items-center h-[60vh]">
      <Loader2 className="w-12 h-12 animate-spin text-neon-blue" />
    </div>
  );

  if (error) return (
    <div className="flex justify-center items-center h-[60vh]">
      <div className="glass-card p-8 flex flex-col items-center gap-4 border-red-500/50">
        <AlertCircle className="w-16 h-16 text-red-500" />
        <h2 className="text-xl font-bold text-white">Error Loading Data</h2>
        <p className="text-red-400">{error}</p>
      </div>
    </div>
  );

  return (
    <div>
      <div className="matchday-header">
        <div className="mh-left">
          <div className="mh-eyebrow">
            <div className="mh-dot" />
            Live predictions
          </div>
          <div className="mh-title">{matches.length > 0 ? "Upcoming Fixtures" : "Off-Season"}</div>
          <div className="mh-sub">
            {matches.length > 0 
              ? `${new Date(matches[0].date).toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })} · ${matches.length} fixtures` 
              : 'The current season has concluded'}
          </div>
        </div>
        <div className="mh-right">
          <div className="mh-badge">
            Powered by <span>Ensemble ML</span>
          </div>
          <div className="mh-league">Bundesliga 2025/26</div>
        </div>
      </div>

      {matches.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 px-4 text-center">
          <div className="w-24 h-24 rounded-full bg-[#1c1f20] border border-[#252829] flex items-center justify-center mb-6 shadow-[0_0_30px_rgba(16,185,129,0.1)]">
            <AlertCircle className="w-10 h-10 text-neon-green opacity-80" />
          </div>
          <h2 className="text-3xl font-extrabold text-white tracking-tight mb-3">
            Season Concluded
          </h2>
          <p className="text-gray-400 max-w-md mx-auto leading-relaxed">
            The current campaign has officially wrapped up. There are no more scheduled fixtures to predict. Check back next season for live predictions, insights, and model updates!
          </p>
          <div className="mt-8 flex gap-4">
            <div className="px-6 py-3 rounded-full bg-[#111314] border border-[#1f2224] text-sm text-gray-300 font-medium">
              Models are currently in hibernation
            </div>
          </div>
        </div>
      ) : (
        <div className="match-grid">
          {matches.map(match => (
            <MatchCard
              key={match.match_id}
              match={match}
              prediction={predictions[match.match_id]}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default Home;
