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
        // Fetch upcoming scheduled matches
        const { data: matchesData, error: matchesError } = await supabase
          .from('matches')
          .select('*')
          .eq('status', 'SCHEDULED')
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
      <div className="mb-10 text-center">
        <h1 className="text-4xl font-black mb-2 text-white">Upcoming Matchday</h1>
        <p className="text-gray-400 max-w-2xl mx-auto">
          AI-driven probabilities for the next Bundesliga fixtures. Powered by Ensemble ML models.
        </p>
      </div>

      {matches.length === 0 ? (
         <div className="text-center text-gray-500 py-12">No upcoming scheduled matches found in the database.</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
