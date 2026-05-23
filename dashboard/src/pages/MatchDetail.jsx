import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { supabase } from '../lib/supabaseClient';
import { Loader2, ArrowLeft, Sparkles, TrendingUp, Shield, Activity, Calendar, Zap, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const MatchDetail = () => {
  const { matchId } = useParams();
  const [match, setMatch] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [generatingInsight, setGeneratingInsight] = useState(false);
  const [insight, setInsight] = useState(null);
  const [insightError, setInsightError] = useState(null);
  const [features, setFeatures] = useState(null);

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

        // Fetch local match features JSON file
        try {
          const fRes = await fetch('/match_features.json');
          if (fRes.ok) {
            const allFeatures = await fRes.json();
            if (allFeatures[matchId]) {
              setFeatures(allFeatures[matchId]);
            }
          }
        } catch (fErr) {
          console.warn('Error loading match features:', fErr);
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
      <Link to="/" className="mb-8 dd-back-link hover:text-neon-cyan transition-colors">
        <ArrowLeft className="w-5 h-5" />
        Back to Fixtures
      </Link>

      <div className="dd-hero relative overflow-hidden">
        <div className="absolute inset-0 opacity-10 bg-gradient-to-r from-neon-blue via-transparent to-neon-cyan pointer-events-none" />
        <div className="dd-eyebrow relative z-10">Deep Dive Analysis</div>
        <div className="dd-teams">
          <div className="dd-team-name">{match.home_team}</div>
          <div className="dd-vs">vs</div>
          <div className="dd-team-name">{match.away_team}</div>
        </div>
        <div className="dd-date relative z-10 flex items-center justify-center gap-2 mt-2">
          <Calendar className="w-4 h-4 text-gray-400" />
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

      {/* Model Features Dashboard */}
      {features && (
        <div className="dd-section">
          <div className="dd-section-title">
            Engineered Model Features & Team Stats
          </div>
          
          <div className="dd-features-grid">
            {/* ELO Rating Card */}
            <div className="dd-features-card">
              <div className="dd-features-card-title">
                <TrendingUp className="w-4 h-4 text-neon-blue" />
                ELO Strength Rating
                {features.promoted_stat_quality === 'goals' && (
                  <span className="badge-neon badge-neon-yellow">B2 Goals Fallback</span>
                )}
              </div>
              
              <div className="elo-header-row">
                <div className="elo-team-box">
                  <div className="elo-team-name">{match.home_team}</div>
                  <div className="elo-team-val val-h">{Math.round(features.home_elo)}</div>
                </div>
                
                <div className="elo-vs-badge">VS</div>
                
                <div className="elo-team-box text-right">
                  <div className="elo-team-name">{match.away_team}</div>
                  <div className="elo-team-val val-a">{Math.round(features.away_elo)}</div>
                </div>
              </div>
              
              <div className="comparison-slider-bar mb-4">
                <div 
                  className="comparison-slider-fill bg-[#10b981]" 
                  style={{ width: `${(features.home_elo / (features.home_elo + features.away_elo)) * 100}%` }}
                />
                <div 
                  className="comparison-slider-fill bg-[#ef4444]" 
                  style={{ width: `${(features.away_elo / (features.home_elo + features.away_elo)) * 100}%` }}
                />
              </div>
              
              <div className="elo-gap-indicator">
                {features.elo_diff > 0 ? (
                  <>
                    <span className="home-lead">{match.home_team}</span> holds a +<span>{Math.round(features.elo_diff)}</span> point ELO quality advantage.
                  </>
                ) : features.elo_diff < 0 ? (
                  <>
                    <span className="away-lead">{match.away_team}</span> holds a +<span>{Math.round(Math.abs(features.elo_diff))}</span> point ELO quality advantage.
                  </>
                ) : (
                  <>Teams are perfectly balanced in historical quality ELO.</>
                )}
              </div>
            </div>
            
            {/* Rolling xG and Form comparison */}
            <div className="dd-features-card">
              <div className="dd-features-card-title">
                <Activity className="w-4 h-4 text-neon-green" />
                Performance Metrics (Last 5 Games)
              </div>
              
              {/* xG Scored */}
              <div className="comparison-row">
                <div className="comparison-label-row">
                  <span className="comparison-val-home">{(features.home_xg_avg5 || 0).toFixed(2)} xG</span>
                  <span className="comparison-label-text">Expected Goals Scored</span>
                  <span className="comparison-val-away">{(features.away_xg_avg5 || 0).toFixed(2)} xG</span>
                </div>
                <div className="comparison-slider-bar">
                  <div 
                    className="comparison-slider-fill bg-[#10b981]" 
                    style={{ width: `${(features.home_xg_avg5 / (features.home_xg_avg5 + features.away_xg_avg5 || 1)) * 100}%` }}
                  />
                  <div 
                    className="comparison-slider-fill bg-[#ef4444]" 
                    style={{ width: `${(features.away_xg_avg5 / (features.home_xg_avg5 + features.away_xg_avg5 || 1)) * 100}%` }}
                  />
                </div>
              </div>

              {/* xG Conceded */}
              <div className="comparison-row">
                <div className="comparison-label-row">
                  <span className="comparison-val-home">{(features.home_xga_avg5 || 0).toFixed(2)} xGA</span>
                  <span className="comparison-label-text">Expected Goals Conceded</span>
                  <span className="comparison-val-away">{(features.away_xga_avg5 || 0).toFixed(2)} xGA</span>
                </div>
                <div className="comparison-slider-bar">
                  <div 
                    className="comparison-slider-fill bg-[#10b981]" 
                    style={{ width: `${(features.away_xga_avg5 / (features.home_xga_avg5 + features.away_xga_avg5 || 1)) * 100}%` }}
                  />
                  <div 
                    className="comparison-slider-fill bg-[#ef4444]" 
                    style={{ width: `${(features.home_xga_avg5 / (features.home_xga_avg5 + features.away_xga_avg5 || 1)) * 100}%` }}
                  />
                </div>
              </div>

              {/* Recent Form points */}
              <div className="comparison-row">
                <div className="comparison-label-row">
                  <span className="comparison-val-home">{(features.home_form_pts || 0).toFixed(1)} Pts</span>
                  <span className="comparison-label-text">Recent Form Avg</span>
                  <span className="comparison-val-away">{(features.away_form_pts || 0).toFixed(1)} Pts</span>
                </div>
                <div className="comparison-slider-bar">
                  <div 
                    className="comparison-slider-fill bg-[#10b981]" 
                    style={{ width: `${(features.home_form_pts / (features.home_form_pts + features.away_form_pts || 1)) * 100}%` }}
                  />
                  <div 
                    className="comparison-slider-fill bg-[#ef4444]" 
                    style={{ width: `${(features.away_form_pts / (features.home_form_pts + features.away_form_pts || 1)) * 100}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Match Dynamics Rest & Home/Away Card */}
            <div className="dd-features-card">
              <div className="dd-features-card-title">
                <Calendar className="w-4 h-4 text-neon-cyan" />
                Match Dynamics & Rest Advantage
              </div>

              <div className="flex flex-col gap-5">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 text-sm">Days of Rest:</span>
                  <div className="flex items-center gap-3 text-sm">
                    <span className="text-white font-semibold">{features.days_rest_home} days</span>
                    <span className="text-gray-600">vs</span>
                    <span className="text-white font-semibold">{features.days_rest_away} days</span>
                  </div>
                </div>

                <div className="flex justify-center">
                  {features.rest_diff > 0 ? (
                    <span className="badge-neon badge-neon-green">
                      <Zap className="w-3 h-3 mr-1" /> {match.home_team} has +{features.rest_diff} days rest advantage
                    </span>
                  ) : features.rest_diff < 0 ? (
                    <span className="badge-neon badge-neon-red">
                      <Zap className="w-3 h-3 mr-1" /> {match.away_team} has +{Math.abs(features.rest_diff)} days rest advantage
                    </span>
                  ) : (
                    <span className="badge-neon badge-neon-blue">Equal Rest periods (balanced freshness)</span>
                  )}
                </div>

                <div className="border-t border-[#1a1d1e] pt-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-gray-400 text-sm">Home vs Away Record (Win Rate):</span>
                  </div>
                  <div className="comparison-row">
                    <div className="comparison-label-row">
                      <span className="comparison-val-home">{(features.home_is_home_record * 100).toFixed(0)}% Home Win</span>
                      <span className="comparison-val-away">{(features.away_is_away_record * 100).toFixed(0)}% Away Win</span>
                    </div>
                    <div className="comparison-slider-bar">
                      <div 
                        className="comparison-slider-fill bg-[#10b981]" 
                        style={{ width: `${(features.home_is_home_record / (features.home_is_home_record + features.away_is_away_record || 1)) * 100}%` }}
                      />
                      <div 
                        className="comparison-slider-fill bg-[#ef4444]" 
                        style={{ width: `${(features.away_is_away_record / (features.home_is_home_record + features.away_is_away_record || 1)) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Head-to-Head (H2H) Card */}
            <div className="dd-features-card">
              <div className="dd-features-card-title">
                <Shield className="w-4 h-4 text-neon-red" />
                Head-to-Head History (H2H)
                {features.h2h_data_quality === 'none' && (
                  <span className="badge-neon badge-neon-red">No Meetings</span>
                )}
                {features.h2h_data_quality === 'partial' && (
                  <span className="badge-neon badge-neon-yellow">Partial Data</span>
                )}
                {features.h2h_data_quality === 'full' && (
                  <span className="badge-neon badge-neon-green">Full History</span>
                )}
              </div>

              <div className="flex flex-col gap-4">
                <div className="text-gray-400 text-xs">
                  Last 5 matchups history breakdown (Relative to Home side):
                </div>

                <div className="flex items-center justify-between bg-[#1c1f20] border border-[#252829] rounded-xl p-3">
                  <div className="text-center">
                    <div className="text-[10px] text-gray-500 uppercase font-bold">Home Wins</div>
                    <div className="text-lg font-extrabold text-[#4ade80]">{Math.round(features.h2h_home_wins)}</div>
                  </div>
                  <div className="text-center border-x border-[#252829] px-6">
                    <div className="text-[10px] text-gray-500 uppercase font-bold">Draws</div>
                    <div className="text-lg font-extrabold text-[#facc15]">{Math.round(features.h2h_draws)}</div>
                  </div>
                  <div className="text-center">
                    <div className="text-[10px] text-gray-500 uppercase font-bold">Away Wins</div>
                    <div className="text-lg font-extrabold text-[#f87171]">
                      {Math.round(Math.max(0, 5 - features.h2h_home_wins - features.h2h_draws))}
                    </div>
                  </div>
                </div>

                <div className="h2h-timeline justify-center py-2">
                  {(() => {
                    const timeline = [];
                    const hWins = Math.round(features.h2h_home_wins);
                    const draws = Math.round(features.h2h_draws);
                    const aWins = Math.max(0, 5 - hWins - draws);

                    for (let h = 0; h < hWins; h++) timeline.push(<div key={`h-${h}`} className="h2h-bubble win-h">W</div>);
                    for (let d = 0; d < draws; d++) timeline.push(<div key={`d-${d}`} className="h2h-bubble draw">D</div>);
                    for (let a = 0; a < aWins; a++) timeline.push(<div key={`a-${a}`} className="h2h-bubble win-a">L</div>);
                    
                    while (timeline.length < 5) {
                      timeline.push(<div key={`empty-${timeline.length}`} className="h2h-bubble empty">-</div>);
                    }

                    return timeline;
                  })()}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

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
              Generate a comprehensive AI analysis for {match.home_team} vs {match.away_team}, driven by ensemble predictions and recent team form.
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
