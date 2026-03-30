import { createClient } from '@supabase/supabase-js';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { matchId } = req.body;
  if (!matchId) {
    return res.status(400).json({ error: 'matchId is required' });
  }

  const supabaseUrl = process.env.VITE_SUPABASE_URL || process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_SERVICE_KEY || process.env.VITE_SUPABASE_ANON_KEY;
  const groqApiKey = process.env.GROQ_API_KEY;

  if (!supabaseUrl || !supabaseKey || !groqApiKey) {
    return res.status(500).json({ error: 'Server misconfiguration: Missing API keys' });
  }

  const supabase = createClient(supabaseUrl, supabaseKey);

  try {
    // 1. Atomically check and set "GENERATING..."
    // Supabase JS client doesn't support complex conditional updates safely unless using RPC.
    // However we can fetch the current status first, or just rely on a simple update if it's currently NULL.
    // For simplicity, we just check if it's already generating or not.
    const { data: matchData, error: fetchError } = await supabase
      .from('matches')
      .select('ai_insight, home_team, away_team')
      .eq('match_id', matchId)
      .single();

    if (fetchError || !matchData) {
      return res.status(404).json({ error: 'Match not found' });
    }

    if (matchData.ai_insight === 'GENERATING...' || matchData.ai_insight) {
      // If it exists or is generating, we can just return what we have (or tell client it's processing)
      return res.status(200).json({ 
        status: matchData.ai_insight === 'GENERATING...' ? 'processing' : 'ready',
        insight: matchData.ai_insight 
      });
    }

    // Lock the row (Provisional state)
    await supabase
      .from('matches')
      .update({ ai_insight: 'GENERATING...' })
      .eq('match_id', matchId);

    // Fetch predictions to provide context to LLM
    const { data: predsData } = await supabase
      .from('predictions')
      .select('*')
      .eq('match_id', matchId);

    let contextText = `Analyze the upcoming match between ${matchData.home_team} (Home) and ${matchData.away_team} (Away).\n\nHere are the AI model predictions for this match:\n`;
    if (predsData && predsData.length > 0) {
      predsData.forEach(p => {
        contextText += `- ${p.model_name}: Home Win ${(p.prob_home * 100).toFixed(1)}%, Draw ${(p.prob_draw * 100).toFixed(1)}%, Away Win ${(p.prob_away * 100).toFixed(1)}%, Confidence: ${(p.confidence * 100).toFixed(1)}%\n`;
      });
    } else {
      contextText += "No specific model predictions available right now.\n";
    }

    contextText += "\nPlease act as a professional football data analyst. Based on these probabilities (especially the Ensemble model if present), write a very short, engaging 2-3 sentence analysis of what the data is suggesting for this fixture. Don't mention the raw probabilities too mechanically, but focus on the narrative (e.g. 'Strong favorite...', 'Looks like a tight match...'). Do not use formatting like bolding or bullet points.";

    // 2. Call Groq API
    const groqResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${groqApiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'llama-3.1-8b-instant',
        messages: [{ role: 'user', content: contextText }],
        temperature: 0.7,
        max_tokens: 250
      })
    });

    if (!groqResponse.ok) {
      const errText = await groqResponse.text();
      // Revert the lock
      await supabase.from('matches').update({ ai_insight: null }).eq('match_id', matchId);
      console.error("Groq API Error:", errText);
      return res.status(502).json({ error: 'Failed to generate insight from LLM' });
    }

    const groqData = await groqResponse.json();
    const generatedText = groqData.choices[0].message.content.trim();

    // 3. Update the matches table with the generated text
    await supabase
      .from('matches')
      .update({ ai_insight: generatedText })
      .eq('match_id', matchId);

    return res.status(200).json({ status: 'success', insight: generatedText });

  } catch (err) {
    console.error('Server error:', err);
    // In case of error, try to unlock if we locked
    await supabase.from('matches').update({ ai_insight: null }).eq('match_id', matchId);
    return res.status(500).json({ error: 'Internal server error' });
  }
}
