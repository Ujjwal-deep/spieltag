-- Drop existing tables to ensure a clean schema creation
DROP TABLE IF EXISTS public.predictions CASCADE;
DROP TABLE IF EXISTS public.matches CASCADE;

-- Create the matches table
CREATE TABLE public.matches (
    match_id TEXT PRIMARY KEY,
    date TIMESTAMP WITH TIME ZONE NOT NULL,
    season TEXT,
    division TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_goals INTEGER,
    away_goals INTEGER,
    result INTEGER,
    status TEXT DEFAULT 'SCHEDULED'
);

-- Create the predictions table
CREATE TABLE public.predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    match_id TEXT REFERENCES public.matches(match_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    prob_home NUMERIC,
    prob_draw NUMERIC,
    prob_away NUMERIC,
    confidence NUMERIC,
    is_ensemble BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    UNIQUE(match_id, model_name)
);

-- Note: model_metrics is optional and omitted for simplicity but can be added if needed

-- Enable RLS
ALTER TABLE public.matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access
CREATE POLICY "Allow public read access to matches" 
    ON public.matches FOR SELECT 
    USING (true);

CREATE POLICY "Allow public read access to predictions" 
    ON public.predictions FOR SELECT 
    USING (true);

-- Allow authenticated users or service role to insert/update
CREATE POLICY "Allow service role insert updates matches"
    ON public.matches FOR ALL
    USING (auth.role() = 'service_role');

CREATE POLICY "Allow service role insert updates predictions"
    ON public.predictions FOR ALL
    USING (auth.role() = 'service_role');
