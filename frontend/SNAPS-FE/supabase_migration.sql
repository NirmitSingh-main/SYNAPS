-- =============================================================
-- SYNAPS Database Migration
-- Run this in the Supabase SQL Editor (Project > SQL Editor)
-- =============================================================

-- ─────────────────────────────────────────────────────────────
-- 1. PROFILES TABLE
-- Stores user role. 'user' = normal, 'admin' = platform admin.
-- DO NOT create if it already exists with the same structure.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.profiles (
  id          uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email       text,
  role        text NOT NULL DEFAULT 'user',
  created_at  timestamptz NOT NULL DEFAULT now()
);

-- Auto-create a profile row when a new user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  INSERT INTO public.profiles (id, email, role)
  VALUES (NEW.id, NEW.email, 'user')
  ON CONFLICT (id) DO NOTHING;
  RETURN NEW;
END;
$$;

-- Attach trigger to auth.users (safe to re-run)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


-- ─────────────────────────────────────────────────────────────
-- 2. ANALYSIS REPORTS TABLE
-- Only create if it does not already exist.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.analysis_reports (
  id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id              uuid REFERENCES auth.users(id) ON DELETE CASCADE,
  created_at           timestamptz NOT NULL DEFAULT now(),
  filename             text,
  format               text,
  classification       text,
  confidence           numeric,
  sample_rate          numeric,
  duration             numeric,
  bandwidth            numeric,
  snr                  numeric,
  peak_frequency       numeric,
  num_samples          bigint,
  prediction_breakdown jsonb,
  features             jsonb,
  explanation          text,
  raw_report           jsonb
);


-- ─────────────────────────────────────────────────────────────
-- 3. ROW LEVEL SECURITY — profiles
-- ─────────────────────────────────────────────────────────────
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Users can read their own profile
DROP POLICY IF EXISTS "profiles: own read" ON public.profiles;
CREATE POLICY "profiles: own read"
  ON public.profiles
  FOR SELECT
  USING (auth.uid() = id);

-- Admins can read all profiles
DROP POLICY IF EXISTS "profiles: admin read all" ON public.profiles;
CREATE POLICY "profiles: admin read all"
  ON public.profiles
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role = 'admin'
    )
  );


-- ─────────────────────────────────────────────────────────────
-- 4. ROW LEVEL SECURITY — analysis_reports
-- ─────────────────────────────────────────────────────────────
ALTER TABLE public.analysis_reports ENABLE ROW LEVEL SECURITY;

-- Normal users: full CRUD on own rows only
DROP POLICY IF EXISTS "reports: own select" ON public.analysis_reports;
CREATE POLICY "reports: own select"
  ON public.analysis_reports
  FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "reports: own insert" ON public.analysis_reports;
CREATE POLICY "reports: own insert"
  ON public.analysis_reports
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "reports: own update" ON public.analysis_reports;
CREATE POLICY "reports: own update"
  ON public.analysis_reports
  FOR UPDATE
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "reports: own delete" ON public.analysis_reports;
CREATE POLICY "reports: own delete"
  ON public.analysis_reports
  FOR DELETE
  USING (auth.uid() = user_id);

-- Admins: read all reports for dashboard
DROP POLICY IF EXISTS "reports: admin read all" ON public.analysis_reports;
CREATE POLICY "reports: admin read all"
  ON public.analysis_reports
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.profiles
      WHERE id = auth.uid() AND role = 'admin'
    )
  );


-- ─────────────────────────────────────────────────────────────
-- 5. GRANT A USER ADMIN ROLE
-- Replace 'your-email@example.com' with the target user's email.
-- Run this manually after the user has signed up.
-- ─────────────────────────────────────────────────────────────
-- UPDATE public.profiles
-- SET role = 'admin'
-- WHERE email = 'your-email@example.com';


-- ─────────────────────────────────────────────────────────────
-- DONE
-- After running this migration:
--   1. All new sign-ups will automatically get a profile row.
--   2. RLS ensures users only see their own analysis_reports.
--   3. Admins (role='admin') can read all profiles + reports.
--   4. No service-role key is needed or exposed in the frontend.
-- ─────────────────────────────────────────────────────────────
