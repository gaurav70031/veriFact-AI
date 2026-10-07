import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider }      from '@/context/AuthContext'
import { ThemeProvider }     from '@/context/ThemeContext'
import { ProtectedRoute }    from '@/components/auth/ProtectedRoute'
import { Layout }            from '@/components/layout/Layout'
import Home                  from '@/pages/Home'
import Analyze               from '@/pages/Analyze'
import Results               from '@/pages/Results'
import LiveNews              from '@/pages/LiveNews'
import History               from '@/pages/History'
import Analytics             from '@/pages/Analytics'
import ModelPerformance      from '@/pages/ModelPerformance'
import About                 from '@/pages/About'
import Methodology           from '@/pages/Methodology'
import Login                 from '@/pages/Login'
import Register              from '@/pages/Register'

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            {/* ── Auth pages (no main layout) ────────────────────────────── */}
            <Route path="/login"    element={<Login />} />
            <Route path="/register" element={<Register />} />

            {/* ── Main layout ──────────────────────────────────────────────── */}
            <Route element={<Layout />}>
              <Route path="/"            element={<Home />} />
              <Route path="/analyze"     element={<Analyze />} />
              <Route path="/results/:id" element={<Results />} />
              <Route path="/live-news"   element={<LiveNews />} />
              <Route path="/analytics"   element={<Analytics />} />
              <Route path="/performance" element={<ModelPerformance />} />
              <Route path="/about"       element={<About />} />
              <Route path="/methodology" element={<Methodology />} />

              {/* ── Protected routes (requires auth) ─────────────────────── */}
              <Route
                path="/history"
                element={
                  <ProtectedRoute>
                    <History />
                  </ProtectedRoute>
                }
              />

              {/* 404 fallback */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  )
}
