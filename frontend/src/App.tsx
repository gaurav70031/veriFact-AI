import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider }      from '@/context/AuthContext'
import { ThemeProvider }     from '@/context/ThemeContext'
import { ProtectedRoute }    from '@/components/auth/ProtectedRoute'
import { Layout }            from '@/components/layout/Layout'

// Pages
import Home             from '@/pages/Home'
import Analyze          from '@/pages/Analyze'
import Results          from '@/pages/Results'
import LiveNews         from '@/pages/LiveNews'
import History          from '@/pages/History'
import Analytics        from '@/pages/Analytics'
import ModelLaboratory  from '@/pages/ModelLaboratory'
import ModelComparison  from '@/pages/ModelComparison'
import DatasetExplorer  from '@/pages/DatasetExplorer'
import ExplainableAI    from '@/pages/ExplainableAI'
import About            from '@/pages/About'
import Methodology      from '@/pages/Methodology'
import Settings         from '@/pages/Settings'
import Login            from '@/pages/Login'
import Register         from '@/pages/Register'

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <AuthProvider>
          <Routes>
            {/* ── Auth pages (layout wraps them in a centered container) ── */}
            <Route element={<Layout />}>
              <Route path="/login"    element={<Login />} />
              <Route path="/register" element={<Register />} />

              {/* ── Main navigation pages ─────────────────────────────── */}

              {/* 1. Overview Dashboard */}
              <Route path="/"  element={<Home />} />

              {/* 2. News Verification */}
              <Route path="/live-news" element={<LiveNews />} />

              {/* 3. Text Analysis */}
              <Route path="/analyze"       element={<Analyze />} />
              <Route path="/analyze/text"  element={<Analyze />} />

              {/* 4. URL Verification */}
              <Route path="/analyze/url"   element={<Analyze />} />

              {/* Results (shared by all analysis types) */}
              <Route path="/results/:id"   element={<Results />} />

              {/* 5. Dataset Explorer */}
              <Route path="/dataset" element={<DatasetExplorer />} />

              {/* 6. Model Laboratory */}
              <Route path="/lab"         element={<ModelLaboratory />} />
              {/* Legacy alias */}
              <Route path="/performance" element={<ModelLaboratory />} />

              {/* 7. Model Comparison */}
              <Route path="/comparison" element={<ModelComparison />} />

              {/* 8. Explainable AI */}
              <Route path="/explainable"   element={<ExplainableAI />} />
              {/* Legacy alias */}
              <Route path="/methodology"   element={<Methodology />} />

              {/* 9. Misinformation Analytics */}
              <Route path="/analytics" element={<Analytics />} />

              {/* 10. Verification History (protected) */}
              <Route
                path="/history"
                element={
                  <ProtectedRoute>
                    <History />
                  </ProtectedRoute>
                }
              />

              {/* 11. About Project */}
              <Route path="/about" element={<About />} />

              {/* 12. Settings */}
              <Route path="/settings" element={<Settings />} />

              {/* 404 */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </AuthProvider>
      </BrowserRouter>
    </ThemeProvider>
  )
}
