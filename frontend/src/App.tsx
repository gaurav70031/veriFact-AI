import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from '@/components/layout/Layout'
import Home            from '@/pages/Home'
import Analyze         from '@/pages/Analyze'
import Results         from '@/pages/Results'
import LiveNews        from '@/pages/LiveNews'
import History         from '@/pages/History'
import Analytics       from '@/pages/Analytics'
import ModelPerformance from '@/pages/ModelPerformance'
import About           from '@/pages/About'
import Methodology     from '@/pages/Methodology'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/"             element={<Home />} />
          <Route path="/analyze"      element={<Analyze />} />
          <Route path="/results/:id"  element={<Results />} />
          <Route path="/live-news"    element={<LiveNews />} />
          <Route path="/history"      element={<History />} />
          <Route path="/analytics"    element={<Analytics />} />
          <Route path="/performance"  element={<ModelPerformance />} />
          <Route path="/about"        element={<About />} />
          <Route path="/methodology"  element={<Methodology />} />
          {/* 404 fallback */}
          <Route path="*"             element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
