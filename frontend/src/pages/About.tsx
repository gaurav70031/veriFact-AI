import { ShieldCheck, Github, Database, Cpu, Search } from 'lucide-react'
import { useHealth } from '@/hooks/useApi'

export default function About() {
  const { data: health } = useHealth()

  const stack = [
    { icon: <Cpu className="w-4 h-4 text-brand-400" />, layer: 'ML Pipeline', items: ['scikit-learn (TF-IDF + LR / SVM / NB)', 'PyTorch + HuggingFace DistilBERT', 'LIME explainability', 'Attention attribution'] },
    { icon: <Database className="w-4 h-4 text-emerald-400" />, layer: 'Backend', items: ['FastAPI + Python 3.11', 'PostgreSQL 15 + SQLAlchemy', 'Alembic migrations', 'Async evidence retrieval'] },
    { icon: <Search className="w-4 h-4 text-violet-400" />, layer: 'Evidence', items: ['NewsAPI', 'GNews', 'RSS feeds (configurable)', 'SerpAPI / Brave Search'] },
    { icon: <ShieldCheck className="w-4 h-4 text-amber-400" />, layer: 'Frontend', items: ['React 18 + TypeScript', 'Vite + Tailwind CSS', 'TanStack Query', 'Recharts'] },
  ]

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 animate-fade-in">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-slate-100 mb-3">About VeriFact AI</h1>
        <p className="text-slate-400 leading-relaxed max-w-2xl">
          A final-year B.Tech CSE project — a production-oriented fake news detection system
          using NLP and transformer-based models. Built to demonstrate real ML pipelines,
          evidence retrieval, and explainability — not UI mockups.
        </p>
      </div>

      {/* System status */}
      {health && (
        <div className="glass-card p-5 mb-8">
          <h2 className="section-title mb-4">System Status</h2>
          <div className="grid sm:grid-cols-3 gap-4">
            {Object.entries(health.components).map(([name, comp]) => (
              <div key={name} className="flex items-center gap-3">
                <div className={`w-2 h-2 rounded-full ${
                  comp.status === 'ok' ? 'bg-emerald-400' :
                  comp.status === 'degraded' ? 'bg-amber-400' : 'bg-red-400'
                }`} />
                <div>
                  <div className="text-sm text-slate-300 capitalize">{name.replace('_', ' ')}</div>
                  <div className="text-xs text-slate-600">
                    {comp.status}{comp.latency_ms ? ` · ${comp.latency_ms.toFixed(1)}ms` : ''}
                  </div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-slate-800 text-xs text-slate-600">
            API v{health.version} · {health.environment}
          </div>
        </div>
      )}

      {/* Tech stack */}
      <div className="glass-card p-6 mb-8">
        <h2 className="section-title mb-5">Technology Stack</h2>
        <div className="grid sm:grid-cols-2 gap-6">
          {stack.map((s) => (
            <div key={s.layer}>
              <div className="flex items-center gap-2 mb-2">
                {s.icon}
                <span className="text-sm font-semibold text-slate-300">{s.layer}</span>
              </div>
              <ul className="space-y-1">
                {s.items.map((item) => (
                  <li key={item} className="text-xs text-slate-500 flex items-center gap-2">
                    <span className="w-1 h-1 rounded-full bg-slate-600 shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="glass-card p-5 border-amber-500/20">
        <h2 className="text-sm font-semibold text-amber-400 mb-2">Important Disclaimer</h2>
        <p className="text-xs text-slate-400 leading-relaxed">
          Model predictions are probabilistic estimates based on statistical patterns learned
          during training — they are <strong>not factual determinations</strong>. Confidence scores
          reflect pattern matching, not ground truth. Evidence assessments are based on retrieved
          sources and are clearly distinguished from ML predictions. Absence of evidence is never
          interpreted as proof of falsehood.
        </p>
      </div>
    </div>
  )
}
