import { Link } from 'react-router-dom'
import { Shield, Search, Zap, BookOpen, ArrowRight, CheckCircle, AlertTriangle, HelpCircle } from 'lucide-react'
import { useStats, useHealth } from '@/hooks/useApi'
import { Spinner } from '@/components/ui/Spinner'
import { cn } from '@/lib/utils'

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="glass-card p-5 text-center">
      <div className="text-3xl font-bold text-slate-100 tracking-tight">{value}</div>
      <div className="text-sm text-slate-400 mt-1">{label}</div>
      {sub && <div className="text-xs text-slate-600 mt-0.5">{sub}</div>}
    </div>
  )
}

const features = [
  {
    icon: <Zap className="w-5 h-5 text-brand-400" />,
    title: 'Multiple ML Models',
    desc: 'TF-IDF + Logistic Regression, SVM, Naive Bayes, and fine-tuned DistilBERT run concurrently on every submission.',
  },
  {
    icon: <Search className="w-5 h-5 text-emerald-400" />,
    title: 'Real Evidence Retrieval',
    desc: 'NewsAPI, GNews, RSS feeds, and web search are queried for corroborating or contradicting sources.',
  },
  {
    icon: <BookOpen className="w-5 h-5 text-violet-400" />,
    title: 'Explainable AI',
    desc: 'LIME and attention attribution show which words drove the model — clearly distinguished from factual evidence.',
  },
  {
    icon: <Shield className="w-5 h-5 text-amber-400" />,
    title: 'Transparent Uncertainty',
    desc: 'ML confidence and evidence assessments are reported separately. Absence of evidence is never interpreted as falsehood.',
  },
]

const verdictExamples = [
  { icon: <CheckCircle className="w-4 h-4" />, label: 'Likely Credible', color: 'text-emerald-400', desc: 'Multiple corroborating sources, no contradictions' },
  { icon: <AlertTriangle className="w-4 h-4" />, label: 'Contradicted', color: 'text-red-400', desc: 'Independent sources dispute the claim' },
  { icon: <AlertTriangle className="w-4 h-4" />, label: 'Likely Misleading', color: 'text-orange-400', desc: 'High fake-probability + contradicting evidence' },
  { icon: <HelpCircle className="w-4 h-4" />, label: 'Insufficient Evidence', color: 'text-slate-400', desc: 'Not enough sources to make an assessment' },
]

export default function Home() {
  const { data: stats, isLoading: statsLoading } = useStats()
  const { data: health } = useHealth()
  const isOnline = health?.status === 'ok' || health?.status === 'degraded'

  return (
    <div className="animate-fade-in">
      {/* Hero */}
      <section className="relative overflow-hidden py-24 px-4">
        {/* Background glow */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-brand-600/10 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-4xl mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/20 text-xs text-brand-400 font-medium mb-8">
            <span className={cn('w-1.5 h-1.5 rounded-full', isOnline ? 'bg-emerald-400' : 'bg-slate-600')} />
            {isOnline ? 'API Online · Real predictions · Real evidence' : 'Connecting to API…'}
          </div>

          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight text-slate-100 leading-tight mb-6">
            Detect Misinformation<br />
            <span className="gradient-text">With Explainable AI</span>
          </h1>

          <p className="text-lg text-slate-400 max-w-2xl mx-auto leading-relaxed mb-10">
            Submit any news article, URL, or claim. Get real ML predictions,
            evidence-backed assessments, and transparent explanations — not black-box verdicts.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/analyze" className="btn-primary text-base px-7 py-3">
              Start Analyzing
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link to="/methodology" className="btn-secondary text-base px-7 py-3">
              How It Works
            </Link>
          </div>
        </div>
      </section>

      {/* Live stats */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-16">
          {statsLoading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="glass-card p-5 flex items-center justify-center h-24">
                <Spinner size="sm" />
              </div>
            ))
          ) : stats ? (
            <>
              <StatCard label="Total Analyses" value={stats.total_analyses.toLocaleString()} />
              <StatCard
                label="Fake Detected"
                value={`${stats.fake_percentage.toFixed(1)}%`}
                sub={`${stats.verdict_counts.FAKE.toLocaleString()} articles`}
              />
              <StatCard
                label="Avg Confidence"
                value={`${(stats.avg_confidence * 100).toFixed(1)}%`}
              />
              <StatCard
                label="Avg Processing"
                value={`${(stats.avg_processing_ms / 1000).toFixed(1)}s`}
              />
            </>
          ) : (
            <div className="col-span-4 glass-card p-6 text-center text-sm text-slate-500">
              Stats unavailable — backend may be offline.
            </div>
          )}
        </div>

        {/* Features */}
        <div className="mb-16">
          <h2 className="text-2xl font-bold text-center text-slate-100 mb-2">How It Works</h2>
          <p className="text-center text-slate-500 text-sm mb-10">
            Four independent signals, assembled into one transparent report.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
            {features.map((f) => (
              <div key={f.title} className="glass-card p-5 hover:border-slate-700/80 transition-colors">
                <div className="p-2.5 rounded-lg bg-slate-800 w-fit mb-4">{f.icon}</div>
                <h3 className="text-sm font-semibold text-slate-200 mb-2">{f.title}</h3>
                <p className="text-xs text-slate-500 leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Verdict explanations */}
        <div className="glass-card p-6">
          <h2 className="section-title mb-1">Understanding Verdicts</h2>
          <p className="text-xs text-slate-500 mb-5">
            ML predictions and evidence assessments are always reported separately.
          </p>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {verdictExamples.map((v) => (
              <div key={v.label} className="flex flex-col gap-1.5">
                <div className={cn('flex items-center gap-2 font-medium text-sm', v.color)}>
                  {v.icon}
                  {v.label}
                </div>
                <p className="text-xs text-slate-500 leading-relaxed">{v.desc}</p>
              </div>
            ))}
          </div>
          <div className="mt-5 pt-4 border-t border-slate-800 text-xs text-slate-600">
            ⚠ Absence of evidence is never interpreted as proof of falsehood.
          </div>
        </div>
      </section>
    </div>
  )
}
