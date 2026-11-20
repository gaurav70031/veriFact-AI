/**
 * Explainable AI Studio
 *
 * Live interactive XAI page with:
 *   1. Live analysis selector — pick any completed analysis from history
 *   2. Highlighted article text — inline colour heatmap
 *   3. Top-10 influential words bar chart
 *   4. Positive vs. negative contribution chart per model
 *   5. Per-model token heatmaps (LIME, SHAP, Attention, IG)
 *   6. Cross-model token comparison
 *   7. Report download
 *   8. Method documentation cards
 *
 * All data from live API. No fabricated numbers.
 */

import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Lightbulb, AlertTriangle, Info, ArrowRight,
  Zap, Brain, BarChart3, Layers, ChevronDown,
  BookOpen, Download, ExternalLink, FlaskConical,
} from 'lucide-react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip, ReferenceLine, Legend,
} from 'recharts'
import {
  useHistory, useExplanation, useAnnotatedText,
  useExplanationSummary,
} from '@/hooks/useApi'
import { getExplanationReportUrl } from '@/lib/api'
import { Spinner } from '@/components/ui/Spinner'
import { cn } from '@/lib/utils'
import type {
  AnalysisListItem, PerModelExplanation,
  AnnotatedWord, ExplanationSummary, CrossModelToken,
} from '@/types/api'

// ── Tooltip style ─────────────────────────────────────────────────────────────
const TT = {
  contentStyle: {
    backgroundColor: 'var(--bg-surface-2)',
    border: '1px solid var(--border)',
    borderRadius: 8,
  },
  labelStyle: { color: 'var(--text-primary)', fontWeight: 600 },
  itemStyle:  { color: 'var(--text-secondary)' },
}

// ── Method documentation ──────────────────────────────────────────────────────
const XAI_METHODS = [
  {
    icon: <Zap className="w-6 h-6" />,
    color: '#818cf8',
    name: 'LIME',
    fullName: 'Local Interpretable Model-Agnostic Explanations',
    models: ['TF-IDF + LR', 'Linear SVM', 'Naive Bayes'],
    desc: 'LIME perturbs the input text (1,000 samples) and trains a local linear model to approximate which tokens most influenced the prediction.',
    pros: ['Model-agnostic — works with any classifier', 'Token-level granularity', 'Interpretable signed weights'],
    cons: ['Stochastic — varies slightly between runs', 'Computationally expensive (~5–15s)', 'Local approximation only'],
    output: 'Token weights in [−1, +1]; positive → toward FAKE, negative → toward REAL',
    citation: 'Ribeiro et al. (2016). "Why Should I Trust You?" SIGKDD.',
  },
  {
    icon: <FlaskConical className="w-6 h-6" />,
    color: '#34d399',
    name: 'SHAP',
    fullName: 'SHapley Additive exPlanations',
    models: ['TF-IDF + LR (LinearExplainer)', 'Linear SVM, Naive Bayes (KernelExplainer)'],
    desc: 'SHAP assigns each feature a Shapley value — its average marginal contribution across all feature subsets. LinearExplainer is O(d) for linear models; KernelExplainer uses sampling for black-box models.',
    pros: ['Satisfies Efficiency, Symmetry, Null-player axioms', 'Exact for linear models', 'Theoretically grounded'],
    cons: ['KernelExplainer is O(n_samples × d) — slow for large vocab', 'Values relative to a chosen baseline', 'Approximate for non-linear models'],
    output: 'Shapley values per feature — positive = supports FAKE relative to baseline',
    citation: 'Lundberg & Lee (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.',
  },
  {
    icon: <Brain className="w-6 h-6" />,
    color: '#a78bfa',
    name: 'Attention Rollout',
    fullName: 'Transformer Self-Attention Attribution',
    models: ['DistilBERT (fine-tuned)'],
    desc: 'Attention rollout propagates attention weights from the [CLS] token through the last transformer layer, aggregating sub-word tokens into whole-word contributions.',
    pros: ['Single forward pass — fast', 'No external library required', 'Native to transformer'],
    cons: ['Attention ≠ importance (Jain & Wallace, 2019)', 'Last-layer only', 'May highlight syntactic rather than semantic tokens'],
    output: 'L1-normalised CLS attention scores; signed by predicted label',
    citation: 'Abnar & Zuidema (2020). "Quantifying Attention Flow in Transformers." ACL.',
  },
  {
    icon: <BarChart3 className="w-6 h-6" />,
    color: '#fbbf24',
    name: 'Integrated Gradients',
    fullName: 'Gradient-Based Attribution for DistilBERT',
    models: ['DistilBERT (fine-tuned)'],
    desc: 'IG interpolates N steps between a zero embedding (baseline) and the actual input, computing gradients at each step. The integral approximates each token\'s contribution to the predicted class.',
    pros: ['Satisfies Implementation Invariance', 'Gradient-based — more faithful than attention', 'Well-studied theoretical foundation'],
    cons: ['N forward passes (default 20) — 10–20x slower than attention', 'Sensitive to baseline choice', 'Sum property only holds approximately'],
    output: 'Token attribution scores; positive = supports predicted label',
    citation: 'Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks." ICML.',
  },
  {
    icon: <Layers className="w-6 h-6" />,
    color: '#60a5fa',
    name: 'TF-IDF Weights',
    fullName: 'Feature Weight Fallback',
    models: ['All TF-IDF models (when LIME/SHAP unavailable)'],
    desc: 'When LIME and SHAP cannot run, the top features from the learned TF-IDF weight vector are shown. This is a global explanation — the same for all inputs containing a given word.',
    pros: ['Always available', 'Deterministic', 'No additional computation'],
    cons: ['Global, not local', 'Reflects training distribution, not specific input', 'All weights have the same sign'],
    output: 'TF-IDF scores signed by predicted label — not Shapley values',
    citation: 'Fallback method — no specific citation.',
  },
]

// ── Analysis selector ─────────────────────────────────────────────────────────
function AnalysisSelector({
  items,
  selectedId,
  onSelect,
}: {
  items: AnalysisListItem[]
  selectedId: number | null
  onSelect: (id: number) => void
}) {
  if (!items.length) return null
  return (
    <div className="card p-5">
      <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>
        Select Analysis to Explain
      </h3>
      <div className="space-y-1.5 max-h-64 overflow-y-auto">
        {items.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelect(item.id)}
            className={cn(
              'w-full text-left px-3 py-2.5 rounded-xl text-sm transition-all',
              selectedId === item.id ? 'ring-1' : 'hover:bg-surface-2',
            )}
            style={selectedId === item.id
              ? { backgroundColor: 'var(--brand-light)', outline: '1px solid var(--brand)' }
              : { color: 'var(--text-secondary)' }
            }
          >
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono shrink-0" style={{ color: 'var(--text-faint)' }}>#{item.id}</span>
              <span className="truncate flex-1" style={{ color: selectedId === item.id ? 'var(--brand)' : 'var(--text-secondary)' }}>
                {item.article_title ?? item.original_input.slice(0, 80)}
              </span>
              {item.final_verdict && (
                <span className={cn(
                  'text-xs font-semibold shrink-0',
                  item.final_verdict === 'FAKE' ? 'text-red-400' : 'text-emerald-400'
                )}>
                  {item.final_verdict}
                </span>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

// ── Token heatmap (inline) ────────────────────────────────────────────────────
function InlineHeatmap({ tokens }: { tokens: Array<{ token: string; weight: number }> }) {
  if (!tokens.length) return null
  const maxAbs = Math.max(...tokens.map((t) => Math.abs(t.weight)), 0.01)

  return (
    <div className="flex flex-wrap gap-2">
      {tokens.map((tw, i) => {
        const intensity = Math.abs(tw.weight) / maxAbs
        const isFake    = tw.weight > 0
        const base      = isFake ? '248,113,113' : '52,211,153'
        const textClr   = isFake
          ? `rgba(252,165,165,${0.6 + intensity * 0.4})`
          : `rgba(110,231,183,${0.6 + intensity * 0.4})`
        return (
          <span
            key={i}
            className="px-2.5 py-1 rounded-md text-xs font-mono border cursor-help select-none"
            title={`${tw.token}: ${tw.weight > 0 ? '+' : ''}${tw.weight.toFixed(3)}`}
            style={{
              color:           textClr,
              borderColor:     `rgba(${base},${0.1 + intensity * 0.3})`,
              backgroundColor: `rgba(${base},${0.05 + intensity * 0.18})`,
            }}
          >
            {tw.token}
          </span>
        )
      })}
    </div>
  )
}

// ── Live analysis panel ───────────────────────────────────────────────────────
function LiveAnalysisPanel({ analysisId }: { analysisId: number }) {
  const { data: expData,   isLoading: expL  } = useExplanation(analysisId)
  const { data: annData,   isLoading: annL  } = useAnnotatedText(analysisId)
  const { data: summData,  isLoading: sumL  } = useExplanationSummary(analysisId)
  const [showText, setShowText] = useState(false)

  if (expL) return <div className="flex justify-center py-16"><Spinner /></div>
  if (!expData) return null

  const available = expData.model_explanations.filter((e) => e.method !== 'unavailable')

  // Aggregate tokens
  const tokenMap: Record<string, number[]> = {}
  for (const exp of available) {
    for (const t of exp.top_tokens) {
      if (!tokenMap[t.token]) tokenMap[t.token] = []
      tokenMap[t.token].push(t.weight)
    }
  }
  const aggTokens = Object.entries(tokenMap)
    .map(([token, ws]) => ({ token, weight: ws.reduce((a, b) => a + b, 0) / ws.length }))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))

  const top10 = aggTokens.slice(0, 10).map((t) => ({
    ...t,
    fake: t.weight > 0 ? t.weight : 0,
    real: t.weight < 0 ? Math.abs(t.weight) : 0,
  }))

  // Contribution data
  const contribData = available
    .filter((e) => e.top_tokens.length > 0)
    .map((e) => ({
      model:      e.model_id.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
      fakeSignal: Math.round(e.top_tokens.filter((t) => t.weight > 0).reduce((s, t) => s + t.weight, 0) * 100) / 100,
      realSignal: -Math.round(e.top_tokens.filter((t) => t.weight < 0).reduce((s, t) => s + Math.abs(t.weight), 0) * 100) / 100,
    }))

  const MODEL_COLORS = ['#818cf8', '#34d399', '#f87171', '#fbbf24', '#a78bfa']

  return (
    <div className="space-y-5">
      {/* Signal-vs-evidence warning */}
      <div className="card p-4 flex items-start gap-3" style={{ borderColor: 'rgba(251,191,36,0.2)', backgroundColor: 'rgba(251,191,36,0.03)' }}>
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <p className="text-xs text-amber-300/80 leading-relaxed">
          <strong className="text-amber-300">Model signal — not factual evidence.</strong>{' '}
          {expData.signal_vs_evidence_warning}
        </p>
      </div>

      {/* Verdict summary */}
      <div className="card p-5 flex items-center gap-4">
        <div>
          <div className="text-xs font-semibold mb-0.5" style={{ color: 'var(--text-subtle)' }}>ML Verdict</div>
          <div className={cn('text-2xl font-bold', expData.ml_verdict === 'FAKE' ? 'text-red-400' : expData.ml_verdict === 'REAL' ? 'text-emerald-400' : 'text-amber-400')}>
            {expData.ml_verdict}
          </div>
        </div>
        {expData.ml_confidence !== null && expData.ml_confidence !== undefined && (
          <div>
            <div className="text-xs font-semibold mb-0.5" style={{ color: 'var(--text-subtle)' }}>Confidence</div>
            <div className="text-2xl font-bold font-mono" style={{ color: 'var(--brand)' }}>
              {(expData.ml_confidence * 100).toFixed(1)}%
            </div>
          </div>
        )}
        <div className="ml-auto flex gap-2">
          <a
            href={getExplanationReportUrl(analysisId)}
            download={`xai_report_${analysisId}.json`}
            className="btn-secondary text-xs"
          >
            <Download className="w-3.5 h-3.5" /> Report JSON
          </a>
          <Link to={`/results/${analysisId}`} className="btn-primary text-xs">
            Full Analysis <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>

      {/* Highlighted article text */}
      {!annL && annData?.annotated?.length > 0 && (
        <div className="card p-5">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="section-title">Highlighted Article Text</h3>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>
                Words coloured by aggregate explanation weight
              </p>
            </div>
            <button onClick={() => setShowText(!showText)} className="btn-ghost text-xs">
              {showText ? 'Hide' : 'Show text'}
            </button>
          </div>
          <div className="flex gap-5 text-xs mb-3" style={{ color: 'var(--text-faint)' }}>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-sm bg-red-500/25 border border-red-500/30" />
              Fake-pattern
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-sm bg-emerald-500/25 border border-emerald-500/30" />
              Real-pattern
            </span>
          </div>
          {showText && (
            <div
              className="text-sm font-mono leading-7 rounded-xl p-4 overflow-y-auto max-h-64"
              style={{ backgroundColor: 'var(--bg-surface-2)', lineHeight: '2rem' }}
            >
              {(annData.annotated as AnnotatedWord[]).map((w, i) => {
                if (w.is_space) return <span key={i}>{w.word}</span>
                const isFake = w.weight > 0.05
                const isReal = w.weight < -0.05
                const abs = Math.abs(w.weight)
                if (!isFake && !isReal) return <span key={i} style={{ color: 'var(--text-secondary)' }}>{w.word}</span>
                const base = isFake ? '248,113,113' : '52,211,153'
                return (
                  <span
                    key={i}
                    title={`Weight: ${w.weight.toFixed(3)}`}
                    style={{
                      color: isFake ? `rgba(252,165,165,${0.6 + abs * 0.4})` : `rgba(110,231,183,${0.6 + abs * 0.4})`,
                      backgroundColor: `rgba(${base},${0.05 + abs * 0.18})`,
                      borderRadius: 3,
                      padding: '0 1px',
                    }}
                  >
                    {w.word}
                  </span>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* Top-10 influential words */}
      {top10.length > 0 && (
        <div className="card p-5">
          <h3 className="section-title mb-1">Top 10 Influential Words</h3>
          <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>
            Aggregate importance across all models — positive = fake-pattern, negative = real-pattern
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={top10} layout="vertical" margin={{ left: 70, right: 30, top: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[-1, 1]} tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="token" tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'monospace' }} tickLine={false} axisLine={false} width={70} />
              <Tooltip {...TT} formatter={(v: number) => [`${v > 0 ? '+' : ''}${v.toFixed(3)}`]} />
              <ReferenceLine x={0} stroke="var(--border)" />
              <Bar dataKey="weight" radius={[0, 3, 3, 0]}>
                {top10.map((entry, i) => (
                  <Cell key={i} fill={entry.weight > 0 ? '#f87171' : '#34d399'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Positive vs negative contributions */}
      {contribData.length > 0 && (
        <div className="card p-5">
          <h3 className="section-title mb-1">Positive vs. Negative Contributions</h3>
          <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>
            Sum of fake-pattern (red) and real-pattern (green) token weights per model
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={contribData} margin={{ top: 4, right: 10, bottom: 20, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="model" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
              <ReferenceLine y={0} stroke="var(--border)" />
              <Tooltip {...TT} formatter={(v: number) => [v.toFixed(3)]} />
              <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{v}</span>} />
              <Bar dataKey="fakeSignal" fill="#f87171" name="Fake signal" radius={[3, 3, 0, 0]} />
              <Bar dataKey="realSignal" fill="#34d399" name="Real signal" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Per-model heatmaps */}
      {available.length > 0 && (
        <div className="card p-5">
          <h3 className="section-title mb-4">Per-Model Word Attributions</h3>
          <div className="space-y-5">
            {available.map((exp) => (
              <div key={exp.model_id}>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                    {exp.model_name}
                  </span>
                  <span className={cn(
                    'text-xs px-2 py-0.5 rounded border border-current/20 font-medium',
                    exp.method.startsWith('shap') ? 'text-emerald-400' :
                    exp.method === 'lime' ? 'text-brand-400' :
                    exp.method === 'integrated_gradients' ? 'text-amber-400' :
                    exp.method === 'attention' ? 'text-violet-400' :
                    'text-sky-400'
                  )}>
                    {exp.method.replace(/_/g, ' ').toUpperCase()}
                  </span>
                  <span className={cn('text-xs font-semibold', exp.label === 'FAKE' ? 'text-red-400' : 'text-emerald-400')}>
                    {exp.label}
                  </span>
                </div>
                <InlineHeatmap tokens={exp.top_tokens} />
                {exp.plain_text && (
                  <p className="text-xs mt-2 leading-relaxed" style={{ color: 'var(--text-subtle)' }}>
                    {exp.plain_text.slice(0, 200)}
                    {exp.plain_text.length > 200 && '…'}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Cross-model comparison */}
      {!sumL && summData?.tokens?.length > 0 && summData.models?.length >= 2 && (
        <div className="card p-5">
          <h3 className="section-title mb-1">Explanation Comparison Across Models</h3>
          <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>
            Agreement between models = consistent signal · Disagreement = uncertain attribution
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={(summData.tokens as CrossModelToken[]).slice(0, 10).map((t) => ({
                token: t.token,
                ...Object.fromEntries(
                  (summData.models as string[]).map((m) => [
                    m.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()),
                    Math.round((t.weights[m] ?? 0) * 1000) / 1000,
                  ])
                ),
              }))}
              layout="vertical"
              margin={{ left: 70, right: 10, top: 4, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[-1, 1]} tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="token" tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'monospace' }} tickLine={false} axisLine={false} width={70} />
              <ReferenceLine x={0} stroke="var(--border)" />
              <Tooltip {...TT} formatter={(v: number) => [`${v > 0 ? '+' : ''}${v.toFixed(3)}`]} />
              <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{v}</span>} />
              {(summData.models as string[]).map((m: string, idx: number) => (
                <Bar
                  key={m}
                  dataKey={m.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())}
                  fill={MODEL_COLORS[idx % MODEL_COLORS.length]}
                  radius={[0, 2, 2, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

const MODEL_COLORS = ['#818cf8', '#34d399', '#f87171', '#fbbf24', '#a78bfa']

// ── Main page ─────────────────────────────────────────────────────────────────
export default function ExplainableAI() {
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [showMethods, setShowMethods] = useState(false)

  const { data: history, isLoading: histLoading } = useHistory({ page: 1, page_size: 20 })
  const completedItems = history?.items.filter((i) => i.status === 'completed') ?? []

  // Auto-select first completed item if none selected
  if (!selectedId && completedItems.length > 0 && !histLoading) {
    setSelectedId(completedItems[0].id)
  }

  return (
    <div className="page-content">
      {/* Header */}
      <div className="flex items-start gap-4 mb-8">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center shrink-0">
          <Lightbulb className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="page-title mb-1">Explainable AI Studio</h1>
          <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
            Live token-level explanations for real model predictions
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left: selector + method docs */}
        <div className="lg:col-span-1 space-y-5">
          {histLoading ? (
            <div className="card p-6 flex justify-center"><Spinner /></div>
          ) : completedItems.length > 0 ? (
            <AnalysisSelector
              items={completedItems}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          ) : (
            <div className="card p-6 text-center space-y-3">
              <Lightbulb className="w-8 h-8 mx-auto opacity-20" style={{ color: 'var(--text-faint)' }} />
              <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
                No completed analyses yet.
              </p>
              <Link to="/analyze/text" className="btn-primary text-xs justify-center">
                Start an Analysis <ArrowRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          )}

          {/* Method docs toggle */}
          <button
            onClick={() => setShowMethods(!showMethods)}
            className="w-full flex items-center justify-between px-5 py-3.5 rounded-xl transition-colors"
            style={{ backgroundColor: 'var(--bg-surface)', border: '1px solid var(--border)', color: 'var(--text-secondary)' }}
          >
            <span className="flex items-center gap-2 text-sm font-medium">
              <BookOpen className="w-4 h-4" />
              Method Documentation
            </span>
            <ChevronDown className={cn('w-4 h-4 transition-transform', showMethods && 'rotate-180')} />
          </button>

          {showMethods && (
            <div className="space-y-3">
              {XAI_METHODS.map((method) => (
                <div key={method.name} className="card p-4">
                  <div className="flex items-start gap-3 mb-3">
                    <div
                      className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0"
                      style={{ backgroundColor: `${method.color}15`, color: method.color }}
                    >
                      {method.icon}
                    </div>
                    <div>
                      <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{method.name}</div>
                      <div className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>{method.fullName}</div>
                    </div>
                  </div>
                  <p className="text-xs mb-3 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{method.desc}</p>
                  <div className="grid grid-cols-2 gap-3 mb-2">
                    <div>
                      <div className="text-xs font-semibold text-emerald-400 mb-1">Advantages</div>
                      {method.pros.map((p) => <div key={p} className="text-xs flex items-start gap-1" style={{ color: 'var(--text-secondary)' }}><span className="text-emerald-400 shrink-0">+</span>{p}</div>)}
                    </div>
                    <div>
                      <div className="text-xs font-semibold text-red-400 mb-1">Limitations</div>
                      {method.cons.map((c) => <div key={c} className="text-xs flex items-start gap-1" style={{ color: 'var(--text-secondary)' }}><span className="text-red-400 shrink-0">−</span>{c}</div>)}
                    </div>
                  </div>
                  <div className="text-xs rounded px-2 py-1.5" style={{ backgroundColor: 'var(--bg-surface-2)', color: 'var(--text-subtle)' }}>
                    <strong style={{ color: 'var(--text-secondary)' }}>Output: </strong>{method.output}
                  </div>
                  <div className="text-xs mt-1.5 italic" style={{ color: 'var(--text-faint)' }}>{method.citation}</div>
                </div>
              ))}
            </div>
          )}

          {/* Limitations card */}
          <div className="card p-5" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
            <div className="flex items-center gap-2 mb-3">
              <Info className="w-4 h-4" style={{ color: 'var(--text-subtle)' }} />
              <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Important Limitations</h3>
            </div>
            <ul className="space-y-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
              {[
                'Token weights are statistical associations from training data — not factual determinations.',
                'High weight ≠ the word is factually wrong or the article is false.',
                'LIME and SHAP are approximations — results may vary slightly.',
                'Attention weights show correlation, not causation.',
                'Models trained on 2015–2018 US news may not generalise.',
              ].map((l, i) => (
                <li key={i} className="flex items-start gap-1.5">
                  <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0 mt-0.5" />
                  {l}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Right: live analysis panel */}
        <div className="lg:col-span-2">
          {selectedId ? (
            <LiveAnalysisPanel key={selectedId} analysisId={selectedId} />
          ) : (
            <div className="card p-16 text-center">
              <Lightbulb className="w-12 h-12 mx-auto mb-4 opacity-20" style={{ color: 'var(--text-faint)' }} />
              <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
                Select a completed analysis from the left panel to view explanations.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
