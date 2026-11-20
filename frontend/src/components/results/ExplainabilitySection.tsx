/**
 * ExplainabilitySection
 *
 * Renders the complete XAI panel for a completed analysis:
 *
 * 1. Signal-vs-evidence warning (always first)
 * 2. Highlighted article text (inline colour heatmap)
 * 3. Top-10 influential words bar chart (pos + neg contributions)
 * 4. Positive vs. negative contribution chart (per model)
 * 5. Per-model token heatmaps with method badges
 * 6. Cross-model token comparison chart
 * 7. Download report button
 *
 * Data sources:
 *   GET /api/v1/explanation/{id}            → per-model attributions
 *   GET /api/v1/explanation/{id}/annotated  → word-level inline weights
 *   GET /api/v1/explanation/{id}/summary    → cross-model token table
 *
 * Critical: signal-vs-evidence disclaimer is never omitted.
 */

import { useState } from 'react'
import {
  AlertTriangle, Info, ChevronDown, ChevronRight, Cpu,
  Download, Eye, EyeOff, BarChart3, Layers,
} from 'lucide-react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid,
  ResponsiveContainer, Tooltip, Legend, ReferenceLine,
} from 'recharts'
import { cn } from '@/lib/utils'
import { useExplanation, useAnnotatedText, useExplanationSummary } from '@/hooks/useApi'
import { getExplanationReportUrl } from '@/lib/api'
import { Spinner } from '@/components/ui/Spinner'
import type {
  PerModelExplanation, AnnotatedWord, CrossModelToken, ExplanationSummary,
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

// ── Method metadata ───────────────────────────────────────────────────────────
const METHOD_META: Record<string, { label: string; color: string; desc: string }> = {
  lime:                  { label: 'LIME',          color: 'text-brand-400',  desc: 'Local Interpretable Model-Agnostic Explanations (perturbation-based)' },
  shap_linear:           { label: 'SHAP Linear',   color: 'text-emerald-400', desc: 'SHAP LinearExplainer — exact Shapley values for linear models' },
  shap_kernel:           { label: 'SHAP Kernel',   color: 'text-teal-400',   desc: 'SHAP KernelExplainer — model-agnostic Shapley approximation' },
  attention:             { label: 'Attention',     color: 'text-violet-400', desc: 'Transformer last-layer CLS attention rollout' },
  integrated_gradients:  { label: 'IG',            color: 'text-amber-400',  desc: 'Integrated Gradients — gradient-based attribution' },
  tfidf_weights:         { label: 'TF-IDF',        color: 'text-sky-400',    desc: 'Raw TF-IDF feature importance (fallback)' },
  unavailable:           { label: 'N/A',           color: 'text-slate-600',  desc: 'Explanation not computed' },
}

function MethodBadge({ method }: { method: string }) {
  const meta = METHOD_META[method] ?? METHOD_META.unavailable
  return (
    <span
      className={cn('text-xs font-medium px-2 py-0.5 rounded border border-current/20 bg-current/8', meta.color)}
      title={meta.desc}
    >
      {meta.label}
    </span>
  )
}

// ── Token heatmap (top tokens) ────────────────────────────────────────────────
function TokenHeatmap({ tokens }: { tokens: PerModelExplanation['top_tokens'] }) {
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
            title={`${tw.token}: ${tw.weight > 0 ? '+' : ''}${tw.weight.toFixed(3)} → ${isFake ? 'FAKE pattern' : 'REAL pattern'}`}
            className="px-2.5 py-1 rounded-md text-xs font-mono border cursor-help select-none"
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

// ── Inline annotated article text ─────────────────────────────────────────────
function AnnotatedText({
  words,
  visible,
  onToggle,
}: {
  words: AnnotatedWord[]
  visible: boolean
  onToggle: () => void
}) {
  const hasWeights = words.some((w) => !w.is_space && Math.abs(w.weight) > 0.05)
  if (!words.length) return null

  const maxAbs = Math.max(...words.filter((w) => !w.is_space).map((w) => Math.abs(w.weight)), 0.01)

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Highlighted Article Text
          </h3>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>
            Words coloured by aggregate explanation weight across all models
          </p>
        </div>
        <button
          onClick={onToggle}
          className="btn-ghost text-xs"
        >
          {visible ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
          {visible ? 'Hide' : 'Show'}
        </button>
      </div>

      {!hasWeights && (
        <p className="text-xs mb-3" style={{ color: 'var(--text-subtle)' }}>
          No significant word weights available for this analysis.
        </p>
      )}

      {/* Legend */}
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

      {visible && (
        <div
          className="text-sm leading-7 font-mono rounded-xl p-4"
          style={{ backgroundColor: 'var(--bg-surface-2)', lineHeight: '2rem' }}
        >
          {words.map((w, i) => {
            if (w.is_space) return <span key={i}>{w.word}</span>
            const abs = Math.abs(w.weight) / maxAbs
            const isFake = w.weight > 0.05
            const isReal = w.weight < -0.05
            const base   = isFake ? '248,113,113' : '52,211,153'
            const style: React.CSSProperties = abs > 0.1 ? {
              backgroundColor: `rgba(${base},${0.05 + abs * 0.2})`,
              borderRadius: 4,
              color: isFake
                ? `rgba(252,165,165,${0.6 + abs * 0.4})`
                : isReal
                ? `rgba(110,231,183,${0.6 + abs * 0.4})`
                : 'var(--text-secondary)',
              padding: '0 2px',
            } : { color: 'var(--text-secondary)' }

            return (
              <span
                key={i}
                style={style}
                title={abs > 0.1 ? `Weight: ${w.weight.toFixed(3)}` : undefined}
              >
                {w.word}
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Top-10 words bar chart ────────────────────────────────────────────────────
function Top10Chart({ tokens }: { tokens: Array<{ token: string; weight: number }> }) {
  if (!tokens.length) return null

  const top10 = [...tokens]
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, 10)
    .map((t) => ({
      token:  t.token,
      weight: t.weight,
      fake:   t.weight > 0 ? t.weight : 0,
      real:   t.weight < 0 ? Math.abs(t.weight) : 0,
    }))

  return (
    <div className="card p-5">
      <h3 className="section-title mb-1">Top 10 Influential Words</h3>
      <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>
        Positive = fake-pattern association · Negative = real-pattern association (aggregate across models)
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart
          data={top10}
          layout="vertical"
          margin={{ left: 70, right: 30, top: 4, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            domain={[-1, 1]}
            tick={{ fontSize: 10, fill: 'var(--text-faint)' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => v.toFixed(1)}
          />
          <YAxis
            type="category"
            dataKey="token"
            tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            width={70}
          />
          <Tooltip
            {...TT}
            formatter={(v: number, name: string) => [
              `${v > 0 ? '+' : ''}${v.toFixed(3)}`,
              name === 'fake' ? 'Fake association' : 'Real association',
            ]}
          />
          <ReferenceLine x={0} stroke="var(--border)" />
          <Bar dataKey="weight" radius={[0, 3, 3, 0]}>
            {top10.map((entry, i) => (
              <Cell key={i} fill={entry.weight > 0 ? '#f87171' : '#34d399'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Pos/neg contribution chart (per model) ────────────────────────────────────
function ContributionChart({ explanations }: { explanations: PerModelExplanation[] }) {
  const data = explanations
    .filter((e) => e.method !== 'unavailable' && e.top_tokens.length > 0)
    .map((e) => {
      const fakeSignal = e.top_tokens.filter((t) => t.weight > 0).reduce((s, t) => s + t.weight, 0)
      const realSignal = e.top_tokens.filter((t) => t.weight < 0).reduce((s, t) => s + Math.abs(t.weight), 0)
      return {
        model:       e.model_id.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
        fakeSignal:  Math.round(fakeSignal * 100) / 100,
        realSignal:  -Math.round(realSignal * 100) / 100,  // negative for chart axis
        label:       e.label,
      }
    })

  if (!data.length) return null

  return (
    <div className="card p-5">
      <h3 className="section-title mb-1">Positive vs. Negative Contributions</h3>
      <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>
        Sum of fake-pattern weights (red) and real-pattern weights (green) per model
      </p>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 4, right: 10, bottom: 20, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="model"
            tick={{ fontSize: 10, fill: 'var(--text-secondary)' }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: 'var(--text-faint)' }}
            tickLine={false}
            axisLine={false}
          />
          <ReferenceLine y={0} stroke="var(--border)" />
          <Tooltip {...TT} formatter={(v: number) => [v.toFixed(3), v > 0 ? 'Fake signal' : 'Real signal']} />
          <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{v}</span>} />
          <Bar dataKey="fakeSignal" fill="#f87171" name="Fake signal" radius={[3, 3, 0, 0]} />
          <Bar dataKey="realSignal" fill="#34d399" name="Real signal" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Cross-model comparison chart ──────────────────────────────────────────────
function CrossModelChart({ summary }: { summary: ExplanationSummary }) {
  if (!summary.tokens.length || summary.models.length < 2) return null

  // Show top 10 tokens and each model's weight
  const top10 = summary.tokens.slice(0, 10)
  const data = top10.map((t) => ({
    token: t.token,
    ...Object.fromEntries(
      summary.models.map((m) => [
        m.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
        Math.round((t.weights[m] ?? 0) * 1000) / 1000,
      ])
    ),
  }))

  const MODEL_COLORS = ['#818cf8', '#34d399', '#f87171', '#fbbf24', '#a78bfa']

  return (
    <div className="card p-5">
      <h3 className="section-title mb-1">Explanation Comparison Across Models</h3>
      <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>
        Each model's weight for the top tokens — agreement = consistent signal, disagreement = uncertain
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ left: 70, right: 10, top: 4, bottom: 4 }}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            domain={[-1, 1]}
            tick={{ fontSize: 10, fill: 'var(--text-faint)' }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            type="category"
            dataKey="token"
            tick={{ fontSize: 11, fill: 'var(--text-secondary)', fontFamily: 'monospace' }}
            tickLine={false}
            axisLine={false}
            width={70}
          />
          <ReferenceLine x={0} stroke="var(--border)" />
          <Tooltip {...TT} formatter={(v: number) => [`${v > 0 ? '+' : ''}${v.toFixed(3)}`]} />
          <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{v}</span>} />
          {summary.models.map((m, idx) => (
            <Bar
              key={m}
              dataKey={m.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
              fill={MODEL_COLORS[idx % MODEL_COLORS.length]}
              radius={[0, 2, 2, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Single model block ────────────────────────────────────────────────────────
function ModelBlock({ exp }: { exp: PerModelExplanation }) {
  const [open, setOpen] = useState(false)

  if (exp.method === 'unavailable') {
    return (
      <div className="rounded-xl p-4" style={{ border: '1px solid var(--border)', backgroundColor: 'var(--bg-surface-2)' }}>
        <div className="flex items-center gap-3">
          <Cpu className="w-4 h-4" style={{ color: 'var(--text-faint)' }} />
          <span className="text-sm" style={{ color: 'var(--text-subtle)' }}>{exp.model_name}</span>
          <MethodBadge method={exp.method} />
        </div>
        {exp.error && (
          <p className="text-xs mt-2 ml-7" style={{ color: 'var(--text-faint)' }}>{exp.error}</p>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      <button
        className="w-full flex items-center gap-3 px-4 py-3 transition-colors text-left"
        style={{ backgroundColor: 'var(--bg-surface-2)' }}
        onClick={() => setOpen(!open)}
        onMouseOver={(e) => { (e.currentTarget as HTMLElement).style.backgroundColor = 'var(--bg-surface-3)' }}
        onMouseOut={(e) => { (e.currentTarget as HTMLElement).style.backgroundColor = 'var(--bg-surface-2)' }}
      >
        <Cpu className="w-4 h-4 shrink-0" style={{ color: 'var(--text-subtle)' }} />
        <span className="text-sm font-medium flex-1" style={{ color: 'var(--text-secondary)' }}>
          {exp.model_name}
        </span>
        <MethodBadge method={exp.method} />
        <span
          className={cn('text-xs font-semibold ml-1 mr-1', exp.label === 'FAKE' ? 'text-red-400' : 'text-emerald-400')}
        >
          {exp.label}
        </span>
        {open
          ? <ChevronDown className="w-4 h-4 shrink-0" style={{ color: 'var(--text-subtle)' }} />
          : <ChevronRight className="w-4 h-4 shrink-0" style={{ color: 'var(--text-subtle)' }} />}
      </button>

      {open && (
        <div className="px-4 pb-4 pt-3 space-y-4" style={{ borderTop: '1px solid var(--border)' }}>
          {exp.top_tokens.length > 0 && (
            <div>
              <div className="label-sm mb-2">Word attributions</div>
              <TokenHeatmap tokens={exp.top_tokens} />
              <div className="flex gap-5 mt-2 text-xs" style={{ color: 'var(--text-faint)' }}>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm bg-red-500/25 border border-red-500/30" />
                  Fake-pattern association
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-3 rounded-sm bg-emerald-500/25 border border-emerald-500/30" />
                  Real-pattern association
                </span>
              </div>
            </div>
          )}

          {exp.plain_text && (
            <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              {exp.plain_text}
            </p>
          )}

          {exp.disclaimer && (
            <div className="flex items-start gap-2">
              <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" style={{ color: 'var(--text-faint)' }} />
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-subtle)' }}>{exp.disclaimer}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────

interface ExplainabilitySectionProps {
  analysisId: number
}

export function ExplainabilitySection({ analysisId }: ExplainabilitySectionProps) {
  const [showFullText, setShowFullText] = useState(false)

  const { data: expData,     isLoading: expLoading,   error: expError   } = useExplanation(analysisId)
  const { data: annotated,   isLoading: annLoading                       } = useAnnotatedText(analysisId)
  const { data: summaryData, isLoading: sumLoading                       } = useExplanationSummary(analysisId)

  if (expLoading) {
    return (
      <section aria-label="Explainability">
        <div className="card p-6 flex items-center gap-3">
          <Spinner size="sm" />
          <span className="text-sm" style={{ color: 'var(--text-subtle)' }}>
            Loading model explanations…
          </span>
        </div>
      </section>
    )
  }

  if (expError || !expData) return null

  const available = expData.model_explanations.filter((e) => e.method !== 'unavailable')
  if (!available.length && !expData.model_explanations.length) return null

  // Aggregate top tokens from available explanations
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

  return (
    <section aria-label="Explainability">
      <div className="card overflow-hidden">
        {/* Header */}
        <div className="px-5 py-4 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)' }}>
          <div>
            <h2 className="section-title">Explainability</h2>
            <p className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>
              Which words influenced each model's prediction
            </p>
          </div>
          <a
            href={getExplanationReportUrl(analysisId)}
            download={`xai_report_${analysisId}.json`}
            className="btn-secondary text-xs"
          >
            <Download className="w-3.5 h-3.5" />
            Download Report
          </a>
        </div>

        {/* Signal-vs-evidence warning — ALWAYS FIRST */}
        <div className="mx-5 mt-5 flex items-start gap-3 p-3.5 rounded-xl"
          style={{ backgroundColor: 'rgba(251,191,36,0.05)', border: '1px solid rgba(251,191,36,0.2)' }}>
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-300/80 leading-relaxed">
            <span className="font-semibold text-amber-300">Model signal — not factual evidence.</span>{' '}
            {expData.signal_vs_evidence_warning}
          </p>
        </div>

        <div className="p-5 space-y-5">
          {/* 1. Highlighted article text */}
          {!annLoading && annotated?.annotated && (
            <AnnotatedText
              words={annotated.annotated}
              visible={showFullText}
              onToggle={() => setShowFullText(!showFullText)}
            />
          )}

          {/* 2. Top-10 influential words bar chart */}
          {aggTokens.length > 0 && (
            <Top10Chart tokens={aggTokens} />
          )}

          {/* 3. Positive vs negative contribution chart */}
          {available.length > 0 && (
            <ContributionChart explanations={available} />
          )}

          {/* 4. Per-model token heatmaps */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Layers className="w-4 h-4" style={{ color: 'var(--text-subtle)' }} />
              <h3 className="section-title">Per-Model Explanations</h3>
            </div>
            <div className="space-y-3">
              {expData.model_explanations.map((exp) => (
                <ModelBlock key={exp.model_id} exp={exp} />
              ))}
            </div>
          </div>

          {/* 5. Cross-model comparison */}
          {!sumLoading && summaryData && summaryData.models?.length >= 2 && (
            <CrossModelChart summary={summaryData as ExplanationSummary} />
          )}

          {/* Method inventory */}
          {available.length > 0 && (
            <div className="flex flex-wrap gap-2 pt-2" style={{ borderTop: '1px solid var(--border)' }}>
              <span className="text-xs" style={{ color: 'var(--text-subtle)' }}>Methods used:</span>
              {[...new Set(available.map((e) => e.method))].map((m) => (
                <MethodBadge key={m} method={m} />
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
