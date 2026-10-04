/**
 * ExplainabilitySection
 *
 * Fetches explainability data from GET /api/v1/explanation/{id} and renders:
 *  • Per-model token heatmaps (LIME / attention / tfidf_weights)
 *  • Method badge (shows which algorithm produced the weights)
 *  • Plain-language explanation per model
 *  • Model-vs-evidence disclaimer (always visible)
 *  • "Not available" graceful state when explanation wasn't computed
 *
 * The signal-vs-evidence warning is shown at the top of every render —
 * never omitted.
 */

import { useState } from 'react'
import { AlertTriangle, Info, ChevronDown, ChevronRight, Cpu } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useExplanation } from '@/hooks/useApi'
import { Spinner } from '@/components/ui/Spinner'
import type { PerModelExplanation } from '@/types/api'

// ── Method badge ──────────────────────────────────────────────────────────────

const METHOD_META: Record<string, { label: string; color: string; desc: string }> = {
  lime:          { label: 'LIME',           color: 'text-brand-400',   desc: 'Local Interpretable Model-Agnostic Explanations' },
  attention:     { label: 'Attention',      color: 'text-violet-400',  desc: 'Transformer last-layer attention attribution'    },
  tfidf_weights: { label: 'TF-IDF Weights', color: 'text-amber-400',   desc: 'Raw TF-IDF feature importance (LIME fallback)'  },
  unavailable:   { label: 'Unavailable',    color: 'text-slate-600',   desc: 'Explanation not computed for this model'        },
}

function MethodBadge({ method }: { method: string }) {
  const meta = METHOD_META[method] ?? METHOD_META.unavailable
  return (
    <span
      className={cn('text-xs font-medium px-2 py-0.5 rounded border border-current/20 bg-current/5', meta.color)}
      title={meta.desc}
    >
      {meta.label}
    </span>
  )
}

// ── Token heatmap ─────────────────────────────────────────────────────────────

function TokenHeatmap({ tokens }: { tokens: PerModelExplanation['top_tokens'] }) {
  if (!tokens.length) return null

  const maxAbs = Math.max(...tokens.map((t) => Math.abs(t.weight)), 0.01)

  return (
    <div className="flex flex-wrap gap-2">
      {tokens.map((tw, i) => {
        const intensity = Math.abs(tw.weight) / maxAbs
        const isFake    = tw.weight > 0
        const base      = isFake ? '239,68,68' : '52,211,153'
        const textColor = isFake
          ? `rgba(252,165,165,${0.55 + intensity * 0.45})`
          : `rgba(110,231,183,${0.55 + intensity * 0.45})`

        return (
          <span
            key={i}
            title={`${tw.token}: ${tw.weight > 0 ? '+' : ''}${tw.weight.toFixed(3)} (${isFake ? 'fake-pattern' : 'real-pattern'})`}
            className="px-2.5 py-1 rounded-md text-xs font-mono border cursor-help select-none transition-all"
            style={{
              color:           textColor,
              borderColor:     `rgba(${base},${0.08 + intensity * 0.3})`,
              backgroundColor: `rgba(${base},${0.04 + intensity * 0.16})`,
            }}
          >
            {tw.token}
          </span>
        )
      })}
    </div>
  )
}

// ── Single model block ────────────────────────────────────────────────────────

function ModelBlock({ exp }: { exp: PerModelExplanation }) {
  const [open, setOpen] = useState(true)

  if (exp.method === 'unavailable') {
    return (
      <div className="rounded-xl border border-slate-800/40 bg-slate-900/20 p-4">
        <div className="flex items-center gap-3">
          <Cpu className="w-4 h-4 text-slate-700" />
          <span className="text-sm text-slate-500">{exp.model_name}</span>
          <MethodBadge method={exp.method} />
        </div>
        {exp.error && (
          <p className="text-xs text-slate-700 mt-2 ml-7">{exp.error}</p>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-slate-800/50 bg-slate-900/30 overflow-hidden">
      {/* Header */}
      <button
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-slate-800/20 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <Cpu className="w-4 h-4 text-slate-600 shrink-0" />
        <span className="text-sm font-medium text-slate-300 flex-1 text-left">
          {exp.model_name}
        </span>
        <MethodBadge method={exp.method} />
        <span className={cn(
          'text-xs font-semibold ml-1 mr-1',
          exp.label === 'FAKE' ? 'text-red-400' : 'text-emerald-400',
        )}>
          {exp.label}
        </span>
        {open
          ? <ChevronDown className="w-4 h-4 text-slate-600 shrink-0" />
          : <ChevronRight className="w-4 h-4 text-slate-600 shrink-0" />}
      </button>

      {open && (
        <div className="px-4 pb-4 pt-1 space-y-4 border-t border-slate-800/40">
          {/* Token heatmap */}
          {exp.top_tokens.length > 0 && (
            <div>
              <div className="label-sm mb-2">Word attributions</div>
              <TokenHeatmap tokens={exp.top_tokens} />

              {/* Colour legend */}
              <div className="flex gap-5 mt-2.5 text-xs text-slate-700">
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

          {/* Plain-text explanation */}
          {exp.plain_text && (
            <p className="text-sm text-slate-400 leading-relaxed">
              {exp.plain_text}
            </p>
          )}

          {/* Per-model disclaimer */}
          {exp.disclaimer && (
            <div className="flex items-start gap-2">
              <Info className="w-3.5 h-3.5 text-slate-600 shrink-0 mt-0.5" />
              <p className="text-xs text-slate-600 leading-relaxed">{exp.disclaimer}</p>
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
  const { data, isLoading, error } = useExplanation(analysisId)

  // Loading state
  if (isLoading) {
    return (
      <section aria-label="Explainability">
        <div className="glass-card p-6 flex items-center gap-3">
          <Spinner size="sm" />
          <span className="text-sm text-slate-500">Loading model explanations…</span>
        </div>
      </section>
    )
  }

  // Silently skip if unavailable — don't show a broken state
  if (error || !data) return null

  const available = data.model_explanations.filter((e) => e.method !== 'unavailable')
  // Still show the section even if some models are unavailable, as long as there's something
  if (!available.length && !data.model_explanations.length) return null

  return (
    <section aria-label="Explainability">
      <div className="glass-card overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800">
          <h2 className="section-title">Explainability</h2>
          <p className="text-xs text-slate-500 mt-1">
            Which words influenced each model's prediction.
          </p>
        </div>

        {/* Prominent signal-vs-evidence warning */}
        <div className="mx-5 mt-4 flex items-start gap-3 p-3.5 rounded-xl bg-amber-500/5 border border-amber-500/15">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-300/75 leading-relaxed">
            <span className="font-semibold text-amber-300">Model signal — not factual evidence.</span>{' '}
            {data.signal_vs_evidence_warning}
          </p>
        </div>

        <div className="p-5 space-y-3">
          {data.model_explanations.map((exp) => (
            <ModelBlock key={exp.model_id} exp={exp} />
          ))}
        </div>
      </div>
    </section>
  )
}
