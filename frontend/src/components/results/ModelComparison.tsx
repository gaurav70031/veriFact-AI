/**
 * ModelComparison
 *
 * Per-model prediction breakdown with probability bars and inference timing.
 * Renders nothing when model_predictions is empty.
 */

import { Cpu, Timer } from 'lucide-react'
import { cn, formatPct } from '@/lib/utils'
import type { ModelPrediction } from '@/types/api'

// ── Dual probability bar ──────────────────────────────────────────────────────

function DualBar({
  fakeProb,
  realProb,
}: {
  fakeProb: number
  realProb: number
}) {
  const fakeW = Math.round(fakeProb * 100)
  const realW = Math.round(realProb * 100)

  return (
    <div className="space-y-1.5">
      {/* FAKE bar */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500 w-8 shrink-0">FAKE</span>
        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all duration-500',
              fakeProb > 0.65 ? 'bg-red-500' : fakeProb > 0.45 ? 'bg-orange-500' : 'bg-slate-600',
            )}
            style={{ width: `${fakeW}%` }}
          />
        </div>
        <span className={cn(
          'text-xs font-mono w-10 text-right shrink-0',
          fakeProb > 0.65 ? 'text-red-400' : 'text-slate-500',
        )}>
          {fakeW}%
        </span>
      </div>

      {/* REAL bar */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-slate-500 w-8 shrink-0">REAL</span>
        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all duration-500',
              realProb > 0.65 ? 'bg-emerald-500' : realProb > 0.45 ? 'bg-teal-600' : 'bg-slate-600',
            )}
            style={{ width: `${realW}%` }}
          />
        </div>
        <span className={cn(
          'text-xs font-mono w-10 text-right shrink-0',
          realProb > 0.65 ? 'text-emerald-400' : 'text-slate-500',
        )}>
          {realW}%
        </span>
      </div>
    </div>
  )
}

// ── Single model card ─────────────────────────────────────────────────────────

function ModelCard({ pred }: { pred: ModelPrediction }) {
  const isFake     = pred.is_fake
  const isHighConf = pred.confidence > 0.8
  const isMed      = pred.confidence > 0.6

  return (
    <div className="p-4 rounded-xl border border-slate-800/60 bg-slate-900/40 space-y-3 hover:border-slate-700/80 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-1.5 mb-0.5">
            <Cpu className="w-3.5 h-3.5 text-slate-600" />
            <span className="text-xs text-slate-500 font-mono">{pred.model_id}</span>
          </div>
          <div className="text-sm font-semibold text-slate-200">{pred.model_name}</div>
        </div>

        {/* Verdict + confidence pill */}
        <div className={cn(
          'flex flex-col items-end gap-0.5 shrink-0',
        )}>
          <span className={cn(
            'text-sm font-bold',
            isFake ? 'text-red-400' : 'text-emerald-400',
          )}>
            {pred.label}
          </span>
          <span className={cn(
            'text-xs font-mono',
            isHighConf ? (isFake ? 'text-red-500/70' : 'text-emerald-500/70') :
            isMed      ? 'text-amber-500/70' :
            'text-slate-600',
          )}>
            {formatPct(pred.confidence)}
          </span>
        </div>
      </div>

      {/* Probability bars */}
      <DualBar fakeProb={pred.fake_probability} realProb={pred.real_probability} />

      {/* Inference time */}
      <div className="flex items-center gap-1 pt-1 border-t border-slate-800/40">
        <Timer className="w-3 h-3 text-slate-700" />
        <span className="text-xs text-slate-700 font-mono">
          {pred.inference_time_ms.toFixed(0)} ms
        </span>
      </div>
    </div>
  )
}

// ── Ensemble summary row ──────────────────────────────────────────────────────

function EnsembleSummary({ predictions }: { predictions: ModelPrediction[] }) {
  if (predictions.length < 2) return null

  const fakeVotes  = predictions.filter((p) => p.is_fake).length
  const realVotes  = predictions.length - fakeVotes
  const avgFake    = predictions.reduce((s, p) => s + p.fake_probability, 0) / predictions.length
  const avgReal    = predictions.reduce((s, p) => s + p.real_probability, 0) / predictions.length

  return (
    <div className="rounded-xl border border-slate-700/40 bg-slate-800/20 p-4">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <div>
          <div className="label-sm mb-1">Model votes</div>
          <div className="flex items-center gap-2 text-sm">
            {fakeVotes > 0 && (
              <span className="font-semibold text-red-400">{fakeVotes} FAKE</span>
            )}
            {fakeVotes > 0 && realVotes > 0 && (
              <span className="text-slate-700">·</span>
            )}
            {realVotes > 0 && (
              <span className="font-semibold text-emerald-400">{realVotes} REAL</span>
            )}
          </div>
        </div>

        <div>
          <div className="label-sm mb-1">Average P(FAKE)</div>
          <span className={cn(
            'text-sm font-mono font-semibold',
            avgFake > 0.6 ? 'text-red-400' : avgFake > 0.4 ? 'text-amber-400' : 'text-emerald-400',
          )}>
            {formatPct(avgFake)}
          </span>
        </div>

        <div>
          <div className="label-sm mb-1">Average P(REAL)</div>
          <span className={cn(
            'text-sm font-mono font-semibold',
            avgReal > 0.6 ? 'text-emerald-400' : avgReal > 0.4 ? 'text-amber-400' : 'text-red-400',
          )}>
            {formatPct(avgReal)}
          </span>
        </div>

        <div>
          <div className="label-sm mb-1">Total models</div>
          <span className="text-sm font-mono font-semibold text-slate-300">
            {predictions.length}
          </span>
        </div>
      </div>
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────

interface ModelComparisonProps {
  predictions: ModelPrediction[]
}

export function ModelComparison({ predictions }: ModelComparisonProps) {
  if (!predictions.length) return null

  return (
    <section aria-label="Model predictions">
      <div className="glass-card overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800">
          <h2 className="section-title">Model Predictions</h2>
          <p className="text-xs text-slate-500 mt-1">
            Each model's independent probability estimate. The ensemble
            combines these with weighted averaging — it is a statistical
            signal, not a factual judgement.
          </p>
        </div>

        <div className="p-5 space-y-4">
          <div className="grid sm:grid-cols-2 xl:grid-cols-3 gap-3">
            {predictions.map((p) => (
              <ModelCard key={p.model_id} pred={p} />
            ))}
          </div>

          <EnsembleSummary predictions={predictions} />
        </div>
      </div>
    </section>
  )
}
