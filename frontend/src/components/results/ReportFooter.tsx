/**
 * ReportFooter
 *
 * Displays metadata and limitations at the bottom of the results page:
 *  • Timestamp (created_at)
 *  • Processing time (ms / s)
 *  • Per-model version information
 *  • Static system limitations that always apply
 *  • Dynamic evidence limitations from the analysis
 *
 * Renders only fields that exist in the response — nothing fabricated.
 */

import { Clock, Cpu, AlertTriangle, Info, Timer, Database } from 'lucide-react'
import { cn, formatDateTime, formatMs } from '@/lib/utils'
import type { ModelPrediction } from '@/types/api'

// ── Static system limitations ─────────────────────────────────────────────────
// These always apply regardless of the individual result.

const SYSTEM_LIMITATIONS = [
  'ML predictions reflect statistical patterns from training data (2015–2018 ISOT dataset) and may not generalise to newer events, scientific topics, or non-English content.',
  'High confidence scores indicate strong pattern match — not that the content is factually incorrect.',
  'Evidence search depends on configured providers. Results are limited by API availability and search quality.',
  'Absence of supporting evidence cannot be interpreted as proof of falsehood.',
  'The explainability output shows which words the model attended to, not which words are factually wrong.',
]

// ── Metadata row ──────────────────────────────────────────────────────────────

function MetaRow({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode
  label: string
  value: React.ReactNode
}) {
  return (
    <div className="flex items-start gap-3 py-2.5 border-b border-slate-800/50 last:border-0">
      <span className="text-slate-600 mt-0.5 shrink-0">{icon}</span>
      <div className="flex-1 min-w-0 flex flex-col sm:flex-row sm:items-center gap-0.5 sm:gap-3">
        <span className="text-xs text-slate-500 sm:w-36 shrink-0">{label}</span>
        <span className="text-xs text-slate-300 font-mono">{value}</span>
      </div>
    </div>
  )
}

// ── Model version table ───────────────────────────────────────────────────────

function ModelVersions({ predictions }: { predictions: ModelPrediction[] }) {
  if (!predictions.length) return null

  return (
    <div>
      <div className="label-sm mb-3 flex items-center gap-2">
        <Cpu className="w-3.5 h-3.5" />
        Model Versions Used
      </div>
      <div className="rounded-xl border border-slate-800/50 bg-slate-900/30 divide-y divide-slate-800/40">
        {predictions.map((p) => (
          <div key={p.model_id} className="flex items-center justify-between px-4 py-3">
            <div>
              <div className="text-sm text-slate-300">{p.model_name}</div>
              <div className="text-xs text-slate-600 font-mono mt-0.5">{p.model_id}</div>
            </div>
            <div className="text-right">
              <div className={cn(
                'text-xs font-semibold',
                p.is_fake ? 'text-red-400' : 'text-emerald-400',
              )}>
                {p.label}
              </div>
              <div className="text-xs text-slate-700 font-mono mt-0.5">
                {p.inference_time_ms.toFixed(0)} ms
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Limitations list ──────────────────────────────────────────────────────────

function LimitationsList({
  dynamic,
}: {
  dynamic: string[]
}) {
  const all = [...new Set([...dynamic, ...SYSTEM_LIMITATIONS])]

  return (
    <div>
      <div className="label-sm mb-3 flex items-center gap-2">
        <AlertTriangle className="w-3.5 h-3.5 text-amber-600" />
        Limitations & Caveats
      </div>
      <ul className="space-y-2">
        {all.map((lim, i) => (
          <li key={i} className="flex items-start gap-2.5">
            <Info className="w-3 h-3 text-slate-700 shrink-0 mt-0.5" />
            <span className="text-xs text-slate-500 leading-relaxed">{lim}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────

interface ReportFooterProps {
  analysisId:         number
  createdAt:          string
  processingTimeMs?:  number
  predictions:        ModelPrediction[]
  evidenceLimitations: string[]
  analysisStatus:     string
}

export function ReportFooter({
  analysisId,
  createdAt,
  processingTimeMs,
  predictions,
  evidenceLimitations,
  analysisStatus,
}: ReportFooterProps) {
  return (
    <section aria-label="Report metadata">
      <div className="glass-card overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800">
          <h2 className="section-title">Report Details</h2>
        </div>

        <div className="p-5 space-y-6">
          {/* Metadata */}
          <div>
            <div className="label-sm mb-3 flex items-center gap-2">
              <Database className="w-3.5 h-3.5" />
              Analysis Metadata
            </div>
            <div className="rounded-xl border border-slate-800/50 bg-slate-900/30 px-4">
              <MetaRow
                icon={<Info className="w-3.5 h-3.5" />}
                label="Analysis ID"
                value={`#${analysisId}`}
              />
              <MetaRow
                icon={<Clock className="w-3.5 h-3.5" />}
                label="Timestamp"
                value={formatDateTime(createdAt)}
              />
              {processingTimeMs !== undefined && (
                <MetaRow
                  icon={<Timer className="w-3.5 h-3.5" />}
                  label="Processing time"
                  value={`${formatMs(processingTimeMs)} (${processingTimeMs.toLocaleString()} ms)`}
                />
              )}
              <MetaRow
                icon={<Info className="w-3.5 h-3.5" />}
                label="Status"
                value={
                  <span className={cn(
                    'capitalize',
                    analysisStatus === 'completed' ? 'text-emerald-400' :
                    analysisStatus === 'failed'    ? 'text-red-400'     :
                    'text-amber-400',
                  )}>
                    {analysisStatus}
                  </span>
                }
              />
            </div>
          </div>

          {/* Model versions */}
          <ModelVersions predictions={predictions} />

          {/* Limitations */}
          <LimitationsList dynamic={evidenceLimitations} />
        </div>
      </div>
    </section>
  )
}
