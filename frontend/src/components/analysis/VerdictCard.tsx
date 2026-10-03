import { cn, getVerdictConfig, formatPct } from '@/lib/utils'
import { ProgressBar } from '@/components/ui/ProgressBar'
import type { MLVerdict, EvidenceVerdict } from '@/types/api'
import { AlertCircle, Info } from 'lucide-react'

interface VerdictCardProps {
  mlVerdict: MLVerdict
  mlConfidence?: number
  evidenceVerdict?: EvidenceVerdict
  evidenceExplanation?: string
  processingTimeMs?: number
  className?: string
}

export function VerdictCard({
  mlVerdict,
  mlConfidence,
  evidenceVerdict,
  evidenceExplanation,
  processingTimeMs,
  className,
}: VerdictCardProps) {
  const ml = getVerdictConfig(mlVerdict)
  const ev = evidenceVerdict ? getVerdictConfig(evidenceVerdict) : null

  return (
    <div className={cn('glass-card p-6 animate-slide-up', className)}>
      <h2 className="section-title mb-5">Analysis Result</h2>

      <div className="grid sm:grid-cols-2 gap-4">
        {/* ML Verdict */}
        <div className={cn('rounded-xl p-5 border', ml.bg, ml.border)}>
          <div className="label-sm mb-2">ML Model Prediction</div>
          <div className={cn('text-2xl font-bold tracking-tight', ml.color)}>{ml.label}</div>
          {mlConfidence !== undefined && (
            <div className="mt-3 space-y-1.5">
              <div className="flex justify-between text-xs text-slate-500">
                <span>Confidence</span>
                <span className={ml.color}>{formatPct(mlConfidence)}</span>
              </div>
              <ProgressBar
                value={mlConfidence}
                color={mlConfidence > 0.7 ? 'bg-brand-500' : mlConfidence > 0.55 ? 'bg-amber-500' : 'bg-slate-600'}
                size="md"
              />
            </div>
          )}
          <div className="mt-3 flex items-start gap-1.5 text-xs text-slate-500">
            <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span>Statistical prediction — not a factual determination</span>
          </div>
        </div>

        {/* Evidence Verdict */}
        {ev ? (
          <div className={cn('rounded-xl p-5 border', ev.bg, ev.border)}>
            <div className="label-sm mb-2">Evidence Assessment</div>
            <div className={cn('text-2xl font-bold tracking-tight', ev.color)}>{ev.label}</div>
            {evidenceExplanation && (
              <p className="mt-3 text-xs text-slate-400 leading-relaxed line-clamp-4">
                {evidenceExplanation}
              </p>
            )}
          </div>
        ) : (
          <div className="rounded-xl p-5 border bg-slate-800/40 border-slate-700/40 flex flex-col items-center justify-center text-center gap-2">
            <AlertCircle className="w-8 h-8 text-slate-600" />
            <div className="label-sm">Evidence Assessment</div>
            <p className="text-xs text-slate-500 leading-relaxed">
              No evidence assessment available.
            </p>
          </div>
        )}
      </div>

      {processingTimeMs && (
        <div className="mt-4 pt-4 border-t border-slate-800 text-xs text-slate-600">
          Processed in {processingTimeMs.toLocaleString()}ms
        </div>
      )}
    </div>
  )
}
