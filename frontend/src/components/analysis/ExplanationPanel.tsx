import { Info, AlertTriangle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useExplanation } from '@/hooks/useApi'
import { Spinner } from '@/components/ui/Spinner'
import type { PerModelExplanation } from '@/types/api'

interface ExplanationPanelProps {
  analysisId: number
}

export function ExplanationPanel({ analysisId }: ExplanationPanelProps) {
  const { data, isLoading, error } = useExplanation(analysisId)

  if (isLoading) return (
    <div className="glass-card p-8 flex items-center justify-center gap-3">
      <Spinner size="sm" />
      <span className="text-sm text-slate-500">Loading explanations…</span>
    </div>
  )

  if (error || !data) return null

  const hasAny = data.model_explanations.some((e) => e.method !== 'unavailable')
  if (!hasAny) return null

  return (
    <div className="glass-card overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-800">
        <h3 className="section-title">Model Explainability</h3>
        <p className="text-xs text-slate-500 mt-1 leading-relaxed">
          {data.signal_vs_evidence_warning}
        </p>
      </div>

      {/* Signal vs Evidence warning banner */}
      <div className="mx-5 mt-4 p-3 rounded-lg bg-amber-500/5 border border-amber-500/20 flex gap-2.5">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <p className="text-xs text-amber-300/80 leading-relaxed">
          Highlighted words show model attention patterns from training data.
          They are <strong>not</strong> proof that any word is factually incorrect.
        </p>
      </div>

      <div className="p-5 space-y-5">
        {data.model_explanations
          .filter((e) => e.method !== 'unavailable' && e.top_tokens.length > 0)
          .map((exp) => (
            <ModelExplanationBlock key={exp.model_id} exp={exp} />
          ))}
      </div>
    </div>
  )
}

function ModelExplanationBlock({ exp }: { exp: PerModelExplanation }) {
  const maxAbs = Math.max(...exp.top_tokens.map((t) => Math.abs(t.weight)), 0.01)

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-sm font-medium text-slate-300">{exp.model_name}</span>
          <span className="ml-2 text-xs text-slate-600 uppercase tracking-wider">{exp.method}</span>
        </div>
        <span className={cn(
          'text-xs font-semibold',
          exp.label === 'FAKE' ? 'text-red-400' : 'text-emerald-400'
        )}>
          {exp.label}
        </span>
      </div>

      {/* Token heatmap */}
      <div className="flex flex-wrap gap-2">
        {exp.top_tokens.map((tw, i) => {
          const intensity = Math.abs(tw.weight) / maxAbs
          const isFake = tw.weight > 0
          return (
            <span
              key={i}
              title={`Weight: ${tw.weight.toFixed(3)}`}
              className={cn(
                'px-2.5 py-1 rounded-md text-xs font-mono border transition-all',
                isFake
                  ? 'text-red-300 border-red-500/20'
                  : 'text-emerald-300 border-emerald-500/20',
              )}
              style={{
                backgroundColor: isFake
                  ? `rgba(239,68,68,${0.05 + intensity * 0.25})`
                  : `rgba(34,197,94,${0.05 + intensity * 0.25})`,
              }}
            >
              {tw.token}
            </span>
          )
        })}
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs text-slate-600">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-red-500/30 border border-red-500/30" />
          Associated with fake patterns
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-emerald-500/30 border border-emerald-500/30" />
          Associated with real patterns
        </span>
      </div>

      {exp.disclaimer && (
        <div className="flex items-start gap-1.5 text-xs text-slate-600">
          <Info className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          <span>{exp.disclaimer}</span>
        </div>
      )}
    </div>
  )
}
