import { cn, formatPct } from '@/lib/utils'
import { ProgressBar } from '@/components/ui/ProgressBar'
import type { ModelPrediction } from '@/types/api'

interface ModelPredictionsTableProps {
  predictions: ModelPrediction[]
}

export function ModelPredictionsTable({ predictions }: ModelPredictionsTableProps) {
  if (!predictions.length) return null

  return (
    <div className="glass-card overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-800">
        <h3 className="section-title">Per-Model Predictions</h3>
        <p className="text-xs text-slate-500 mt-1">
          Each model's independent probability estimate — ensemble combines these.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-800">
              <th className="px-5 py-3 text-left label-sm">Model</th>
              <th className="px-5 py-3 text-left label-sm">Prediction</th>
              <th className="px-5 py-3 text-left label-sm w-32">Fake Prob.</th>
              <th className="px-5 py-3 text-left label-sm w-32">Real Prob.</th>
              <th className="px-5 py-3 text-right label-sm">Time</th>
            </tr>
          </thead>
          <tbody>
            {predictions.map((p) => (
              <tr key={p.model_id} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
                <td className="px-5 py-4">
                  <div className="font-medium text-sm text-slate-200">{p.model_name}</div>
                  <div className="text-xs text-slate-600 mt-0.5 font-mono">{p.model_id}</div>
                </td>
                <td className="px-5 py-4">
                  <span className={cn(
                    'font-semibold text-sm',
                    p.is_fake ? 'text-red-400' : 'text-emerald-400'
                  )}>
                    {p.label}
                  </span>
                  <div className="text-xs text-slate-500 mt-0.5">{formatPct(p.confidence)}</div>
                </td>
                <td className="px-5 py-4">
                  <div className="text-xs text-slate-300 mb-1">{formatPct(p.fake_probability)}</div>
                  <ProgressBar
                    value={p.fake_probability}
                    color={p.fake_probability > 0.6 ? 'bg-red-500' : 'bg-slate-600'}
                  />
                </td>
                <td className="px-5 py-4">
                  <div className="text-xs text-slate-300 mb-1">{formatPct(p.real_probability)}</div>
                  <ProgressBar
                    value={p.real_probability}
                    color={p.real_probability > 0.6 ? 'bg-emerald-500' : 'bg-slate-600'}
                  />
                </td>
                <td className="px-5 py-4 text-right text-xs text-slate-500 font-mono">
                  {p.inference_time_ms.toFixed(0)}ms
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
