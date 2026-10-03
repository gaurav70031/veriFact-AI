import { useState } from 'react'
import { useModelPerformance, useModels } from '@/hooks/useApi'
import { ModelRadar } from '@/components/charts/ModelRadar'
import { PerformanceBar } from '@/components/charts/PerformanceBar'
import { Spinner } from '@/components/ui/Spinner'
import { ErrorState } from '@/components/ui/ErrorState'
import { Badge } from '@/components/ui/Badge'
import { formatPct } from '@/lib/utils'

type Metric = 'accuracy' | 'f1_score' | 'roc_auc'
const METRICS: { id: Metric; label: string }[] = [
  { id: 'accuracy', label: 'Accuracy' },
  { id: 'f1_score', label: 'F1 Score' },
  { id: 'roc_auc',  label: 'ROC-AUC'  },
]

export default function ModelPerformance() {
  const [metric, setMetric] = useState<Metric>('f1_score')
  const { data: perf, isLoading: pl, error: pe, refetch: pr } = useModelPerformance()
  const { data: models } = useModels()

  if (pl) return <div className="flex justify-center py-32"><Spinner size="lg" /></div>
  if (pe) return (
    <div className="max-w-xl mx-auto py-24 px-4">
      <ErrorState message={(pe as Error).message} onRetry={() => pr()} />
    </div>
  )

  const perfModels = perf?.models ?? []

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-100 mb-2">Model Performance</h1>
        <p className="text-sm text-slate-500">
          Evaluation metrics computed on the held-out test set during training — not estimates or placeholders.
        </p>
      </div>

      {perfModels.length === 0 ? (
        <div className="glass-card p-10 text-center">
          <p className="text-slate-500 text-sm">No model performance data available yet.</p>
          <p className="text-xs text-slate-600 mt-1">
            Run training scripts and register models in the database to see metrics here.
          </p>
        </div>
      ) : (
        <>
          {/* Metric selector */}
          <div className="flex gap-2 mb-6">
            {METRICS.map((m) => (
              <button
                key={m.id}
                onClick={() => setMetric(m.id)}
                className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                  metric === m.id
                    ? 'bg-brand-600 text-white'
                    : 'bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>

          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Bar chart */}
            <div className="glass-card p-6">
              <h2 className="section-title mb-4">
                {METRICS.find((m) => m.id === metric)?.label} Comparison
              </h2>
              <PerformanceBar models={perfModels} metric={metric} />
            </div>

            {/* Radar chart */}
            <div className="glass-card p-6">
              <h2 className="section-title mb-4">Multi-Metric Comparison</h2>
              <ModelRadar models={perfModels} />
            </div>
          </div>

          {/* Detailed table */}
          <div className="glass-card overflow-hidden">
            <div className="px-5 py-4 border-b border-slate-800">
              <h2 className="section-title">Full Metrics Table</h2>
              <p className="text-xs text-slate-500 mt-1">ISOT Fake News Dataset — held-out test set</p>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-800">
                    <th className="px-5 py-3 text-left label-sm">Model</th>
                    <th className="px-5 py-3 text-right label-sm">Accuracy</th>
                    <th className="px-5 py-3 text-right label-sm">Precision</th>
                    <th className="px-5 py-3 text-right label-sm">Recall</th>
                    <th className="px-5 py-3 text-right label-sm">F1</th>
                    <th className="px-5 py-3 text-right label-sm">ROC-AUC</th>
                    <th className="px-5 py-3 text-right label-sm">Test Samples</th>
                  </tr>
                </thead>
                <tbody>
                  {perfModels.map((m) => (
                    <tr key={m.model_name} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
                      <td className="px-5 py-4">
                        <div className="font-medium text-sm text-slate-200">
                          {m.model_name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                        </div>
                        <div className="text-xs text-slate-600 mt-0.5">{m.version}</div>
                      </td>
                      {['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'].map((key) => {
                        const val = m[key as keyof typeof m] as number | undefined
                        return (
                          <td key={key} className="px-5 py-4 text-right text-sm font-mono">
                            {val !== undefined ? formatPct(val) : '—'}
                          </td>
                        )
                      })}
                      <td className="px-5 py-4 text-right text-sm text-slate-500 font-mono">
                        {m.test_samples?.toLocaleString() ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Registered models */}
          {models && models.length > 0 && (
            <div className="glass-card p-6 mt-6">
              <h2 className="section-title mb-4">Registered Model Versions</h2>
              <div className="space-y-3">
                {models.map((m) => (
                  <div key={m.id} className="flex items-center justify-between py-3 border-b border-slate-800/50 last:border-0">
                    <div>
                      <span className="text-sm font-medium text-slate-300">{m.model_name}</span>
                      <span className="text-xs text-slate-600 ml-2">v{m.version}</span>
                      {m.description && (
                        <p className="text-xs text-slate-600 mt-0.5">{m.description}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant={m.model_type === 'transformer' ? 'info' : 'muted'}>
                        {m.model_type}
                      </Badge>
                      <Badge variant={m.is_active ? 'success' : 'danger'}>
                        {m.is_active ? 'Active' : 'Inactive'}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
