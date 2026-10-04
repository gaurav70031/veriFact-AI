/**
 * Model Performance page
 *
 * Data:
 *   GET /api/v1/model-performance  → ModelPerformanceList
 *   GET /api/v1/models             → ModelInfo[]
 *
 * Displays actual offline evaluation results generated from ML experiments.
 * Every metric shown was computed on a held-out test set during training.
 * Nothing is estimated or placeholder.
 *
 * Sections:
 *   1. Metric selector         — Accuracy / F1 / ROC-AUC
 *   2. Bar chart comparison    — selected metric across all models
 *   3. Radar chart             — all metrics for all models overlaid
 *   4. Full metrics table      — all metrics, test sample counts, version
 *   5. Registered model cards  — from model_versions table (active + inactive)
 *   6. Dataset provenance note — ISOT dataset, test set description
 *
 * Empty state: shown when no model versions are in the DB yet,
 * with instructions to run training scripts.
 */

import { useState } from 'react'
import { useModelPerformance, useModels } from '@/hooks/useApi'
import { ModelRadar } from '@/components/charts/ModelRadar'
import { PerformanceBar } from '@/components/charts/PerformanceBar'
import { Spinner } from '@/components/ui/Spinner'
import { ErrorState } from '@/components/ui/ErrorState'
import { Badge } from '@/components/ui/Badge'
import { ProgressBar } from '@/components/ui/ProgressBar'
import {
  Database, Cpu, FlaskConical, AlertTriangle, Info,
  ChevronDown, ChevronUp, RefreshCw,
} from 'lucide-react'
import { cn, formatPct, formatDateTime } from '@/lib/utils'
import type { ModelPerformance, ModelInfo } from '@/types/api'

// ─── Metric selector ──────────────────────────────────────────────────────────

type Metric = 'accuracy' | 'f1_score' | 'roc_auc' | 'precision' | 'recall'

const METRICS: { id: Metric; label: string; desc: string }[] = [
  { id: 'accuracy',  label: 'Accuracy',   desc: 'Fraction of correct predictions' },
  { id: 'f1_score',  label: 'F1 Score',   desc: 'Harmonic mean of precision and recall' },
  { id: 'roc_auc',   label: 'ROC-AUC',    desc: 'Area under the ROC curve' },
  { id: 'precision', label: 'Precision',  desc: 'TP / (TP + FP)' },
  { id: 'recall',    label: 'Recall',     desc: 'TP / (TP + FN)' },
]

// ─── Metric value cell ────────────────────────────────────────────────────────

function MetricCell({ value, highlight }: { value?: number; highlight?: boolean }) {
  if (value == null) {
    return <span className="text-slate-700">—</span>
  }
  const pct = value * 100
  const color =
    pct >= 98 ? 'text-emerald-400' :
    pct >= 95 ? 'text-emerald-500' :
    pct >= 90 ? 'text-brand-400'   :
    pct >= 80 ? 'text-amber-400'   :
    'text-red-400'

  return (
    <span className={cn('font-mono text-sm', highlight ? color : 'text-slate-300')}>
      {formatPct(value)}
    </span>
  )
}

// ─── Model detail card ────────────────────────────────────────────────────────

function ModelCard({ model }: { model: ModelInfo }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="rounded-xl border border-slate-800/50 bg-slate-900/30 overflow-hidden">
      <button
        className="w-full flex items-center gap-4 px-5 py-4 hover:bg-slate-800/20 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <div className="p-2 rounded-lg bg-slate-800/60">
          <Cpu className="w-4 h-4 text-slate-400" />
        </div>
        <div className="flex-1 min-w-0 text-left">
          <div className="flex flex-wrap items-center gap-2 mb-0.5">
            <span className="text-sm font-semibold text-slate-200">{model.model_name}</span>
            <span className="text-xs text-slate-600 font-mono">v{model.version}</span>
          </div>
          {model.description && (
            <p className="text-xs text-slate-500 line-clamp-1">{model.description}</p>
          )}
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Badge variant={model.model_type === 'transformer' ? 'info' : model.model_type === 'ensemble' ? 'default' : 'muted'}>
            {model.model_type}
          </Badge>
          <Badge variant={model.is_active ? 'success' : 'danger'}>
            {model.is_active ? 'Active' : 'Inactive'}
          </Badge>
          {open
            ? <ChevronUp className="w-4 h-4 text-slate-600" />
            : <ChevronDown className="w-4 h-4 text-slate-600" />}
        </div>
      </button>

      {open && (
        <div className="px-5 pb-4 pt-1 border-t border-slate-800/40 grid sm:grid-cols-2 gap-x-8 gap-y-2 text-xs">
          <div className="flex justify-between py-1.5 border-b border-slate-800/30">
            <span className="text-slate-500">Algorithm</span>
            <span className="text-slate-300 font-mono">{model.algorithm}</span>
          </div>
          <div className="flex justify-between py-1.5 border-b border-slate-800/30">
            <span className="text-slate-500">Version</span>
            <span className="text-slate-300 font-mono">{model.version}</span>
          </div>
          {model.dataset_name && (
            <div className="flex justify-between py-1.5 border-b border-slate-800/30">
              <span className="text-slate-500">Dataset</span>
              <span className="text-slate-300">{model.dataset_name}</span>
            </div>
          )}
          {model.training_samples != null && (
            <div className="flex justify-between py-1.5 border-b border-slate-800/30">
              <span className="text-slate-500">Training samples</span>
              <span className="text-slate-300 font-mono">{model.training_samples.toLocaleString()}</span>
            </div>
          )}
          <div className="flex justify-between py-1.5 border-b border-slate-800/30">
            <span className="text-slate-500">Registered</span>
            <span className="text-slate-300">{formatDateTime(model.created_at)}</span>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Best model highlight ─────────────────────────────────────────────────────

function BestModelBanner({ models, metric }: { models: ModelPerformance[]; metric: Metric }) {
  const best = models.reduce<ModelPerformance | null>((prev, cur) => {
    const a = prev?.[metric] ?? 0
    const b = cur[metric]   ?? 0
    return b > a ? cur : prev
  }, null)

  if (!best || best[metric] == null) return null

  return (
    <div className="flex items-center gap-3 p-3.5 rounded-xl bg-emerald-500/5 border border-emerald-500/15 mb-6">
      <FlaskConical className="w-4 h-4 text-emerald-400 shrink-0" />
      <p className="text-sm text-slate-300">
        Best {METRICS.find((m) => m.id === metric)?.label}:{' '}
        <span className="font-semibold text-emerald-400">
          {best.model_name.replace(/_/g, ' ')}
        </span>{' '}
        <span className="text-emerald-500 font-mono">{formatPct(best[metric]!)}</span>
        {' '}— from offline evaluation on held-out test set.
      </p>
    </div>
  )
}

// ─── Empty state ──────────────────────────────────────────────────────────────

function NoModelsState() {
  return (
    <div className="glass-card p-10 text-center space-y-4">
      <div className="w-14 h-14 rounded-2xl bg-slate-800/60 border border-slate-700/40 flex items-center justify-center mx-auto">
        <Database className="w-7 h-7 text-slate-600" />
      </div>
      <div>
        <h3 className="text-base font-semibold text-slate-300 mb-2">
          No model performance data
        </h3>
        <p className="text-sm text-slate-500 max-w-md mx-auto leading-relaxed mb-3">
          Model evaluation metrics appear here after training. Run the training
          and evaluation scripts to generate metrics and register model versions
          in the database.
        </p>
      </div>
      <div className="bg-slate-900/60 rounded-xl p-4 text-left max-w-sm mx-auto">
        <div className="label-sm mb-2">To populate this page</div>
        <ol className="text-xs text-slate-500 space-y-1.5 list-decimal list-inside">
          <li>Run <code className="text-brand-400 font-mono">python scripts/prepare_dataset.py</code></li>
          <li>Run <code className="text-brand-400 font-mono">python scripts/train_baseline.py</code></li>
          <li>Run <code className="text-brand-400 font-mono">python scripts/evaluate_baseline.py</code></li>
          <li>Apply the Alembic migration and reseed model versions</li>
        </ol>
      </div>
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function ModelPerformance() {
  const [metric, setMetric] = useState<Metric>('f1_score')

  const { data: perf, isLoading: pl, error: pe, refetch: pr } = useModelPerformance()
  const { data: models, isLoading: ml } = useModels()

  const isLoading = pl || ml

  if (isLoading && !perf) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-3">
        <Spinner size="lg" />
        <p className="text-sm text-slate-500">Loading evaluation metrics from database…</p>
      </div>
    )
  }

  if (pe && !perf) {
    return (
      <div className="max-w-xl mx-auto py-24 px-4">
        <ErrorState message={(pe as Error).message} onRetry={() => pr()} />
      </div>
    )
  }

  const perfModels = perf?.models ?? []
  const modelInfos  = models ?? []

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fade-in">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex items-start justify-between mb-7">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 mb-1">Model Performance</h1>
          <p className="text-sm text-slate-500">
            Offline evaluation metrics from ML experiments — held-out test set.
            No placeholders or estimates.
          </p>
        </div>
        <button
          onClick={() => pr()}
          className="p-2 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* ── No data ─────────────────────────────────────────────────────── */}
      {perfModels.length === 0 && <NoModelsState />}

      {/* ── Has data ────────────────────────────────────────────────────── */}
      {perfModels.length > 0 && (
        <>
          {/* Metric selector */}
          <div className="flex flex-wrap gap-2 mb-6">
            {METRICS.map((m) => (
              <button
                key={m.id}
                onClick={() => setMetric(m.id)}
                title={m.desc}
                className={cn(
                  'px-4 py-1.5 rounded-lg text-sm font-medium transition-all',
                  metric === m.id
                    ? 'bg-brand-600 text-white shadow-sm'
                    : 'bg-slate-800 text-slate-400 hover:text-slate-200 border border-slate-700',
                )}
              >
                {m.label}
              </button>
            ))}
          </div>

          {/* Best model banner */}
          <BestModelBanner models={perfModels} metric={metric} />

          {/* Charts */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            <div className="glass-card p-6">
              <h2 className="section-title mb-1">
                {METRICS.find((m) => m.id === metric)?.label} by Model
              </h2>
              <p className="text-xs text-slate-600 mb-4">
                {METRICS.find((m) => m.id === metric)?.desc}
              </p>
              <PerformanceBar models={perfModels} metric={metric as 'accuracy' | 'f1_score' | 'roc_auc'} />
            </div>

            <div className="glass-card p-6">
              <h2 className="section-title mb-1">Multi-Metric Comparison</h2>
              <p className="text-xs text-slate-600 mb-4">
                Accuracy, Precision, Recall, F1, ROC-AUC — all models overlaid
              </p>
              <ModelRadar models={perfModels} />
            </div>
          </div>

          {/* Full metrics table */}
          <div className="glass-card overflow-hidden mb-6">
            <div className="px-5 py-4 border-b border-slate-800">
              <h2 className="section-title">Full Metrics Table</h2>
              <p className="text-xs text-slate-500 mt-1">
                ISOT Fake News Dataset — held-out test set (10% of data, stratified split)
              </p>
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
                    <th className="px-5 py-3 text-right label-sm">Test N</th>
                  </tr>
                </thead>
                <tbody>
                  {perfModels.map((m) => (
                    <tr
                      key={`${m.model_name}-${m.version}`}
                      className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors"
                    >
                      <td className="px-5 py-4">
                        <div className="font-medium text-sm text-slate-200">
                          {m.model_name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                        </div>
                        <div className="text-xs text-slate-700 font-mono mt-0.5">
                          v{m.version} · {m.algorithm.replace(/_/g, ' ')}
                        </div>
                      </td>
                      {(['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'] as Metric[]).map((key) => (
                        <td key={key} className="px-5 py-4 text-right">
                          <MetricCell value={m[key as keyof ModelPerformance] as number | undefined} highlight={key === metric} />
                        </td>
                      ))}
                      <td className="px-5 py-4 text-right text-sm text-slate-500 font-mono">
                        {m.test_samples?.toLocaleString() ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Per-metric mini bars */}
          <div className="glass-card p-6 mb-6">
            <h2 className="section-title mb-4">Metric Comparison Bars</h2>
            <div className="space-y-5">
              {METRICS.map((met) => (
                <div key={met.id}>
                  <div className="label-sm mb-2">{met.label}</div>
                  <div className="space-y-2">
                    {perfModels
                      .filter((m) => m[met.id as keyof ModelPerformance] != null)
                      .sort((a, b) => (b[met.id as keyof ModelPerformance] as number) - (a[met.id as keyof ModelPerformance] as number))
                      .map((m) => {
                        const val = m[met.id as keyof ModelPerformance] as number
                        return (
                          <div key={m.model_name} className="flex items-center gap-3">
                            <span className="text-xs text-slate-500 w-48 shrink-0 truncate">
                              {m.model_name.replace(/_/g, ' ')}
                            </span>
                            <div className="flex-1">
                              <ProgressBar
                                value={val}
                                color={val >= 0.95 ? 'bg-emerald-500' : val >= 0.85 ? 'bg-brand-500' : 'bg-amber-500'}
                                size="sm"
                                showLabel
                              />
                            </div>
                          </div>
                        )
                      })}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Registered model cards */}
          {modelInfos.length > 0 && (
            <div className="mb-6">
              <h2 className="section-title mb-4">Registered Model Versions</h2>
              <div className="space-y-3">
                {modelInfos.map((m) => (
                  <ModelCard key={m.id} model={m} />
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* ── Dataset note (always shown) ──────────────────────────────────── */}
      <div className="glass-card p-5 flex items-start gap-3">
        <Info className="w-4 h-4 text-slate-600 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-600 leading-relaxed space-y-1">
          <p>
            <span className="text-slate-400 font-medium">Training dataset:</span>{' '}
            ISOT Fake News Dataset (Ahmed et al., 2018) — ~44,900 articles,
            stratified 80/10/10 train/val/test split.
            Metrics were computed on the held-out 10% test set.
          </p>
          <p>
            <span className="text-slate-400 font-medium">These are offline evaluation results</span>{' '}
            from ML experiments — they describe how each model performed on the test dataset,
            not how it will perform on arbitrary real-world content.
          </p>
          <p>
            <span className="text-slate-400 font-medium">High accuracy does not mean</span>{' '}
            the model will correctly classify every submission. Treat model predictions as
            probabilistic signals, not factual verdicts.
          </p>
        </div>
      </div>
    </div>
  )
}
