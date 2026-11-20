/**
 * Model Comparison
 *
 * Side-by-side comparison of all registered model versions.
 * Data: GET /api/v1/model-performance
 *
 * Features:
 *   - Metric selector
 *   - Grouped bar chart comparing all models across all metrics
 *   - Sortable table
 *   - Best/worst model highlights
 */

import { useState } from 'react'
import { useModelPerformance } from '@/hooks/useApi'
import { Spinner } from '@/components/ui/Spinner'
import { ErrorState } from '@/components/ui/ErrorState'
import { SkeletonChart } from '@/components/ui/Skeleton'
import {
  GitCompare, RefreshCw, Trophy, TrendingDown,
  ArrowUpDown,
} from 'lucide-react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, Cell,
  ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, Legend,
} from 'recharts'
import { cn, formatPct } from '@/lib/utils'
import type { ModelPerformance } from '@/types/api'

type Metric = 'accuracy' | 'f1_score' | 'roc_auc' | 'precision' | 'recall'
type SortDir = 'asc' | 'desc'

const METRICS: { id: Metric; label: string; color: string }[] = [
  { id: 'accuracy',  label: 'Accuracy',  color: '#818cf8' },
  { id: 'f1_score',  label: 'F1 Score',  color: '#34d399' },
  { id: 'roc_auc',   label: 'ROC-AUC',   color: '#a78bfa' },
  { id: 'precision', label: 'Precision', color: '#60a5fa' },
  { id: 'recall',    label: 'Recall',    color: '#fbbf24' },
]

const MODEL_COLORS = ['#818cf8', '#34d399', '#f87171', '#fbbf24', '#a78bfa', '#60a5fa']

const TOOLTIP_STYLE = {
  contentStyle: { backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)', borderRadius: 8 },
  labelStyle: { color: 'var(--text-primary)', fontWeight: 600 },
  itemStyle:  { color: 'var(--text-secondary)' },
}

// ── Grouped bar chart ─────────────────────────────────────────────────────────

function MetricGroupedBar({ models }: { models: ModelPerformance[] }) {
  // Build data: one entry per metric, bars per model
  const data = METRICS.map(({ id, label }) => {
    const entry: Record<string, unknown> = { metric: label }
    models.forEach((m) => {
      const val = m[id as keyof ModelPerformance]
      entry[m.model_name] = typeof val === 'number' ? Math.round(val * 10000) / 100 : null
    })
    return entry
  })

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="metric" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} tickLine={false} axisLine={false} />
        <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} unit="%" />
        <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(2)}%`]} />
        <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{v.replace(/_/g, ' ')}</span>} />
        {models.map((m, idx) => (
          <Bar key={m.model_name} dataKey={m.model_name} fill={MODEL_COLORS[idx % MODEL_COLORS.length]} radius={[2, 2, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Radar chart ───────────────────────────────────────────────────────────────

function MultiRadar({ models }: { models: ModelPerformance[] }) {
  const data = METRICS.map(({ id, label }) => {
    const entry: Record<string, unknown> = { metric: label }
    models.forEach((m) => {
      const val = m[id as keyof ModelPerformance]
      entry[m.model_name] = typeof val === 'number' ? Math.round(val * 10000) / 100 : 0
    })
    return entry
  })

  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart data={data}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
        <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 9, fill: 'var(--text-faint)' }} tickCount={4} />
        {models.map((m, idx) => (
          <Radar
            key={m.model_name}
            name={m.model_name.replace(/_/g, ' ')}
            dataKey={m.model_name}
            stroke={MODEL_COLORS[idx % MODEL_COLORS.length]}
            fill={MODEL_COLORS[idx % MODEL_COLORS.length]}
            fillOpacity={0.08}
            strokeWidth={1.5}
          />
        ))}
        <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{v}</span>} />
        <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(2)}%`]} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ModelComparison() {
  const [sortMetric, setSortMetric] = useState<Metric>('f1_score')
  const [sortDir,    setSortDir]    = useState<SortDir>('desc')

  const { data: perf, isLoading, error, refetch } = useModelPerformance()

  if (isLoading) {
    return (
      <div className="page-content">
        <div className="mb-7"><div className="h-7 w-48 skeleton mb-2" /><div className="h-4 w-64 skeleton" /></div>
        <div className="grid lg:grid-cols-2 gap-6">
          <SkeletonChart className="h-72" />
          <SkeletonChart className="h-72" />
        </div>
      </div>
    )
  }

  if (error) return <div className="page-content max-w-xl"><ErrorState message={(error as Error).message} onRetry={() => refetch()} /></div>

  const models = perf?.models ?? []

  if (models.length === 0) {
    return (
      <div className="page-content">
        <h1 className="page-title mb-2">Model Comparison</h1>
        <div className="card p-10 text-center mt-8">
          <GitCompare className="w-10 h-10 mx-auto mb-4" style={{ color: 'var(--text-faint)' }} />
          <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>No model data available. Run evaluation scripts first.</p>
        </div>
      </div>
    )
  }

  // Sort models for ranking table
  const sorted = [...models].sort((a, b) => {
    const av = (a[sortMetric] ?? 0) as number
    const bv = (b[sortMetric] ?? 0) as number
    return sortDir === 'desc' ? bv - av : av - bv
  })

  const best  = sorted[0]
  const worst = sorted[sorted.length - 1]

  const handleSort = (m: Metric) => {
    if (m === sortMetric) setSortDir(sortDir === 'desc' ? 'asc' : 'desc')
    else { setSortMetric(m); setSortDir('desc') }
  }

  return (
    <div className="page-content">
      {/* Header */}
      <div className="flex items-start justify-between mb-7">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-brand-500 to-emerald-500 flex items-center justify-center shrink-0">
            <GitCompare className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="page-title mb-1">Model Comparison</h1>
            <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
              Head-to-head comparison of all {models.length} registered models
            </p>
          </div>
        </div>
        <button onClick={() => refetch()} className="btn-ghost" title="Refresh">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Best / worst highlight */}
      <div className="grid sm:grid-cols-2 gap-4 mb-7">
        <div className="card p-4 flex items-center gap-3" style={{ borderColor: 'rgba(52,211,153,0.2)', backgroundColor: 'rgba(52,211,153,0.04)' }}>
          <Trophy className="w-5 h-5 text-emerald-400 shrink-0" />
          <div>
            <div className="text-xs font-semibold text-emerald-400 mb-0.5">Best by {METRICS.find((m) => m.id === sortMetric)?.label}</div>
            <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
              {best.model_name.replace(/_/g, ' ')}
            </div>
            <div className="text-xs font-mono text-emerald-400">
              {formatPct((best[sortMetric] ?? 0) as number)}
            </div>
          </div>
        </div>
        <div className="card p-4 flex items-center gap-3" style={{ borderColor: 'rgba(248,113,113,0.2)', backgroundColor: 'rgba(248,113,113,0.04)' }}>
          <TrendingDown className="w-5 h-5 text-red-400 shrink-0" />
          <div>
            <div className="text-xs font-semibold text-red-400 mb-0.5">Lowest by {METRICS.find((m) => m.id === sortMetric)?.label}</div>
            <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
              {worst.model_name.replace(/_/g, ' ')}
            </div>
            <div className="text-xs font-mono text-red-400">
              {formatPct((worst[sortMetric] ?? 0) as number)}
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid lg:grid-cols-2 gap-6 mb-7">
        <div className="card p-5">
          <h2 className="section-title mb-4">All Metrics — Grouped Bars</h2>
          <MetricGroupedBar models={models} />
        </div>
        <div className="card p-5">
          <h2 className="section-title mb-4">Multi-Metric Radar</h2>
          <MultiRadar models={models} />
        </div>
      </div>

      {/* Comparison table */}
      <div className="card overflow-hidden">
        <div className="px-5 py-4 flex items-center gap-3" style={{ borderBottom: '1px solid var(--border)' }}>
          <h2 className="section-title">Sortable Comparison Table</h2>
          <span className="text-xs" style={{ color: 'var(--text-subtle)' }}>Click column headers to sort</span>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                {METRICS.map((m) => (
                  <th
                    key={m.id}
                    className="text-right cursor-pointer hover:text-primary transition-colors select-none"
                    onClick={() => handleSort(m.id)}
                    style={{ color: sortMetric === m.id ? 'var(--brand)' : undefined }}
                  >
                    <span className="inline-flex items-center gap-1">
                      {m.label}
                      <ArrowUpDown className="w-3 h-3 opacity-50" />
                    </span>
                  </th>
                ))}
                <th className="text-right">Test N</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((m, idx) => (
                <tr key={`${m.model_name}-${m.version}`}>
                  <td>
                    <div className={cn(
                      'w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold',
                    )} style={{
                      backgroundColor: idx === 0 ? 'rgba(52,211,153,0.15)' : idx === sorted.length - 1 ? 'rgba(248,113,113,0.1)' : 'var(--bg-surface-2)',
                      color: idx === 0 ? '#34d399' : idx === sorted.length - 1 ? '#f87171' : 'var(--text-subtle)',
                    }}>
                      {idx + 1}
                    </div>
                  </td>
                  <td>
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: MODEL_COLORS[models.indexOf(m) % MODEL_COLORS.length] }} />
                      <div>
                        <div className="font-medium text-sm" style={{ color: 'var(--text-secondary)' }}>
                          {m.model_name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                        </div>
                        <div className="text-xs font-mono" style={{ color: 'var(--text-faint)' }}>v{m.version}</div>
                      </div>
                    </div>
                  </td>
                  {METRICS.map((met) => {
                    const val = m[met.id as keyof ModelPerformance] as number | undefined
                    return (
                      <td key={met.id} className="text-right">
                        <span
                          className="font-mono text-sm"
                          style={{ color: met.id === sortMetric ? met.color : 'var(--text-secondary)' }}
                        >
                          {val != null ? formatPct(val) : '—'}
                        </span>
                      </td>
                    )
                  })}
                  <td className="text-right text-sm font-mono" style={{ color: 'var(--text-subtle)' }}>
                    {m.test_samples?.toLocaleString() ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
