/**
 * Model Laboratory
 *
 * Complete training, evaluation, and inspection interface.
 * Data from: GET /api/v1/lab/models, /lab/comparison, /lab/report, /lab/train
 *
 * Tabs:
 *   Overview   — metric summary, best model banners, metric selectors
 *   Training   — configure + trigger training, monitor progress, view logs
 *   Evaluation — per-model detail: confusion matrix, loss curves, per-class metrics
 *   Report     — automated research comparison narrative
 *
 * All numbers from live API. No hardcoded metrics.
 */

import { useState, useEffect } from 'react'
import {
  FlaskConical, Trophy, RefreshCw, Play, Terminal,
  CheckCircle2, XCircle, Loader2, ChevronDown, ChevronUp,
  AlertTriangle, Info, Database, Clock, Cpu, BarChart3,
  TrendingUp, FileText, Award, BookOpen, Zap,
} from 'lucide-react'
import {
  BarChart, Bar, Cell, LineChart, Line,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
  CartesianGrid, Legend, ReferenceLine,
} from 'recharts'
import {
  useLabModels, useLabComparison, useLabReport,
  useLabCurves, useLabConfusion, useLabRoc,
  useStartTraining, useJobStatus,
} from '@/hooks/useApi'
import { Spinner } from '@/components/ui/Spinner'
import { SkeletonCard, SkeletonChart } from '@/components/ui/Skeleton'
import { cn } from '@/lib/utils'
import type {
  ModelDetailResponse, ResearchReport, BestModelEntry,
} from '@/types/api'

const TT = {
  contentStyle: { backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)', borderRadius: 8 },
  labelStyle: { color: 'var(--text-primary)', fontWeight: 600 },
  itemStyle:  { color: 'var(--text-secondary)' },
}
const MODEL_COLORS = ['#818cf8', '#34d399', '#f87171', '#fbbf24', '#a78bfa', '#60a5fa']

const METRICS = [
  { id: 'accuracy',  label: 'Accuracy',  color: '#818cf8' },
  { id: 'f1_score',  label: 'F1 Score',  color: '#34d399' },
  { id: 'roc_auc',   label: 'ROC-AUC',   color: '#a78bfa' },
  { id: 'precision', label: 'Precision', color: '#60a5fa' },
  { id: 'recall',    label: 'Recall',    color: '#fbbf24' },
]

function fmt(v: number | null | undefined, pct = true): string {
  if (v == null) return '—'
  return pct ? `${(v * 100).toFixed(2)}%` : v.toString()
}

function fmtBytes(b: number | null): string {
  if (b == null) return '—'
  if (b < 1024) return `${b} B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`
  return `${(b / 1024 / 1024).toFixed(1)} MB`
}

// ── Overview Tab ──────────────────────────────────────────────────────────────

function OverviewTab({ models }: { models: ModelDetailResponse[] }) {
  const [selectedMetric, setSelectedMetric] = useState('f1_score')
  const metric = METRICS.find((m) => m.id === selectedMetric)!

  const sorted = [...models].sort((a, b) => {
    const av = (a[selectedMetric as keyof ModelDetailResponse] as number | null) ?? 0
    const bv = (b[selectedMetric as keyof ModelDetailResponse] as number | null) ?? 0
    return bv - av
  })

  const best = sorted[0]

  // Bar chart data
  const barData = models.map((m, idx) => ({
    name:  m.model_name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()).replace('Logistic Regression','LR').replace('Linear Svm','SVM').replace('Naive Bayes','NB').replace('Distilbert','DistilBERT').replace('Bert', 'BERT'),
    value: ((m[selectedMetric as keyof ModelDetailResponse] as number | null) ?? 0) * 100,
    fill:  MODEL_COLORS[idx % MODEL_COLORS.length],
  }))

  // Multi-metric grouped bar
  const groupedData = METRICS.map(({ id, label }) => {
    const entry: Record<string, unknown> = { metric: label }
    models.forEach((m, idx) => {
      const v = m[id as keyof ModelDetailResponse] as number | null
      entry[`m${idx}`] = v != null ? Math.round(v * 10000) / 100 : null
    })
    return entry
  })

  return (
    <div className="space-y-6">
      {/* Best model banner */}
      {best && best[selectedMetric as keyof ModelDetailResponse] != null && (
        <div className="card p-4 flex items-center gap-3" style={{ borderColor: 'rgba(52,211,153,0.2)', backgroundColor: 'rgba(52,211,153,0.04)' }}>
          <Trophy className="w-5 h-5 text-emerald-400 shrink-0" />
          <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Best {metric.label}:{' '}
            <span className="font-semibold text-emerald-400">{best.model_name.replace(/_/g, ' ')}</span>
            {' '}<span className="font-mono text-emerald-400">{fmt(best[selectedMetric as keyof ModelDetailResponse] as number)}</span>
            {' '}— held-out test set
          </p>
        </div>
      )}

      {/* Metric selector */}
      <div className="flex flex-wrap gap-2">
        {METRICS.map((m) => (
          <button
            key={m.id}
            onClick={() => setSelectedMetric(m.id)}
            className={cn('px-4 py-1.5 rounded-lg text-sm font-medium transition-all', selectedMetric === m.id ? 'text-white' : 'hover:bg-surface-2')}
            style={selectedMetric === m.id
              ? { background: 'linear-gradient(135deg, var(--brand), var(--accent))' }
              : { backgroundColor: 'var(--bg-surface-2)', color: 'var(--text-secondary)', border: '1px solid var(--border)' }
            }
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Single-metric bar */}
        <div className="card p-5">
          <h3 className="section-title mb-4">{metric.label} by Model</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 4, right: 4, bottom: 20, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} tickLine={false} axisLine={false} angle={-20} textAnchor="end" />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} unit="%" />
              <Tooltip {...TT} formatter={(v: number) => [`${v.toFixed(2)}%`, metric.label]} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {barData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* All-metrics grouped */}
        <div className="card p-5">
          <h3 className="section-title mb-4">All Metrics Grouped</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={groupedData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="metric" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} tickLine={false} axisLine={false} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} unit="%" />
              <Tooltip {...TT} formatter={(v: number) => [`${v?.toFixed(2)}%`]} />
              <Legend iconType="circle" iconSize={8} formatter={(v) => {
                const idx = parseInt(v.replace('m', ''))
                const m = models[idx]
                return <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{m?.model_name.replace(/_/g, ' ')}</span>
              }} />
              {models.map((_, idx) => (
                <Bar key={idx} dataKey={`m${idx}`} fill={MODEL_COLORS[idx % MODEL_COLORS.length]} radius={[2, 2, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Full metrics table */}
      <div className="card overflow-hidden">
        <div className="px-5 py-4" style={{ borderBottom: '1px solid var(--border)' }}>
          <h3 className="section-title">Full Metrics Table</h3>
          <p className="text-xs mt-1" style={{ color: 'var(--text-subtle)' }}>
            ISOT — 10% held-out test set, stratified split, same seed for all models
          </p>
        </div>
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead><tr>
              <th>Model</th>
              <th className="text-right">Acc</th>
              <th className="text-right">Prec</th>
              <th className="text-right">Recall</th>
              <th className="text-right">F1</th>
              <th className="text-right">AUC</th>
              <th className="text-right">Train(s)</th>
              <th className="text-right">Infer(ms)</th>
              <th className="text-right">Size</th>
              <th className="text-right">Test N</th>
            </tr></thead>
            <tbody>
              {sorted.map((m, idx) => (
                <tr key={m.id}>
                  <td>
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: MODEL_COLORS[models.indexOf(m) % MODEL_COLORS.length] }} />
                      <div>
                        <div className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
                          {m.model_name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                        </div>
                        <div className="text-xs font-mono" style={{ color: 'var(--text-faint)' }}>v{m.version} · {m.model_type}</div>
                      </div>
                    </div>
                  </td>
                  {['accuracy','precision','recall','f1_score','roc_auc'].map((key) => {
                    const v = m[key as keyof ModelDetailResponse] as number | null
                    const isBest = sorted[0].model_name === m.model_name && key === selectedMetric
                    return (
                      <td key={key} className="text-right">
                        <span className={cn('font-mono text-sm', isBest && 'font-bold')} style={{ color: isBest ? METRICS.find(x=>x.id===key)?.color : 'var(--text-secondary)' }}>
                          {fmt(v)}
                        </span>
                      </td>
                    )
                  })}
                  <td className="text-right font-mono text-sm" style={{ color: 'var(--text-subtle)' }}>
                    {m.train_time_s != null ? `${m.train_time_s.toFixed(1)}` : '—'}
                  </td>
                  <td className="text-right font-mono text-sm" style={{ color: 'var(--text-subtle)' }}>
                    {m.inference_time_ms != null ? `${m.inference_time_ms.toFixed(3)}` : '—'}
                  </td>
                  <td className="text-right text-sm" style={{ color: 'var(--text-subtle)' }}>
                    {fmtBytes(m.model_size_bytes)}
                  </td>
                  <td className="text-right font-mono text-sm" style={{ color: 'var(--text-subtle)' }}>
                    {m.test_samples?.toLocaleString() ?? '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Speed comparison */}
      {models.some((m) => m.inference_time_ms != null) && (
        <div className="grid sm:grid-cols-2 gap-6">
          <div className="card p-5">
            <h3 className="section-title mb-4">Training Time (seconds)</h3>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={models.filter(m=>m.train_time_s!=null).map((m,i)=>({name:m.model_name.replace(/_/g,' '),value:m.train_time_s,fill:MODEL_COLORS[i%MODEL_COLORS.length]}))} layout="vertical" margin={{left:80,right:20,top:4,bottom:4}}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{fontSize:10,fill:'var(--text-faint)'}} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" tick={{fontSize:11,fill:'var(--text-secondary)'}} tickLine={false} axisLine={false} width={80} />
                <Tooltip {...TT} formatter={(v:number)=>[`${v.toFixed(1)}s`,'Train time']} />
                <Bar dataKey="value" radius={[0,3,3,0]}>{models.filter(m=>m.train_time_s!=null).map((_,i)=><Cell key={i} fill={MODEL_COLORS[i%MODEL_COLORS.length]} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card p-5">
            <h3 className="section-title mb-4">Inference Latency (ms/sample)</h3>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={models.filter(m=>m.inference_time_ms!=null).map((m,i)=>({name:m.model_name.replace(/_/g,' '),value:m.inference_time_ms,fill:MODEL_COLORS[i%MODEL_COLORS.length]}))} layout="vertical" margin={{left:80,right:20,top:4,bottom:4}}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" tick={{fontSize:10,fill:'var(--text-faint)'}} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" tick={{fontSize:11,fill:'var(--text-secondary)'}} tickLine={false} axisLine={false} width={80} />
                <Tooltip {...TT} formatter={(v:number)=>[`${v.toFixed(3)}ms`,'Inference']} />
                <Bar dataKey="value" radius={[0,3,3,0]}>{models.filter(m=>m.inference_time_ms!=null).map((_,i)=><Cell key={i} fill={MODEL_COLORS[i%MODEL_COLORS.length]} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Training Tab ──────────────────────────────────────────────────────────────

function TrainingTab() {
  const [selectedModel, setSelectedModel] = useState('logistic_regression')
  const [dataDir, setDataDir] = useState('ml/datasets/processed')
  const [activeJob, setActiveJob] = useState<string | null>(null)

  const trainMutation = useStartTraining()
  const { data: jobData, isLoading: jobLoading } = useJobStatus(activeJob)

  const handleTrain = async () => {
    const result = await trainMutation.mutateAsync({ model_id: selectedModel, data_dir: dataDir })
    setActiveJob(result.job_id)
  }

  const isRunning = jobData?.status === 'running' || jobData?.status === 'queued'
  const isComplete = jobData?.status === 'completed'
  const isFailed = jobData?.status === 'failed'

  const TRAINABLE = [
    { id: 'logistic_regression', label: 'Logistic Regression', desc: 'TF-IDF + LR (C=1.0, balanced)' },
    { id: 'linear_svm',          label: 'Linear SVM',          desc: 'TF-IDF + LinearSVC (calibrated)' },
    { id: 'naive_bayes',         label: 'Naive Bayes',          desc: 'TF-IDF + MultinomialNB (α=0.1)' },
  ]

  return (
    <div className="space-y-5">
      {/* Info banner */}
      <div className="card p-4 flex items-start gap-3" style={{ borderColor: 'rgba(99,102,241,0.2)', backgroundColor: 'rgba(99,102,241,0.04)' }}>
        <Info className="w-4 h-4 shrink-0 mt-0.5" style={{ color: 'var(--brand)' }} />
        <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          <p className="font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Prerequisites</p>
          <p>Processed dataset splits must exist. Run <code className="font-mono text-xs" style={{ color: 'var(--brand)' }}>python scripts/prepare_dataset.py --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv</code> first.</p>
          <p className="mt-1">Training runs on the server in a background thread. After completion, metrics are written to the database.</p>
        </div>
      </div>

      {/* Model selector */}
      <div className="card p-5">
        <h3 className="section-title mb-4">Select Model to Train</h3>
        <div className="space-y-2 mb-5">
          {TRAINABLE.map((m) => (
            <label key={m.id} className={cn('flex items-start gap-3 p-3 rounded-xl cursor-pointer border transition-all', selectedModel === m.id ? 'ring-1' : '')}
              style={{
                borderColor: selectedModel === m.id ? 'var(--brand)' : 'var(--border)',
                backgroundColor: selectedModel === m.id ? 'var(--brand-light)' : 'var(--bg-surface-2)',
              }}>
              <input type="radio" value={m.id} checked={selectedModel === m.id} onChange={() => setSelectedModel(m.id)} className="mt-0.5" />
              <div>
                <div className="text-sm font-medium" style={{ color: selectedModel === m.id ? 'var(--brand)' : 'var(--text-primary)' }}>{m.label}</div>
                <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>{m.desc}</div>
              </div>
            </label>
          ))}
        </div>

        <div className="mb-4">
          <label className="label-sm block mb-1.5">Data directory</label>
          <input
            className="input-field font-mono text-sm"
            value={dataDir}
            onChange={(e) => setDataDir(e.target.value)}
          />
        </div>

        <div className="card p-3 mb-4" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
          <div className="label-sm mb-2">Training configuration</div>
          <div className="text-xs space-y-1 font-mono" style={{ color: 'var(--text-secondary)' }}>
            {selectedModel === 'logistic_regression' && <>
              <div>C=1.0 · solver=lbfgs · max_iter=1000 · class_weight=balanced</div>
              <div>TF-IDF: 50k features · 1–2 ngrams · sublinear_tf=True</div>
            </>}
            {selectedModel === 'linear_svm' && <>
              <div>C=1.0 · max_iter=2000 · class_weight=balanced</div>
              <div>CalibratedClassifierCV(cv=5, method=sigmoid)</div>
              <div>TF-IDF: 50k features · 1–2 ngrams · sublinear_tf=True</div>
            </>}
            {selectedModel === 'naive_bayes' && <>
              <div>MultinomialNB(alpha=0.1)</div>
              <div>TF-IDF: 50k features · 1–2 ngrams · sublinear_tf=True</div>
            </>}
            <div>random_seed=42 · stratified 80/10/10 split</div>
          </div>
        </div>

        <button
          onClick={handleTrain}
          disabled={trainMutation.isPending || isRunning}
          className="btn-primary w-full justify-center"
        >
          {trainMutation.isPending || isRunning
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Training…</>
            : <><Play className="w-4 h-4" /> Start Training</>
          }
        </button>
      </div>

      {/* Job monitor */}
      {jobData && (
        <div className="card overflow-hidden">
          <div className="px-5 py-4 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)' }}>
            <div className="flex items-center gap-2">
              <Terminal className="w-4 h-4" style={{ color: 'var(--brand)' }} />
              <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                Job {jobData.job_id}
              </span>
              <span className={cn(
                'text-xs px-2 py-0.5 rounded-full font-medium',
                isComplete ? 'text-emerald-400 bg-emerald-400/10' :
                isFailed   ? 'text-red-400 bg-red-400/10' :
                isRunning  ? 'text-amber-400 bg-amber-400/10' :
                'text-slate-400 bg-slate-400/10',
              )}>
                {jobData.status}
              </span>
            </div>
            {isComplete && (
              <span className="text-xs text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Complete — metrics updated in DB
              </span>
            )}
            {isFailed && (
              <span className="text-xs text-red-400 flex items-center gap-1">
                <XCircle className="w-3.5 h-3.5" /> {jobData.error}
              </span>
            )}
          </div>

          {/* Log output */}
          <div
            className="p-4 font-mono text-xs overflow-y-auto max-h-64"
            style={{ backgroundColor: 'var(--bg-sidebar)', color: '#94d82d' }}
          >
            {jobData.log_lines.length === 0 ? (
              <span style={{ color: 'var(--text-faint)' }}>Waiting for log output…</span>
            ) : (
              jobData.log_lines.map((line: string, i: number) => (
                <div key={i} className="leading-5">{line}</div>
              ))
            )}
            {isRunning && (
              <div className="flex items-center gap-1.5 mt-1" style={{ color: '#94d82d' }}>
                <Loader2 className="w-3 h-3 animate-spin" /> running…
              </div>
            )}
          </div>
        </div>
      )}

      {/* Transformer note */}
      <div className="card p-4 flex items-start gap-3" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
        <Cpu className="w-4 h-4 shrink-0 mt-0.5" style={{ color: 'var(--accent)' }} />
        <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
          <p className="font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Training DistilBERT / BERT</p>
          <p>Transformer training requires a GPU and is not available through this UI. Use the CLI:</p>
          <code className="block mt-1.5 text-xs font-mono p-2 rounded" style={{ backgroundColor: 'var(--bg-sidebar)', color: '#94d82d' }}>
            python scripts/train_transformer.py --epochs 3<br/>
            python scripts/evaluate_transformer.py<br/>
            python scripts/register_models.py
          </code>
        </div>
      </div>
    </div>
  )
}

// ── Evaluation Tab ────────────────────────────────────────────────────────────

function EvaluationTab({ models }: { models: ModelDetailResponse[] }) {
  const [selectedId, setSelectedId] = useState<number | null>(models[0]?.id ?? null)
  const { data: curveData } = useLabCurves(selectedId)
  const { data: confData }  = useLabConfusion(selectedId)
  const { data: rocData }   = useLabRoc(selectedId)

  const model = models.find((m) => m.id === selectedId)

  return (
    <div className="grid lg:grid-cols-4 gap-6">
      {/* Model selector */}
      <div className="lg:col-span-1">
        <div className="label-sm mb-2">Select Model</div>
        <div className="space-y-1.5">
          {models.map((m) => (
            <button
              key={m.id}
              onClick={() => setSelectedId(m.id)}
              className={cn('w-full text-left px-3 py-2.5 rounded-xl text-sm transition-all', selectedId === m.id ? 'ring-1' : 'hover:bg-surface-2')}
              style={selectedId === m.id
                ? { backgroundColor: 'var(--brand-light)', outline: '1px solid var(--brand)', color: 'var(--brand)' }
                : { color: 'var(--text-secondary)' }
              }
            >
              <div className="font-medium">{m.model_name.replace(/_/g, ' ')}</div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>
                {m.accuracy != null ? `Acc ${(m.accuracy * 100).toFixed(1)}%` : 'No metrics yet'}
              </div>
            </button>
          ))}
        </div>

        {model && (
          <div className="mt-4 card p-4 space-y-2 text-xs">
            <div className="label-sm mb-2">Model Details</div>
            {[
              ['Algorithm', model.algorithm],
              ['Version', model.version],
              ['Dataset', model.dataset_name ?? '—'],
              ['Train N', model.training_samples?.toLocaleString() ?? '—'],
              ['Test N', model.test_samples?.toLocaleString() ?? '—'],
              ['Random seed', String(model.random_seed ?? 42)],
              ['Size', fmtBytes(model.model_size_bytes)],
              ['Train time', model.train_time_s != null ? `${model.train_time_s.toFixed(1)}s` : '—'],
              ['Inference', model.inference_time_ms != null ? `${model.inference_time_ms.toFixed(3)}ms/sample` : '—'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span style={{ color: 'var(--text-subtle)' }}>{k}</span>
                <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>{v}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Charts */}
      <div className="lg:col-span-3 space-y-5">
        {/* Confusion matrix */}
        <div className="card p-5">
          <h3 className="section-title mb-4">Confusion Matrix</h3>
          {confData?.matrix ? (
            <div className="flex flex-col items-center gap-4">
              <div className="grid grid-cols-3 gap-1 w-64">
                {/* Header */}
                <div />
                <div className="text-center text-xs font-semibold text-red-400 pb-1">Predicted FAKE</div>
                <div className="text-center text-xs font-semibold text-emerald-400 pb-1">Predicted REAL</div>
                {/* Row 1: Actual FAKE */}
                <div className="text-xs font-semibold text-red-400 flex items-center pr-2">Actual FAKE</div>
                <div className="p-4 rounded-lg text-center" style={{ backgroundColor: 'rgba(52,211,153,0.15)' }}>
                  <div className="text-2xl font-bold text-emerald-400">{confData.matrix[0]?.[0] ?? 0}</div>
                  <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>TN</div>
                </div>
                <div className="p-4 rounded-lg text-center" style={{ backgroundColor: 'rgba(248,113,113,0.1)' }}>
                  <div className="text-2xl font-bold text-red-400">{confData.matrix[0]?.[1] ?? 0}</div>
                  <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>FP</div>
                </div>
                {/* Row 2: Actual REAL */}
                <div className="text-xs font-semibold text-emerald-400 flex items-center pr-2">Actual REAL</div>
                <div className="p-4 rounded-lg text-center" style={{ backgroundColor: 'rgba(248,113,113,0.1)' }}>
                  <div className="text-2xl font-bold text-red-400">{confData.matrix[1]?.[0] ?? 0}</div>
                  <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>FN</div>
                </div>
                <div className="p-4 rounded-lg text-center" style={{ backgroundColor: 'rgba(52,211,153,0.15)' }}>
                  <div className="text-2xl font-bold text-emerald-400">{confData.matrix[1]?.[1] ?? 0}</div>
                  <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>TP</div>
                </div>
              </div>
              <p className="text-xs" style={{ color: 'var(--text-subtle)' }}>
                Test samples: {confData.test_samples?.toLocaleString() ?? '—'}
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 gap-2" style={{ color: 'var(--text-faint)' }}>
              <BarChart3 className="w-8 h-8 opacity-30" />
              <p className="text-sm">Run evaluation to generate confusion matrix</p>
            </div>
          )}
        </div>

        {/* ROC Curve */}
        <div className="card p-5">
          <h3 className="section-title mb-1">ROC Curve</h3>
          <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>
            {rocData?.note ?? 'AUC: ' + fmt(model?.roc_auc)}
          </p>
          {rocData?.points?.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={rocData.points} margin={{ top: 4, right: 20, bottom: 4, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="fpr" type="number" domain={[0,1]} tick={{fontSize:10,fill:'var(--text-faint)'}} tickLine={false} axisLine={false} label={{value:'FPR',position:'insideBottom',offset:-2,style:{fill:'var(--text-faint)',fontSize:10}}} />
                <YAxis domain={[0,1]} tick={{fontSize:10,fill:'var(--text-faint)'}} tickLine={false} axisLine={false} label={{value:'TPR',angle:-90,position:'insideLeft',style:{fill:'var(--text-faint)',fontSize:10}}} />
                <Tooltip {...TT} formatter={(v:number)=>[v.toFixed(3)]} />
                <ReferenceLine stroke="var(--border)" strokeDasharray="4 4" segment={[{x:0,y:0},{x:1,y:1}]} />
                <Line type="monotone" dataKey="tpr" stroke="#818cf8" strokeWidth={2} dot={false} name={`AUC=${fmt(model?.roc_auc)}`} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex justify-center py-8" style={{ color: 'var(--text-faint)' }}>
              <p className="text-sm">No ROC data — run evaluation first</p>
            </div>
          )}
        </div>

        {/* Loss curves */}
        <div className="card p-5">
          <h3 className="section-title mb-1">Training & Validation Loss</h3>
          <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>
            {curveData?.note ?? 'Per-epoch loss curves (transformer models only)'}
          </p>
          {curveData?.training_history?.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={curveData.training_history} margin={{ top: 4, right: 10, bottom: 4, left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" tick={{fontSize:10,fill:'var(--text-faint)'}} tickLine={false} axisLine={false} />
                <YAxis tick={{fontSize:10,fill:'var(--text-faint)'}} tickLine={false} axisLine={false} />
                <Tooltip {...TT} />
                <Legend iconType="circle" iconSize={8} formatter={(v)=><span style={{color:'var(--text-secondary)',fontSize:11}}>{v}</span>} />
                <Line type="monotone" dataKey="train_loss" stroke="#f87171" strokeWidth={2} dot={false} name="Train loss" />
                <Line type="monotone" dataKey="val_loss"   stroke="#818cf8" strokeWidth={2} dot={false} name="Val loss"   />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 gap-2" style={{ color: 'var(--text-faint)' }}>
              <TrendingUp className="w-8 h-8 opacity-30" />
              <p className="text-sm">
                {model?.model_type === 'transformer'
                  ? 'No training history stored. Re-train to capture loss curves.'
                  : 'Baseline models are single-pass — no per-epoch loss curves.'}
              </p>
            </div>
          )}
        </div>

        {/* Per-class metrics */}
        {model?.per_class_metrics && (
          <div className="card overflow-hidden">
            <div className="px-5 py-4" style={{ borderBottom: '1px solid var(--border)' }}>
              <h3 className="section-title">Per-Class Metrics</h3>
            </div>
            <table className="data-table">
              <thead><tr>
                <th>Class</th>
                <th className="text-right">Precision</th>
                <th className="text-right">Recall</th>
                <th className="text-right">F1</th>
                <th className="text-right">Support</th>
              </tr></thead>
              <tbody>
                {Object.entries(model.per_class_metrics).map(([cls, m]) => (
                  <tr key={cls}>
                    <td><span className={cn('font-semibold text-sm', cls === 'FAKE' ? 'text-red-400' : 'text-emerald-400')}>{cls}</span></td>
                    <td className="text-right font-mono">{fmt(m.precision)}</td>
                    <td className="text-right font-mono">{fmt(m.recall)}</td>
                    <td className="text-right font-mono">{fmt(m.f1)}</td>
                    <td className="text-right font-mono" style={{ color: 'var(--text-subtle)' }}>{m.support?.toLocaleString() ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Report Tab ────────────────────────────────────────────────────────────────

function BestCard({ entry, label, icon }: { entry: BestModelEntry; label: string; icon: React.ReactNode }) {
  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ backgroundColor: 'var(--brand-light)' }}>
          <span style={{ color: 'var(--brand)' }}>{icon}</span>
        </div>
        <span className="text-xs font-semibold" style={{ color: 'var(--text-subtle)' }}>{label}</span>
      </div>
      <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
        {entry.model_name === 'N/A' ? '—' : entry.model_name.replace(/_/g, ' ')}
      </div>
      <div className="text-sm font-mono text-emerald-400">
        {entry.value != null ? (
          entry.metric === 'train_time_s' ? `${entry.value.toFixed(1)}s` :
          entry.metric === 'inference_time_ms' ? `${entry.value.toFixed(3)}ms` :
          `${(entry.value * 100).toFixed(2)}%`
        ) : '—'}
      </div>
    </div>
  )
}

function ReportTab({ report }: { report: ResearchReport }) {
  return (
    <div className="space-y-6">
      {/* Summary strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <BestCard entry={report.best_by_accuracy}  label="Best Accuracy"  icon={<Award className="w-4 h-4" />} />
        <BestCard entry={report.best_by_f1}        label="Best F1 Score"  icon={<BarChart3 className="w-4 h-4" />} />
        <BestCard entry={report.best_by_roc_auc}   label="Best ROC-AUC"  icon={<TrendingUp className="w-4 h-4" />} />
        <BestCard entry={report.fastest_inference}  label="Fastest Inference" icon={<Zap className="w-4 h-4" />} />
        <BestCard entry={report.best_by_precision} label="Best Precision" icon={<CheckCircle2 className="w-4 h-4" />} />
        <BestCard entry={report.best_by_recall}    label="Best Recall"   icon={<Database className="w-4 h-4" />} />
        <BestCard entry={report.fastest_training}   label="Fastest Training" icon={<Clock className="w-4 h-4" />} />
      </div>

      {/* Group comparison */}
      <div className="grid sm:grid-cols-2 gap-4">
        {[report.traditional_summary, report.transformer_summary].map((g) => (
          <div key={g.group} className="card p-5">
            <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>{g.group}</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span style={{ color: 'var(--text-subtle)' }}>Models</span>
                <span style={{ color: 'var(--text-secondary)' }}>{g.models.join(', ').replace(/_/g, ' ') || '—'}</span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: 'var(--text-subtle)' }}>Avg Accuracy</span>
                <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>{g.avg_accuracy != null ? `${(g.avg_accuracy * 100).toFixed(2)}%` : '—'}</span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: 'var(--text-subtle)' }}>Avg F1</span>
                <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>{g.avg_f1 != null ? `${(g.avg_f1 * 100).toFixed(2)}%` : '—'}</span>
              </div>
              <div className="flex justify-between">
                <span style={{ color: 'var(--text-subtle)' }}>Avg Train Time</span>
                <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>{g.avg_train_time != null ? `${g.avg_train_time.toFixed(1)}s` : '—'}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Narrative */}
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-4">
          <BookOpen className="w-4 h-4" style={{ color: 'var(--brand)' }} />
          <h3 className="section-title">Research Narrative</h3>
        </div>
        <div className="space-y-3 text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          {report.narrative.map((para, i) => <p key={i}>{para}</p>)}
        </div>
      </div>

      {/* Evaluation conditions */}
      <div className="card p-5">
        <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Evaluation Conditions</h3>
        <div className="space-y-2 text-xs font-mono">
          {Object.entries(report.evaluation_conditions).map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span style={{ color: 'var(--text-subtle)' }}>{k}</span>
              <span style={{ color: 'var(--text-secondary)' }}>{String(v)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Limitations */}
      <div className="card p-5" style={{ borderColor: 'rgba(251,191,36,0.2)', backgroundColor: 'rgba(251,191,36,0.03)' }}>
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle className="w-4 h-4 text-amber-400" />
          <h3 className="text-sm font-semibold text-amber-400">Limitations</h3>
        </div>
        <ul className="space-y-1.5">
          {report.limitations.map((l, i) => (
            <li key={i} className="text-xs flex items-start gap-2" style={{ color: 'var(--text-secondary)' }}>
              <span className="text-amber-400 shrink-0 mt-0.5">•</span> {l}
            </li>
          ))}
        </ul>
        <p className="text-xs mt-3" style={{ color: 'var(--text-faint)' }}>
          Generated {new Date(report.generated_at).toLocaleString()} · {report.total_models} models
        </p>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

type Tab = 'overview' | 'training' | 'evaluation' | 'report'

export default function ModelLaboratory() {
  const [activeTab, setActiveTab] = useState<Tab>('overview')

  const { data: models, isLoading: modelsLoading, refetch } = useLabModels()
  const { data: report, isLoading: reportLoading } = useLabReport()

  const modelList: ModelDetailResponse[] = models ?? []

  const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview',   label: 'Overview',   icon: <BarChart3   className="w-4 h-4" /> },
    { id: 'training',   label: 'Training',   icon: <Play        className="w-4 h-4" /> },
    { id: 'evaluation', label: 'Evaluation', icon: <FlaskConical className="w-4 h-4" /> },
    { id: 'report',     label: 'Report',     icon: <FileText    className="w-4 h-4" /> },
  ]

  return (
    <div className="page-content">
      {/* Header */}
      <div className="flex items-start justify-between mb-7">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-600 to-brand-500 flex items-center justify-center shrink-0">
            <FlaskConical className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="page-title mb-1">Model Laboratory</h1>
            <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
              Train, evaluate, and compare fake news detection models · all metrics from actual evaluation
            </p>
          </div>
        </div>
        <button onClick={() => refetch()} className="btn-ghost" title="Refresh">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-6 p-1 rounded-xl" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
        {TABS.map(({ id, label, icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={cn('flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-sm font-medium transition-all', activeTab === id ? 'text-white shadow-sm' : '')}
            style={activeTab === id
              ? { background: 'linear-gradient(135deg, var(--brand), var(--accent))' }
              : { color: 'var(--text-subtle)' }
            }
          >
            {icon}
            <span className="hidden sm:inline">{label}</span>
          </button>
        ))}
      </div>

      {/* Tab content */}
      {modelsLoading ? (
        <div className="space-y-4">
          {Array.from({ length: 3 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      ) : (
        <>
          {activeTab === 'overview' && (
            modelList.length === 0 ? (
              <div className="card p-10 text-center space-y-4">
                <Database className="w-12 h-12 mx-auto opacity-20" style={{ color: 'var(--text-faint)' }} />
                <p className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>No models with metrics yet</p>
                <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
                  Go to the Training tab to train a model, or run evaluation scripts.
                </p>
                <button onClick={() => setActiveTab('training')} className="btn-primary text-sm mx-auto">
                  <Play className="w-4 h-4" /> Go to Training
                </button>
              </div>
            ) : <OverviewTab models={modelList} />
          )}
          {activeTab === 'training'   && <TrainingTab />}
          {activeTab === 'evaluation' && <EvaluationTab models={modelList} />}
          {activeTab === 'report'     && (
            reportLoading ? <div className="flex justify-center py-16"><Spinner /></div> :
            report ? <ReportTab report={report as ResearchReport} /> :
            <div className="card p-10 text-center" style={{ color: 'var(--text-faint)' }}>No report available</div>
          )}
        </>
      )}

      {/* Footer note */}
      <div className="mt-8 card p-4 flex items-start gap-3">
        <Info className="w-4 h-4 shrink-0 mt-0.5" style={{ color: 'var(--text-faint)' }} />
        <div className="text-xs leading-relaxed space-y-1" style={{ color: 'var(--text-subtle)' }}>
          <p><strong style={{ color: 'var(--text-secondary)' }}>Research integrity:</strong> All models were evaluated on the same 10% held-out test split (random_seed=42, stratified). No test data was used during training or hyperparameter selection.</p>
          <p><strong style={{ color: 'var(--text-secondary)' }}>Dataset:</strong> ISOT Fake News Dataset (Ahmed et al., 2018) — ~44,900 articles, 2015–2018 US political news. High metrics on this dataset do not guarantee generalisation.</p>
        </div>
      </div>
    </div>
  )
}
