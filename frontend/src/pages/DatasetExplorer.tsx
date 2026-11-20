/**
 * Dataset Explorer
 *
 * All data loaded from live API endpoints:
 *   GET /api/v1/datasets                    — catalog
 *   GET /api/v1/datasets/{name}/stats       — split statistics
 *   GET /api/v1/datasets/{name}/analytics   — word freq, quality, distribution
 *   GET /api/v1/datasets/{name}/samples     — row samples
 *   GET /api/v1/datasets/{name}/audit       — preprocessing audit log
 *   POST /api/v1/datasets/upload            — file upload
 *   POST /api/v1/datasets/preprocess        — trigger pipeline
 *
 * No hardcoded numbers anywhere.
 */

import { useState, useRef } from 'react'
import {
  Database, FileText, ChevronDown, ChevronUp, ExternalLink,
  Info, BarChart3, Upload, RefreshCw, CheckCircle2, XCircle,
  AlertTriangle, Clock, Download, Play, Shield, Eye,
  BookOpen, ChevronRight, Loader2, X, FileUp,
} from 'lucide-react'
import {
  BarChart, Bar, Cell, PieChart, Pie,
  ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, Legend,
} from 'recharts'
import {
  useDatasetCatalog, useDatasetStats, useDatasetAnalytics,
  useDatasetSamples, useDatasetAudit, useUploadDataset,
  usePreprocessDataset,
} from '@/hooks/useApi'
import { Spinner } from '@/components/ui/Spinner'
import { SkeletonCard, SkeletonChart } from '@/components/ui/Skeleton'
import { cn } from '@/lib/utils'
import type { DatasetCatalogEntry, WordFreqEntry, AuditStep } from '@/types/api'

// ── Tooltip style ─────────────────────────────────────────────────────────────
const TT_STYLE = {
  contentStyle: {
    backgroundColor: 'var(--bg-surface-2)',
    border: '1px solid var(--border)',
    borderRadius: 8,
  },
  labelStyle: { color: 'var(--text-primary)', fontWeight: 600 },
  itemStyle:  { color: 'var(--text-secondary)' },
}

// ── License badge ─────────────────────────────────────────────────────────────
function LicenseBadge({ license }: { license: DatasetCatalogEntry['license'] }) {
  const colorMap: Record<string, string> = {
    research_only: 'text-amber-400 bg-amber-400/10',
    cc_by:         'text-emerald-400 bg-emerald-400/10',
    unknown:       'text-slate-400 bg-slate-400/10',
  }
  const cls = colorMap[license.type] ?? colorMap.unknown
  return (
    <span className={cn('text-xs px-2.5 py-0.5 rounded-full font-medium', cls)}>
      {license.type === 'research_only' ? 'Research Only' :
       license.type === 'cc_by'         ? 'CC-BY' : 'Check License'}
    </span>
  )
}

// ── Dataset card in catalog ───────────────────────────────────────────────────
function DatasetCard({
  entry,
  selected,
  onSelect,
}: {
  entry: DatasetCatalogEntry
  selected: boolean
  onSelect: () => void
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={cn(
        'card transition-all duration-200',
        selected && 'ring-2',
      )}
      style={selected ? { ringColor: 'var(--brand)' } as React.CSSProperties : undefined}
    >
      {/* Header row */}
      <button
        className="w-full flex items-start gap-4 p-5 text-left"
        onClick={onSelect}
      >
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center shrink-0 mt-0.5"
          style={{
            backgroundColor: entry.is_downloaded ? 'rgba(52,211,153,0.12)' : 'var(--bg-surface-2)',
            border: `1px solid ${entry.is_downloaded ? 'rgba(52,211,153,0.2)' : 'var(--border)'}`,
          }}
        >
          <Database className="w-5 h-5" style={{ color: entry.is_downloaded ? '#34d399' : 'var(--text-faint)' }} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2 mb-1">
            <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
              {entry.display_name}
            </span>
            <LicenseBadge license={entry.license} />
            {entry.is_downloaded && (
              <span className="text-xs text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full font-medium flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> Downloaded
              </span>
            )}
            {entry.is_processed && (
              <span className="text-xs px-2 py-0.5 rounded-full font-medium flex items-center gap-1" style={{ color: 'var(--brand)', backgroundColor: 'var(--brand-light)' }}>
                <CheckCircle2 className="w-3 h-3" /> Processed
              </span>
            )}
            {!entry.is_downloaded && (
              <span className="text-xs text-slate-500 bg-slate-500/10 px-2 py-0.5 rounded-full font-medium">
                Not Downloaded
              </span>
            )}
          </div>
          <p className="text-xs line-clamp-2" style={{ color: 'var(--text-subtle)' }}>
            {entry.description}
          </p>
        </div>

        <div className="shrink-0 flex items-center gap-2">
          {entry.total_articles && (
            <span className="text-sm font-bold" style={{ color: 'var(--text-secondary)' }}>
              {entry.total_articles.toLocaleString()}
            </span>
          )}
          <button
            onClick={(e) => { e.stopPropagation(); setExpanded(!expanded) }}
            className="p-1.5 rounded-lg hover:bg-surface-2 transition-colors"
            style={{ color: 'var(--text-subtle)' }}
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </button>

      {/* Stats row */}
      {entry.total_articles && (
        <div className="px-5 pb-4 grid grid-cols-3 gap-3">
          {[
            { label: 'Total',       value: entry.total_articles?.toLocaleString() ?? '—',  color: 'var(--brand)' },
            { label: 'Fake',        value: entry.fake_articles?.toLocaleString()  ?? '—',  color: '#f87171' },
            { label: 'Real',        value: entry.real_articles?.toLocaleString()  ?? '—',  color: '#34d399' },
          ].map((s) => (
            <div key={s.label} className="rounded-xl p-3 text-center" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
              <div className="text-lg font-bold" style={{ color: s.color }}>{s.value}</div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>{s.label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Expanded detail */}
      {expanded && (
        <div className="px-5 pb-5 space-y-3 text-sm" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="pt-3 grid sm:grid-cols-2 gap-x-6 gap-y-2">
            {[
              ['Domain', entry.domain],
              ['Language', entry.language],
              ['License', entry.license.summary],
            ].map(([k, v]) => (
              <div key={k}>
                <span className="text-xs font-semibold" style={{ color: 'var(--text-subtle)' }}>{k}: </span>
                <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>{v}</span>
              </div>
            ))}
          </div>

          <div>
            <div className="text-xs font-semibold mb-1" style={{ color: 'var(--text-subtle)' }}>Citation</div>
            <p className="text-xs italic leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{entry.citation}</p>
          </div>

          <div className="flex flex-wrap gap-2">
            <a
              href={entry.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs flex items-center gap-1"
              style={{ color: 'var(--brand)' }}
            >
              Dataset page <ExternalLink className="w-3 h-3" />
            </a>
            {entry.paper_url && (
              <a
                href={entry.paper_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs flex items-center gap-1"
                style={{ color: 'var(--accent)' }}
              >
                Paper <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </div>

          {!entry.is_downloaded && (
            <div className="card p-3" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
              <div className="text-xs font-semibold mb-1.5" style={{ color: 'var(--text-secondary)' }}>Download command</div>
              <pre className="text-xs font-mono overflow-x-auto whitespace-pre-wrap" style={{ color: 'var(--brand)' }}>
                {entry.download_command}
              </pre>
            </div>
          )}

          <div className="flex flex-wrap gap-1.5">
            <span className="text-xs" style={{ color: 'var(--text-subtle)' }}>Expected files:</span>
            {entry.expected_files.map((f) => (
              <span
                key={f}
                className="text-xs px-2 py-0.5 rounded font-mono"
                style={{ backgroundColor: 'var(--bg-surface-3)', color: 'var(--text-secondary)' }}
              >
                {f}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Word frequency chart ──────────────────────────────────────────────────────
function WordFreqChart({
  data,
  title,
  color,
}: {
  data: WordFreqEntry[]
  title: string
  color: string
}) {
  const top = data.slice(0, 20)
  if (!top.length) return <div className="flex items-center justify-center h-32 text-sm" style={{ color: 'var(--text-faint)' }}>No data</div>

  return (
    <div>
      <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>{title}</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={top} layout="vertical" margin={{ left: 70, right: 10, top: 4, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
          <YAxis type="category" dataKey="word" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} tickLine={false} axisLine={false} width={70} />
          <Tooltip {...TT_STYLE} formatter={(v: number) => [v.toLocaleString(), 'count']} />
          <Bar dataKey="count" fill={color} radius={[0, 3, 3, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ── Length distribution chart ─────────────────────────────────────────────────
function LengthHistogram({ histogram }: { histogram: { bucket: string; count: number }[] }) {
  if (!histogram?.length) return null
  return (
    <ResponsiveContainer width="100%" height={160}>
      <BarChart data={histogram} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} allowDecimals={false} />
        <Tooltip {...TT_STYLE} />
        <Bar dataKey="count" fill="var(--brand)" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Audit log timeline ────────────────────────────────────────────────────────
function AuditTimeline({ steps }: { steps: AuditStep[] }) {
  const OP_COLORS: Record<string, string> = {
    load:                 '#818cf8',
    label_normalisation:  '#a78bfa',
    text_cleaning:        '#60a5fa',
    deduplication:        '#fbbf24',
    split:                '#34d399',
    statistics:           '#94a3b8',
  }

  return (
    <div className="space-y-3">
      {steps.map((step, idx) => (
        <div key={step.step} className="flex gap-3">
          <div className="flex flex-col items-center">
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold text-white shrink-0"
              style={{ backgroundColor: OP_COLORS[step.operation] ?? '#94a3b8' }}
            >
              {step.step}
            </div>
            {idx < steps.length - 1 && (
              <div className="flex-1 w-px mt-1" style={{ backgroundColor: 'var(--border)' }} />
            )}
          </div>
          <div className="pb-3 flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-0.5">
              <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>
                {step.operation.replace(/_/g, ' ')}
              </span>
              {step.dropped > 0 && (
                <span className="text-xs text-amber-400">−{step.dropped} rows</span>
              )}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>
              {step.rows_before.toLocaleString()} → {step.rows_after.toLocaleString()} rows
            </div>
            {step.params && (
              <pre className="text-xs mt-1 font-mono overflow-x-auto" style={{ color: 'var(--text-faint)' }}>
                {JSON.stringify(step.params, null, 2).slice(0, 200)}
              </pre>
            )}
            {step.warnings?.map((w, i) => (
              <div key={i} className="text-xs text-amber-400 flex items-start gap-1 mt-1">
                <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" /> {w}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// ── Upload panel ──────────────────────────────────────────────────────────────
function UploadPanel({ onSuccess }: { onSuccess: () => void }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const uploadMutation = useUploadDataset()

  const handleFile = (file: File) => {
    uploadMutation.mutate(file, { onSuccess })
  }

  return (
    <div
      className={cn(
        'border-2 border-dashed rounded-xl p-8 text-center transition-colors',
        dragging ? 'border-brand-400 bg-brand-500/5' : 'border-border',
      )}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const file = e.dataTransfer.files[0]
        if (file) handleFile(file)
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.tsv,.json,.jsonl"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
      />

      {uploadMutation.isPending ? (
        <div className="flex flex-col items-center gap-2">
          <Spinner />
          <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>Uploading…</p>
        </div>
      ) : uploadMutation.isSuccess ? (
        <div className="space-y-2">
          <CheckCircle2 className="w-10 h-10 text-emerald-400 mx-auto" />
          <p className="text-sm font-medium text-emerald-400">Upload successful</p>
          <p className="text-xs" style={{ color: 'var(--text-subtle)' }}>
            File: <strong>{uploadMutation.data?.filename}</strong> —{' '}
            {uploadMutation.data?.rows_detected?.toLocaleString() ?? '?'} rows detected
          </p>
          {uploadMutation.data?.detected_label_col && (
            <p className="text-xs" style={{ color: 'var(--text-subtle)' }}>
              Label column: <code className="font-mono" style={{ color: 'var(--brand)' }}>{uploadMutation.data.detected_label_col}</code>
            </p>
          )}
          <button onClick={() => uploadMutation.reset()} className="btn-secondary text-xs mt-2">Upload another</button>
        </div>
      ) : (
        <>
          <FileUp className="w-10 h-10 mx-auto mb-3" style={{ color: 'var(--text-faint)' }} />
          <p className="text-sm font-medium mb-1" style={{ color: 'var(--text-secondary)' }}>
            Drag & drop a dataset file here
          </p>
          <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>
            CSV, TSV, JSON, or JSONL — max 200 MB
          </p>
          <button onClick={() => inputRef.current?.click()} className="btn-secondary text-sm">
            <Upload className="w-4 h-4" /> Browse files
          </button>
          {uploadMutation.isError && (
            <p className="text-xs text-red-400 mt-3">
              {(uploadMutation.error as Error).message}
            </p>
          )}
        </>
      )}
    </div>
  )
}

// ── Preprocess panel ──────────────────────────────────────────────────────────
function PreprocessPanel({ selectedEntry }: { selectedEntry: DatasetCatalogEntry | null }) {
  const preprocessMutation = usePreprocessDataset()
  const [lemmatize, setLemmatize] = useState(false)
  const [mode, setMode] = useState<'baseline' | 'transformer'>('baseline')

  const handleRun = () => {
    if (!selectedEntry) return
    preprocessMutation.mutate({
      files: selectedEntry.expected_files,
      cleaning_mode: mode,
      lemmatize,
      force: true,
    })
  }

  return (
    <div className="card p-5 space-y-4">
      <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
        Run Preprocessing Pipeline
      </h3>

      {!selectedEntry ? (
        <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
          Select a downloaded dataset from the catalog to preprocess it.
        </p>
      ) : !selectedEntry.is_downloaded ? (
        <div className="flex items-start gap-2 p-3 rounded-xl" style={{ backgroundColor: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.2)' }}>
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <p className="text-sm text-amber-400">
            {selectedEntry.display_name} has not been downloaded yet.
            Run the download command shown in the dataset card first.
          </p>
        </div>
      ) : (
        <>
          <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>
            Dataset: <strong>{selectedEntry.display_name}</strong>
          </div>

          {/* Options */}
          <div className="space-y-3">
            <div>
              <label className="label-sm block mb-2">Cleaning Mode</label>
              <div className="flex gap-2">
                {(['baseline', 'transformer'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={cn(
                      'px-3 py-1.5 rounded-lg text-xs font-medium border transition-all',
                      mode === m ? 'text-white' : '',
                    )}
                    style={mode === m
                      ? { background: 'linear-gradient(135deg, var(--brand), var(--accent))', borderColor: 'transparent' }
                      : { backgroundColor: 'var(--bg-surface-2)', color: 'var(--text-secondary)', borderColor: 'var(--border)' }
                    }
                  >
                    {m}
                  </button>
                ))}
              </div>
              <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
                {mode === 'baseline'
                  ? 'Heavy normalisation for TF-IDF models (lowercase, strip URLs, remove non-alpha)'
                  : 'Light cleaning for BERT/DistilBERT (preserves casing and punctuation)'}
              </p>
            </div>

            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={lemmatize}
                onChange={(e) => setLemmatize(e.target.checked)}
                className="w-4 h-4 rounded"
              />
              <div>
                <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>Lemmatization</div>
                <div className="text-xs" style={{ color: 'var(--text-faint)' }}>
                  Apply stopword removal + WordNet lemmatization (requires NLTK data)
                </div>
              </div>
            </label>
          </div>

          {preprocessMutation.isPending && (
            <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-subtle)' }}>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running pipeline… this may take a minute.
            </div>
          )}

          {preprocessMutation.isSuccess && (
            <div className="flex items-start gap-2 p-3 rounded-xl" style={{ backgroundColor: 'rgba(52,211,153,0.06)', border: '1px solid rgba(52,211,153,0.2)' }}>
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div className="text-sm text-emerald-400">
                Pipeline completed — train: {preprocessMutation.data?.train_rows?.toLocaleString()},
                val: {preprocessMutation.data?.val_rows?.toLocaleString()},
                test: {preprocessMutation.data?.test_rows?.toLocaleString()} rows
              </div>
            </div>
          )}

          {preprocessMutation.isError && (
            <div className="flex items-start gap-2 p-3 rounded-xl" style={{ backgroundColor: 'rgba(248,113,113,0.06)', border: '1px solid rgba(248,113,113,0.2)' }}>
              <XCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <div className="text-sm text-red-400">{(preprocessMutation.error as Error).message}</div>
            </div>
          )}

          <button
            onClick={handleRun}
            disabled={preprocessMutation.isPending}
            className="btn-primary w-full justify-center"
          >
            {preprocessMutation.isPending
              ? <><Loader2 className="w-4 h-4 animate-spin" /> Running…</>
              : <><Play className="w-4 h-4" /> Run Preprocessing</>
            }
          </button>
        </>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function DatasetExplorer() {
  const [selectedName, setSelectedName] = useState<string>('isot')
  const [activeTab, setActiveTab] = useState<'overview' | 'analytics' | 'samples' | 'audit' | 'upload'>('overview')
  const [analyticsSplit, setAnalyticsSplit] = useState<'train' | 'val' | 'test'>('train')
  const [sampleLabel, setSampleLabel] = useState<number | undefined>(undefined)

  const { data: catalog, isLoading: catalogLoading, refetch: refetchCatalog } = useDatasetCatalog()
  const { data: stats,   isLoading: statsLoading  } = useDatasetStats(selectedName)
  const { data: analytics, isLoading: analyticsLoading } = useDatasetAnalytics(selectedName, analyticsSplit)
  const { data: samples,   isLoading: samplesLoading   } = useDatasetSamples(selectedName, 'train', 10, sampleLabel)
  const { data: audit,     isLoading: auditLoading     } = useDatasetAudit(selectedName)

  const selectedEntry = catalog?.datasets?.find((d: DatasetCatalogEntry) => d.name === selectedName) ?? null

  const TABS = [
    { id: 'overview',  label: 'Overview',  icon: BarChart3 },
    { id: 'analytics', label: 'Analytics', icon: Database  },
    { id: 'samples',   label: 'Samples',   icon: Eye       },
    { id: 'audit',     label: 'Audit Log', icon: Clock     },
    { id: 'upload',    label: 'Upload',    icon: Upload    },
  ] as const

  return (
    <div className="page-content">
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center shrink-0">
            <Database className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="page-title mb-1">Dataset Explorer</h1>
            <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
              Manage, inspect, and preprocess training datasets
            </p>
          </div>
        </div>
        <button onClick={() => refetchCatalog()} className="btn-ghost" title="Refresh">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left: catalog */}
        <div className="lg:col-span-1 space-y-4">
          <h2 className="section-title">Supported Datasets</h2>
          {catalogLoading ? (
            Array.from({ length: 3 }).map((_, i) => <SkeletonCard key={i} />)
          ) : catalog?.datasets?.length ? (
            catalog.datasets.map((entry: DatasetCatalogEntry) => (
              <DatasetCard
                key={entry.name}
                entry={entry}
                selected={selectedName === entry.name}
                onSelect={() => setSelectedName(entry.name)}
              />
            ))
          ) : (
            <div className="card p-6 text-center" style={{ color: 'var(--text-faint)' }}>
              <Database className="w-8 h-8 mx-auto mb-2 opacity-30" />
              <p className="text-sm">No datasets in catalog</p>
            </div>
          )}

          {/* Preprocess panel */}
          <PreprocessPanel selectedEntry={selectedEntry} />
        </div>

        {/* Right: detail */}
        <div className="lg:col-span-2">
          {/* Tabs */}
          <div className="flex gap-1 mb-6 p-1 rounded-xl" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
            {TABS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={cn(
                  'flex-1 flex items-center justify-center gap-1.5 py-2 px-3 rounded-lg text-sm font-medium transition-all',
                  activeTab === id ? 'text-white shadow-sm' : '',
                )}
                style={activeTab === id
                  ? { background: 'linear-gradient(135deg, var(--brand), var(--accent))' }
                  : { color: 'var(--text-subtle)' }
                }
              >
                <Icon className="w-3.5 h-3.5 hidden sm:block" />
                <span className="hidden sm:inline">{label}</span>
                <span className="sm:hidden">{label.slice(0, 3)}</span>
              </button>
            ))}
          </div>

          {/* Tab content */}
          {activeTab === 'overview' && (
            <div className="space-y-5">
              {statsLoading ? (
                Array.from({ length: 3 }).map((_, i) => <SkeletonCard key={i} />)
              ) : stats ? (
                <>
                  {/* Split summary */}
                  <div className="grid grid-cols-3 gap-4">
                    {Object.entries(stats.splits ?? {}).map(([split, s]: [string, any]) => (
                      <div key={split} className="stat-card text-center">
                        <div className="text-2xl font-bold" style={{ color: 'var(--brand)' }}>
                          {s.total_rows.toLocaleString()}
                        </div>
                        <div className="text-xs mt-1 capitalize" style={{ color: 'var(--text-secondary)' }}>{split}</div>
                        <div className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>
                          Fake: {s.fake_count.toLocaleString()} / Real: {s.real_count.toLocaleString()}
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Class distribution donut */}
                  {stats.splits?.train && (
                    <div className="card p-5">
                      <h3 className="section-title mb-4">Train Split — Class Distribution</h3>
                      <ResponsiveContainer width="100%" height={180}>
                        <PieChart>
                          <Pie
                            data={[
                              { name: 'Fake', value: stats.splits.train.fake_count, fill: '#f87171' },
                              { name: 'Real', value: stats.splits.train.real_count, fill: '#34d399' },
                            ]}
                            cx="50%" cy="50%" innerRadius={50} outerRadius={72}
                            paddingAngle={4} dataKey="value"
                          >
                            {[{ fill: '#f87171' }, { fill: '#34d399' }].map((entry, i) => (
                              <Cell key={i} fill={entry.fill} stroke="transparent" />
                            ))}
                          </Pie>
                          <Tooltip {...TT_STYLE} formatter={(v: number) => [v.toLocaleString(), 'articles']} />
                          <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{v}</span>} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Dedup + leakage */}
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div className="card p-5">
                      <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Deduplication</h3>
                      <div className="space-y-2 text-sm">
                        {Object.entries(stats.deduplication ?? {}).map(([k, v]) => (
                          <div key={k} className="flex justify-between">
                            <span style={{ color: 'var(--text-subtle)' }}>{k.replace(/_/g, ' ')}</span>
                            <span className="font-mono" style={{ color: (v as number) > 0 ? '#fbbf24' : 'var(--text-secondary)' }}>
                              {(v as number).toLocaleString()}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div className="card p-5">
                      <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Leakage Check</h3>
                      <div className="space-y-2 text-sm">
                        {Object.entries(stats.leakage ?? {}).map(([k, v]) => (
                          <div key={k} className="flex justify-between">
                            <span style={{ color: 'var(--text-subtle)' }}>{k.replace(/_/g, ' ')}</span>
                            <span className="font-mono flex items-center gap-1" style={{ color: (v as number) === 0 ? '#34d399' : '#f87171' }}>
                              {(v as number) === 0 && <CheckCircle2 className="w-3 h-3" />}
                              {(v as number) > 0 && <AlertTriangle className="w-3 h-3" />}
                              {(v as number).toLocaleString()}
                            </span>
                          </div>
                        ))}
                      </div>
                      {Object.values(stats.leakage ?? {}).every((v) => v === 0) && (
                        <p className="text-xs text-emerald-400 mt-2">No leakage detected ✓</p>
                      )}
                    </div>
                  </div>

                  {stats.last_processed && (
                    <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
                      Last processed: {new Date(stats.last_processed).toLocaleString()}
                    </p>
                  )}
                </>
              ) : (
                <div className="card p-10 text-center">
                  <Database className="w-10 h-10 mx-auto mb-3 opacity-20" style={{ color: 'var(--text-faint)' }} />
                  <p className="text-sm mb-2" style={{ color: 'var(--text-subtle)' }}>No processed data for this dataset.</p>
                  <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
                    Download the dataset files and run the preprocessing pipeline.
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'analytics' && (
            <div className="space-y-5">
              {/* Split selector */}
              <div className="flex gap-2">
                {(['train', 'val', 'test'] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setAnalyticsSplit(s)}
                    className={cn('px-3 py-1.5 rounded-lg text-xs font-medium border transition-all', analyticsSplit === s ? 'text-white border-transparent' : '')}
                    style={analyticsSplit === s
                      ? { background: 'linear-gradient(135deg, var(--brand), var(--accent))' }
                      : { backgroundColor: 'var(--bg-surface-2)', color: 'var(--text-secondary)', borderColor: 'var(--border)' }
                    }
                  >
                    {s.toUpperCase()}
                  </button>
                ))}
              </div>

              {analyticsLoading ? (
                <div className="space-y-4">
                  <SkeletonChart className="h-56" />
                  <SkeletonChart className="h-56" />
                </div>
              ) : analytics ? (
                <>
                  {/* Quality score */}
                  {analytics.quality_report?.quality_score !== undefined && (
                    <div className="card p-5 flex items-center gap-4">
                      <div>
                        <div className="text-3xl font-bold" style={{ color: analytics.quality_report.quality_score >= 0.8 ? '#34d399' : analytics.quality_report.quality_score >= 0.6 ? '#fbbf24' : '#f87171' }}>
                          {(analytics.quality_report.quality_score * 100).toFixed(0)}
                        </div>
                        <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>Quality Score / 100</div>
                      </div>
                      <div className="flex-1 space-y-1">
                        {(analytics.quality_report.issues as string[] ?? []).slice(0, 3).map((issue: string, i: number) => (
                          <div key={i} className="flex items-start gap-1.5 text-xs text-amber-400">
                            <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" /> {issue}
                          </div>
                        ))}
                        {!(analytics.quality_report.issues as string[])?.length && (
                          <div className="flex items-center gap-1.5 text-xs text-emerald-400">
                            <CheckCircle2 className="w-3 h-3" /> No issues detected
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Article length histogram */}
                  <div className="card p-5">
                    <h3 className="section-title mb-4">Article Length Distribution (characters)</h3>
                    <LengthHistogram histogram={(analytics.length_distribution as any)?.histogram ?? []} />
                  </div>

                  {/* Word frequency charts */}
                  <div className="card p-5">
                    <h3 className="section-title mb-4">Word Frequency Analysis</h3>
                    <div className="grid md:grid-cols-2 gap-6">
                      <WordFreqChart
                        data={analytics.word_frequency?.fake ?? []}
                        title="Most Common Fake News Terms"
                        color="#f87171"
                      />
                      <WordFreqChart
                        data={analytics.word_frequency?.real ?? []}
                        title="Most Common Real News Terms"
                        color="#34d399"
                      />
                    </div>
                  </div>

                  {/* Duplicate & missing value info */}
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div className="card p-5">
                      <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Duplicate Records</h3>
                      {Object.entries(analytics.duplicates ?? {}).map(([k, v]) => (
                        <div key={k} className="flex justify-between text-sm py-1" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                          <span style={{ color: 'var(--text-subtle)' }}>{k.replace(/_/g, ' ')}</span>
                          <span className="font-mono" style={{ color: 'var(--text-secondary)' }}>{(v as number).toLocaleString?.() ?? v}</span>
                        </div>
                      ))}
                    </div>
                    <div className="card p-5">
                      <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Missing Values</h3>
                      {((analytics.missing_values as any)?.columns ?? []).slice(0, 5).map((col: { name: string; missing: number; pct: number }) => (
                        <div key={col.name} className="flex justify-between text-sm py-1" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                          <span className="font-mono text-xs" style={{ color: 'var(--text-subtle)' }}>{col.name}</span>
                          <span style={{ color: col.missing > 0 ? '#fbbf24' : '#34d399' }}>
                            {col.missing.toLocaleString()} ({col.pct}%)
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="card p-10 text-center">
                  <BarChart3 className="w-10 h-10 mx-auto mb-3 opacity-20" style={{ color: 'var(--text-faint)' }} />
                  <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
                    No analytics data. Process the dataset first.
                  </p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'samples' && (
            <div className="space-y-4">
              {/* Filter */}
              <div className="flex gap-2">
                {[
                  { label: 'All',  value: undefined },
                  { label: 'Fake', value: 0 },
                  { label: 'Real', value: 1 },
                ].map((opt) => (
                  <button
                    key={opt.label}
                    onClick={() => setSampleLabel(opt.value)}
                    className={cn('px-3 py-1.5 rounded-lg text-xs font-medium border transition-all', sampleLabel === opt.value ? 'text-white border-transparent' : '')}
                    style={sampleLabel === opt.value
                      ? { background: 'linear-gradient(135deg, var(--brand), var(--accent))' }
                      : { backgroundColor: 'var(--bg-surface-2)', color: 'var(--text-secondary)', borderColor: 'var(--border)' }
                    }
                  >
                    {opt.label}
                  </button>
                ))}
              </div>

              {samplesLoading ? (
                <div className="flex justify-center py-12"><Spinner /></div>
              ) : samples?.samples?.length ? (
                <div className="space-y-3">
                  {samples.samples.map((row: Record<string, unknown>, idx: number) => (
                    <div key={idx} className="card p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span
                          className={cn('text-xs px-2 py-0.5 rounded-full font-medium', row.label === 0 ? 'text-red-400 bg-red-400/10' : 'text-emerald-400 bg-emerald-400/10')}
                        >
                          {row.label === 0 ? 'FAKE' : 'REAL'}
                        </span>
                        <span className="text-xs font-mono" style={{ color: 'var(--text-faint)' }}>row {idx + 1}</span>
                      </div>
                      <p className="text-xs leading-relaxed line-clamp-4" style={{ color: 'var(--text-secondary)' }}>
                        {String(row.cleaned_text ?? row.text ?? '').slice(0, 400)}
                        {String(row.cleaned_text ?? row.text ?? '').length > 400 && '…'}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="card p-10 text-center">
                  <Eye className="w-10 h-10 mx-auto mb-3 opacity-20" style={{ color: 'var(--text-faint)' }} />
                  <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>No samples available. Process the dataset first.</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'audit' && (
            <div>
              {auditLoading ? (
                <div className="flex justify-center py-12"><Spinner /></div>
              ) : audit ? (
                <div className="space-y-5">
                  <div className="card p-5">
                    <div className="grid sm:grid-cols-3 gap-4 mb-4">
                      {[
                        { label: 'Dataset',       value: audit.dataset_name },
                        { label: 'Cleaning Mode', value: audit.cleaning_mode },
                        { label: 'Total Dropped', value: audit.total_dropped?.toLocaleString() },
                      ].map((s) => (
                        <div key={s.label}>
                          <div className="text-xs font-semibold mb-0.5" style={{ color: 'var(--text-subtle)' }}>{s.label}</div>
                          <div className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{s.value}</div>
                        </div>
                      ))}
                    </div>
                    <div className="text-xs" style={{ color: 'var(--text-faint)' }}>
                      Started: {audit.started_at && new Date(audit.started_at).toLocaleString()}
                      {audit.completed_at && ` · Completed: ${new Date(audit.completed_at).toLocaleString()}`}
                    </div>
                  </div>

                  <div className="card p-5">
                    <h3 className="section-title mb-4">Pipeline Steps</h3>
                    <AuditTimeline steps={audit.steps ?? []} />
                  </div>

                  {/* Source files */}
                  <div className="card p-5">
                    <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Source Files</h3>
                    <div className="space-y-1">
                      {(audit.source_files ?? []).map((f: string) => (
                        <div key={f} className="flex items-center gap-2 text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
                          <FileText className="w-3.5 h-3.5" style={{ color: 'var(--text-subtle)' }} />
                          {f}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="card p-10 text-center">
                  <Clock className="w-10 h-10 mx-auto mb-3 opacity-20" style={{ color: 'var(--text-faint)' }} />
                  <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>No audit log. Run the preprocessing pipeline first.</p>
                </div>
              )}
            </div>
          )}

          {activeTab === 'upload' && (
            <div className="space-y-5">
              <div className="card p-5">
                <div className="flex items-start gap-3 mb-4">
                  <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div className="text-sm space-y-1" style={{ color: 'var(--text-secondary)' }}>
                    <p className="font-medium text-amber-400">Copyright Notice</p>
                    <p>
                      By uploading a dataset you confirm you have the right to use it for research purposes
                      and that it is not restricted material. Do not upload datasets that prohibit
                      redistribution or processing without explicit permission.
                    </p>
                    <p>
                      Uploaded files are saved to <code className="font-mono text-xs" style={{ color: 'var(--brand)' }}>ml/datasets/raw/</code> on the server.
                    </p>
                  </div>
                </div>
                <UploadPanel onSuccess={() => refetchCatalog()} />
              </div>

              {/* Export */}
              <div className="card p-5">
                <h3 className="section-title mb-3">Export Processed Data</h3>
                <p className="text-sm mb-4" style={{ color: 'var(--text-subtle)' }}>
                  Download processed splits (cleaned text + labels).
                  Raw article text is never exported to respect dataset copyright.
                </p>
                <div className="flex flex-wrap gap-2">
                  {(['train', 'val', 'test'] as const).map((split) => (
                    <a
                      key={split}
                      href={`/api/v1/datasets/${selectedName}/export/${split}`}
                      download={`${selectedName}_${split}.csv`}
                      className="btn-secondary text-xs"
                    >
                      <Download className="w-3.5 h-3.5" />
                      {split.toUpperCase()} CSV
                    </a>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Copyright footer */}
      <div className="mt-8 card p-4 flex items-start gap-3" style={{ borderColor: 'rgba(99,102,241,0.15)', backgroundColor: 'rgba(99,102,241,0.03)' }}>
        <Shield className="w-4 h-4 shrink-0 mt-0.5" style={{ color: 'var(--brand)' }} />
        <p className="text-xs leading-relaxed" style={{ color: 'var(--text-subtle)' }}>
          <strong style={{ color: 'var(--text-secondary)' }}>Copyright notice:</strong>{' '}
          No dataset files are included in this repository. All datasets must be obtained from their original
          sources. Users are responsible for complying with each dataset's terms of use.
          ISOT real articles originate from Reuters — do not redistribute raw text.
          LIAR statements originate from PolitiFact — research use only.
        </p>
      </div>
    </div>
  )
}
