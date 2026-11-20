/**
 * Verification History
 *
 * Full history management with:
 *   - Text search across input and title
 *   - Verdict, input type, date range, model, confidence filters
 *   - Pagination with URL sync
 *   - Export to CSV
 *   - Open previous verification reports
 *   - Compare two analyses
 *   - Download PDF report per analysis
 *
 * Data: GET /api/v1/history (paginated, filterable)
 */

import { useState, useCallback, useEffect } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  Clock, ChevronLeft, ChevronRight, Filter, FileText,
  Link2, MessageSquareText, ArrowRight, AlertCircle,
  Loader2, RefreshCw, History as HistoryIcon, Search, X,
  Download, FileDown, GitMerge,
} from 'lucide-react'
import { useHistory, useModels } from '@/hooks/useApi'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { ErrorState } from '@/components/ui/ErrorState'
import { EmptyState } from '@/components/ui/EmptyState'
import { SkeletonTable } from '@/components/ui/Skeleton'
import { getReportPdfUrl, exportHistory } from '@/lib/api'
import { cn, formatRelativeTime, formatDateTime, formatMs, getVerdictConfig, truncate } from '@/lib/utils'
import type { AnalysisListItem, ModelInfo } from '@/types/api'

const PAGE_SIZE = 20

const VERDICT_OPTS = [
  { value: '',           label: 'All verdicts' },
  { value: 'FAKE',       label: 'Fake' },
  { value: 'REAL',       label: 'Real' },
  { value: 'UNVERIFIED', label: 'Unverified' },
  { value: 'MIXED',      label: 'Mixed' },
]

const TYPE_OPTS = [
  { value: '',      label: 'All types' },
  { value: 'text',  label: 'Text' },
  { value: 'url',   label: 'URL' },
  { value: 'query', label: 'Claim' },
]

function InputTypeIcon({ type }: { type: string }) {
  const cls = 'w-3.5 h-3.5 shrink-0'
  if (type === 'url')   return <Link2             className={cls} />
  if (type === 'query') return <MessageSquareText className={cls} />
  return                        <FileText          className={cls} />
}

function HistoryRow({
  item,
  selected,
  onSelect,
}: {
  item: AnalysisListItem
  selected: boolean
  onSelect: (id: number) => void
}) {
  const navigate    = useNavigate()
  const vc          = getVerdictConfig(item.final_verdict ?? undefined)
  const isCompleted = item.status === 'completed'
  const content     = item.article_title || truncate(item.original_input, 80)

  return (
    <tr
      onClick={() => navigate(`/results/${item.id}`)}
      className="cursor-pointer group"
    >
      {/* Checkbox for compare */}
      <td className="px-3 py-3.5" onClick={(e) => e.stopPropagation()}>
        <input
          type="checkbox"
          checked={selected}
          onChange={() => onSelect(item.id)}
          className="w-4 h-4 cursor-pointer"
        />
      </td>
      {/* ID */}
      <td className="px-3 py-3.5">
        <div className="text-xs font-mono" style={{ color: 'var(--text-faint)' }}>#{item.id}</div>
      </td>
      {/* Type */}
      <td className="px-3 py-3.5">
        <div className={cn(
          'inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full',
          item.input_type === 'url'   ? 'text-violet-400 bg-violet-500/10' :
          item.input_type === 'query' ? 'text-amber-400  bg-amber-500/10'  :
          'text-blue-400 bg-blue-500/10',
        )}>
          <InputTypeIcon type={item.input_type} />
          <span className="capitalize">{item.input_type}</span>
        </div>
      </td>
      {/* Content */}
      <td className="px-3 py-3.5 max-w-xs">
        <div className="text-sm truncate" style={{ color: 'var(--text-secondary)' }}>{content}</div>
      </td>
      {/* Assessment */}
      <td className="px-3 py-3.5">
        {isCompleted && item.final_verdict ? (
          <span className={cn('text-xs font-semibold px-2.5 py-1 rounded-full', vc.color, vc.bg)}>
            {vc.label}
          </span>
        ) : (
          <Badge variant="warning">{item.status}</Badge>
        )}
      </td>
      {/* Confidence */}
      <td className="px-3 py-3.5 hidden md:table-cell">
        {isCompleted && item.final_confidence != null && (
          <div className="flex items-center gap-2">
            <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-surface-3)' }}>
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.round(item.final_confidence * 100)}%`,
                  backgroundColor: item.final_confidence > 0.8 ? '#818cf8' : item.final_confidence > 0.6 ? '#fbbf24' : 'var(--text-faint)',
                }}
              />
            </div>
            <span className="text-xs font-mono" style={{ color: 'var(--text-secondary)' }}>
              {Math.round(item.final_confidence * 100)}%
            </span>
          </div>
        )}
      </td>
      {/* Time */}
      <td className="px-3 py-3.5 hidden lg:table-cell">
        <span className="text-xs" title={formatDateTime(item.created_at)} style={{ color: 'var(--text-subtle)' }}>
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {formatRelativeTime(item.created_at)}
          </span>
        </span>
      </td>
      {/* Duration */}
      <td className="px-3 py-3.5 hidden xl:table-cell">
        <span className="text-xs font-mono" style={{ color: 'var(--text-faint)' }}>
          {item.processing_time_ms ? formatMs(item.processing_time_ms) : '—'}
        </span>
      </td>
      {/* Actions */}
      <td className="px-3 py-3.5 text-right" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-end gap-1.5">
          <a
            href={getReportPdfUrl(item.id)}
            download={`verifact_report_${item.id}.pdf`}
            className="p-1.5 rounded-lg transition-colors hover:bg-surface-2"
            style={{ color: 'var(--text-subtle)' }}
            title="Download PDF report"
          >
            <FileDown className="w-3.5 h-3.5" />
          </a>
          <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: 'var(--brand)' }} />
        </div>
      </td>
    </tr>
  )
}

function Pagination({ page, totalPages, onChange }: {
  page: number; totalPages: number; onChange: (p: number) => void
}) {
  if (totalPages <= 1) return null
  const win = 2
  const start = Math.max(1, page - win)
  const end = Math.min(totalPages, page + win)
  const pages = Array.from({ length: end - start + 1 }, (_, i) => start + i)

  return (
    <div className="flex items-center justify-between pt-4">
      <span className="text-xs" style={{ color: 'var(--text-subtle)' }}>
        Page <span style={{ color: 'var(--text-secondary)' }}>{page}</span> of{' '}
        <span style={{ color: 'var(--text-secondary)' }}>{totalPages}</span>
      </span>
      <div className="flex items-center gap-1">
        <button className="btn-secondary px-2.5 py-1.5 text-xs" disabled={page === 1} onClick={() => onChange(page - 1)}>
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>
        {start > 1 && (
          <>
            <button onClick={() => onChange(1)} className="px-2.5 py-1.5 text-xs rounded-lg transition-colors" style={{ color: 'var(--text-subtle)' }}>1</button>
            {start > 2 && <span style={{ color: 'var(--text-faint)' }} className="px-1">…</span>}
          </>
        )}
        {pages.map((p) => (
          <button
            key={p}
            onClick={() => onChange(p)}
            className={cn('px-2.5 py-1.5 text-xs rounded-lg transition-colors', p === page ? 'text-white' : '')}
            style={p === page
              ? { background: 'linear-gradient(135deg, var(--brand), var(--accent))' }
              : { color: 'var(--text-subtle)' }
            }
          >
            {p}
          </button>
        ))}
        {end < totalPages && (
          <>
            {end < totalPages - 1 && <span style={{ color: 'var(--text-faint)' }} className="px-1">…</span>}
            <button onClick={() => onChange(totalPages)} className="px-2.5 py-1.5 text-xs rounded-lg transition-colors" style={{ color: 'var(--text-subtle)' }}>{totalPages}</button>
          </>
        )}
        <button className="btn-secondary px-2.5 py-1.5 text-xs" disabled={page === totalPages} onClick={() => onChange(page + 1)}>
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  )
}

export default function History() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [page,      setPage]      = useState(Number(searchParams.get('page') ?? 1))
  const [verdict,   setVerdict]   = useState(searchParams.get('verdict') ?? '')
  const [inputType, setInputType] = useState(searchParams.get('type') ?? '')
  const [search,    setSearch]    = useState(searchParams.get('q') ?? '')
  const [dateFrom,  setDateFrom]  = useState(searchParams.get('from') ?? '')
  const [dateTo,    setDateTo]    = useState(searchParams.get('to') ?? '')
  const [modelFilter, setModelFilter] = useState(searchParams.get('model') ?? '')
  const [confMin,   setConfMin]   = useState(searchParams.get('cmin') ?? '')
  const [confMax,   setConfMax]   = useState(searchParams.get('cmax') ?? '')
  const [selectedIds, setSelectedIds] = useState<number[]>([])
  const [showAdvanced, setShowAdvanced] = useState(false)

  const { data, isLoading, isPlaceholderData, error, refetch } = useHistory({
    page,
    page_size: PAGE_SIZE,
    verdict:   verdict || undefined,
    input_type: inputType || undefined,
    search:    search || undefined,
    date_from: dateFrom || undefined,
    date_to:   dateTo || undefined,
    model_id:  modelFilter || undefined,
    confidence_min: confMin ? parseFloat(confMin) : undefined,
    confidence_max: confMax ? parseFloat(confMax) : undefined,
  })

  const { data: models } = useModels()

  const pushParams = useCallback((overrides: Record<string, string>) => {
    const sp: Record<string, string> = {}
    const all: Record<string, string> = {
      page:    overrides.page     ?? String(page),
      verdict: overrides.verdict  ?? verdict,
      type:    overrides.type     ?? inputType,
      q:       overrides.q        ?? search,
      from:    overrides.from     ?? dateFrom,
      to:      overrides.to       ?? dateTo,
      model:   overrides.model    ?? modelFilter,
      cmin:    overrides.cmin     ?? confMin,
      cmax:    overrides.cmax     ?? confMax,
    }
    Object.entries(all).forEach(([k, v]) => {
      if (v && v !== '1' && v !== '') sp[k] = v
    })
    setSearchParams(sp, { replace: true })
  }, [page, verdict, inputType, search, dateFrom, dateTo, modelFilter, confMin, confMax, setSearchParams])

  const handleSelect = (id: number) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id].slice(-2)
    )
  }

  const handleExport = async () => {
    try {
      const blob = await exportHistory({
        verdict: verdict || undefined,
        input_type: inputType || undefined,
        search: search || undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'verifact_history.csv'
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Export failed:', err)
    }
  }

  const hasFilters = !!(verdict || inputType || search || dateFrom || dateTo || modelFilter || confMin || confMax)

  const clearAll = () => {
    setVerdict(''); setInputType(''); setSearch(''); setDateFrom(''); setDateTo('')
    setModelFilter(''); setConfMin(''); setConfMax(''); setPage(1)
    setSearchParams({}, { replace: true })
  }

  return (
    <div className="page-content">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 mb-7">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-brand-600 flex items-center justify-center shrink-0">
            <HistoryIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="page-title mb-1">Verification History</h1>
            <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
              {data
                ? <><span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{data.total.toLocaleString()}</span> analyses in database</>
                : 'Loading…'}
            </p>
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Export CSV */}
          <button onClick={handleExport} className="btn-secondary text-xs" title="Export filtered history as CSV">
            <Download className="w-3.5 h-3.5" /> Export CSV
          </button>

          {/* Compare */}
          {selectedIds.length === 2 && (
            <Link to={`/results/compare?id1=${selectedIds[0]}&id2=${selectedIds[1]}`} className="btn-primary text-xs">
              <GitMerge className="w-3.5 h-3.5" /> Compare #{selectedIds[0]} vs #{selectedIds[1]}
            </Link>
          )}

          {/* Refresh */}
          <button onClick={() => refetch()} className="btn-ghost p-1.5" title="Refresh">
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Search bar */}
      <div className="card p-4 mb-5">
        <div className="flex flex-wrap items-center gap-3">
          {/* Search input */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-faint)' }} />
            <input
              className="input-field pl-9 text-sm"
              placeholder="Search across headlines and article text…"
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1) }}
            />
            {search && (
              <button onClick={() => setSearch('')} className="absolute right-3 top-1/2 -translate-y-1/2">
                <X className="w-3.5 h-3.5" style={{ color: 'var(--text-faint)' }} />
              </button>
            )}
          </div>

          {/* Verdict filter */}
          <select
            className="input-field w-auto text-xs py-1.5"
            value={verdict}
            onChange={(e) => { setVerdict(e.target.value); setPage(1) }}
          >
            {VERDICT_OPTS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>

          {/* Input type filter */}
          <select
            className="input-field w-auto text-xs py-1.5"
            value={inputType}
            onChange={(e) => { setInputType(e.target.value); setPage(1) }}
          >
            {TYPE_OPTS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>

          {/* Advanced filters toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className={cn('btn-ghost text-xs', showAdvanced && 'ring-1')}
            style={showAdvanced ? { outline: '1px solid var(--brand)', color: 'var(--brand)' } : undefined}
          >
            <Filter className="w-3.5 h-3.5" /> Advanced
          </button>

          {/* Clear all */}
          {hasFilters && (
            <button onClick={clearAll} className="text-xs flex items-center gap-1 hover:opacity-80" style={{ color: 'var(--text-subtle)' }}>
              <X className="w-3.5 h-3.5" /> Clear
            </button>
          )}
        </div>

        {/* Advanced filters */}
        {showAdvanced && (
          <div className="flex flex-wrap items-center gap-3 mt-3 pt-3" style={{ borderTop: '1px solid var(--border-subtle)' }}>
            <div>
              <label className="text-xs block mb-0.5" style={{ color: 'var(--text-subtle)' }}>From date</label>
              <input type="date" className="input-field text-xs py-1.5" value={dateFrom} onChange={(e) => { setDateFrom(e.target.value); setPage(1) }} />
            </div>
            <div>
              <label className="text-xs block mb-0.5" style={{ color: 'var(--text-subtle)' }}>To date</label>
              <input type="date" className="input-field text-xs py-1.5" value={dateTo} onChange={(e) => { setDateTo(e.target.value); setPage(1) }} />
            </div>
            <div>
              <label className="text-xs block mb-0.5" style={{ color: 'var(--text-subtle)' }}>Model</label>
              <select className="input-field text-xs py-1.5" value={modelFilter} onChange={(e) => { setModelFilter(e.target.value); setPage(1) }}>
                <option value="">All models</option>
                {(models ?? []).map((m: ModelInfo) => (
                  <option key={m.id} value={m.model_name}>{m.model_name.replace(/_/g, ' ')}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs block mb-0.5" style={{ color: 'var(--text-subtle)' }}>Confidence min</label>
              <input type="number" min="0" max="1" step="0.1" className="input-field text-xs py-1.5 w-20" value={confMin}
                onChange={(e) => { setConfMin(e.target.value); setPage(1) }} placeholder="0.0" />
            </div>
            <div>
              <label className="text-xs block mb-0.5" style={{ color: 'var(--text-subtle)' }}>Confidence max</label>
              <input type="number" min="0" max="1" step="0.1" className="input-field text-xs py-1.5 w-20" value={confMax}
                onChange={(e) => { setConfMax(e.target.value); setPage(1) }} placeholder="1.0" />
            </div>
          </div>
        )}
      </div>

      {/* Loading */}
      {isLoading && !data && <SkeletonTable rows={8} />}

      {/* Error */}
      {error && <ErrorState message={(error as Error).message} onRetry={() => refetch()} />}

      {/* Empty */}
      {!isLoading && !error && data?.items.length === 0 && (
        <EmptyState
          icon={<HistoryIcon />}
          title={hasFilters ? 'No results for these filters' : 'No analyses yet'}
          description={hasFilters
            ? 'Try adjusting or clearing filters.'
            : 'Submit an article, URL, or claim to start building history.'}
          action={hasFilters
            ? <button className="btn-secondary" onClick={clearAll}>Clear filters</button>
            : <Link to="/analyze/text" className="btn-primary">Start Analyzing</Link>}
        />
      )}

      {/* Table */}
      {!error && data && data.items.length > 0 && (
        <>
          <div className={cn('card overflow-hidden mb-5 transition-opacity duration-200', isPlaceholderData && 'opacity-60')}>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead>
                  <tr>
                    <th style={{ width: 40 }} />
                    <th>ID</th>
                    <th>Type</th>
                    <th>Content</th>
                    <th>Assessment</th>
                    <th className="hidden md:table-cell">Confidence</th>
                    <th className="hidden lg:table-cell">Time</th>
                    <th className="hidden xl:table-cell">Duration</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((item) => (
                    <HistoryRow
                      key={item.id}
                      item={item}
                      selected={selectedIds.includes(item.id)}
                      onSelect={handleSelect}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <Pagination page={data.page} totalPages={data.total_pages} onChange={(p) => { setPage(p); window.scrollTo({ top: 0, behavior: 'smooth' }) }} />
        </>
      )}

      {/* Research report download */}
      <div className="mt-6 card p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Research Reports</h3>
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>
            Model evaluation PDF with metrics and architecture · Full academic research paper (Markdown)
          </p>
        </div>
        <div className="flex gap-2 shrink-0">
          <a href="/api/v1/report/research" download="verifact_research_report.pdf" className="btn-secondary text-xs">
            <FileDown className="w-3.5 h-3.5" /> PDF Report
          </a>
          <a href="/api/v1/report/paper" download="VeriFact_AI_Research_Paper.md" className="btn-primary text-xs">
            <FileDown className="w-3.5 h-3.5" /> Research Paper
          </a>
        </div>
      </div>
    </div>
  )
}
