/**
 * Analysis History
 *
 * Data: GET /api/v1/history  (paginated, filterable)
 *
 * Displays per row — all from actual DB records:
 *   • Analysis ID
 *   • Input type badge (text / url / claim)
 *   • Content preview (article title or truncated input)
 *   • ML Assessment (verdict label + colour, or status badge)
 *   • Confidence %
 *   • Timestamp (relative + absolute on hover)
 *   • Processing time
 *
 * Clicking any row navigates to /results/{id}.
 * Filters: verdict, input_type — both push to URL so they're bookmarkable.
 */

import { useState, useCallback } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  Clock, ChevronLeft, ChevronRight, Filter,
  FileText, Link2, MessageSquareText, ArrowRight,
  AlertCircle, CheckCircle, Timer, Hash,
  Loader2, RefreshCw,
} from 'lucide-react'
import { useHistory } from '@/hooks/useApi'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { ErrorState } from '@/components/ui/ErrorState'
import { EmptyState } from '@/components/ui/EmptyState'
import { ProgressBar } from '@/components/ui/ProgressBar'
import {
  cn, formatRelativeTime, formatDateTime, formatMs,
  getVerdictConfig, truncate,
} from '@/lib/utils'
import type { AnalysisListItem } from '@/types/api'

// ── Constants ─────────────────────────────────────────────────────────────────

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
  { value: 'claim', label: 'Claim' },
]

// ── Sub-components ────────────────────────────────────────────────────────────

function InputTypeIcon({ type }: { type: string }) {
  const cls = 'w-3.5 h-3.5 shrink-0'
  if (type === 'url')   return <Link2            className={cls} />
  if (type === 'query') return <MessageSquareText className={cls} />
  return                        <FileText         className={cls} />
}

function StatusBadge({ status }: { status: string }) {
  if (status === 'completed') return null   // verdict shown instead
  if (status === 'failed')    return <Badge variant="danger">Failed</Badge>
  if (status === 'processing') return (
    <span className="inline-flex items-center gap-1 text-xs text-amber-400">
      <Loader2 className="w-3 h-3 animate-spin" />Processing
    </span>
  )
  return <Badge variant="warning">{status}</Badge>
}

function ConfidenceCell({ confidence }: { confidence?: number }) {
  if (confidence == null) return <span className="text-slate-700">—</span>
  const pct = Math.round(confidence * 100)
  const color = pct >= 80 ? 'bg-brand-500' : pct >= 60 ? 'bg-amber-500' : 'bg-slate-600'
  return (
    <div className="flex flex-col gap-1 min-w-[64px]">
      <span className="text-xs font-mono text-slate-300">{pct}%</span>
      <ProgressBar value={confidence} color={color} />
    </div>
  )
}

// ── Table row (clickable) ─────────────────────────────────────────────────────

function HistoryRow({ item }: { item: AnalysisListItem }) {
  const navigate = useNavigate()
  const vc = getVerdictConfig(item.final_verdict ?? undefined)
  const isCompleted = item.status === 'completed'
  const content = item.article_title || truncate(item.original_input, 90)
  const relTime = formatRelativeTime(item.created_at)
  const absTime = formatDateTime(item.created_at)

  return (
    <tr
      onClick={() => navigate(`/results/${item.id}`)}
      className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors cursor-pointer group"
    >
      {/* ID */}
      <td className="px-4 py-3.5">
        <div className="flex items-center gap-1 text-xs text-slate-600 font-mono group-hover:text-slate-400 transition-colors">
          <Hash className="w-3 h-3" />
          {item.id}
        </div>
      </td>

      {/* Type */}
      <td className="px-4 py-3.5">
        <div className={cn(
          'inline-flex items-center gap-1.5 text-xs px-2 py-0.5 rounded-full border border-current/15',
          item.input_type === 'url'   ? 'text-violet-400 bg-violet-500/8' :
          item.input_type === 'query' ? 'text-amber-400  bg-amber-500/8'  :
          'text-brand-400 bg-brand-500/8',
        )}>
          <InputTypeIcon type={item.input_type} />
          <span className="capitalize">{item.input_type}</span>
        </div>
      </td>

      {/* Content preview */}
      <td className="px-4 py-3.5 max-w-xs">
        <div className="text-sm text-slate-300 line-clamp-2 leading-snug group-hover:text-slate-200 transition-colors">
          {content}
        </div>
      </td>

      {/* Assessment */}
      <td className="px-4 py-3.5">
        {isCompleted && item.final_verdict ? (
          <span className={cn('text-sm font-semibold', vc.color)}>{vc.label}</span>
        ) : (
          <StatusBadge status={item.status} />
        )}
      </td>

      {/* Confidence */}
      <td className="px-4 py-3.5 hidden md:table-cell">
        {isCompleted && (
          <ConfidenceCell confidence={item.final_confidence ?? undefined} />
        )}
      </td>

      {/* Timestamp */}
      <td className="px-4 py-3.5 hidden lg:table-cell">
        <span
          className="text-xs text-slate-500 cursor-default"
          title={absTime}
        >
          <span className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {relTime}
          </span>
        </span>
      </td>

      {/* Processing time */}
      <td className="px-4 py-3.5 hidden xl:table-cell">
        <span className="text-xs text-slate-600 font-mono">
          {item.processing_time_ms ? formatMs(item.processing_time_ms) : '—'}
        </span>
      </td>

      {/* Arrow */}
      <td className="px-4 py-3.5 text-right">
        <ArrowRight className="w-4 h-4 text-slate-700 group-hover:text-brand-400 transition-colors ml-auto" />
      </td>
    </tr>
  )
}

// ── Pagination ────────────────────────────────────────────────────────────────

function Pagination({
  page, totalPages, onChange,
}: {
  page: number
  totalPages: number
  onChange: (p: number) => void
}) {
  if (totalPages <= 1) return null

  // Show a window of page numbers around the current page
  const window = 2
  const start  = Math.max(1, page - window)
  const end    = Math.min(totalPages, page + window)
  const pages  = Array.from({ length: end - start + 1 }, (_, i) => start + i)

  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-slate-500">
        Page <span className="text-slate-300">{page}</span> of{' '}
        <span className="text-slate-300">{totalPages}</span>
      </span>

      <div className="flex items-center gap-1">
        <button
          className="btn-secondary px-2.5 py-1.5 text-xs"
          disabled={page === 1}
          onClick={() => onChange(page - 1)}
        >
          <ChevronLeft className="w-3.5 h-3.5" />
        </button>

        {start > 1 && (
          <>
            <button onClick={() => onChange(1)} className="px-2.5 py-1.5 text-xs rounded-lg text-slate-400 hover:bg-slate-800 transition-colors">1</button>
            {start > 2 && <span className="text-slate-700 px-1">…</span>}
          </>
        )}

        {pages.map((p) => (
          <button
            key={p}
            onClick={() => onChange(p)}
            className={cn(
              'px-2.5 py-1.5 text-xs rounded-lg transition-colors',
              p === page
                ? 'bg-brand-600 text-white'
                : 'text-slate-400 hover:bg-slate-800',
            )}
          >
            {p}
          </button>
        ))}

        {end < totalPages && (
          <>
            {end < totalPages - 1 && <span className="text-slate-700 px-1">…</span>}
            <button onClick={() => onChange(totalPages)} className="px-2.5 py-1.5 text-xs rounded-lg text-slate-400 hover:bg-slate-800 transition-colors">{totalPages}</button>
          </>
        )}

        <button
          className="btn-secondary px-2.5 py-1.5 text-xs"
          disabled={page === totalPages}
          onClick={() => onChange(page + 1)}
        >
          <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function History() {
  const [searchParams, setSearchParams] = useSearchParams()

  const [page,      setPage]      = useState(Number(searchParams.get('page') ?? 1))
  const [verdict,   setVerdict]   = useState(searchParams.get('verdict') ?? '')
  const [inputType, setInputType] = useState(searchParams.get('type') ?? '')

  const { data, isLoading, isPlaceholderData, error, refetch } = useHistory({
    page,
    page_size: PAGE_SIZE,
    verdict:    verdict    || undefined,
    input_type: inputType  || undefined,
  })

  const pushParams = useCallback((p: number, v: string, t: string) => {
    const sp: Record<string, string> = {}
    if (p > 1)    sp.page    = String(p)
    if (v)        sp.verdict  = v
    if (t)        sp.type     = t
    setSearchParams(sp, { replace: true })
  }, [setSearchParams])

  const handleVerdictChange = (v: string) => {
    setVerdict(v); setPage(1)
    pushParams(1, v, inputType)
  }
  const handleTypeChange = (t: string) => {
    setInputType(t); setPage(1)
    pushParams(1, verdict, t)
  }
  const handlePageChange = (p: number) => {
    setPage(p)
    pushParams(p, verdict, inputType)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const hasFilters = verdict || inputType

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fade-in">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 mb-7">
        <div>
          <h1 className="text-2xl font-bold text-slate-100 mb-1">Analysis History</h1>
          <p className="text-sm text-slate-500">
            {data
              ? <><span className="text-slate-300 font-medium">{data.total.toLocaleString()}</span> analyses stored in PostgreSQL</>
              : 'Loading…'}
          </p>
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-slate-600" />

          <select
            className="input-field w-auto text-xs py-1.5"
            value={verdict}
            onChange={(e) => handleVerdictChange(e.target.value)}
          >
            {VERDICT_OPTS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>

          <select
            className="input-field w-auto text-xs py-1.5"
            value={inputType}
            onChange={(e) => handleTypeChange(e.target.value)}
          >
            {TYPE_OPTS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>

          {hasFilters && (
            <button
              className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
              onClick={() => { handleVerdictChange(''); handleTypeChange('') }}
            >
              Clear
            </button>
          )}

          <button
            onClick={() => refetch()}
            className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── Loading ──────────────────────────────────────────────────────── */}
      {isLoading && !data && (
        <div className="flex justify-center py-24"><Spinner size="lg" /></div>
      )}

      {/* ── Error ────────────────────────────────────────────────────────── */}
      {error && (
        <ErrorState
          message={(error as Error).message}
          onRetry={() => refetch()}
        />
      )}

      {/* ── Empty ────────────────────────────────────────────────────────── */}
      {!isLoading && !error && data?.items.length === 0 && (
        <EmptyState
          icon={<Clock />}
          title={hasFilters ? 'No results for these filters' : 'No analyses yet'}
          description={
            hasFilters
              ? 'Try adjusting your filters or clearing them.'
              : 'Submit an article, URL, or claim to begin building your history.'
          }
          action={
            hasFilters
              ? <button className="btn-secondary" onClick={() => { handleVerdictChange(''); handleTypeChange('') }}>Clear filters</button>
              : <Link to="/analyze" className="btn-primary">Start Analyzing</Link>
          }
        />
      )}

      {/* ── Table ────────────────────────────────────────────────────────── */}
      {!error && data && data.items.length > 0 && (
        <>
          <div className={cn(
            'glass-card overflow-hidden mb-5 transition-opacity duration-200',
            isPlaceholderData && 'opacity-60',
          )}>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-slate-800">
                    <th className="px-4 py-3 text-left label-sm">ID</th>
                    <th className="px-4 py-3 text-left label-sm">Type</th>
                    <th className="px-4 py-3 text-left label-sm">Content</th>
                    <th className="px-4 py-3 text-left label-sm">Assessment</th>
                    <th className="px-4 py-3 text-left label-sm hidden md:table-cell">Confidence</th>
                    <th className="px-4 py-3 text-left label-sm hidden lg:table-cell">Time</th>
                    <th className="px-4 py-3 text-left label-sm hidden xl:table-cell">Duration</th>
                    <th className="px-4 py-3" />
                  </tr>
                </thead>
                <tbody>
                  {data.items.map((item) => (
                    <HistoryRow key={item.id} item={item} />
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <Pagination
            page={data.page}
            totalPages={data.total_pages}
            onChange={handlePageChange}
          />
        </>
      )}
    </div>
  )
}
