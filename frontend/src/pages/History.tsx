import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Clock, ChevronLeft, ChevronRight, Filter } from 'lucide-react'
import { useHistory } from '@/hooks/useApi'
import { Badge } from '@/components/ui/Badge'
import { Spinner } from '@/components/ui/Spinner'
import { ErrorState } from '@/components/ui/ErrorState'
import { EmptyState } from '@/components/ui/EmptyState'
import { cn, formatRelativeTime, getVerdictConfig, truncate } from '@/lib/utils'

const VERDICT_OPTS = ['', 'FAKE', 'REAL', 'UNVERIFIED', 'MIXED']
const TYPE_OPTS    = ['', 'text', 'url', 'claim']

export default function History() {
  const [page, setPage]   = useState(1)
  const [verdict, setVerdict] = useState('')
  const [inputType, setInputType] = useState('')

  const { data, isLoading, error, refetch } = useHistory({
    page,
    page_size: 20,
    verdict:    verdict    || undefined,
    input_type: inputType  || undefined,
  })

  const handleFilterChange = () => setPage(1)

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-slate-100">Analysis History</h1>
          {data && (
            <p className="text-sm text-slate-500 mt-1">
              {data.total.toLocaleString()} total analyses
            </p>
          )}
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-500" />
          <select
            className="input-field w-auto text-xs py-1.5"
            value={verdict}
            onChange={(e) => { setVerdict(e.target.value); handleFilterChange() }}
          >
            <option value="">All verdicts</option>
            {VERDICT_OPTS.slice(1).map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
          <select
            className="input-field w-auto text-xs py-1.5"
            value={inputType}
            onChange={(e) => { setInputType(e.target.value); handleFilterChange() }}
          >
            <option value="">All types</option>
            {TYPE_OPTS.slice(1).map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
      </div>

      {isLoading && (
        <div className="flex justify-center py-24">
          <Spinner size="lg" />
        </div>
      )}

      {error && (
        <ErrorState
          message={(error as Error).message}
          onRetry={() => refetch()}
        />
      )}

      {!isLoading && !error && data?.items.length === 0 && (
        <EmptyState
          icon={<Clock />}
          title="No analyses yet"
          description="Submit an article, URL, or claim on the Analyze page to get started."
          action={<Link to="/analyze" className="btn-primary">Start Analyzing</Link>}
        />
      )}

      {!isLoading && !error && data && data.items.length > 0 && (
        <>
          <div className="glass-card overflow-hidden mb-6">
            <table className="w-full">
              <thead>
                <tr className="border-b border-slate-800">
                  <th className="px-5 py-3 text-left label-sm">Input</th>
                  <th className="px-5 py-3 text-left label-sm hidden sm:table-cell">Type</th>
                  <th className="px-5 py-3 text-left label-sm">ML Verdict</th>
                  <th className="px-5 py-3 text-right label-sm hidden md:table-cell">Time</th>
                  <th className="px-5 py-3 text-right label-sm"></th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item) => {
                  const vc = getVerdictConfig(item.final_verdict ?? undefined)
                  const statusBadge = item.status === 'failed' ? 'danger' :
                    item.status === 'pending' || item.status === 'processing' ? 'warning' : 'muted'
                  return (
                    <tr key={item.id} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
                      <td className="px-5 py-4 max-w-xs">
                        <div className="text-sm text-slate-300 line-clamp-2">
                          {item.article_title || truncate(item.original_input, 100)}
                        </div>
                        <div className="text-xs text-slate-600 mt-0.5 flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {formatRelativeTime(item.created_at)}
                        </div>
                      </td>
                      <td className="px-5 py-4 hidden sm:table-cell">
                        <Badge variant="muted">{item.input_type}</Badge>
                      </td>
                      <td className="px-5 py-4">
                        {item.status === 'completed' && item.final_verdict ? (
                          <span className={cn('text-sm font-semibold', vc.color)}>{vc.label}</span>
                        ) : (
                          <Badge variant={statusBadge}>{item.status}</Badge>
                        )}
                      </td>
                      <td className="px-5 py-4 text-right text-xs text-slate-600 hidden md:table-cell font-mono">
                        {item.processing_time_ms ? `${item.processing_time_ms}ms` : '—'}
                      </td>
                      <td className="px-5 py-4 text-right">
                        <Link
                          to={`/results/${item.id}`}
                          className="text-xs text-brand-400 hover:text-brand-300 font-medium transition-colors"
                        >
                          View →
                        </Link>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {data.total_pages > 1 && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-500">
                Page {data.page} of {data.total_pages}
              </span>
              <div className="flex gap-2">
                <button
                  className="btn-secondary px-3 py-1.5 text-xs"
                  disabled={page === 1}
                  onClick={() => setPage((p) => p - 1)}
                >
                  <ChevronLeft className="w-4 h-4" />
                  Prev
                </button>
                <button
                  className="btn-secondary px-3 py-1.5 text-xs"
                  disabled={page === data.total_pages}
                  onClick={() => setPage((p) => p + 1)}
                >
                  Next
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
