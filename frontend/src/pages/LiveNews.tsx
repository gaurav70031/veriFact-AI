/**
 * Live News page
 *
 * Data flow (as required):
 *   User types query
 *   → GET /api/v1/news/search?q=...  (backend proxy)
 *   → backend calls NewsAPI / GNews / RSS / Search
 *   → normalised EvidenceItem[] returned
 *   → rendered here
 *
 * The frontend NEVER calls news APIs directly.
 * All provider keys remain server-side.
 *
 * States handled:
 *   • Idle (before first search)
 *   • Loading / in-flight
 *   • Success with results
 *   • Insufficient evidence (no results, providers ok)
 *   • All providers failed
 *   • Rate-limit error (HTTP 429)
 *   • Generic API error
 *   • Stale results (search_time_ms > STALE_THRESHOLD)
 *   • Partial failure (some providers failed, others ok)
 *
 * Each result card has a [ Verify this claim ] button that sends
 * the article title + URL to the Analyze pipeline.
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  Search, ExternalLink, Calendar, RefreshCw,
  AlertCircle, Wifi, WifiOff, Clock, ChevronDown,
  Newspaper, Zap, ShieldCheck, Info, AlertTriangle,
  Loader2, ArrowRight, Radio,
} from 'lucide-react'
import { cn, formatDate, formatRelativeTime } from '@/lib/utils'
import { useNewsSearch, useAnalyzeUrl, useAnalyzeClaim } from '@/hooks/useApi'
import { Badge } from '@/components/ui/Badge'
import type { EvidenceItem, NewsSearchResult } from '@/types/api'

// ─── Constants ────────────────────────────────────────────────────────────────

/** Results older than this (in ms of search time) are considered stale */
const STALE_THRESHOLD_MS = 5 * 60_000   // 5 minutes
/** ms since the query ran after which we flag results as potentially stale */
const RESULT_AGE_STALE   = 3 * 60_000   // 3 minutes

const MAX_RESULTS_OPTIONS = [10, 20, 50] as const
const DATE_RANGE_OPTIONS = [
  { label: 'Last 24 hours', days: 1  },
  { label: 'Last 7 days',   days: 7  },
  { label: 'Last 30 days',  days: 30 },
  { label: 'Last 90 days',  days: 90 },
  { label: 'Last year',     days: 365 },
]

// ─── Source type config ───────────────────────────────────────────────────────

const SOURCE_TYPE_CONFIG: Record<string, {
  label: string
  bg: string
  text: string
  icon: React.ReactNode
}> = {
  news_api:   { label: 'NewsAPI',    bg: 'bg-brand-500/10',   text: 'text-brand-400',   icon: <Newspaper   className="w-3 h-3" /> },
  gnews:      { label: 'GNews',      bg: 'bg-violet-500/10',  text: 'text-violet-400',  icon: <Newspaper   className="w-3 h-3" /> },
  rss_feed:   { label: 'RSS',        bg: 'bg-emerald-500/10', text: 'text-emerald-400', icon: <Radio       className="w-3 h-3" /> },
  web_search: { label: 'Web',        bg: 'bg-amber-500/10',   text: 'text-amber-400',   icon: <Search      className="w-3 h-3" /> },
  official:   { label: 'Official',   bg: 'bg-sky-500/10',     text: 'text-sky-400',     icon: <ShieldCheck className="w-3 h-3" /> },
  fact_check: { label: 'Fact Check', bg: 'bg-rose-500/10',    text: 'text-rose-400',    icon: <ShieldCheck className="w-3 h-3" /> },
}

function SourceTypeBadge({ sourceType }: { sourceType: string }) {
  const cfg = SOURCE_TYPE_CONFIG[sourceType] ?? {
    label: sourceType.replace('_', ' '),
    bg:    'bg-slate-700/50',
    text:  'text-slate-400',
    icon:  <Newspaper className="w-3 h-3" />,
  }
  return (
    <span className={cn(
      'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border border-current/10',
      cfg.bg, cfg.text,
    )}>
      {cfg.icon}
      {cfg.label}
    </span>
  )
}

// ─── Verify button + state ────────────────────────────────────────────────────

interface VerifyButtonProps {
  item: EvidenceItem
  onVerifyStart: () => void
}

function VerifyButton({ item, onVerifyStart }: VerifyButtonProps) {
  const navigate    = useNavigate()
  const urlMutation = useAnalyzeUrl()
  const claimMut    = useAnalyzeClaim()
  const [busy, setBusy] = useState(false)

  const handleVerify = async () => {
    setBusy(true)
    onVerifyStart()

    try {
      // If we have a proper URL, send it for extraction + analysis
      if (item.url && item.url.startsWith('http')) {
        const result = await urlMutation.mutateAsync({ url: item.url })
        navigate(`/results/${result.id}`, { state: { result } })
      } else {
        // Fall back to claim analysis using title + snippet
        const claim = item.title
        const context = item.description ?? undefined
        const result = await claimMut.mutateAsync({ claim, context })
        navigate(`/results/${result.id}`, { state: { result } })
      }
    } catch {
      setBusy(false)
    }
  }

  return (
    <button
      onClick={handleVerify}
      disabled={busy}
      className={cn(
        'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium',
        'border border-brand-500/30 bg-brand-500/10 text-brand-400',
        'hover:bg-brand-500/20 hover:border-brand-500/50 transition-all duration-150',
        'focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500',
        'disabled:opacity-50 disabled:cursor-not-allowed',
      )}
      title="Run fake-news analysis on this article"
    >
      {busy ? (
        <>
          <Loader2 className="w-3 h-3 animate-spin" />
          Analysing…
        </>
      ) : (
        <>
          <ShieldCheck className="w-3 h-3" />
          Verify this claim
        </>
      )}
    </button>
  )
}

// ─── News card ────────────────────────────────────────────────────────────────

function NewsCard({
  item,
  index,
  onVerifyStart,
}: {
  item: EvidenceItem
  index: number
  onVerifyStart: () => void
}) {
  const relTime = item.published_at ? formatRelativeTime(item.published_at) : null
  const absDate = item.published_at ? formatDate(item.published_at)         : null
  const relScore = Math.round(item.relevance_score * 100)

  return (
    <article
      className="glass-card p-5 hover:border-slate-700/70 transition-all duration-200 group animate-fade-in"
      style={{ animationDelay: `${index * 30}ms` }}
    >
      {/* Top row: headline + source type badge */}
      <div className="flex items-start gap-3 mb-3">
        <div className="flex-1 min-w-0">
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-semibold text-slate-200 hover:text-brand-400 transition-colors leading-snug block group-hover:text-brand-300"
          >
            {item.title}
            <ExternalLink className="inline-block w-3 h-3 ml-1.5 opacity-0 group-hover:opacity-60 transition-opacity shrink-0 align-middle" />
          </a>
        </div>
        <SourceTypeBadge sourceType={item.source_type} />
      </div>

      {/* Description / snippet */}
      {item.description && (
        <p className="text-xs text-slate-500 leading-relaxed mb-3 line-clamp-3">
          {item.description}
        </p>
      )}

      {/* Source + date + relevance row */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 mb-4">
        {/* Source name */}
        <span className="text-xs font-medium text-slate-400">{item.source_name}</span>

        {/* Publication time */}
        {relTime && (
          <span
            className="text-xs text-slate-600 flex items-center gap-1"
            title={absDate ?? ''}
          >
            <Clock className="w-3 h-3" />
            {relTime}
            {absDate && (
              <span className="text-slate-700 ml-0.5">({absDate})</span>
            )}
          </span>
        )}

        {/* Provider */}
        <span className="text-xs text-slate-700 flex items-center gap-1">
          <Wifi className="w-3 h-3" />
          {item.provider_name}
        </span>

        {/* Relevance score */}
        <span className="text-xs text-slate-700 font-mono ml-auto">
          {relScore}% relevance
        </span>
      </div>

      {/* Footer: URL + verify button */}
      <div className="flex items-center justify-between gap-3 pt-3 border-t border-slate-800/50">
        <span className="text-xs text-slate-700 font-mono truncate flex-1 min-w-0">
          {item.url}
        </span>
        <VerifyButton item={item} onVerifyStart={onVerifyStart} />
      </div>
    </article>
  )
}

// ─── Empty / error states ──────────────────────────────────────────────────────

function IdleState() {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center px-4">
      <div className="w-14 h-14 rounded-2xl bg-brand-500/10 border border-brand-500/20 flex items-center justify-center mb-4">
        <Search className="w-7 h-7 text-brand-500/60" />
      </div>
      <h3 className="text-base font-semibold text-slate-300 mb-2">
        Search current news
      </h3>
      <p className="text-sm text-slate-500 max-w-sm leading-relaxed">
        Enter a topic, claim, or keyword above. Results come directly from
        configured news providers — no cached or fabricated content.
      </p>
    </div>
  )
}

function LoadingState({ query }: { query: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4">
      <div className="relative">
        <Loader2 className="w-8 h-8 text-brand-500 animate-spin" />
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-slate-300 mb-1">
          Searching news providers…
        </p>
        <p className="text-xs text-slate-600">
          Querying NewsAPI, GNews, RSS feeds and web search for{' '}
          <span className="text-slate-400">"{query}"</span>
        </p>
      </div>
    </div>
  )
}

function InsufficientState({ query, onRetry }: { query: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center px-4 gap-4">
      <div className="w-14 h-14 rounded-2xl bg-slate-800/60 border border-slate-700/40 flex items-center justify-center">
        <Info className="w-7 h-7 text-slate-500" />
      </div>
      <div>
        <h3 className="text-base font-semibold text-slate-300 mb-2">
          No articles found
        </h3>
        <p className="text-sm text-slate-500 max-w-md leading-relaxed mb-1">
          No current articles matching{' '}
          <span className="text-slate-400">"{query}"</span>{' '}
          were found in configured providers.
        </p>
        <p className="text-xs text-amber-600/80 max-w-md leading-relaxed">
          Absence of results does not indicate the topic is false or
          non-existent. Try broader search terms or a wider date range.
        </p>
      </div>
      <button onClick={onRetry} className="btn-secondary text-sm">
        <RefreshCw className="w-3.5 h-3.5" />
        Try again
      </button>
    </div>
  )
}

function AllFailedState({ providersFailed, onRetry }: {
  providersFailed: string[]
  onRetry: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center px-4 gap-4">
      <div className="w-14 h-14 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center justify-center">
        <WifiOff className="w-7 h-7 text-red-400" />
      </div>
      <div>
        <h3 className="text-base font-semibold text-slate-300 mb-2">
          Evidence providers unavailable
        </h3>
        <p className="text-sm text-slate-500 max-w-md leading-relaxed">
          All configured news providers failed to respond. This is a temporary
          connectivity issue — no inference about the content can be made.
        </p>
        {providersFailed.length > 0 && (
          <p className="text-xs text-slate-700 mt-2">
            Failed: {providersFailed.join(', ')}
          </p>
        )}
      </div>
      <button onClick={onRetry} className="btn-secondary text-sm">
        <RefreshCw className="w-3.5 h-3.5" />
        Retry
      </button>
    </div>
  )
}

function ApiErrorState({ message, isRateLimit, onRetry }: {
  message: string
  isRateLimit: boolean
  onRetry: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center px-4 gap-4">
      <div className={cn(
        'w-14 h-14 rounded-2xl flex items-center justify-center',
        isRateLimit
          ? 'bg-amber-500/10 border border-amber-500/20'
          : 'bg-red-500/10 border border-red-500/20',
      )}>
        <AlertCircle className={cn('w-7 h-7', isRateLimit ? 'text-amber-400' : 'text-red-400')} />
      </div>
      <div>
        <h3 className="text-base font-semibold text-slate-300 mb-2">
          {isRateLimit ? 'Rate limit reached' : 'Search failed'}
        </h3>
        <p className="text-sm text-slate-500 max-w-md leading-relaxed">
          {isRateLimit
            ? 'The news API rate limit has been reached. Please wait a moment before searching again.'
            : message}
        </p>
      </div>
      {!isRateLimit && (
        <button onClick={onRetry} className="btn-secondary text-sm">
          <RefreshCw className="w-3.5 h-3.5" />
          Try again
        </button>
      )}
    </div>
  )
}

// ─── Results metadata bar ─────────────────────────────────────────────────────

function ResultsMeta({
  result,
  queryTime,
}: {
  result: NewsSearchResult
  queryTime: number | null
}) {
  const now     = Date.now()
  const ageMs   = queryTime ? now - queryTime : 0
  const isStale = ageMs > RESULT_AGE_STALE

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 py-3 px-1 text-xs">
      {/* Count */}
      <span className="text-slate-400 font-medium">
        {result.after_dedup}{' '}
        {result.after_dedup === 1 ? 'article' : 'articles'}
        {result.total_found > result.after_dedup && (
          <span className="text-slate-600">
            {' '}({result.total_found} before dedup)
          </span>
        )}
      </span>

      {/* Providers used */}
      {result.providers_used.length > 0 && (
        <span className="flex items-center gap-1 text-slate-600">
          <Wifi className="w-3 h-3" />
          {result.providers_used.join(', ')}
        </span>
      )}

      {/* Search latency */}
      <span className="text-slate-700 font-mono">
        {result.search_time_ms.toFixed(0)} ms
      </span>

      {/* Partial failure */}
      {result.providers_failed.length > 0 && (
        <span className="flex items-center gap-1 text-amber-600">
          <AlertTriangle className="w-3 h-3" />
          {result.providers_failed.length} provider{result.providers_failed.length > 1 ? 's' : ''} failed
          <span className="text-amber-800">({result.providers_failed.join(', ')})</span>
        </span>
      )}

      {/* Stale results badge */}
      {isStale && (
        <span className="flex items-center gap-1 text-amber-600 ml-auto">
          <Clock className="w-3 h-3" />
          Results from {Math.round(ageMs / 60_000)} min ago — may be stale
        </span>
      )}
    </div>
  )
}

// ─── Search bar ───────────────────────────────────────────────────────────────

interface SearchBarProps {
  query:         string
  fromDays:      number
  maxResults:    number
  onQueryChange:      (v: string)  => void
  onFromDaysChange:   (v: number)  => void
  onMaxResultsChange: (v: number)  => void
  onSubmit:      (e: React.FormEvent) => void
  isLoading:     boolean
}

function SearchBar({
  query, fromDays, maxResults,
  onQueryChange, onFromDaysChange, onMaxResultsChange,
  onSubmit, isLoading,
}: SearchBarProps) {
  return (
    <form onSubmit={onSubmit} className="glass-card p-4">
      {/* Main input row */}
      <div className="flex gap-2 mb-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600 pointer-events-none" />
          <input
            className="input-field pl-9 pr-4"
            placeholder="Search current news — e.g. 'ECB raised interest rates 2024'"
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            autoComplete="off"
            spellCheck={false}
            autoFocus
          />
        </div>
        <button
          type="submit"
          disabled={isLoading || query.trim().length < 3}
          className="btn-primary shrink-0"
        >
          {isLoading ? (
            <><Loader2 className="w-4 h-4 animate-spin" />Searching…</>
          ) : (
            <><Search className="w-4 h-4" />Search</>
          )}
        </button>
      </div>

      {/* Filters row */}
      <div className="flex flex-wrap gap-2 items-center">
        <span className="text-xs text-slate-600">Filters:</span>
        <select
          className="input-field py-1.5 text-xs w-auto"
          value={fromDays}
          onChange={(e) => onFromDaysChange(Number(e.target.value))}
        >
          {DATE_RANGE_OPTIONS.map((opt) => (
            <option key={opt.days} value={opt.days}>{opt.label}</option>
          ))}
        </select>
        <select
          className="input-field py-1.5 text-xs w-auto"
          value={maxResults}
          onChange={(e) => onMaxResultsChange(Number(e.target.value))}
        >
          {MAX_RESULTS_OPTIONS.map((n) => (
            <option key={n} value={n}>{n} results</option>
          ))}
        </select>

        <span className="text-xs text-slate-700 ml-auto flex items-center gap-1">
          <Info className="w-3 h-3" />
          Results from backend evidence providers only — no direct API calls
        </span>
      </div>
    </form>
  )
}

// ─── Verify-in-progress toast ─────────────────────────────────────────────────

function VerifyingToast({ visible }: { visible: boolean }) {
  if (!visible) return null
  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 animate-slide-up">
      <div className="flex items-center gap-3 px-5 py-3.5 rounded-xl bg-brand-600 text-white text-sm font-medium shadow-xl shadow-brand-900/40 border border-brand-500/50">
        <Loader2 className="w-4 h-4 animate-spin" />
        Sending to analysis pipeline…
        <ArrowRight className="w-4 h-4" />
      </div>
    </div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────

export default function LiveNews() {
  const [searchParams, setSearchParams] = useSearchParams()

  // Initialise from URL so searches are bookmarkable
  const [inputQuery,  setInputQuery]   = useState(searchParams.get('q') ?? '')
  const [fromDays,    setFromDays]      = useState(Number(searchParams.get('d') ?? 30))
  const [maxResults,  setMaxResults]    = useState(Number(searchParams.get('n') ?? 20))

  // The committed query (what was actually searched)
  const [activeParams, setActiveParams] = useState<{
    q: string; from_days: number; max_results: number
  } | null>(
    searchParams.get('q')
      ? { q: searchParams.get('q')!, from_days: fromDays, max_results: maxResults }
      : null,
  )

  const [queryTime,    setQueryTime]    = useState<number | null>(null)
  const [verifying,    setVerifying]    = useState(false)

  const { data, isLoading, error, refetch } = useNewsSearch(activeParams)

  // When a new search is committed, record the time and push to URL
  const handleSearch = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    const q = inputQuery.trim()
    if (q.length < 3) return
    const params = { q, from_days: fromDays, max_results: maxResults }
    setActiveParams(params)
    setQueryTime(Date.now())
    setSearchParams({ q, d: String(fromDays), n: String(maxResults) }, { replace: true })
  }, [inputQuery, fromDays, maxResults, setSearchParams])

  const handleRetry = useCallback(() => {
    setQueryTime(Date.now())
    refetch()
  }, [refetch])

  // Determine error type
  const errorMsg     = (error as Error)?.message ?? ''
  const isRateLimit  = errorMsg.toLowerCase().includes('429') ||
                       errorMsg.toLowerCase().includes('rate limit')

  // Determine which state to show
  const showIdle        = !activeParams
  const showLoading     = isLoading
  const showError       = !isLoading && !!error
  const showNoResults   = !isLoading && !error && data?.items.length === 0
  const showResults     = !isLoading && !error && (data?.items.length ?? 0) > 0

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fade-in">

      {/* ── Page header ─────────────────────────────────────────────────── */}
      <div className="mb-7">
        <div className="flex items-center gap-2.5 mb-2">
          <div className="p-1.5 rounded-lg bg-brand-500/10 border border-brand-500/20">
            <Newspaper className="w-5 h-5 text-brand-400" />
          </div>
          <h1 className="text-2xl font-bold text-slate-100">Live News</h1>
          <span className="flex items-center gap-1 text-xs text-emerald-500 font-medium ml-1">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Live
          </span>
        </div>
        <p className="text-sm text-slate-500 leading-relaxed">
          Search current articles from configured backend providers.
          The frontend never calls news APIs directly — all requests route
          through the backend evidence service.
        </p>
      </div>

      {/* ── Search bar ──────────────────────────────────────────────────── */}
      <div className="mb-5">
        <SearchBar
          query={inputQuery}
          fromDays={fromDays}
          maxResults={maxResults}
          onQueryChange={setInputQuery}
          onFromDaysChange={setFromDays}
          onMaxResultsChange={setMaxResults}
          onSubmit={handleSearch}
          isLoading={isLoading}
        />
      </div>

      {/* ── Results metadata ─────────────────────────────────────────────── */}
      {showResults && data && (
        <ResultsMeta result={data} queryTime={queryTime} />
      )}

      {/* ── Content area ─────────────────────────────────────────────────── */}

      {showIdle && <IdleState />}

      {showLoading && activeParams && (
        <LoadingState query={activeParams.q} />
      )}

      {showError && (
        <ApiErrorState
          message={errorMsg || 'An unexpected error occurred.'}
          isRateLimit={isRateLimit}
          onRetry={handleRetry}
        />
      )}

      {showNoResults && data && activeParams && (
        data.status === 'all_providers_failed' ? (
          <AllFailedState
            providersFailed={data.providers_failed}
            onRetry={handleRetry}
          />
        ) : (
          <InsufficientState
            query={activeParams.q}
            onRetry={handleRetry}
          />
        )
      )}

      {showResults && data && (
        <div className="space-y-3">
          {data.items.map((item, i) => (
            <NewsCard
              key={`${item.url}-${i}`}
              item={item}
              index={i}
              onVerifyStart={() => setVerifying(true)}
            />
          ))}
        </div>
      )}

      {/* Verify toast */}
      <VerifyingToast visible={verifying} />
    </div>
  )
}
