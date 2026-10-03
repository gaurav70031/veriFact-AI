import { useState } from 'react'
import { Search, Loader2, ExternalLink, Calendar, AlertCircle, RefreshCw } from 'lucide-react'
import { useSearchEvidence } from '@/hooks/useApi'
import { Badge } from '@/components/ui/Badge'
import { EmptyState } from '@/components/ui/EmptyState'
import { formatDate, cn } from '@/lib/utils'
import type { EvidenceItem } from '@/types/api'

function NewsCard({ item }: { item: EvidenceItem }) {
  const typeColors: Record<string, string> = {
    news_api: 'bg-brand-500/15 text-brand-400',
    gnews:    'bg-violet-500/15 text-violet-400',
    rss_feed: 'bg-emerald-500/15 text-emerald-400',
    web_search: 'bg-amber-500/15 text-amber-400',
  }

  return (
    <div className="glass-card p-5 hover:border-slate-700/80 transition-colors group">
      <div className="flex items-start justify-between gap-3 mb-2">
        <a
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm font-medium text-slate-200 hover:text-brand-400 transition-colors line-clamp-2 flex-1"
        >
          {item.title}
          <ExternalLink className="inline-block w-3 h-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
        </a>
        <span className={cn('text-xs px-2 py-0.5 rounded-full font-medium shrink-0', typeColors[item.source_type] ?? 'bg-slate-700 text-slate-400')}>
          {item.source_type.replace('_', ' ')}
        </span>
      </div>

      {item.description && (
        <p className="text-xs text-slate-500 line-clamp-2 mb-3">{item.description}</p>
      )}

      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 text-xs text-slate-600">
          <span className="font-medium text-slate-500">{item.source_name}</span>
          {item.published_at && (
            <span className="flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              {formatDate(item.published_at)}
            </span>
          )}
        </div>
        <div className="text-xs text-slate-600 font-mono">
          {(item.relevance_score * 100).toFixed(0)}% relevance
        </div>
      </div>
    </div>
  )
}

export default function LiveNews() {
  const [query, setQuery] = useState('')
  const [submitted, setSubmitted] = useState('')
  const [fromDays, setFromDays] = useState(30)

  const { mutate, data, isPending, error } = useSearchEvidence()

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return
    setSubmitted(query)
    mutate({ claim: query, max_results: 20, from_days: fromDays })
  }

  const noResults = data && data.items.length === 0
  const hasResults = data && data.items.length > 0

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-slate-100 mb-2">Live News Search</h1>
        <p className="text-sm text-slate-500">
          Search real-time news from configured providers (NewsAPI, GNews, RSS feeds).
          Results come directly from external APIs — no cached or fabricated content.
        </p>
      </div>

      {/* Search form */}
      <form onSubmit={handleSearch} className="glass-card p-5 mb-6">
        <div className="flex gap-3">
          <div className="flex-1">
            <input
              className="input-field"
              placeholder="Enter a claim, topic, or search query…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <select
            className="input-field w-auto"
            value={fromDays}
            onChange={(e) => setFromDays(Number(e.target.value))}
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
            <option value={365}>Last year</option>
          </select>
          <button type="submit" disabled={isPending || !query.trim()} className="btn-primary">
            {isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            {isPending ? 'Searching…' : 'Search'}
          </button>
        </div>
      </form>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/20 mb-6">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
          <p className="text-sm text-red-300">{(error as Error).message}</p>
        </div>
      )}

      {/* Results meta */}
      {data && (
        <div className="flex flex-wrap items-center gap-3 mb-5 text-sm text-slate-500">
          {hasResults && (
            <span>
              Found <span className="text-slate-300 font-medium">{data.after_dedup}</span> unique results
              {data.total_found > data.after_dedup && ` (${data.total_found} before dedup)`}
            </span>
          )}
          {data.providers_used.length > 0 && (
            <span>via {data.providers_used.join(', ')}</span>
          )}
          <span className="text-xs text-slate-700">in {data.search_time_ms.toFixed(0)}ms</span>
          {data.providers_failed.length > 0 && (
            <Badge variant="warning" className="flex items-center gap-1">
              <AlertCircle className="w-2.5 h-2.5" />
              {data.providers_failed.length} provider{data.providers_failed.length > 1 ? 's' : ''} failed
            </Badge>
          )}
        </div>
      )}

      {/* No results */}
      {noResults && (
        <EmptyState
          icon={<Search />}
          title="No results found"
          description={
            data.status === 'INSUFFICIENT_EVIDENCE'
              ? `No articles found for "${submitted}". Absence of results does not indicate the claim is false.`
              : data.status === 'all_providers_failed'
              ? 'All evidence providers failed to respond. Check your API keys or try again later.'
              : `No matching articles found for this search.`
          }
          action={
            <button className="btn-secondary" onClick={() => setQuery(submitted)}>
              <RefreshCw className="w-4 h-4" />
              Try again
            </button>
          }
        />
      )}

      {/* Results grid */}
      {hasResults && (
        <div className="space-y-3">
          {data.items.map((item, i) => (
            <NewsCard key={`${item.url}-${i}`} item={item} />
          ))}
        </div>
      )}

      {/* Initial empty state */}
      {!data && !isPending && (
        <EmptyState
          icon={<Search />}
          title="Search for current news"
          description="Enter a topic or claim above to search live news sources."
        />
      )}
    </div>
  )
}
