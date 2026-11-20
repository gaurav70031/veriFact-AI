/**
 * Misinformation Pattern Analysis
 *
 * Full analytics dashboard driven by GET /api/v1/analytics/patterns
 * with dedicated endpoints for word freq, temporal, sentiment, and topics.
 *
 * All data from live API. No hardcoded statistics.
 * Insights distinguish correlations from causal conclusions.
 */

import { useState } from 'react'
import {
  TrendingUp, BarChart3, Database, RefreshCw, AlertCircle,
  PieChart as PieChartIcon, Activity, Layers, Hash,
  Filter, Lightbulb, AlertTriangle, Type, FileText, Clock,
} from 'lucide-react'
import {
  BarChart, Bar, Cell, PieChart, Pie, AreaChart, Area,
  ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, Legend, ReferenceLine,
} from 'recharts'
import { usePatterns, useTemporalTrend, useSentiment, useWordFreq, useTopics } from '@/hooks/useApi'
import { WordCloud } from '@/components/charts/WordCloud'
import { Spinner } from '@/components/ui/Spinner'
import { SkeletonCard, SkeletonChart } from '@/components/ui/Skeleton'
import { cn } from '@/lib/utils'
import type { PatternsResponse, Insight } from '@/types/api'

// ── Tooltip style ─────────────────────────────────────────────────────────────
const TT = {
  contentStyle: { backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)', borderRadius: 8 },
  labelStyle: { color: 'var(--text-primary)', fontWeight: 600 },
  itemStyle:  { color: 'var(--text-secondary)' },
}

const MODEL_COLORS = ['#818cf8', '#34d399', '#f87171', '#fbbf24', '#a78bfa', '#60a5fa']

// ── Filter bar ────────────────────────────────────────────────────────────────

interface FilterState {
  source:    string    // "live" | "dataset" | "both"
  verdict:   string
  date_from: string
  date_to:   string
  limit:     number
}

function FilterBar({ filters, onFilter }: { filters: FilterState; onFilter: (f: FilterState) => void }) {
  return (
    <div className="card p-4 mb-6">
      <div className="flex flex-wrap items-center gap-3">
        <Filter className="w-4 h-4" style={{ color: 'var(--text-subtle)' }} />

        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-subtle)' }}>Source</label>
          <select className="input-field text-sm py-1.5" value={filters.source}
            onChange={(e) => onFilter({ ...filters, source: e.target.value })}>
            <option value="live">Live analyses</option>
            <option value="dataset">Dataset (ISOT)</option>
            <option value="both">Both</option>
          </select>
        </div>

        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-subtle)' }}>Verdict</label>
          <select className="input-field text-sm py-1.5" value={filters.verdict}
            onChange={(e) => onFilter({ ...filters, verdict: e.target.value })}>
            <option value="">All</option>
            <option value="FAKE">FAKE</option>
            <option value="REAL">REAL</option>
            <option value="MIXED">MIXED</option>
            <option value="UNVERIFIED">UNVERIFIED</option>
          </select>
        </div>

        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-subtle)' }}>From date</label>
          <input type="date" className="input-field text-sm py-1.5" value={filters.date_from}
            onChange={(e) => onFilter({ ...filters, date_from: e.target.value })} />
        </div>

        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-subtle)' }}>To date</label>
          <input type="date" className="input-field text-sm py-1.5" value={filters.date_to}
            onChange={(e) => onFilter({ ...filters, date_to: e.target.value })} />
        </div>

        <div>
          <label className="text-xs block mb-1" style={{ color: 'var(--text-subtle)' }}>Max records</label>
          <select className="input-field text-sm py-1.5" value={String(filters.limit)}
            onChange={(e) => onFilter({ ...filters, limit: parseInt(e.target.value) })}>
            <option value="100">100</option>
            <option value="500">500</option>
            <option value="1000">1000</option>
            <option value="2000">2000</option>
          </select>
        </div>
      </div>
    </div>
  )
}

// ── Section wrapper ───────────────────────────────────────────────────────────
function Section({ title, icon, children, subtitle }: { title: string; icon: React.ReactNode; children: React.ReactNode; subtitle?: string }) {
  return (
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-1">
        <span style={{ color: 'var(--brand)' }}>{icon}</span>
        <h2 className="section-title">{title}</h2>
      </div>
      {subtitle && <p className="text-xs mb-4" style={{ color: 'var(--text-subtle)' }}>{subtitle}</p>}
      <div className={subtitle ? '' : 'mt-4'}>{children}</div>
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────
function EmptyState({ msg }: { msg: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 gap-2" style={{ color: 'var(--text-faint)' }}>
      <AlertCircle className="w-6 h-6 opacity-40" />
      <p className="text-sm">{msg}</p>
    </div>
  )
}

// ── Insight card ──────────────────────────────────────────────────────────────
function InsightCard({ insight }: { insight: Insight }) {
  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 mb-2">
        <Lightbulb className="w-4 h-4" style={{ color: 'var(--brand)' }} />
        <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{insight.category}</span>
      </div>
      <p className="text-sm leading-relaxed mb-2" style={{ color: 'var(--text-secondary)' }}>{insight.observation}</p>
      {insight.caveat && (
        <div className="flex items-start gap-1.5 mt-2 pt-2" style={{ borderTop: '1px solid var(--border-subtle)' }}>
          <AlertTriangle className="w-3 h-3 text-amber-400 shrink-0 mt-0.5" />
          <p className="text-xs" style={{ color: 'var(--text-subtle)' }}>{insight.caveat}</p>
        </div>
      )}
      <p className="text-xs mt-1 font-mono" style={{ color: 'var(--text-faint)' }}>Basis: {insight.data_basis}</p>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Analytics() {
  const [filters, setFilters] = useState<FilterState>({
    source:    'live',
    verdict:   '',
    date_from: '',
    date_to:   '',
    limit:     500,
  })

  const apiFilters = {
    source:    filters.source,
    verdict:   filters.verdict || undefined,
    date_from: filters.date_from || undefined,
    date_to:   filters.date_to || undefined,
    limit:     filters.limit,
  }

  const { data: patterns, isLoading, refetch } = usePatterns(apiFilters)
  const { data: temporal } = useTemporalTrend(apiFilters)
  const { data: sentiment } = useSentiment(apiFilters)
  const { data: wordFreq } = useWordFreq(apiFilters, 80)
  const { data: topics }   = useTopics(apiFilters, 5)

  if (isLoading && !patterns) {
    return (
      <div className="page-content">
        <div className="mb-7"><div className="h-7 w-56 skeleton mb-2" /><div className="h-4 w-72 skeleton" /></div>
        <div className="grid lg:grid-cols-2 gap-6">
          {Array.from({ length: 4 }).map((_, i) => <SkeletonChart key={i} className="h-64" />)}
        </div>
      </div>
    )
  }

  const data = patterns as PatternsResponse | undefined

  return (
    <div className="page-content">
      {/* Header */}
      <div className="flex items-start justify-between mb-7">
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center shrink-0">
            <BarChart3 className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="page-title mb-1">Misinformation Pattern Analysis</h1>
            <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
              Real-time pattern detection from {data?.document_count ?? 0} documents · all metrics from actual data
            </p>
          </div>
        </div>
        <button onClick={() => refetch()} className="btn-ghost" title="Refresh">
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {/* Filters */}
      <FilterBar filters={filters} onFilter={setFilters} />

      {data?.error ? (
        <div className="card p-6 text-center" style={{ color: 'var(--text-subtle)' }}>
          <AlertCircle className="w-8 h-8 mx-auto mb-2 opacity-40" />
          <p className="text-sm">{data.error}</p>
        </div>
      ) : (data?.document_count ?? 0) === 0 ? (
        <div className="card p-10 text-center space-y-3">
          <Database className="w-10 h-10 mx-auto opacity-20" style={{ color: 'var(--text-faint)' }} />
          <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
            No records found matching the current filters.
          </p>
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
            Submit articles through the analysis pipeline, or switch the source to "Dataset (ISOT)".
          </p>
        </div>
      ) : (
        <>
          {/* ── Row 1: Verdict distribution + Sentiment ─────────────────────── */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Verdict distribution */}
            <Section title="Fake vs Real Distribution" icon={<PieChartIcon className="w-5 h-5" />}
              subtitle="Classification label distribution across analysed documents">
              {data?.verdict_distribution?.counts && Object.keys(data.verdict_distribution.counts).length > 0 ? (
                <div className="grid sm:grid-cols-2 gap-4 items-center">
                  <ResponsiveContainer width="100%" height={160}>
                    <PieChart>
                      <Pie data={Object.entries(data.verdict_distribution.counts).map(([k, v]) => ({
                        name: k,
                        value: v as number,
                        fill: k === 'FAKE' ? '#f87171' : k === 'REAL' ? '#34d399' : k === 'MIXED' ? '#fbbf24' : '#94a3b8',
                      }))} cx="50%" cy="50%" innerRadius={45} outerRadius={70} paddingAngle={3} dataKey="value">
                        {Object.entries(data.verdict_distribution.counts).map(([k], i) => (
                          <Cell key={i} fill={k === 'FAKE' ? '#f87171' : k === 'REAL' ? '#34d399' : k === 'MIXED' ? '#fbbf24' : '#94a3b8'} />
                        ))}
                      </Pie>
                      <Tooltip {...TT} />
                      <Legend iconType="circle" iconSize={8} formatter={(v) => <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{v}</span>} />
                    </PieChart>
                  </ResponsiveContainer>
                  <div className="space-y-2">
                    {Object.entries(data.verdict_distribution.counts).map(([k, v]) => (
                      <div key={k} className="flex justify-between text-sm">
                        <span style={{ color: 'var(--text-secondary)' }}>{k}</span>
                        <span className="font-mono" style={{ color: 'var(--text-primary)' }}>{v as number}</span>
                      </div>
                    ))}
                    <div className="flex justify-between text-sm pt-1" style={{ borderTop: '1px solid var(--border-subtle)' }}>
                      <span style={{ color: 'var(--text-subtle)' }}>Total</span>
                      <span className="font-mono font-bold" style={{ color: 'var(--brand)' }}>{data.verdict_distribution.total}</span>
                    </div>
                  </div>
                </div>
              ) : <EmptyState msg="No verdict data available" />}
            </Section>

            {/* Sentiment */}
            <Section title="Sentiment Distribution" icon={<Activity className="w-5 h-5" />}
              subtitle={sentiment?.all?.method ? `Method: ${sentiment.all.method}` : 'VADER sentiment analysis'}>
              {sentiment?.all?.distribution?.length ? (
                <>
                  <ResponsiveContainer width="100%" height={140}>
                    <BarChart data={sentiment.all.distribution} margin={{ top: 4, right: 10, bottom: 4, left: -20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="label" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} tickLine={false} axisLine={false} />
                      <YAxis tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} unit="%" />
                      <Tooltip {...TT} />
                      <Bar dataKey="pct" radius={[4, 4, 0, 0]}>
                        {sentiment.all.distribution.map((d: { label: string; count: number; pct: number }, i: number) => (
                          <Cell key={i} fill={d.label === 'positive' ? '#34d399' : d.label === 'negative' ? '#f87171' : '#94a3b8'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="flex justify-between text-xs mt-2">
                    <span style={{ color: 'var(--text-subtle)' }}>Avg compound score:</span>
                    <span className="font-mono" style={{ color: sentiment.all.avg_compound > 0 ? '#34d399' : sentiment.all.avg_compound < 0 ? '#f87171' : 'var(--text-subtle)' }}>
                      {sentiment.all.avg_compound.toFixed(4)}
                    </span>
                  </div>
                </>
              ) : <EmptyState msg="No sentiment data" />}
            </Section>
          </div>

          {/* ── Row 2: Word cloud + Word freq bar ────────────────────────────── */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Word cloud */}
            <Section title="Word Frequency Cloud" icon={<Type className="w-5 h-5" />}
              subtitle="Most frequent terms — size = frequency, colour = label">
              {wordFreq?.all?.length ? (
                <WordCloud words={wordFreq.all} label="all" maxWords={40} height={250} />
              ) : <EmptyState msg="No word frequency data" />}
            </Section>

            {/* Top fake vs real terms */}
            <Section title="Most Common Terms by Label" icon={<Hash className="w-5 h-5" />}
              subtitle="Top 15 words in FAKE vs REAL labelled content">
              {wordFreq && (wordFreq.FAKE?.length || wordFreq.REAL?.length) ? (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs font-semibold text-red-400 mb-2">FAKE</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={wordFreq.FAKE.slice(0, 10)} layout="vertical" margin={{ left: 60, right: 5, top: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" tick={{ fontSize: 9, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
                        <YAxis type="category" dataKey="word" tick={{ fontSize: 10, fill: 'var(--text-secondary)', fontFamily: 'monospace' }} tickLine={false} axisLine={false} width={60} />
                        <Tooltip {...TT} formatter={(v: number) => [v, 'count']} />
                        <Bar dataKey="count" fill="#f87171" radius={[0, 3, 3, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div>
                    <div className="text-xs font-semibold text-emerald-400 mb-2">REAL</div>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={wordFreq.REAL.slice(0, 10)} layout="vertical" margin={{ left: 60, right: 5, top: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                        <XAxis type="number" tick={{ fontSize: 9, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
                        <YAxis type="category" dataKey="word" tick={{ fontSize: 10, fill: 'var(--text-secondary)', fontFamily: 'monospace' }} tickLine={false} axisLine={false} width={60} />
                        <Tooltip {...TT} formatter={(v: number) => [v, 'count']} />
                        <Bar dataKey="count" fill="#34d399" radius={[0, 3, 3, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ) : <EmptyState msg="No label-grouped word data" />}
            </Section>
          </div>

          {/* ── Row 3: Temporal trend + Article length ──────────────────────── */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Temporal */}
            <Section title="Temporal Distribution" icon={<Clock className="w-5 h-5" />}
              subtitle="Article count by date — stacked by verdict">
              {temporal?.series?.length >= 2 ? (
                <ResponsiveContainer width="100%" height={180}>
                  <AreaChart data={temporal.series} margin={{ top: 4, right: 10, bottom: 4, left: -20 }}>
                    <defs>
                      <linearGradient id="gT" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#818cf8" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
                    <YAxis tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} allowDecimals={false} />
                    <Tooltip {...TT} />
                    <Area type="monotone" dataKey="total" stroke="#818cf8" fill="url(#gT)" strokeWidth={2} dot={false} name="Total" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : <EmptyState msg="Insufficient temporal data (need ≥ 2 date points)" />}
            </Section>

            {/* Article length */}
            <Section title="Article Length Distribution" icon={<FileText className="w-5 h-5" />}
              subtitle={data?.text_length?.observation}>
              {data?.text_length?.histogram?.length ? (
                <>
                  <ResponsiveContainer width="100%" height={140}>
                    <BarChart data={data.text_length.histogram} margin={{ top: 4, right: 10, bottom: 4, left: -20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="bucket" tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
                      <YAxis tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} allowDecimals={false} />
                      <Tooltip {...TT} />
                      <Bar dataKey="count" fill="var(--brand)" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                  {data.text_length.by_label && (
                    <div className="grid grid-cols-2 gap-3 mt-3">
                      {Object.entries(data.text_length.by_label).map(([label, stats]) => (
                        <div key={label} className="rounded-lg p-3" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                          <div className={cn('text-xs font-semibold mb-1', label === 'FAKE' ? 'text-red-400' : 'text-emerald-400')}>{label}</div>
                          <div className="text-xs space-y-0.5" style={{ color: 'var(--text-subtle)' }}>
                            <div className="flex justify-between"><span>Chars</span><span className="font-mono">{(stats as any).char?.mean ?? '—'}</span></div>
                            <div className="flex justify-between"><span>Words</span><span className="font-mono">{(stats as any).word?.mean ?? '—'}</span></div>
                            <div className="flex justify-between"><span>Count</span><span className="font-mono">{(stats as any).count ?? '—'}</span></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              ) : <EmptyState msg="No length data" />}
            </Section>
          </div>

          {/* ── Row 4: N-grams + Source distribution ────────────────────────── */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* Bigrams */}
            <Section title="Most Frequent Bigrams" icon={<Hash className="w-5 h-5" />}
              subtitle="Top 2-word phrases across all content">
              {data?.ngrams?.bigrams?.length ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={data.ngrams.bigrams.slice(0, 10)} layout="vertical" margin={{ left: 90, right: 10, top: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 9, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
                    <YAxis type="category" dataKey="ngram" tick={{ fontSize: 10, fill: 'var(--text-secondary)', fontFamily: 'monospace' }} tickLine={false} axisLine={false} width={90} />
                    <Tooltip {...TT} formatter={(v: number) => [v, 'count']} />
                    <Bar dataKey="count" fill="#a78bfa" radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <EmptyState msg="No n-gram data" />}
            </Section>

            {/* Source distribution */}
            <Section title="Source Distribution" icon={<Database className="w-5 h-5" />}
              subtitle="Evidence sources by publisher">
              {data?.source_distribution?.length ? (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={data.source_distribution.slice(0, 8)} layout="vertical" margin={{ left: 90, right: 10, top: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 9, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
                    <YAxis type="category" dataKey="source" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} tickLine={false} axisLine={false} width={90} />
                    <Tooltip {...TT} />
                    <Bar dataKey="count" fill="#60a5fa" radius={[0, 3, 3, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : <EmptyState msg="No evidence sources retrieved" />}
            </Section>
          </div>

          {/* ── Row 5: LDA topics + Repeated claims ─────────────────────────── */}
          <div className="grid lg:grid-cols-2 gap-6 mb-6">
            {/* LDA topics */}
            <Section title="Topic Distribution (LDA)" icon={<Layers className="w-5 h-5" />}
              subtitle={topics?.note ?? 'Latent topic discovery'}>
              {topics?.topics?.length ? (
                <div className="space-y-3">
                  {topics.topics.map((topic: import('@/types/api').LDATopic) => (
                    <div key={topic.topic_id} className="rounded-lg p-3" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-semibold" style={{ color: 'var(--brand)' }}>
                          Topic {topic.topic_id + 1}: {topic.label}
                        </span>
                        <span className="text-xs font-mono" style={{ color: 'var(--text-faint)' }}>
                          {topic.doc_count} docs
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {topic.words.slice(0, 8).map((w: import('@/types/api').LDAWord, i: number) => (
                          <span key={i} className="text-xs px-2 py-0.5 rounded font-mono"
                            style={{ backgroundColor: `${MODEL_COLORS[topic.topic_id % MODEL_COLORS.length]}20`, color: MODEL_COLORS[topic.topic_id % MODEL_COLORS.length] }}>
                            {w.word}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
                  {topics.perplexity != null && (
                    <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Perplexity: {topics.perplexity} (lower = better fit)</p>
                  )}
                </div>
              ) : <EmptyState msg={topics?.note ?? 'Insufficient documents for LDA (need ≥ 20)'} />}
            </Section>

            {/* Repeated claims */}
            <Section title="Repeated / Similar Claims" icon={<AlertCircle className="w-5 h-5" />}
              subtitle="Near-duplicate articles detected by TF-IDF cosine similarity">
              {data?.repeated_claims?.length ? (
                <div className="space-y-3">
                  {data.repeated_claims.slice(0, 5).map((rc) => (
                    <div key={rc.group_id} className="rounded-lg p-3" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-xs font-bold px-2 py-0.5 rounded text-amber-400" style={{ backgroundColor: 'rgba(251,191,36,0.1)' }}>
                          {rc.count} articles
                        </span>
                        <span className="text-xs font-mono" style={{ color: 'var(--text-faint)' }}>
                          similarity: {(rc.similarity * 100).toFixed(0)}%
                        </span>
                      </div>
                      <p className="text-xs line-clamp-2" style={{ color: 'var(--text-secondary)' }}>{rc.representative}</p>
                      <p className="text-xs mt-1 font-mono" style={{ color: 'var(--text-faint)' }}>IDs: {rc.ids.join(', ')}</p>
                    </div>
                  ))}
                </div>
              ) : <EmptyState msg="No repeated claims detected" />}
            </Section>
          </div>

          {/* ── Row 6: Linguistic features ──────────────────────────────────── */}
          {data?.linguistic_features?.all && Object.keys(data.linguistic_features.all).length > 0 && (
            <Section title="Linguistic Features" icon={<Type className="w-5 h-5" />}
              subtitle="Stylometric features — computed from raw text, correlational only">
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                {[
                  ['Avg word length', `${(data.linguistic_features.all as any).avg_word_length ?? '—'} chars`],
                  ['Type-token ratio', (data.linguistic_features.all as any).type_token_ratio ?? '—'],
                  ['Avg sentence len', `${(data.linguistic_features.all as any).avg_sentence_length ?? '—'} words`],
                  ['Uppercase ratio', `${(((data.linguistic_features.all as any).uppercase_ratio ?? 0) * 100).toFixed(1)}%`],
                  ['Exclam. rate', `${(data.linguistic_features.all as any).exclamation_rate ?? 0}/1k chars`],
                ].map(([label, value]) => (
                  <div key={label} className="rounded-lg p-3 text-center" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                    <div className="text-lg font-bold" style={{ color: 'var(--brand)' }}>{value}</div>
                    <div className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>{label}</div>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* ── Research insights ────────────────────────────────────────────── */}
          <div className="mt-6">
            <div className="flex items-center gap-2 mb-4">
              <Lightbulb className="w-5 h-5" style={{ color: 'var(--brand)' }} />
              <h2 className="section-title">Research Insights</h2>
              <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
                — derived from actual computed statistics
              </span>
            </div>
            <div className="grid sm:grid-cols-2 gap-4">
              {data?.insights?.length ? (
                data.insights.map((insight, i) => <InsightCard key={i} insight={insight} />)
              ) : (
                <div className="card p-6 text-center" style={{ color: 'var(--text-faint)' }}>
                  <p className="text-sm">Insufficient data for insights. Submit more articles to enable pattern detection.</p>
                </div>
              )}
            </div>
          </div>

          {/* ── Data integrity footer ────────────────────────────────────────── */}
          <div className="mt-6 card p-4 flex items-start gap-3">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-400" />
            <div className="text-xs leading-relaxed space-y-1" style={{ color: 'var(--text-subtle)' }}>
              <p>
                <strong style={{ color: 'var(--text-secondary)' }}>Research integrity:</strong>{' '}
                All statistics on this page are computed from actual data — nothing is fabricated or estimated.
                Insights explicitly distinguish dataset-level correlations from causal conclusions.
              </p>
              <p>
                <strong style={{ color: 'var(--text-secondary)' }}>Data source:</strong>{' '}
                {data?.source === 'live' ? 'Live analysis history (PostgreSQL)' : data?.source === 'dataset' ? 'ISOT training data' : 'Combined live + dataset'}
                · {data?.document_count ?? 0} documents · computed at {data?.computed_at ? new Date(data.computed_at).toLocaleTimeString() : '—'}
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
