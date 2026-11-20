/**
 * Overview Dashboard
 *
 * All statistics sourced from live API:
 *   GET /api/v1/stats   → StatsResponse
 *   GET /api/v1/history → PaginatedHistory (recent activity)
 *   GET /api/v1/health  → HealthResponse
 *
 * No hardcoded data.
 */

import { Link } from 'react-router-dom'
import {
  Database, CheckCircle, XCircle, HelpCircle, TrendingUp,
  ArrowRight, ArrowUpRight, BarChart3, Activity, Zap,
  ScanText, Newspaper, FlaskConical, Clock, Shield,
  RefreshCw, ExternalLink, ChevronRight, AlertTriangle,
} from 'lucide-react'
import {
  AreaChart, Area, PieChart, Pie, Cell, ResponsiveContainer,
  Tooltip, XAxis, YAxis, CartesianGrid, Legend,
} from 'recharts'
import { useStats, useHistory, useHealth } from '@/hooks/useApi'
import { Spinner } from '@/components/ui/Spinner'
import { SkeletonCard } from '@/components/ui/Skeleton'
import { cn, formatMs, formatRelativeTime, getVerdictConfig, truncate } from '@/lib/utils'
import type { StatsResponse, PaginatedHistory } from '@/types/api'

// ── KPI card ──────────────────────────────────────────────────────────────────

interface KpiProps {
  icon: React.ReactNode
  label: string
  value: string | number
  sub?: string
  trend?: { dir: 'up' | 'down' | 'neutral'; label: string }
  accentColor?: string
  loading?: boolean
}

function KpiCard({ icon, label, value, sub, trend, accentColor = 'var(--brand)', loading }: KpiProps) {
  if (loading) return <SkeletonCard />
  return (
    <div className="stat-card">
      <div className="flex items-start justify-between mb-4">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center"
          style={{ backgroundColor: `${accentColor}18`, border: `1px solid ${accentColor}25` }}
        >
          {icon}
        </div>
        {trend && (
          <span className={cn(
            'inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full',
            trend.dir === 'up'      ? 'text-emerald-400 bg-emerald-400/10' :
            trend.dir === 'down'    ? 'text-red-400 bg-red-400/10' :
                                      'text-slate-500 bg-slate-500/10',
          )}>
            <TrendingUp className="w-3 h-3" />
            {trend.label}
          </span>
        )}
      </div>
      <div className="metric-value" style={{ color: accentColor }}>
        {value}
      </div>
      <div className="text-sm mt-1" style={{ color: 'var(--text-secondary)' }}>{label}</div>
      {sub && <div className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>{sub}</div>}
    </div>
  )
}

// ── Verdict donut ─────────────────────────────────────────────────────────────

function VerdictDonutChart({ stats }: { stats: StatsResponse }) {
  const data = [
    { name: 'Fake',       value: stats.verdict_counts.FAKE,       color: '#f87171' },
    { name: 'Real',       value: stats.verdict_counts.REAL,       color: '#34d399' },
    { name: 'Unverified', value: stats.verdict_counts.UNVERIFIED, color: '#fbbf24' },
    { name: 'Mixed',      value: stats.verdict_counts.MIXED,      color: '#fb923c' },
  ].filter((d) => d.value > 0)

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48" style={{ color: 'var(--text-faint)' }}>
        No data yet
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={58}
          outerRadius={80}
          paddingAngle={3}
          dataKey="value"
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={entry.color} stroke="transparent" />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--bg-surface-2)',
            border: '1px solid var(--border)',
            borderRadius: 8,
          }}
          labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
          itemStyle={{ color: 'var(--text-secondary)' }}
          formatter={(value: number) => [value.toLocaleString(), 'articles']}
        />
        <Legend
          iconType="circle"
          iconSize={8}
          formatter={(value) => <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

// ── Activity trend chart ──────────────────────────────────────────────────────

function ActivityChart({ history }: { history: PaginatedHistory }) {
  // Build a rough day-bucket from recent analyses
  const dayBuckets: Record<string, { total: number; fake: number; real: number }> = {}
  const now = Date.now()

  for (const item of history.items) {
    const daysAgo = Math.floor((now - new Date(item.created_at).getTime()) / 86_400_000)
    if (daysAgo > 13) continue
    const label = daysAgo === 0 ? 'Today' : `${daysAgo}d ago`
    if (!dayBuckets[label]) dayBuckets[label] = { total: 0, fake: 0, real: 0 }
    dayBuckets[label].total++
    if (item.final_verdict === 'FAKE') dayBuckets[label].fake++
    if (item.final_verdict === 'REAL') dayBuckets[label].real++
  }

  const chartData = Object.entries(dayBuckets)
    .reverse()
    .map(([date, v]) => ({ date, ...v }))

  if (chartData.length < 2) {
    return (
      <div className="flex flex-col items-center justify-center h-40 gap-2" style={{ color: 'var(--text-faint)' }}>
        <Activity className="w-8 h-8 opacity-30" />
        <span className="text-sm">Not enough data for trend chart</span>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={160}>
      <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
        <defs>
          <linearGradient id="gTotal" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#818cf8" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#818cf8" stopOpacity={0} />
          </linearGradient>
          <linearGradient id="gFake" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor="#f87171" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} />
        <YAxis tick={{ fontSize: 10, fill: 'var(--text-faint)' }} tickLine={false} axisLine={false} allowDecimals={false} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)', borderRadius: 8 }}
          labelStyle={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}
          itemStyle={{ color: 'var(--text-secondary)' }}
        />
        <Area type="monotone" dataKey="total" stroke="#818cf8" fill="url(#gTotal)" strokeWidth={2} dot={false} name="Total" />
        <Area type="monotone" dataKey="fake"  stroke="#f87171" fill="url(#gFake)"  strokeWidth={1.5} dot={false} name="Fake" />
      </AreaChart>
    </ResponsiveContainer>
  )
}

// ── Recent activity table ─────────────────────────────────────────────────────

function RecentActivity({ history }: { history: PaginatedHistory }) {
  const items = history.items.slice(0, 8)

  if (!items.length) {
    return (
      <div className="flex flex-col items-center justify-center py-12 gap-3" style={{ color: 'var(--text-faint)' }}>
        <Database className="w-8 h-8 opacity-30" />
        <p className="text-sm">No analyses yet</p>
        <Link to="/analyze/text" className="btn-primary text-xs">Start Analyzing</Link>
      </div>
    )
  }

  return (
    <div className="divide-y" style={{ borderColor: 'var(--border-subtle)' }}>
      {items.map((item) => {
        const vc = getVerdictConfig(item.final_verdict ?? undefined)
        const content = item.article_title || truncate(item.original_input, 70)
        return (
          <Link
            key={item.id}
            to={`/results/${item.id}`}
            className="flex items-center gap-3 px-5 py-3.5 group transition-colors"
            style={{ color: 'inherit' }}
            onMouseOver={(e) => { e.currentTarget.style.backgroundColor = 'var(--bg-surface-2)' }}
            onMouseOut={(e) => { e.currentTarget.style.backgroundColor = '' }}
          >
            <div className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 text-xs font-mono font-bold"
              style={{ backgroundColor: 'var(--bg-surface-2)', color: 'var(--text-subtle)' }}>
              {item.id}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm truncate" style={{ color: 'var(--text-secondary)' }}>{content}</p>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-xs capitalize" style={{ color: 'var(--text-faint)' }}>
                  {item.input_type}
                </span>
                <span className="text-xs" style={{ color: 'var(--text-faint)' }}>·</span>
                <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
                  {formatRelativeTime(item.created_at)}
                </span>
              </div>
            </div>
            <div className="text-right shrink-0">
              {item.status === 'completed' && item.final_verdict ? (
                <span className={cn('text-xs font-semibold px-2.5 py-1 rounded-full', vc.color, vc.bg)}>
                  {vc.label}
                </span>
              ) : (
                <span className="text-xs capitalize" style={{ color: 'var(--text-subtle)' }}>{item.status}</span>
              )}
            </div>
            <ChevronRight className="w-4 h-4 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: 'var(--brand)' }} />
          </Link>
        )
      })}
    </div>
  )
}

// ── Quick action cards ────────────────────────────────────────────────────────

const QUICK_ACTIONS = [
  { to: '/analyze/text', icon: ScanText,     label: 'Analyze Text',      desc: 'Paste article or claim', color: '#818cf8' },
  { to: '/analyze/url',  icon: ExternalLink,  label: 'Verify URL',        desc: 'Check a news article',   color: '#34d399' },
  { to: '/live-news',    icon: Newspaper,     label: 'Browse Live News',  desc: 'Search current events',  color: '#60a5fa' },
  { to: '/lab',          icon: FlaskConical,  label: 'Model Lab',         desc: 'View model metrics',     color: '#a78bfa' },
]

// ── System health widget ──────────────────────────────────────────────────────

function SystemHealth() {
  const { data: health } = useHealth()
  if (!health) return null

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>System Health</h3>
        <span className={cn(
          'text-xs font-medium px-2.5 py-0.5 rounded-full',
          health.status === 'ok' ? 'text-emerald-400 bg-emerald-400/10' : 'text-amber-400 bg-amber-400/10',
        )}>
          {health.status === 'ok' ? 'All systems operational' : 'Degraded'}
        </span>
      </div>
      <div className="space-y-2.5">
        {Object.entries(health.components).map(([name, comp]) => (
          <div key={name} className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={cn(
                'w-1.5 h-1.5 rounded-full',
                comp.status === 'ok'       ? 'bg-emerald-400' :
                comp.status === 'degraded' ? 'bg-amber-400'   :
                                              'bg-red-400',
              )} />
              <span className="text-sm capitalize" style={{ color: 'var(--text-secondary)' }}>
                {name.replace(/_/g, ' ')}
              </span>
            </div>
            <span className="text-xs font-mono" style={{ color: 'var(--text-subtle)' }}>
              {comp.latency_ms ? `${comp.latency_ms.toFixed(0)}ms` : comp.status}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-3 pt-3 text-xs" style={{ borderTop: '1px solid var(--border)', color: 'var(--text-faint)' }}>
        v{health.version} · {health.environment}
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Home() {
  const { data: stats, isLoading: sL, refetch: sR } = useStats()
  const { data: history, isLoading: hL }             = useHistory({ page: 1, page_size: 20 })
  const { data: health }                             = useHealth()
  const isLoading                                    = sL || hL

  const fakeCount       = stats?.verdict_counts.FAKE       ?? 0
  const realCount       = stats?.verdict_counts.REAL       ?? 0
  const unverifiedCount = stats?.verdict_counts.UNVERIFIED ?? 0
  const totalCompleted  = stats?.completed                 ?? 0

  const fakeRate  = totalCompleted > 0 ? (fakeCount / totalCompleted * 100).toFixed(1) : '—'
  const realRate  = totalCompleted > 0 ? (realCount / totalCompleted * 100).toFixed(1) : '—'

  // Check if health reports toy-data warning on ML models
  const mlMessage      = (health as any)?.components?.ml_models?.message ?? ''
  const isToyDataModel = mlMessage.includes('toy data')

  return (
    <div className="page-content">
      {/* ── Toy-data model warning (shown prominently when models not properly trained) ── */}
      {isToyDataModel && (
        <div className="mb-6 flex items-start gap-3 p-4 rounded-xl"
          style={{ backgroundColor: 'rgba(248,113,113,0.07)', border: '1px solid rgba(248,113,113,0.3)' }}>
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-red-400 mb-1">
              Setup required — ML models not trained on real data
            </p>
            <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              Models are trained on a 10-word synthetic dataset. Every prediction returns a 50/50 result.
              Evidence retrieval works correctly, but ML classification is unreliable until ISOT training is run.
            </p>
            <details className="mt-2">
              <summary className="text-xs cursor-pointer" style={{ color: 'var(--brand)' }}>
                How to fix this →
              </summary>
              <pre className="text-xs font-mono mt-2 p-2 rounded leading-relaxed" style={{ backgroundColor: 'rgba(0,0,0,0.3)', color: '#94d82d' }}>
{`# 1. Download ISOT from https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# 2. Place Fake.csv + True.csv in ml/datasets/raw/
python scripts/prepare_dataset.py --files ml/datasets/raw/Fake.csv ml/datasets/raw/True.csv
python scripts/train_baseline.py
python scripts/register_models.py --overwrite`}
              </pre>
            </details>
          </div>
        </div>
      )}

      {/* ── Page header ── */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
            Overview Dashboard
          </h1>
          <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
            Live metrics from PostgreSQL · no hardcoded values
          </p>
        </div>
        <button
          onClick={() => sR()}
          className="btn-ghost"
          disabled={isLoading}
          title="Refresh"
        >
          <RefreshCw className={cn('w-4 h-4', isLoading && 'animate-spin')} />
        </button>
      </div>

      {/* ── KPI row ── */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
        <KpiCard
          loading={sL}
          icon={<Database className="w-5 h-5" style={{ color: '#818cf8' }} />}
          label="Total Analysed"
          value={stats?.total_analyses.toLocaleString() ?? '—'}
          sub={`${stats?.completed.toLocaleString() ?? 0} completed`}
          accentColor="#818cf8"
        />
        <KpiCard
          loading={sL}
          icon={<XCircle className="w-5 h-5" style={{ color: '#f87171' }} />}
          label="Predicted Fake"
          value={fakeCount.toLocaleString()}
          sub={`${fakeRate}% of completed`}
          accentColor="#f87171"
        />
        <KpiCard
          loading={sL}
          icon={<CheckCircle className="w-5 h-5" style={{ color: '#34d399' }} />}
          label="Predicted Real"
          value={realCount.toLocaleString()}
          sub={`${realRate}% of completed`}
          accentColor="#34d399"
        />
        <KpiCard
          loading={sL}
          icon={<HelpCircle className="w-5 h-5" style={{ color: '#fbbf24' }} />}
          label="Unverified / Mixed"
          value={(unverifiedCount + (stats?.verdict_counts.MIXED ?? 0)).toLocaleString()}
          sub="Insufficient evidence"
          accentColor="#fbbf24"
        />
      </div>

      {/* ── Secondary KPI row ── */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 mb-8">
        <KpiCard
          loading={sL}
          icon={<Zap className="w-5 h-5" style={{ color: '#a78bfa' }} />}
          label="Avg ML Confidence"
          value={stats ? `${(stats.avg_confidence * 100).toFixed(1)}%` : '—'}
          sub="Ensemble average"
          accentColor="#a78bfa"
        />
        <KpiCard
          loading={sL}
          icon={<Clock className="w-5 h-5" style={{ color: '#60a5fa' }} />}
          label="Avg Processing"
          value={stats?.avg_processing_ms ? formatMs(stats.avg_processing_ms) : '—'}
          sub="All-time average"
          accentColor="#60a5fa"
        />
        <KpiCard
          loading={sL}
          icon={<BarChart3 className="w-5 h-5" style={{ color: '#fb923c' }} />}
          label="Success Rate"
          value={stats && stats.total_analyses > 0
            ? `${(stats.completed / stats.total_analyses * 100).toFixed(0)}%`
            : '—'}
          sub={`${stats?.failed ?? 0} failed`}
          accentColor="#fb923c"
        />
        <KpiCard
          loading={sL}
          icon={<Activity className="w-5 h-5" style={{ color: '#34d399' }} />}
          label="Pending"
          value={stats?.pending.toLocaleString() ?? '—'}
          sub="In queue"
          accentColor="#34d399"
        />
      </div>

      {/* ── Charts row ── */}
      <div className="grid lg:grid-cols-2 gap-6 mb-8">
        {/* Verdict distribution */}
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="section-title">Verdict Distribution</h2>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>
                All completed analyses
              </p>
            </div>
            <Link to="/analytics" className="text-xs flex items-center gap-1" style={{ color: 'var(--brand)' }}>
              Full report <ArrowUpRight className="w-3 h-3" />
            </Link>
          </div>
          {sL ? (
            <div className="flex items-center justify-center h-48"><Spinner /></div>
          ) : (
            <VerdictDonutChart stats={stats ?? {
              total_analyses: 0, completed: 0, failed: 0, pending: 0,
              verdict_counts: { FAKE: 0, REAL: 0, UNVERIFIED: 0, MIXED: 0 },
              fake_percentage: 0, real_percentage: 0, avg_confidence: 0, avg_processing_ms: 0,
            }} />
          )}
        </div>

        {/* Recent activity trend */}
        <div className="card p-5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="section-title">Recent Activity</h2>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>
                Analyses over past 14 days
              </p>
            </div>
          </div>
          {hL ? (
            <div className="flex items-center justify-center h-40"><Spinner /></div>
          ) : history ? (
            <ActivityChart history={history} />
          ) : null}
        </div>
      </div>

      {/* ── Recent analyses + system health ── */}
      <div className="grid lg:grid-cols-3 gap-6 mb-8">
        {/* Recent analyses table */}
        <div className="lg:col-span-2 card overflow-hidden">
          <div className="px-5 py-4 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)' }}>
            <div>
              <h2 className="section-title">Recent Analyses</h2>
              {history && (
                <p className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>
                  {history.total.toLocaleString()} total in database
                </p>
              )}
            </div>
            <Link to="/history" className="text-xs flex items-center gap-1" style={{ color: 'var(--brand)' }}>
              View all <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
          {hL ? (
            <div className="flex justify-center py-12"><Spinner /></div>
          ) : history ? (
            <RecentActivity history={history} />
          ) : null}
        </div>

        {/* Right column */}
        <div className="space-y-6">
          <SystemHealth />

          {/* Model performance summary */}
          <div className="card p-5">
            <h3 className="text-sm font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              Model Performance
            </h3>
            <div className="space-y-2.5 text-sm mb-4">
              {[
                { name: 'DistilBERT',    acc: 'View in lab', color: '#a78bfa' },
                { name: 'TF-IDF + LR',  acc: 'View in lab', color: '#818cf8' },
                { name: 'Linear SVM',   acc: 'View in lab', color: '#60a5fa' },
                { name: 'Naive Bayes',  acc: 'View in lab', color: '#34d399' },
              ].map((m) => (
                <div key={m.name} className="flex items-center justify-between">
                  <span style={{ color: 'var(--text-secondary)' }}>{m.name}</span>
                  <span className="text-xs" style={{ color: m.color }}>{m.acc}</span>
                </div>
              ))}
            </div>
            <Link to="/lab" className="btn-secondary w-full text-xs justify-center">
              <FlaskConical className="w-3.5 h-3.5" />
              Open Model Laboratory
            </Link>
          </div>
        </div>
      </div>

      {/* ── Quick actions ── */}
      <div>
        <h2 className="section-title mb-4">Quick Actions</h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {QUICK_ACTIONS.map((action) => (
            <Link
              key={action.to}
              to={action.to}
              className="card card-hover p-5 group flex flex-col gap-3"
              style={{ color: 'inherit', textDecoration: 'none' }}
            >
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 group-hover:scale-110"
                style={{ backgroundColor: `${action.color}15`, border: `1px solid ${action.color}25` }}
              >
                <action.icon className="w-5 h-5" style={{ color: action.color }} />
              </div>
              <div>
                <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
                  {action.label}
                </div>
                <div className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>
                  {action.desc}
                </div>
              </div>
              <div className="mt-auto flex items-center gap-1 text-xs" style={{ color: action.color }}>
                Get started <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
