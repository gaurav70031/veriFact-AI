import { useStats } from '@/hooks/useApi'
import { VerdictDonut } from '@/components/charts/VerdictDonut'
import { Spinner } from '@/components/ui/Spinner'
import { ErrorState } from '@/components/ui/ErrorState'
import { ProgressBar } from '@/components/ui/ProgressBar'

function StatRow({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-slate-800/60 last:border-0">
      <span className="text-sm text-slate-400">{label}</span>
      <div className="text-right">
        <span className="text-sm font-semibold text-slate-200">{value}</span>
        {sub && <div className="text-xs text-slate-600">{sub}</div>}
      </div>
    </div>
  )
}

export default function Analytics() {
  const { data, isLoading, error, refetch } = useStats()

  if (isLoading) return (
    <div className="flex justify-center py-32"><Spinner size="lg" /></div>
  )

  if (error) return (
    <div className="max-w-xl mx-auto py-24 px-4">
      <ErrorState message={(error as Error).message} onRetry={() => refetch()} />
    </div>
  )

  if (!data) return null

  const total = Math.max(data.completed, 1)

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fade-in">
      <h1 className="text-2xl font-bold text-slate-100 mb-2">Analytics</h1>
      <p className="text-slate-500 text-sm mb-8">
        Real-time statistics computed from the PostgreSQL database — no fabricated numbers.
      </p>

      {/* Overview grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {[
          { label: 'Total Analyses', value: data.total_analyses.toLocaleString(), color: 'text-brand-400' },
          { label: 'Completed',      value: data.completed.toLocaleString(),       color: 'text-emerald-400' },
          { label: 'Failed',         value: data.failed.toLocaleString(),           color: 'text-red-400' },
          { label: 'Pending',        value: data.pending.toLocaleString(),          color: 'text-amber-400' },
        ].map((s) => (
          <div key={s.label} className="glass-card p-5 text-center">
            <div className={`text-3xl font-bold tracking-tight ${s.color}`}>{s.value}</div>
            <div className="text-xs text-slate-500 mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Verdict distribution chart */}
        <div className="glass-card p-6">
          <h2 className="section-title mb-1">Verdict Distribution</h2>
          <p className="text-xs text-slate-500 mb-4">Based on ML ensemble predictions</p>
          <VerdictDonut counts={data.verdict_counts} />
        </div>

        {/* Detailed stats */}
        <div className="glass-card p-6">
          <h2 className="section-title mb-4">Detailed Statistics</h2>
          <StatRow label="Fake Detected"      value={`${data.fake_percentage.toFixed(1)}%`} sub={`${data.verdict_counts.FAKE.toLocaleString()} analyses`} />
          <StatRow label="Real Detected"      value={`${data.real_percentage.toFixed(1)}%`} sub={`${data.verdict_counts.REAL.toLocaleString()} analyses`} />
          <StatRow label="Unverified"         value={`${data.verdict_counts.UNVERIFIED.toLocaleString()}`} />
          <StatRow label="Mixed Signals"      value={`${data.verdict_counts.MIXED.toLocaleString()}`} />
          <StatRow label="Avg ML Confidence"  value={`${(data.avg_confidence * 100).toFixed(1)}%`} />
          <StatRow label="Avg Processing Time" value={`${(data.avg_processing_ms / 1000).toFixed(2)}s`} />

          {/* Fake / Real bar */}
          <div className="mt-5 space-y-3">
            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>Fake</span>
                <span className="text-red-400">{data.fake_percentage.toFixed(1)}%</span>
              </div>
              <ProgressBar value={data.fake_percentage / 100} color="bg-red-500" size="md" />
            </div>
            <div>
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>Real</span>
                <span className="text-emerald-400">{data.real_percentage.toFixed(1)}%</span>
              </div>
              <ProgressBar value={data.real_percentage / 100} color="bg-emerald-500" size="md" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
