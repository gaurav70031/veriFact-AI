import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import type { VerdictCounts } from '@/types/api'

const COLORS = {
  FAKE:       '#f87171',
  REAL:       '#34d399',
  UNVERIFIED: '#fbbf24',
  MIXED:      '#fb923c',
}

interface VerdictDonutProps {
  counts: VerdictCounts
}

export function VerdictDonut({ counts }: VerdictDonutProps) {
  const data = Object.entries(counts)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }))

  if (data.length === 0) return (
    <div className="flex items-center justify-center h-48 text-sm text-slate-500">
      No data yet
    </div>
  )

  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={55}
          outerRadius={80}
          paddingAngle={3}
          dataKey="value"
        >
          {data.map((entry) => (
            <Cell
              key={entry.name}
              fill={COLORS[entry.name as keyof typeof COLORS] ?? '#64748b'}
              stroke="transparent"
            />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '12px' }}
          labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
          itemStyle={{ color: 'var(--text-secondary)' }}
        />
        <Legend
          iconType="circle"
          iconSize={8}
          formatter={(v: string) => <span style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{v}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}
