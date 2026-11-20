import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip, Legend,
} from 'recharts'
import type { ModelPerformance } from '@/types/api'

interface ModelRadarProps {
  models: ModelPerformance[]
}

const COLORS = ['#6172f3', '#34d399', '#fbbf24', '#f87171', '#a78bfa']
const METRICS = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'] as const

export function ModelRadar({ models }: ModelRadarProps) {
  const data = METRICS.map((m) => {
    const row: Record<string, string | number> = { metric: m.replace('_', ' ').replace('f1 score', 'F1') }
    models.forEach((model) => {
      row[model.model_name] = ((model[m] ?? 0) * 100)
    })
    return row
  })

  return (
    <ResponsiveContainer width="100%" height={300}>
      <RadarChart data={data}>
        <PolarGrid stroke="var(--border)" />
        <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
        {models.map((m, i) => (
          <Radar
            key={m.model_name}
            name={m.model_name.replace('_', ' ')}
            dataKey={m.model_name}
            stroke={COLORS[i % COLORS.length]}
            fill={COLORS[i % COLORS.length]}
            fillOpacity={0.12}
            strokeWidth={2}
          />
        ))}
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '12px' }}
          labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
          itemStyle={{ color: 'var(--text-secondary)' }}
          formatter={(val: number) => `${val.toFixed(1)}%`}
        />
        <Legend
          iconType="circle"
          iconSize={8}
          formatter={(v: string) => <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>{v.replace(/_/g, ' ')}</span>}
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}
