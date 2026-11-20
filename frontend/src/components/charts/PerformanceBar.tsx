import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell,
} from 'recharts'
import type { ModelPerformance } from '@/types/api'

interface PerformanceBarProps {
  models: ModelPerformance[]
  metric?: 'accuracy' | 'f1_score' | 'roc_auc'
}

const metricLabels = { accuracy: 'Accuracy', f1_score: 'F1 Score', roc_auc: 'ROC-AUC' }
const COLORS = ['#6172f3', '#34d399', '#fbbf24', '#f87171', '#a78bfa']

export function PerformanceBar({ models, metric = 'f1_score' }: PerformanceBarProps) {
  const data = models.map((m) => ({
    name: m.model_name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
    value: ((m[metric] ?? 0) * 100),
  }))

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="name"
          tick={{ fill: 'var(--text-subtle)', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[0, 100]}
          tick={{ fill: 'var(--text-faint)', fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => `${v}%`}
        />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)', borderRadius: '8px', fontSize: '12px' }}
          labelStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
          itemStyle={{ color: 'var(--text-secondary)' }}
          formatter={(val: number) => [`${val.toFixed(2)}%`, metricLabels[metric]]}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
