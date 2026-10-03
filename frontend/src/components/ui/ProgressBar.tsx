import { cn } from '@/lib/utils'

interface ProgressBarProps {
  value: number  // 0–1
  color?: string
  size?: 'sm' | 'md'
  className?: string
  showLabel?: boolean
}

export function ProgressBar({ value, color = 'bg-brand-500', size = 'sm', className, showLabel }: ProgressBarProps) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100)
  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div className={cn('flex-1 rounded-full bg-slate-800 overflow-hidden', size === 'sm' ? 'h-1.5' : 'h-2.5')}>
        <div
          className={cn('h-full rounded-full transition-all duration-500', color)}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showLabel && (
        <span className="text-xs text-slate-400 w-10 text-right font-mono">{pct}%</span>
      )}
    </div>
  )
}
