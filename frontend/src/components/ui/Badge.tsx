import { cn } from '@/lib/utils'

interface BadgeProps {
  children: React.ReactNode
  variant?: 'default' | 'success' | 'danger' | 'warning' | 'info' | 'muted'
  size?: 'sm' | 'md'
  className?: string
}

const variants = {
  default: 'bg-brand-500/20 text-brand-300 border-brand-500/30',
  success: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  danger:  'bg-red-500/15 text-red-400 border-red-500/30',
  warning: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  info:    'bg-blue-500/15 text-blue-400 border-blue-500/30',
  muted:   'bg-slate-700/50 text-slate-400 border-slate-600/30',
}

export function Badge({ children, variant = 'default', size = 'sm', className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full border font-medium',
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm',
        variants[variant],
        className,
      )}
    >
      {children}
    </span>
  )
}
