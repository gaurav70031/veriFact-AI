import { cn } from '@/lib/utils'

interface SkeletonProps {
  className?: string
  width?: string | number
  height?: string | number
}

export function Skeleton({ className, width, height }: SkeletonProps) {
  return (
    <div
      className={cn('skeleton', className)}
      style={{ width, height: height ?? '1rem' }}
    />
  )
}

export function SkeletonCard({ className }: { className?: string }) {
  return (
    <div className={cn('card p-5 space-y-3', className)}>
      <Skeleton className="w-24 h-3" />
      <Skeleton className="w-16 h-7" />
      <Skeleton className="w-32 h-3" />
    </div>
  )
}

export function SkeletonTable({ rows = 5 }: { rows?: number }) {
  return (
    <div className="card overflow-hidden">
      <div className="p-4" style={{ borderBottom: '1px solid var(--border)' }}>
        <Skeleton className="w-40 h-4" />
      </div>
      <div className="divide-y" style={{ borderColor: 'var(--border-subtle)' }}>
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="px-4 py-3.5 flex items-center gap-4">
            <Skeleton className="w-8 h-3" />
            <Skeleton className="w-20 h-5 rounded-full" />
            <Skeleton className="flex-1 h-3" />
            <Skeleton className="w-16 h-3" />
            <Skeleton className="w-12 h-3" />
          </div>
        ))}
      </div>
    </div>
  )
}

export function SkeletonChart({ className }: { className?: string }) {
  return (
    <div className={cn('card p-5', className)}>
      <Skeleton className="w-32 h-4 mb-4" />
      <div className="flex items-end gap-2 h-32">
        {[60, 80, 45, 90, 70, 55, 85].map((h, i) => (
          <div key={i} className="flex-1 skeleton rounded-t-sm" style={{ height: `${h}%` }} />
        ))}
      </div>
    </div>
  )
}
