import { useState } from 'react'
import { cn } from '@/lib/utils'

interface TooltipProps {
  content: string
  children: React.ReactNode
  className?: string
}

export function Tooltip({ content, children, className }: TooltipProps) {
  const [visible, setVisible] = useState(false)
  return (
    <span
      className={cn('relative inline-flex', className)}
      onMouseEnter={() => setVisible(true)}
      onMouseLeave={() => setVisible(false)}
    >
      {children}
      {visible && (
        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 z-50
          whitespace-nowrap px-2.5 py-1.5 rounded-lg text-xs bg-slate-800 text-slate-200
          border border-slate-700 shadow-xl pointer-events-none">
          {content}
        </span>
      )}
    </span>
  )
}
