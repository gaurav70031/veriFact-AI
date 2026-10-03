import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'
import { formatDistanceToNow, format } from 'date-fns'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatRelativeTime(dateStr: string) {
  try {
    return formatDistanceToNow(new Date(dateStr), { addSuffix: true })
  } catch {
    return dateStr
  }
}

export function formatDate(dateStr: string) {
  try {
    return format(new Date(dateStr), 'MMM d, yyyy')
  } catch {
    return dateStr
  }
}

export function formatDateTime(dateStr: string) {
  try {
    return format(new Date(dateStr), 'MMM d, yyyy HH:mm')
  } catch {
    return dateStr
  }
}

export function formatPct(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

export function formatMs(ms: number) {
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export function truncate(str: string, len = 120) {
  if (str.length <= len) return str
  return str.slice(0, len) + '…'
}

export type VerdictType = 'FAKE' | 'REAL' | 'UNVERIFIED' | 'MIXED' |
  'LIKELY_CREDIBLE' | 'LIKELY_MISLEADING' | 'CONTRADICTED' | 'INSUFFICIENT_EVIDENCE'

export const verdictConfig: Record<VerdictType, {
  label: string
  color: string
  bg: string
  border: string
  icon: string
}> = {
  FAKE: {
    label: 'Likely Fake',
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    icon: '✕',
  },
  REAL: {
    label: 'Likely Real',
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    icon: '✓',
  },
  UNVERIFIED: {
    label: 'Unverified',
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    icon: '?',
  },
  MIXED: {
    label: 'Mixed Signals',
    color: 'text-orange-400',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/30',
    icon: '~',
  },
  LIKELY_CREDIBLE: {
    label: 'Likely Credible',
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    icon: '✓',
  },
  LIKELY_MISLEADING: {
    label: 'Likely Misleading',
    color: 'text-orange-400',
    bg: 'bg-orange-500/10',
    border: 'border-orange-500/30',
    icon: '⚠',
  },
  CONTRADICTED: {
    label: 'Contradicted',
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    icon: '✕',
  },
  INSUFFICIENT_EVIDENCE: {
    label: 'Insufficient Evidence',
    color: 'text-slate-400',
    bg: 'bg-slate-500/10',
    border: 'border-slate-500/30',
    icon: '–',
  },
}

export function getVerdictConfig(verdict?: string) {
  if (!verdict) return verdictConfig.UNVERIFIED
  return verdictConfig[verdict as VerdictType] ?? verdictConfig.UNVERIFIED
}

export const evidenceRelConfig = {
  supporting:    { label: 'Supporting',    color: 'text-emerald-400', dot: 'bg-emerald-400' },
  contradicting: { label: 'Contradicting', color: 'text-red-400',     dot: 'bg-red-400' },
  inconclusive:  { label: 'Inconclusive',  color: 'text-amber-400',   dot: 'bg-amber-400' },
  not_relevant:  { label: 'Not Relevant',  color: 'text-slate-500',   dot: 'bg-slate-500' },
}
