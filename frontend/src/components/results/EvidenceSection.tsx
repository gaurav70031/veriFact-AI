/**
 * EvidenceSection
 *
 * Displays all evidence sources grouped by their relationship to the claim:
 *   • Supporting
 *   • Contradicting
 *   • Inconclusive
 *   • Not Relevant   (collapsed by default — low signal)
 *
 * When no evidence was found:
 *   Renders an explicit "Insufficient reliable evidence found." message.
 *   This state is NEVER conflated with "Fake" or treated as negative.
 *
 * Source links open in new tabs; snippets are shown verbatim (≤ 500 chars).
 */

import { useState } from 'react'
import {
  CheckCircle2, XCircle, MinusCircle, EyeOff,
  ExternalLink, Calendar, BarChart3,
  ChevronDown, ChevronRight, AlertTriangle, Info,
  Wifi, WifiOff,
} from 'lucide-react'
import { cn, formatDate, evidenceRelConfig } from '@/lib/utils'
import type { EvidenceSourceResult, EvidenceSummary } from '@/types/api'

// ── Source card ───────────────────────────────────────────────────────────────

function SourceCard({ src }: { src: EvidenceSourceResult }) {
  const rel = evidenceRelConfig[src.relationship_to_claim as keyof typeof evidenceRelConfig]
    ?? evidenceRelConfig.inconclusive

  const simPct = Math.round(src.comparison_score * 100)
  const relPct = Math.round(src.relevance_score  * 100)

  return (
    <div className="group rounded-xl border border-slate-800/50 bg-slate-900/30 p-4 hover:border-slate-700/70 transition-colors">
      {/* Title + external link */}
      <a
        href={src.url}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-start gap-2 mb-2"
      >
        <span className="text-sm font-medium text-slate-200 hover:text-brand-400 transition-colors leading-snug flex-1">
          {src.title}
        </span>
        <ExternalLink className="w-3.5 h-3.5 text-slate-700 group-hover:text-brand-500 transition-colors shrink-0 mt-0.5" />
      </a>

      {/* Source name + date */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mb-3">
        <span className="text-xs font-medium text-slate-500">{src.source_name}</span>
        {src.published_at && (
          <span className="text-xs text-slate-600 flex items-center gap-1">
            <Calendar className="w-3 h-3" />
            {formatDate(src.published_at)}
          </span>
        )}
        <span className={cn('text-xs font-medium', rel.color)}>
          {rel.label}
        </span>
      </div>

      {/* Snippet */}
      {src.snippet && (
        <blockquote className="text-xs text-slate-500 italic leading-relaxed mb-3 border-l-2 border-slate-700 pl-3 line-clamp-3">
          {src.snippet}
        </blockquote>
      )}

      {/* Score bars */}
      <div className="grid grid-cols-2 gap-2">
        {/* Relevance */}
        <div>
          <div className="flex items-center gap-1 mb-1">
            <BarChart3 className="w-3 h-3 text-slate-700" />
            <span className="text-xs text-slate-700">Relevance</span>
            <span className="text-xs font-mono text-slate-600 ml-auto">{relPct}%</span>
          </div>
          <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-brand-600/50 rounded-full" style={{ width: `${relPct}%` }} />
          </div>
        </div>

        {/* Similarity */}
        {src.comparison_score > 0 && (
          <div>
            <div className="flex items-center gap-1 mb-1">
              <span className="text-xs text-slate-700">Similarity</span>
              <span className="text-xs font-mono text-slate-600 ml-auto">{simPct}%</span>
            </div>
            <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={cn(
                  'h-full rounded-full',
                  src.relationship_to_claim === 'supporting'    ? 'bg-emerald-500/60' :
                  src.relationship_to_claim === 'contradicting' ? 'bg-red-500/60'     :
                  'bg-amber-500/60',
                )}
                style={{ width: `${simPct}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* URL in small text */}
      <div className="mt-2 text-xs text-slate-700 font-mono truncate">{src.url}</div>
    </div>
  )
}

// ── Relationship group ────────────────────────────────────────────────────────

const groupConfig = {
  supporting: {
    icon:    <CheckCircle2 className="w-4 h-4" />,
    color:   'text-emerald-400',
    border:  'border-emerald-500/20',
    bg:      'bg-emerald-500/5',
    label:   'Supporting',
    desc:    'These sources corroborate the claim.',
  },
  contradicting: {
    icon:    <XCircle className="w-4 h-4" />,
    color:   'text-red-400',
    border:  'border-red-500/20',
    bg:      'bg-red-500/5',
    label:   'Contradicting',
    desc:    'These sources dispute or contradict the claim.',
  },
  inconclusive: {
    icon:    <MinusCircle className="w-4 h-4" />,
    color:   'text-amber-400',
    border:  'border-amber-500/20',
    bg:      'bg-amber-500/5',
    label:   'Inconclusive',
    desc:    'Related to the topic but neither supports nor contradicts.',
  },
  not_relevant: {
    icon:    <EyeOff className="w-4 h-4" />,
    color:   'text-slate-500',
    border:  'border-slate-700/30',
    bg:      'bg-slate-800/20',
    label:   'Not Relevant',
    desc:    'Retrieved but had insufficient topic overlap.',
  },
} as const

type RelKey = keyof typeof groupConfig

function EvidenceGroup({
  rel,
  sources,
  defaultOpen,
}: {
  rel: RelKey
  sources: EvidenceSourceResult[]
  defaultOpen: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  const cfg = groupConfig[rel]
  if (!sources.length) return null

  return (
    <div className={cn('rounded-xl border overflow-hidden', cfg.border, cfg.bg)}>
      {/* Group header */}
      <button
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-800/20 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2.5">
          <span className={cfg.color}>{cfg.icon}</span>
          <span className={cn('text-sm font-semibold', cfg.color)}>{cfg.label}</span>
          <span className={cn('text-xs px-2 py-0.5 rounded-full font-mono font-medium', cfg.bg, cfg.color,)}>
            {sources.length}
          </span>
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-600">
          <span className="hidden sm:block">{cfg.desc}</span>
          {open
            ? <ChevronDown className="w-4 h-4 text-slate-500" />
            : <ChevronRight className="w-4 h-4 text-slate-500" />}
        </div>
      </button>

      {/* Source cards */}
      {open && (
        <div className="px-4 pb-4 pt-1 grid sm:grid-cols-2 gap-3">
          {sources.map((src, i) => (
            <SourceCard key={`${src.url}-${i}`} src={src} />
          ))}
        </div>
      )}
    </div>
  )
}

// ── Insufficient evidence state ───────────────────────────────────────────────

function InsufficientState({
  allFailed,
  providersUsed,
  providersFailed,
  limitations,
}: {
  allFailed: boolean
  providersUsed: string[]
  providersFailed: string[]
  limitations: string[]
}) {
  return (
    <div className="glass-card overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-800">
        <h2 className="section-title">Evidence</h2>
      </div>
      <div className="p-6 space-y-4">
        {allFailed ? (
          <div className="flex items-start gap-3 p-4 rounded-xl bg-amber-500/5 border border-amber-500/20">
            <WifiOff className="w-5 h-5 text-amber-500 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-amber-300 mb-1">
                Evidence providers unavailable
              </p>
              <p className="text-xs text-amber-300/60 leading-relaxed">
                All configured evidence providers failed to respond. Evidence assessment
                could not be performed. This is a technical limitation, not an indication
                that the content is false.
              </p>
              {providersFailed.length > 0 && (
                <p className="text-xs text-slate-600 mt-2">
                  Failed: {providersFailed.join(', ')}
                </p>
              )}
            </div>
          </div>
        ) : (
          <div className="flex items-start gap-3 p-4 rounded-xl bg-slate-800/40 border border-slate-700/40">
            <Info className="w-5 h-5 text-slate-500 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-semibold text-slate-300 mb-1">
                Insufficient reliable evidence found.
              </p>
              <p className="text-xs text-slate-500 leading-relaxed">
                The evidence search retrieved no articles with sufficient overlap
                to support or contradict the content.{' '}
                <span className="text-slate-400 font-medium">
                  Absence of evidence cannot be interpreted as proof of falsehood.
                </span>
              </p>
              {providersUsed.length > 0 && (
                <p className="text-xs text-slate-600 mt-2">
                  Searched: {providersUsed.join(', ')}
                </p>
              )}
            </div>
          </div>
        )}

        {/* Limitations */}
        {limitations.length > 0 && (
          <ul className="space-y-1.5">
            {limitations.map((lim, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-slate-600">
                <AlertTriangle className="w-3 h-3 text-amber-700 shrink-0 mt-0.5" />
                {lim}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────

interface EvidenceSectionProps {
  summary?:       EvidenceSummary
  /** All evidence sources from all claims, flattened */
  allSources:     EvidenceSourceResult[]
}

export function EvidenceSection({ summary, allSources }: EvidenceSectionProps) {
  const totalMeaningful = allSources.filter(
    (s) => s.relationship_to_claim !== 'not_relevant',
  ).length

  // No evidence at all
  if (!allSources.length || totalMeaningful === 0) {
    return (
      <InsufficientState
        allFailed={summary?.all_providers_failed ?? false}
        providersUsed={summary?.providers_used ?? []}
        providersFailed={summary?.providers_failed ?? []}
        limitations={summary?.evidence_limitations ?? []}
      />
    )
  }

  // Group by relationship
  const grouped: Record<RelKey, EvidenceSourceResult[]> = {
    supporting:    [],
    contradicting: [],
    inconclusive:  [],
    not_relevant:  [],
  }
  for (const src of allSources) {
    const k = (src.relationship_to_claim ?? 'inconclusive') as RelKey
    if (grouped[k]) grouped[k].push(src)
    else             grouped.inconclusive.push(src)
  }

  // Sort each group by relevance desc
  const ORDER: RelKey[] = ['supporting', 'contradicting', 'inconclusive', 'not_relevant']
  for (const k of ORDER) {
    grouped[k].sort((a, b) => b.relevance_score - a.relevance_score)
  }

  const hasAny = ORDER.some((k) => grouped[k].length > 0)
  if (!hasAny) return null

  return (
    <section aria-label="Evidence">
      <div className="glass-card overflow-hidden">
        {/* Header + provider metadata */}
        <div className="px-5 py-4 border-b border-slate-800">
          <h2 className="section-title">Evidence</h2>
          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 mt-1.5">
            {summary && (
              <>
                <span className="text-xs text-slate-500">
                  {summary.total_evidence} source{summary.total_evidence !== 1 ? 's' : ''} retrieved
                </span>
                {summary.providers_used.length > 0 && (
                  <span className="text-xs text-slate-600 flex items-center gap-1">
                    <Wifi className="w-3 h-3" />
                    {summary.providers_used.join(', ')}
                  </span>
                )}
                {summary.providers_failed.length > 0 && (
                  <span className="text-xs text-amber-600 flex items-center gap-1">
                    <WifiOff className="w-3 h-3" />
                    {summary.providers_failed.length} provider{summary.providers_failed.length > 1 ? 's' : ''} failed
                  </span>
                )}
              </>
            )}
          </div>
        </div>

        {/* Groups */}
        <div className="p-4 space-y-3">
          {ORDER.map((rel) => (
            <EvidenceGroup
              key={rel}
              rel={rel}
              sources={grouped[rel]}
              /* open supporting + contradicting by default; collapse inconclusive/not_relevant */
              defaultOpen={rel === 'supporting' || rel === 'contradicting'}
            />
          ))}
        </div>

        {/* Limitations */}
        {summary?.evidence_limitations && summary.evidence_limitations.length > 0 && (
          <div className="px-5 pb-4 space-y-1.5 border-t border-slate-800 pt-3">
            {summary.evidence_limitations.map((lim, i) => (
              <div key={i} className="flex items-start gap-2 text-xs text-slate-600">
                <AlertTriangle className="w-3 h-3 text-amber-700 shrink-0 mt-0.5" />
                {lim}
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
