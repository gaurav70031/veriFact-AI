/**
 * ClaimsBreakdown
 *
 * Per-claim accordion. Each claim shows:
 *  • Extracted claim text
 *  • ML verdict for that claim
 *  • Evidence-based assessment (with correct insufficient-evidence handling)
 *  • Evidence source count breakdown (supporting / contradicting / inconclusive)
 *  • Evidence explanation text
 *  • Assessment conflict warning when ML and evidence disagree
 *  • Top token weights for that claim (aggregate across models)
 *  • Evidence limitations list
 *  • Expandable source links
 *
 * Renders nothing when claims array is empty.
 */

import { useState } from 'react'
import {
  ChevronDown, ChevronRight, AlertTriangle, ExternalLink,
  CheckCircle2, XCircle, MinusCircle, Calendar,
} from 'lucide-react'
import { cn, getVerdictConfig, formatDate, evidenceRelConfig } from '@/lib/utils'
import type { ClaimResult, EvidenceSourceResult } from '@/types/api'

// ── Evidence count pills ──────────────────────────────────────────────────────

function EvidencePills({ claim }: { claim: ClaimResult }) {
  const pills = [
    { label: 'Supporting',    count: claim.supporting_count,    color: 'text-emerald-400', dot: 'bg-emerald-400' },
    { label: 'Contradicting', count: claim.contradicting_count, color: 'text-red-400',     dot: 'bg-red-400' },
    { label: 'Inconclusive',  count: claim.inconclusive_count,  color: 'text-amber-400',   dot: 'bg-amber-400' },
  ].filter((p) => p.count > 0)

  if (!pills.length) return null

  return (
    <div className="flex flex-wrap gap-2">
      {pills.map((p) => (
        <div key={p.label} className="flex items-center gap-1.5 text-xs">
          <span className={cn('w-1.5 h-1.5 rounded-full shrink-0', p.dot)} />
          <span className={p.color}>{p.count}</span>
          <span className="text-slate-600">{p.label.toLowerCase()}</span>
        </div>
      ))}
    </div>
  )
}

// ── Inline source row ─────────────────────────────────────────────────────────

function SourceRow({ src }: { src: EvidenceSourceResult }) {
  const rel = evidenceRelConfig[src.relationship_to_claim as keyof typeof evidenceRelConfig]
    ?? evidenceRelConfig.inconclusive

  const Icon =
    src.relationship_to_claim === 'supporting'    ? CheckCircle2 :
    src.relationship_to_claim === 'contradicting' ? XCircle      :
    MinusCircle

  return (
    <div className="flex items-start gap-2.5 py-2.5 border-b border-slate-800/40 last:border-0 group">
      <Icon className={cn('w-3.5 h-3.5 shrink-0 mt-0.5', rel.color)} />
      <div className="flex-1 min-w-0">
        <a
          href={src.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-slate-300 hover:text-brand-400 transition-colors line-clamp-2 leading-relaxed"
        >
          {src.title}
          <ExternalLink className="inline-block w-2.5 h-2.5 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
        </a>
        <div className="flex items-center gap-3 mt-0.5">
          <span className="text-xs text-slate-600">{src.source_name}</span>
          {src.published_at && (
            <span className="text-xs text-slate-700 flex items-center gap-0.5">
              <Calendar className="w-2.5 h-2.5" />
              {formatDate(src.published_at)}
            </span>
          )}
        </div>
        {src.snippet && (
          <p className="text-xs text-slate-600 italic mt-1 line-clamp-2">"{src.snippet}"</p>
        )}
      </div>
    </div>
  )
}

// ── Token weight chips ────────────────────────────────────────────────────────

function TokenChips({ tokens }: { tokens: { token: string; weight: number }[] }) {
  if (!tokens.length) return null
  const top = tokens.slice(0, 10)
  const maxAbs = Math.max(...top.map((t) => Math.abs(t.weight)), 0.01)

  return (
    <div>
      <div className="label-sm mb-2">Key terms (model attention)</div>
      <div className="flex flex-wrap gap-1.5">
        {top.map((t, i) => {
          const intensity = Math.abs(t.weight) / maxAbs
          const isFake    = t.weight > 0
          return (
            <span
              key={i}
              title={`Weight: ${t.weight.toFixed(3)} — ${isFake ? 'associated with fake patterns' : 'associated with real patterns'}`}
              className="px-2 py-0.5 rounded text-xs font-mono border cursor-help"
              style={{
                color: isFake ? `rgba(252,165,165,${0.6 + intensity * 0.4})` : `rgba(110,231,183,${0.6 + intensity * 0.4})`,
                borderColor: isFake ? `rgba(239,68,68,${0.1 + intensity * 0.3})` : `rgba(52,211,153,${0.1 + intensity * 0.3})`,
                backgroundColor: isFake ? `rgba(239,68,68,${0.04 + intensity * 0.12})` : `rgba(52,211,153,${0.04 + intensity * 0.12})`,
              }}
            >
              {t.token}
            </span>
          )
        })}
      </div>
      <p className="text-xs text-slate-700 mt-1.5">
        Colour intensity = model weight. Red = fake-associated; Green = real-associated.
        These are model signals, not factual indicators.
      </p>
    </div>
  )
}

// ── Single claim row ──────────────────────────────────────────────────────────

function ClaimRow({ claim, index }: { claim: ClaimResult; index: number }) {
  const [open, setOpen] = useState(index === 0)

  const mlVc = getVerdictConfig(claim.ml_verdict ?? undefined)
  const evVc = claim.evidence_verdict
    ? getVerdictConfig(claim.evidence_verdict)
    : null

  const hasEvidence = claim.total_evidence > 0
  const isInsufficient = claim.evidence_verdict === 'INSUFFICIENT_EVIDENCE' || !hasEvidence

  return (
    <div className="border border-slate-800/60 rounded-xl overflow-hidden">
      {/* Header — always visible */}
      <button
        className="w-full flex items-start gap-4 px-5 py-4 text-left hover:bg-slate-800/20 transition-colors"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {/* Position number */}
        <span className="text-xs font-mono text-slate-700 mt-0.5 shrink-0 w-5">
          {String(index + 1).padStart(2, '0')}
        </span>

        {/* Claim text */}
        <div className="flex-1 min-w-0">
          <p className={cn(
            'text-sm text-slate-300 leading-relaxed',
            !open && 'line-clamp-2',
          )}>
            {claim.claim_text}
          </p>

          {/* Summary chips — collapsed view */}
          {!open && (
            <div className="flex flex-wrap items-center gap-2 mt-2">
              {claim.ml_verdict && (
                <span className={cn('text-xs font-semibold', mlVc.color)}>
                  ML: {mlVc.label}
                </span>
              )}
              {evVc && (
                <span className={cn('text-xs font-semibold', evVc.color)}>
                  Evidence: {evVc.label}
                </span>
              )}
              {!evVc && isInsufficient && (
                <span className="text-xs text-slate-500">
                  Evidence: Insufficient
                </span>
              )}
              {claim.assessment_conflict && (
                <span className="text-xs text-amber-500 flex items-center gap-1">
                  <AlertTriangle className="w-3 h-3" />
                  Conflict
                </span>
              )}
            </div>
          )}
        </div>

        {/* Toggle icon */}
        <span className="shrink-0 mt-0.5">
          {open
            ? <ChevronDown className="w-4 h-4 text-slate-500" />
            : <ChevronRight className="w-4 h-4 text-slate-500" />}
        </span>
      </button>

      {/* Expanded detail */}
      {open && (
        <div className="px-5 pb-5 pt-1 space-y-4 border-t border-slate-800/40">

          {/* Verdict row */}
          <div className="grid sm:grid-cols-2 gap-3">
            {/* ML verdict */}
            {claim.ml_verdict && (
              <div className={cn('rounded-lg p-3 border', mlVc.bg, mlVc.border)}>
                <div className="label-sm mb-1">ML Prediction</div>
                <div className={cn('text-sm font-bold', mlVc.color)}>{mlVc.label}</div>
                {claim.ml_confidence !== undefined && (
                  <div className="text-xs text-slate-500 mt-0.5">
                    {(claim.ml_confidence * 100).toFixed(1)}% confidence
                  </div>
                )}
              </div>
            )}

            {/* Evidence verdict — with explicit insufficient handling */}
            <div className={cn(
              'rounded-lg p-3 border',
              evVc ? cn(evVc.bg, evVc.border) : 'bg-slate-800/30 border-slate-700/30',
            )}>
              <div className="label-sm mb-1">Evidence Assessment</div>
              {claim.evidence_verdict === 'INSUFFICIENT_EVIDENCE' || !claim.evidence_verdict ? (
                <div className="text-sm font-bold text-slate-400">
                  {claim.evidence_verdict === 'INSUFFICIENT_EVIDENCE'
                    ? 'Insufficient Evidence'
                    : 'No Assessment'}
                </div>
              ) : (
                <div className={cn('text-sm font-bold', evVc?.color)}>{evVc?.label}</div>
              )}
            </div>
          </div>

          {/* Conflict warning */}
          {claim.assessment_conflict && (
            <div className="flex items-start gap-2.5 p-3 rounded-lg bg-amber-500/5 border border-amber-500/15">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
              <p className="text-xs text-amber-300/70 leading-relaxed">
                <span className="font-medium text-amber-300">Assessment conflict:</span>{' '}
                ML prediction and evidence assessment disagree. Review both signals independently.
              </p>
            </div>
          )}

          {/* Evidence explanation */}
          {claim.evidence_explanation && (
            <div className="text-sm text-slate-400 leading-relaxed bg-slate-800/30 rounded-lg p-3">
              {claim.evidence_explanation}
            </div>
          )}

          {/* Insufficient evidence explicit message */}
          {isInsufficient && !claim.evidence_explanation && (
            <div className="flex items-start gap-2 p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
              <span className="text-slate-600 text-xs mt-0.5">ℹ</span>
              <p className="text-xs text-slate-500 leading-relaxed">
                <span className="text-slate-400 font-medium">
                  Insufficient reliable evidence found.
                </span>{' '}
                No sources with sufficient topic overlap were retrieved for this claim.
                This does not indicate the claim is false.
              </p>
            </div>
          )}

          {/* Evidence count pills */}
          {hasEvidence && <EvidencePills claim={claim} />}

          {/* Token weights */}
          {claim.top_tokens?.length > 0 && (
            <TokenChips tokens={claim.top_tokens} />
          )}

          {/* Source links — expandable sub-list */}
          {claim.evidence_sources.length > 0 && (
            <SourceList sources={claim.evidence_sources} />
          )}

          {/* Limitations */}
          {claim.evidence_limitations?.length > 0 && (
            <ul className="space-y-1">
              {claim.evidence_limitations.map((lim, i) => (
                <li key={i} className="flex items-start gap-1.5 text-xs text-slate-600">
                  <AlertTriangle className="w-3 h-3 text-amber-700 shrink-0 mt-0.5" />
                  {lim}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

// ── Source list sub-section ───────────────────────────────────────────────────

function SourceList({ sources }: { sources: EvidenceSourceResult[] }) {
  const [expanded, setExpanded] = useState(sources.length <= 3)

  const visible = expanded ? sources : sources.slice(0, 3)

  return (
    <div>
      <div className="label-sm mb-2">
        Sources ({sources.length})
      </div>
      <div className="rounded-lg border border-slate-800/50 bg-slate-900/30 divide-y divide-slate-800/40 px-3">
        {visible.map((src, i) => (
          <SourceRow key={`${src.url}-${i}`} src={src} />
        ))}
      </div>
      {sources.length > 3 && !expanded && (
        <button
          className="mt-2 text-xs text-brand-400 hover:text-brand-300 transition-colors"
          onClick={() => setExpanded(true)}
        >
          Show {sources.length - 3} more source{sources.length - 3 !== 1 ? 's' : ''}…
        </button>
      )}
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────

interface ClaimsBreakdownProps {
  claims: ClaimResult[]
}

export function ClaimsBreakdown({ claims }: ClaimsBreakdownProps) {
  if (!claims.length) return null

  return (
    <section aria-label="Claim analysis">
      <div className="glass-card overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-800">
          <h2 className="section-title">Claims Analysis</h2>
          <p className="text-xs text-slate-500 mt-1">
            {claims.length} claim{claims.length !== 1 ? 's' : ''} extracted and assessed
            independently. Click any claim to expand its evidence and explanation.
          </p>
        </div>

        <div className="p-4 space-y-3">
          {claims.map((claim, i) => (
            <ClaimRow key={`${claim.position}-${i}`} claim={claim} index={i} />
          ))}
        </div>
      </div>
    </section>
  )
}
