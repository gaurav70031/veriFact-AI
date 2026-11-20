/**
 * VerificationPanel
 *
 * Inline evidence verification for a single claim, article text, or URL.
 * Calls POST /api/v1/verify and renders:
 *  - Overall verdict badge
 *  - Per-claim evidence breakdown
 *  - Per-source cards with trust tier, relationship, relevance
 *  - Critical distinction: ML ≠ evidence
 *  - Limitations always shown
 *
 * Props:
 *  claim   – pre-filled claim text
 *  url     – pre-filled URL
 *  onClose – called when user dismisses
 */

import { useState } from 'react'
import {
  ShieldCheck, ShieldAlert, ShieldQuestion, AlertTriangle,
  ExternalLink, ChevronDown, ChevronRight, Info,
  CheckCircle2, XCircle, HelpCircle, Loader2, Clock,
  Star, Award, Building2, Newspaper,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useVerify } from '@/hooks/useApi'
import type {
  VerificationResponse, ClaimVerificationOut, EvidenceSourceOut,
} from '@/types/api'

// ── Verdict config ────────────────────────────────────────────────────────────
const VERDICT_CONFIG: Record<string, { label: string; color: string; bg: string; icon: React.ReactNode }> = {
  LIKELY_CREDIBLE:      { label: 'Likely Credible',      color: 'text-emerald-400', bg: 'rgba(52,211,153,0.08)',  icon: <ShieldCheck   className="w-5 h-5 text-emerald-400" /> },
  CONTRADICTED:         { label: 'Contradicted',         color: 'text-red-400',     bg: 'rgba(248,113,113,0.08)', icon: <ShieldAlert   className="w-5 h-5 text-red-400"     /> },
  LIKELY_MISLEADING:    { label: 'Likely Misleading',    color: 'text-orange-400',  bg: 'rgba(251,146,60,0.08)', icon: <ShieldAlert   className="w-5 h-5 text-orange-400"   /> },
  UNVERIFIED:           { label: 'Unverified',           color: 'text-amber-400',   bg: 'rgba(251,191,36,0.08)', icon: <ShieldQuestion className="w-5 h-5 text-amber-400"  /> },
  INSUFFICIENT_EVIDENCE:{ label: 'Insufficient Evidence',color: 'text-slate-400',   bg: 'rgba(148,163,184,0.08)',icon: <ShieldQuestion className="w-5 h-5 text-slate-400"  /> },
}

// ── Relationship config ───────────────────────────────────────────────────────
const REL_CONFIG: Record<string, { label: string; color: string; icon: React.ReactNode }> = {
  supporting:    { label: 'Supporting',    color: 'text-emerald-400', icon: <CheckCircle2 className="w-3.5 h-3.5" /> },
  contradicting: { label: 'Contradicting', color: 'text-red-400',     icon: <XCircle      className="w-3.5 h-3.5" /> },
  inconclusive:  { label: 'Inconclusive',  color: 'text-amber-400',   icon: <HelpCircle   className="w-3.5 h-3.5" /> },
  not_relevant:  { label: 'Not Relevant',  color: 'text-slate-500',   icon: <HelpCircle   className="w-3.5 h-3.5" /> },
}

// ── Trust tier badge ──────────────────────────────────────────────────────────
function TrustTierBadge({ tier, providerName }: { tier: number; providerName: string }) {
  if (tier === 0) return null
  const configs = [
    null,
    { label: 'Official',    color: 'text-sky-400',     bg: 'bg-sky-400/10',     icon: <Building2  className="w-3 h-3" /> },
    { label: 'Research',    color: 'text-violet-400',  bg: 'bg-violet-400/10',  icon: <Award      className="w-3 h-3" /> },
    { label: 'Established', color: 'text-emerald-400', bg: 'bg-emerald-400/10', icon: <Star       className="w-3 h-3" /> },
    { label: 'News',        color: 'text-blue-400',    bg: 'bg-blue-400/10',    icon: <Newspaper  className="w-3 h-3" /> },
  ]
  const cfg = configs[tier]
  if (!cfg) return null
  return (
    <span className={cn('inline-flex items-center gap-1 text-xs px-1.5 py-0.5 rounded font-medium', cfg.color, cfg.bg)}>
      {cfg.icon} {cfg.label}
    </span>
  )
}

// ── Source card ───────────────────────────────────────────────────────────────
function SourceCard({ src }: { src: EvidenceSourceOut }) {
  const rel = REL_CONFIG[src.relationship_to_claim] ?? REL_CONFIG.not_relevant
  const relevancePct = Math.round(src.relevance_score * 100)

  return (
    <div className="rounded-lg p-3" style={{ backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)' }}>
      <div className="flex items-start gap-2 mb-1.5">
        <div className="flex-1 min-w-0">
          <a
            href={src.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs font-medium hover:underline flex items-start gap-1"
            style={{ color: 'var(--text-primary)' }}
          >
            <span className="line-clamp-2">{src.title}</span>
            <ExternalLink className="w-3 h-3 shrink-0 mt-0.5 opacity-60" />
          </a>
        </div>
        <TrustTierBadge tier={src.trust_tier} providerName={src.provider_name} />
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
        <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>
          {src.source_name}
        </span>
        {src.published_at && (
          <span className="flex items-center gap-1" style={{ color: 'var(--text-faint)' }}>
            <Clock className="w-3 h-3" />
            {new Date(src.published_at).toLocaleDateString()}
          </span>
        )}
        <span className={cn('flex items-center gap-0.5 font-medium', rel.color)}>
          {rel.icon} {rel.label}
        </span>
        <span className="font-mono" style={{ color: 'var(--text-faint)' }}>
          {relevancePct}% relevance
        </span>
      </div>

      {src.snippet && (
        <p className="text-xs mt-1.5 line-clamp-2 leading-relaxed" style={{ color: 'var(--text-subtle)' }}>
          {src.snippet}
        </p>
      )}
    </div>
  )
}

// ── Claim block ───────────────────────────────────────────────────────────────
function ClaimBlock({ claim }: { claim: ClaimVerificationOut }) {
  const [expanded, setExpanded] = useState(false)
  const vc = VERDICT_CONFIG[claim.evidence_verdict] ?? VERDICT_CONFIG.INSUFFICIENT_EVIDENCE

  const meaningfulSources = claim.evidence_sources.filter(
    (s) => s.relationship_to_claim !== 'not_relevant'
  )

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid var(--border)' }}>
      {/* Header */}
      <button
        className="w-full flex items-start gap-3 px-4 py-3 text-left transition-colors"
        style={{ backgroundColor: 'var(--bg-surface-2)' }}
        onClick={() => setExpanded(!expanded)}
        onMouseOver={(e) => { (e.currentTarget as HTMLElement).style.backgroundColor = 'var(--bg-surface-3)' }}
        onMouseOut={(e) => { (e.currentTarget as HTMLElement).style.backgroundColor = 'var(--bg-surface-2)' }}
      >
        {vc.icon}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-0.5 flex-wrap">
            <span className={cn('text-xs font-semibold', vc.color)}>{vc.label}</span>
            <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
              {claim.supporting_count}↑ {claim.contradicting_count}↓ {claim.inconclusive_count}? sources
            </span>
          </div>
          <p className="text-xs line-clamp-2" style={{ color: 'var(--text-secondary)' }}>
            {claim.claim_text}
          </p>
        </div>
        {expanded
          ? <ChevronDown className="w-4 h-4 shrink-0 mt-0.5" style={{ color: 'var(--text-subtle)' }} />
          : <ChevronRight className="w-4 h-4 shrink-0 mt-0.5" style={{ color: 'var(--text-subtle)' }} />
        }
      </button>

      {expanded && (
        <div className="p-4 space-y-3" style={{ borderTop: '1px solid var(--border)' }}>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
            {claim.evidence_explanation}
          </p>

          {meaningfulSources.length > 0 && (
            <div className="space-y-2">
              <div className="label-sm">Evidence Sources</div>
              {meaningfulSources.slice(0, 5).map((src, i) => (
                <SourceCard key={i} src={src} />
              ))}
            </div>
          )}

          {claim.evidence_limitations.length > 0 && (
            <div className="space-y-1">
              {claim.evidence_limitations.map((l, i) => (
                <div key={i} className="flex items-start gap-1.5 text-xs" style={{ color: 'var(--text-subtle)' }}>
                  <Info className="w-3 h-3 shrink-0 mt-0.5" />
                  {l}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

interface VerificationPanelProps {
  claim?:   string
  url?:     string
  text?:    string
  onClose?: () => void
  compact?: boolean
}

export function VerificationPanel({
  claim, url, text, onClose, compact = false,
}: VerificationPanelProps) {
  const verifyMutation = useVerify()
  const [result, setResult] = useState<VerificationResponse | null>(null)

  const handleVerify = () => {
    verifyMutation.mutate(
      { claim, url, text, max_results: 8, from_days: 90, max_claims: 3 },
      {
        onSuccess: (data) => setResult(data as VerificationResponse),
      },
    )
  }

  const vc = result
    ? (VERDICT_CONFIG[result.overall_verdict] ?? VERDICT_CONFIG.INSUFFICIENT_EVIDENCE)
    : null

  if (compact && !result && !verifyMutation.isPending) {
    return (
      <button
        onClick={handleVerify}
        className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
        style={{
          border: '1px solid rgba(99,102,241,0.3)',
          backgroundColor: 'rgba(99,102,241,0.08)',
          color: 'var(--brand)',
        }}
      >
        <ShieldCheck className="w-3.5 h-3.5" />
        Verify this claim
      </button>
    )
  }

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ border: '1px solid var(--border)', backgroundColor: 'var(--bg-surface)' }}
    >
      {/* Panel header */}
      <div
        className="flex items-center justify-between px-4 py-3"
        style={{ borderBottom: '1px solid var(--border)', backgroundColor: 'var(--bg-surface-2)' }}
      >
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4" style={{ color: 'var(--brand)' }} />
          <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>
            Evidence Verification
          </span>
          {vc && (
            <span className={cn('text-xs font-semibold px-2 py-0.5 rounded-full', vc.color)}
              style={{ backgroundColor: vc.bg }}>
              {vc.label}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {!result && !verifyMutation.isPending && (
            <button onClick={handleVerify} className="btn-primary text-xs py-1.5">
              <ShieldCheck className="w-3.5 h-3.5" /> Run Verification
            </button>
          )}
          {onClose && (
            <button onClick={onClose} className="btn-ghost p-1.5">
              <span className="text-xs">✕</span>
            </button>
          )}
        </div>
      </div>

      <div className="p-4">
        {/* Loading */}
        {verifyMutation.isPending && (
          <div className="flex items-center gap-3 py-6 justify-center">
            <Loader2 className="w-5 h-5 animate-spin" style={{ color: 'var(--brand)' }} />
            <span className="text-sm" style={{ color: 'var(--text-subtle)' }}>
              Searching evidence providers…
            </span>
          </div>
        )}

        {/* Error */}
        {verifyMutation.isError && (
          <div className="flex items-start gap-2 p-3 rounded-xl"
            style={{ backgroundColor: 'rgba(248,113,113,0.06)', border: '1px solid rgba(248,113,113,0.2)' }}>
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            <p className="text-sm text-red-400">{(verifyMutation.error as Error).message}</p>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="space-y-4">
            {/* ML/evidence distinction notice */}
            <div className="flex items-start gap-2 p-3 rounded-xl"
              style={{ backgroundColor: 'rgba(251,191,36,0.05)', border: '1px solid rgba(251,191,36,0.15)' }}>
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
              <p className="text-xs text-amber-400/80">
                <strong className="text-amber-400">Evidence-only result.</strong>{' '}
                This verdict is based on retrieved sources — not ML classification.
                INSUFFICIENT_EVIDENCE does not mean the claim is false.
              </p>
            </div>

            {/* Summary row */}
            <div className="grid grid-cols-4 gap-2">
              {[
                { label: 'Supporting',   value: result.supporting,   color: '#34d399' },
                { label: 'Contradicting',value: result.contradicting, color: '#f87171' },
                { label: 'Inconclusive', value: result.inconclusive,  color: '#fbbf24' },
                { label: 'Total',        value: result.total_evidence,color: 'var(--brand)' },
              ].map((s) => (
                <div key={s.label} className="rounded-lg p-2 text-center"
                  style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                  <div className="text-lg font-bold" style={{ color: s.color }}>{s.value}</div>
                  <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>{s.label}</div>
                </div>
              ))}
            </div>

            {/* Overall explanation */}
            <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
              {result.overall_explanation}
            </p>

            {/* Per-claim blocks */}
            {result.claims_checked.length > 0 && (
              <div className="space-y-2">
                <div className="label-sm">Claims Checked ({result.claims_checked.length})</div>
                {result.claims_checked.map((claim, i) => (
                  <ClaimBlock key={i} claim={claim} />
                ))}
              </div>
            )}

            {/* Providers */}
            <div className="flex flex-wrap gap-1.5">
              <span className="text-xs" style={{ color: 'var(--text-subtle)' }}>Sources used:</span>
              {result.providers_used.map((p) => (
                <span key={p} className="text-xs px-2 py-0.5 rounded font-medium"
                  style={{ backgroundColor: 'var(--bg-surface-2)', color: 'var(--text-secondary)' }}>
                  {p.split(':')[0]}
                </span>
              ))}
              {result.providers_failed.length > 0 && (
                <span className="text-xs text-amber-400">
                  {result.providers_failed.length} provider(s) unavailable
                </span>
              )}
            </div>

            {/* Limitations */}
            <details>
              <summary className="text-xs cursor-pointer" style={{ color: 'var(--text-subtle)' }}>
                Show limitations
              </summary>
              <ul className="mt-2 space-y-1">
                {result.limitations.map((l, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-xs" style={{ color: 'var(--text-subtle)' }}>
                    <Info className="w-3 h-3 shrink-0 mt-0.5" /> {l}
                  </li>
                ))}
              </ul>
            </details>

            <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
              {result.processing_time_ms}ms
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
