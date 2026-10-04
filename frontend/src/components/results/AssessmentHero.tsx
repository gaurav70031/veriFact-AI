/**
 * AssessmentHero
 *
 * Top-level verdict display — the first thing a user sees.
 *
 * Shows:
 *  • ML model verdict + weighted confidence gauge
 *  • Evidence-based verdict (or an explicit "Insufficient Evidence" state)
 *  • A separator that makes clear these are two INDEPENDENT signals
 *
 * Does NOT:
 *  • Convert "INSUFFICIENT_EVIDENCE" into "Fake" or "Unverified"
 *  • Synthesise a single combined verdict
 *  • Show evidence verdict when evidence_verdict is null / absent
 */

import { ShieldAlert, ShieldCheck, ShieldQuestion, AlertTriangle, Info } from 'lucide-react'
import { cn, getVerdictConfig, formatPct } from '@/lib/utils'
import type { MLVerdict, EvidenceVerdict } from '@/types/api'

// ── Confidence arc (SVG) ──────────────────────────────────────────────────────

function ConfidenceArc({ value, color }: { value: number; color: string }) {
  const R   = 36
  const circ = Math.PI * R  // half-circle
  const dash = circ * Math.min(Math.max(value, 0), 1)

  return (
    <svg width="88" height="52" viewBox="0 0 88 52" className="overflow-visible">
      {/* Track */}
      <path
        d={`M 8 44 A ${R} ${R} 0 0 1 80 44`}
        fill="none"
        stroke="rgba(100,116,139,0.2)"
        strokeWidth="6"
        strokeLinecap="round"
      />
      {/* Fill */}
      <path
        d={`M 8 44 A ${R} ${R} 0 0 1 80 44`}
        fill="none"
        stroke={color}
        strokeWidth="6"
        strokeLinecap="round"
        strokeDasharray={`${dash} ${circ}`}
        className="transition-all duration-700"
      />
    </svg>
  )
}

// ── Verdict icon ──────────────────────────────────────────────────────────────

function VerdictIcon({ verdict, size = 'md' }: { verdict: string; size?: 'sm' | 'md' }) {
  const cls = size === 'md' ? 'w-7 h-7' : 'w-5 h-5'
  if (verdict === 'FAKE' || verdict === 'LIKELY_MISLEADING' || verdict === 'CONTRADICTED')
    return <ShieldAlert className={cn(cls, 'text-red-400')} />
  if (verdict === 'REAL' || verdict === 'LIKELY_CREDIBLE')
    return <ShieldCheck className={cn(cls, 'text-emerald-400')} />
  return <ShieldQuestion className={cn(cls, 'text-amber-400')} />
}

// ── Confidence colour helper ──────────────────────────────────────────────────

function arcColor(confidence: number, isReal: boolean) {
  if (confidence < 0.55) return '#f59e0b'
  return isReal ? '#34d399' : '#f87171'
}

// ── ML Verdict panel ──────────────────────────────────────────────────────────

function MLPanel({
  verdict,
  confidence,
}: {
  verdict: MLVerdict
  confidence: number | undefined
}) {
  const vc      = getVerdictConfig(verdict)
  const conf    = confidence ?? 0
  const isReal  = verdict === 'REAL'
  const color   = arcColor(conf, isReal)

  return (
    <div className={cn('rounded-2xl p-6 border flex flex-col gap-4', vc.bg, vc.border)}>
      {/* Label row */}
      <div className="flex items-center gap-3">
        <VerdictIcon verdict={verdict} />
        <div>
          <div className="label-sm mb-0.5">ML Model Prediction</div>
          <div className={cn('text-xl font-bold tracking-tight', vc.color)}>{vc.label}</div>
        </div>
      </div>

      {/* Confidence gauge */}
      {confidence !== undefined && (
        <div className="flex items-end gap-4">
          <div className="relative">
            <ConfidenceArc value={conf} color={color} />
            <div className="absolute bottom-0 left-0 right-0 text-center">
              <span className="text-base font-bold font-mono text-slate-200">
                {formatPct(conf)}
              </span>
            </div>
          </div>
          <div className="pb-1">
            <div className="text-xs text-slate-400 font-medium">Confidence</div>
            <div className="text-xs text-slate-600 mt-0.5 leading-relaxed max-w-[140px]">
              {conf > 0.8 ? 'High confidence' : conf > 0.6 ? 'Moderate confidence' : 'Low confidence'}
            </div>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <div className="flex items-start gap-2 pt-1 border-t border-slate-700/30">
        <Info className="w-3 h-3 text-slate-600 shrink-0 mt-0.5" />
        <p className="text-xs text-slate-600 leading-relaxed">
          Statistical pattern match — not a factual determination.
        </p>
      </div>
    </div>
  )
}

// ── Evidence Verdict panel ────────────────────────────────────────────────────

function EvidencePanel({
  verdict,
  explanation,
}: {
  verdict: EvidenceVerdict | undefined
  explanation: string | undefined
}) {
  // Explicit null / absent — do not fabricate
  if (!verdict) {
    return (
      <div className="rounded-2xl p-6 border border-slate-700/40 bg-slate-800/20 flex flex-col justify-between gap-4">
        <div className="flex items-center gap-3">
          <ShieldQuestion className="w-7 h-7 text-slate-600" />
          <div>
            <div className="label-sm mb-0.5">Evidence Assessment</div>
            <div className="text-xl font-bold tracking-tight text-slate-500">Not Available</div>
          </div>
        </div>
        <p className="text-xs text-slate-600 leading-relaxed">
          Evidence assessment was not computed for this analysis.
        </p>
      </div>
    )
  }

  // Insufficient evidence — explicit, never conflated with fake
  if (verdict === 'INSUFFICIENT_EVIDENCE') {
    return (
      <div className="rounded-2xl p-6 border border-slate-600/40 bg-slate-800/30 flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <ShieldQuestion className="w-7 h-7 text-slate-400" />
          <div>
            <div className="label-sm mb-0.5">Evidence Assessment</div>
            <div className="text-xl font-bold tracking-tight text-slate-300">
              Insufficient Evidence
            </div>
          </div>
        </div>

        <div className="p-3 rounded-lg bg-slate-700/30 border border-slate-600/30">
          <p className="text-sm text-slate-300 font-medium mb-1">
            Insufficient reliable evidence found.
          </p>
          <p className="text-xs text-slate-500 leading-relaxed">
            {explanation ||
              'No corroborating or contradicting sources were retrieved. ' +
              'This does not indicate the content is false — absence of evidence ' +
              'cannot be interpreted as proof of falsehood.'}
          </p>
        </div>

        <div className="flex items-start gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0 mt-0.5" />
          <p className="text-xs text-amber-600/80 leading-relaxed">
            Insufficient evidence ≠ false. Consider searching for sources manually.
          </p>
        </div>
      </div>
    )
  }

  const vc = getVerdictConfig(verdict)

  return (
    <div className={cn('rounded-2xl p-6 border flex flex-col gap-4', vc.bg, vc.border)}>
      <div className="flex items-center gap-3">
        <VerdictIcon verdict={verdict} />
        <div>
          <div className="label-sm mb-0.5">Evidence Assessment</div>
          <div className={cn('text-xl font-bold tracking-tight', vc.color)}>{vc.label}</div>
        </div>
      </div>

      {explanation && (
        <p className="text-sm text-slate-400 leading-relaxed">{explanation}</p>
      )}

      <div className="flex items-start gap-2 pt-1 border-t border-slate-700/30">
        <Info className="w-3 h-3 text-slate-600 shrink-0 mt-0.5" />
        <p className="text-xs text-slate-600 leading-relaxed">
          Based on retrieved news sources — see Evidence section below.
        </p>
      </div>
    </div>
  )
}

// ── Conflict banner ───────────────────────────────────────────────────────────

function ConflictBanner() {
  return (
    <div className="flex items-start gap-3 p-3.5 rounded-xl bg-amber-500/5 border border-amber-500/20">
      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
      <div className="text-xs text-amber-300/80 leading-relaxed">
        <span className="font-medium text-amber-300">Assessment conflict.</span>{' '}
        The ML model prediction and the evidence-based assessment disagree.
        This may indicate nuanced or complex content. Review each signal independently.
      </div>
    </div>
  )
}

// ── Main export ───────────────────────────────────────────────────────────────

interface AssessmentHeroProps {
  mlVerdict:            MLVerdict
  mlConfidence?:        number
  evidenceVerdict?:     EvidenceVerdict
  evidenceExplanation?: string
  hasConflict?:         boolean
}

export function AssessmentHero({
  mlVerdict,
  mlConfidence,
  evidenceVerdict,
  evidenceExplanation,
  hasConflict,
}: AssessmentHeroProps) {
  return (
    <section aria-label="Overall assessment">
      <div className="grid sm:grid-cols-2 gap-4">
        <MLPanel verdict={mlVerdict} confidence={mlConfidence} />
        <EvidencePanel verdict={evidenceVerdict} explanation={evidenceExplanation} />
      </div>

      {hasConflict && (
        <div className="mt-3">
          <ConflictBanner />
        </div>
      )}
    </section>
  )
}
