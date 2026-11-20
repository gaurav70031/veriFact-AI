/**
 * AssessmentHero
 *
 * Top-level verdict display — the first thing a user sees.
 *
 * When Gemini is available it becomes the primary verdict panel.
 * The ML style-classifier is demoted to a secondary "signal" panel.
 * When Gemini is unavailable the ML panel is shown as the primary signal.
 */

import { ShieldAlert, ShieldCheck, ShieldQuestion, AlertTriangle, Info, Sparkles } from 'lucide-react'
import { cn, getVerdictConfig, formatPct } from '@/lib/utils'
import type { MLVerdict, EvidenceVerdict } from '@/types/api'

// ── Confidence arc (SVG) ──────────────────────────────────────────────────────

function ConfidenceArc({ value, color }: { value: number; color: string }) {
  const R    = 36
  const circ = Math.PI * R
  const dash = circ * Math.min(Math.max(value, 0), 1)

  return (
    <svg width="88" height="52" viewBox="0 0 88 52" className="overflow-visible">
      <path
        d={`M 8 44 A ${R} ${R} 0 0 1 80 44`}
        fill="none" stroke="rgba(100,116,139,0.2)" strokeWidth="6" strokeLinecap="round"
      />
      <path
        d={`M 8 44 A ${R} ${R} 0 0 1 80 44`}
        fill="none" stroke={color} strokeWidth="6" strokeLinecap="round"
        strokeDasharray={`${dash} ${circ}`}
        className="transition-all duration-700"
      />
    </svg>
  )
}

// ── Verdict icon ──────────────────────────────────────────────────────────────

function VerdictIcon({ verdict, size = 'md' }: { verdict: string; size?: 'sm' | 'md' }) {
  const cls = size === 'md' ? 'w-7 h-7' : 'w-5 h-5'
  if (verdict === 'FAKE' || verdict === 'FALSE' || verdict === 'LIKELY_MISLEADING' || verdict === 'CONTRADICTED' || verdict === 'MISLEADING')
    return <ShieldAlert className={cn(cls, 'text-red-400')} />
  if (verdict === 'REAL' || verdict === 'TRUE' || verdict === 'LIKELY_CREDIBLE')
    return <ShieldCheck className={cn(cls, 'text-emerald-400')} />
  return <ShieldQuestion className={cn(cls, 'text-amber-400')} />
}

function arcColor(confidence: number, isPositive: boolean) {
  if (confidence < 0.55) return '#f59e0b'
  return isPositive ? '#34d399' : '#f87171'
}

// ── Gemini-powered verdict panel (primary when Gemini is available) ───────────

const GEMINI_STYLES: Record<string, { bg: string; border: string; labelColor: string; label: string }> = {
  TRUE:       { bg: 'bg-emerald-900/20', border: 'border-emerald-500/30', labelColor: 'text-emerald-400', label: 'True' },
  FALSE:      { bg: 'bg-red-900/20',     border: 'border-red-500/30',     labelColor: 'text-red-400',     label: 'False' },
  UNCERTAIN:  { bg: 'bg-amber-900/20',   border: 'border-amber-500/30',   labelColor: 'text-amber-400',   label: 'Uncertain' },
  MISLEADING: { bg: 'bg-orange-900/20',  border: 'border-orange-500/30',  labelColor: 'text-orange-400',  label: 'Misleading' },
}

function GeminiPanel({
  verdict,
  confidence,
  explanation,
}: {
  verdict: string
  confidence: number
  explanation: string
}) {
  const style   = GEMINI_STYLES[verdict] ?? GEMINI_STYLES['UNCERTAIN']
  const isPositive = verdict === 'TRUE'
  const color   = arcColor(confidence, isPositive)

  return (
    <div className={cn('rounded-2xl p-6 border flex flex-col gap-4', style.bg, style.border)}>
      {/* Label row */}
      <div className="flex items-center gap-3">
        <VerdictIcon verdict={verdict} />
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="label-sm">Verdict</span>
            <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-300 border border-violet-500/20">
              <Sparkles className="w-3 h-3" />
              Gemini AI
            </span>
          </div>
          <div className={cn('text-xl font-bold tracking-tight', style.labelColor)}>
            {style.label}
          </div>
        </div>
      </div>

      {/* Confidence gauge */}
      <div className="flex items-end gap-4">
        <div className="relative">
          <ConfidenceArc value={confidence} color={color} />
          <div className="absolute bottom-0 left-0 right-0 text-center">
            <span className="text-base font-bold font-mono text-slate-200">
              {formatPct(confidence)}
            </span>
          </div>
        </div>
        <div className="pb-1">
          <div className="text-xs text-slate-400 font-medium">Confidence</div>
          <div className="text-xs text-slate-600 mt-0.5">
            {confidence > 0.8 ? 'High confidence' : confidence > 0.6 ? 'Moderate' : 'Low confidence'}
          </div>
        </div>
      </div>

      {/* Explanation */}
      {explanation && (
        <p className="text-sm text-slate-300 leading-relaxed">{explanation}</p>
      )}

      <div className="flex items-start gap-2 pt-1 border-t border-slate-700/30">
        <Info className="w-3 h-3 text-slate-600 shrink-0 mt-0.5" />
        <p className="text-xs text-slate-600 leading-relaxed">
          Gemini uses live evidence and world knowledge — including Indian context and post-2018 events.
        </p>
      </div>
    </div>
  )
}

// ── ML panel (primary when Gemini unavailable, secondary otherwise) ───────────

function MLPanel({
  verdict,
  confidence,
  isSecondary,
}: {
  verdict: MLVerdict
  confidence: number | undefined
  isSecondary: boolean
}) {
  const vc     = getVerdictConfig(verdict)
  const conf   = confidence ?? 0
  const isReal = verdict === 'REAL'
  const color  = arcColor(conf, isReal)

  if (isSecondary) {
    // Compact, dimmed version shown alongside Gemini
    return (
      <div className="rounded-2xl p-5 border border-slate-700/30 bg-slate-800/20 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <VerdictIcon verdict={verdict} size="sm" />
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <span className="label-sm text-slate-500">ML Style Signal</span>
            </div>
            <div className={cn('text-base font-semibold tracking-tight opacity-70', vc.color)}>
              {vc.label}
              {confidence !== undefined && (
                <span className="text-slate-500 font-normal text-sm ml-2">
                  ({formatPct(conf)})
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-start gap-2">
          <AlertTriangle className="w-3 h-3 text-amber-600/60 shrink-0 mt-0.5" />
          <p className="text-xs text-slate-600 leading-relaxed">
            Style-based classifier only — trained on 2015–2018 US news. Gemini's verdict above supersedes this.
          </p>
        </div>
      </div>
    )
  }

  // Full primary version when Gemini is not available
  return (
    <div className={cn('rounded-2xl p-6 border flex flex-col gap-4', vc.bg, vc.border)}>
      <div className="flex items-center gap-3">
        <VerdictIcon verdict={verdict} />
        <div>
          <div className="label-sm mb-0.5">ML Model Prediction</div>
          <div className={cn('text-xl font-bold tracking-tight', vc.color)}>{vc.label}</div>
        </div>
      </div>

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

  if (verdict === 'INSUFFICIENT_EVIDENCE') {
    return (
      <div className="rounded-2xl p-6 border border-slate-600/40 bg-slate-800/30 flex flex-col gap-4">
        <div className="flex items-center gap-3">
          <ShieldQuestion className="w-7 h-7 text-slate-400" />
          <div>
            <div className="label-sm mb-0.5">Evidence Assessment</div>
            <div className="text-xl font-bold tracking-tight text-slate-300">Insufficient Evidence</div>
          </div>
        </div>
        <div className="p-3 rounded-lg bg-slate-700/30 border border-slate-600/30">
          <p className="text-sm text-slate-300 font-medium mb-1">Insufficient reliable evidence found.</p>
          <p className="text-xs text-slate-500 leading-relaxed">
            {explanation ||
              'No corroborating or contradicting sources were retrieved. ' +
              'This does not indicate the content is false.'}
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
  mlVerdict:             MLVerdict
  mlConfidence?:         number
  evidenceVerdict?:      EvidenceVerdict
  evidenceExplanation?:  string
  hasConflict?:          boolean
  // Gemini fields
  geminiVerdict?:        string
  geminiConfidence?:     number
  geminiExplanation?:    string
  geminiAvailable?:      boolean
}

export function AssessmentHero({
  mlVerdict,
  mlConfidence,
  evidenceVerdict,
  evidenceExplanation,
  hasConflict,
  geminiVerdict,
  geminiConfidence,
  geminiExplanation,
  geminiAvailable,
}: AssessmentHeroProps) {
  const hasGemini = geminiAvailable && !!geminiVerdict

  return (
    <section aria-label="Overall assessment">
      {hasGemini ? (
        // ── Gemini available: 2-col layout — Gemini primary, ML secondary ────
        <div className="grid sm:grid-cols-2 gap-4">
          <GeminiPanel
            verdict={geminiVerdict!}
            confidence={geminiConfidence ?? 0}
            explanation={geminiExplanation ?? ''}
          />
          <div className="flex flex-col gap-4">
            <MLPanel verdict={mlVerdict} confidence={mlConfidence} isSecondary={true} />
            <EvidencePanel verdict={evidenceVerdict} explanation={evidenceExplanation} />
          </div>
        </div>
      ) : (
        // ── No Gemini: original 2-col layout — ML + Evidence side by side ────
        <div className="grid sm:grid-cols-2 gap-4">
          <MLPanel verdict={mlVerdict} confidence={mlConfidence} isSecondary={false} />
          <EvidencePanel verdict={evidenceVerdict} explanation={evidenceExplanation} />
        </div>
      )}

      {hasConflict && !hasGemini && (
        <div className="mt-3">
          <ConflictBanner />
        </div>
      )}
    </section>
  )
}
