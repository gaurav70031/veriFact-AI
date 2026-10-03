import { useState } from 'react'
import { ChevronDown, ChevronUp, AlertTriangle } from 'lucide-react'
import { cn, getVerdictConfig } from '@/lib/utils'
import { Badge } from '@/components/ui/Badge'
import { EvidenceCard } from './EvidenceCard'
import { EmptyState } from '@/components/ui/EmptyState'
import type { ClaimResult } from '@/types/api'

interface ClaimSectionProps {
  claims: ClaimResult[]
}

export function ClaimSection({ claims }: ClaimSectionProps) {
  if (!claims.length) return null

  return (
    <div className="glass-card overflow-hidden">
      <div className="px-5 py-4 border-b border-slate-800">
        <h3 className="section-title">Claim Analysis</h3>
        <p className="text-xs text-slate-500 mt-1">
          {claims.length} claim{claims.length > 1 ? 's' : ''} extracted and assessed independently.
        </p>
      </div>
      <div className="divide-y divide-slate-800/60">
        {claims.map((claim) => (
          <ClaimItem key={claim.position} claim={claim} />
        ))}
      </div>
    </div>
  )
}

function ClaimItem({ claim }: { claim: ClaimResult }) {
  const [expanded, setExpanded] = useState(claim.position === 1)
  const ev = claim.evidence_verdict ? getVerdictConfig(claim.evidence_verdict) : null

  const badgeVariant = claim.evidence_verdict === 'LIKELY_CREDIBLE' ? 'success' :
    claim.evidence_verdict === 'CONTRADICTED' || claim.evidence_verdict === 'LIKELY_MISLEADING' ? 'danger' :
    'warning'

  return (
    <div className="p-5">
      <div className="flex items-start justify-between gap-4 cursor-pointer" onClick={() => setExpanded(!expanded)}>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5">
            <span className="label-sm">Claim {claim.position}</span>
            {claim.assessment_conflict && (
              <Badge variant="warning" className="flex items-center gap-1">
                <AlertTriangle className="w-2.5 h-2.5" />
                ML/Evidence conflict
              </Badge>
            )}
          </div>
          <p className="text-sm text-slate-300 leading-relaxed">{claim.claim_text}</p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {ev && (
            <span className={cn('text-sm font-semibold', ev.color)}>{ev.label}</span>
          )}
          {expanded ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
        </div>
      </div>

      {expanded && (
        <div className="mt-4 space-y-4 animate-fade-in">
          {/* Evidence counts */}
          {claim.total_evidence > 0 && (
            <div className="flex flex-wrap gap-3">
              {claim.supporting_count > 0 && (
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="text-slate-400">{claim.supporting_count} supporting</span>
                </div>
              )}
              {claim.contradicting_count > 0 && (
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="w-2 h-2 rounded-full bg-red-400" />
                  <span className="text-slate-400">{claim.contradicting_count} contradicting</span>
                </div>
              )}
              {claim.inconclusive_count > 0 && (
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="w-2 h-2 rounded-full bg-amber-400" />
                  <span className="text-slate-400">{claim.inconclusive_count} inconclusive</span>
                </div>
              )}
            </div>
          )}

          {/* Evidence explanation */}
          {claim.evidence_explanation && (
            <p className="text-sm text-slate-400 leading-relaxed bg-slate-800/40 rounded-lg p-3">
              {claim.evidence_explanation}
            </p>
          )}

          {/* Limitations */}
          {claim.evidence_limitations.length > 0 && (
            <div className="space-y-1">
              {claim.evidence_limitations.map((lim, i) => (
                <div key={i} className="flex items-start gap-1.5 text-xs text-slate-500">
                  <AlertTriangle className="w-3 h-3 mt-0.5 shrink-0 text-amber-600" />
                  {lim}
                </div>
              ))}
            </div>
          )}

          {/* Evidence cards */}
          {claim.evidence_sources.length > 0 ? (
            <div className="space-y-2">
              <div className="label-sm">Evidence Sources ({claim.evidence_sources.length})</div>
              {claim.evidence_sources.map((src, i) => (
                <EvidenceCard key={i} evidence={src} />
              ))}
            </div>
          ) : (
            <EmptyState
              title="No evidence found"
              description="No relevant sources were retrieved for this claim. This does not indicate the claim is false."
              className="py-8"
            />
          )}
        </div>
      )}
    </div>
  )
}
