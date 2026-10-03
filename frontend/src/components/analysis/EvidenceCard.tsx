import { ExternalLink, Calendar, BarChart3 } from 'lucide-react'
import { cn, formatDate, evidenceRelConfig } from '@/lib/utils'
import { Badge } from '@/components/ui/Badge'
import { ProgressBar } from '@/components/ui/ProgressBar'
import type { EvidenceSourceResult } from '@/types/api'

interface EvidenceCardProps {
  evidence: EvidenceSourceResult
}

export function EvidenceCard({ evidence }: EvidenceCardProps) {
  const rel = evidenceRelConfig[evidence.relationship_to_claim as keyof typeof evidenceRelConfig]
    ?? evidenceRelConfig.inconclusive

  const badgeVariant = evidence.relationship_to_claim === 'supporting' ? 'success' :
    evidence.relationship_to_claim === 'contradicting' ? 'danger' :
    evidence.relationship_to_claim === 'inconclusive' ? 'warning' : 'muted'

  return (
    <div className="glass-card p-4 hover:border-slate-700/80 transition-colors">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1 min-w-0">
          <a
            href={evidence.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-slate-200 hover:text-brand-400 transition-colors line-clamp-2 group"
          >
            {evidence.title}
            <ExternalLink className="inline-block w-3 h-3 ml-1 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>
          <div className="text-xs text-slate-500 mt-1">{evidence.source_name}</div>
        </div>
        <Badge variant={badgeVariant} className="shrink-0">{rel.label}</Badge>
      </div>

      {evidence.snippet && (
        <p className="text-xs text-slate-500 leading-relaxed mb-3 line-clamp-3 italic">
          "{evidence.snippet}"
        </p>
      )}

      <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs text-slate-600">
        {evidence.published_at && (
          <span className="flex items-center gap-1">
            <Calendar className="w-3 h-3" />
            {formatDate(evidence.published_at)}
          </span>
        )}
        <span className="flex items-center gap-1">
          <BarChart3 className="w-3 h-3" />
          Relevance: {(evidence.relevance_score * 100).toFixed(0)}%
        </span>
      </div>

      {evidence.comparison_score > 0 && (
        <div className="mt-2 space-y-1">
          <div className="flex justify-between text-xs text-slate-600">
            <span>Claim similarity</span>
            <span className={cn(rel.color)}>{(evidence.comparison_score * 100).toFixed(0)}%</span>
          </div>
          <ProgressBar
            value={evidence.comparison_score}
            color={
              evidence.relationship_to_claim === 'supporting' ? 'bg-emerald-500' :
              evidence.relationship_to_claim === 'contradicting' ? 'bg-red-500' : 'bg-amber-500'
            }
          />
        </div>
      )}
    </div>
  )
}
