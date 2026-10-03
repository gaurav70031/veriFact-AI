import { useParams, useLocation, Link } from 'react-router-dom'
import { ArrowLeft, ExternalLink, Clock, Database } from 'lucide-react'
import { useAnalysis } from '@/hooks/useApi'
import { VerdictCard } from '@/components/analysis/VerdictCard'
import { ModelPredictionsTable } from '@/components/analysis/ModelPredictionsTable'
import { ClaimSection } from '@/components/analysis/ClaimSection'
import { ExplanationPanel } from '@/components/analysis/ExplanationPanel'
import { Spinner } from '@/components/ui/Spinner'
import { ErrorState } from '@/components/ui/ErrorState'
import { Badge } from '@/components/ui/Badge'
import { formatDateTime, formatMs, cn } from '@/lib/utils'
import type { AnalysisResponse } from '@/types/api'

export default function Results() {
  const { id } = useParams<{ id: string }>()
  const location = useLocation()
  const numId = id ? parseInt(id, 10) : null

  // Use navigation state if available (instant display), otherwise fetch
  const stateResult = location.state?.result as AnalysisResponse | undefined
  const { data: fetched, isLoading, error } = useAnalysis(
    stateResult ? null : numId
  )

  const result = stateResult ?? fetched

  if (!stateResult && isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-4">
        <Spinner size="lg" />
        <p className="text-slate-400 text-sm">Loading analysis result…</p>
      </div>
    )
  }

  if (!stateResult && error) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-24">
        <ErrorState
          title="Could not load analysis"
          message={(error as Error).message}
        />
      </div>
    )
  }

  if (!result) return null

  const inputLabel = result.input_type === 'url' ? 'URL' :
    result.input_type === 'claim' ? 'Claim' : 'Text'

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fade-in space-y-6">
      {/* Back + header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <Link
            to="/analyze"
            className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-300 mb-4 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            New Analysis
          </Link>
          <h1 className="text-2xl font-bold text-slate-100">Analysis Report</h1>
          <div className="flex flex-wrap items-center gap-3 mt-2">
            <Badge variant="muted">#{result.id}</Badge>
            <Badge variant="info">{inputLabel}</Badge>
            <span className="text-xs text-slate-600 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatDateTime(result.created_at)}
            </span>
            {result.processing_time_ms && (
              <span className="text-xs text-slate-600 flex items-center gap-1">
                <Database className="w-3 h-3" />
                {formatMs(result.processing_time_ms)}
              </span>
            )}
          </div>
        </div>
        <Link
          to={`/history`}
          className="btn-secondary text-xs shrink-0"
        >
          View History
        </Link>
      </div>

      {/* Source info */}
      {(result.article_title || result.source_url) && (
        <div className="glass-card p-4">
          {result.article_title && (
            <p className="text-sm font-medium text-slate-200 mb-1">{result.article_title}</p>
          )}
          {result.source_url && (
            <a
              href={result.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-brand-400 hover:text-brand-300 flex items-center gap-1 break-all"
            >
              {result.source_url}
              <ExternalLink className="w-3 h-3 shrink-0" />
            </a>
          )}
        </div>
      )}

      {/* Main verdict */}
      <VerdictCard
        mlVerdict={result.ml_verdict}
        mlConfidence={result.ml_confidence}
        evidenceVerdict={result.evidence_verdict}
        evidenceExplanation={result.evidence_explanation}
        processingTimeMs={result.processing_time_ms}
      />

      {/* Per-model table */}
      <ModelPredictionsTable predictions={result.model_predictions} />

      {/* Explainability */}
      <ExplanationPanel analysisId={result.id} />

      {/* Claim-level breakdown */}
      <ClaimSection claims={result.claims} />

      {/* Evidence summary footer */}
      {result.evidence_summary && (
        <div className="glass-card p-5">
          <h3 className="section-title mb-4">Evidence Summary</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
            {[
              { label: 'Supporting',    value: result.evidence_summary.supporting_count,    color: 'text-emerald-400' },
              { label: 'Contradicting', value: result.evidence_summary.contradicting_count, color: 'text-red-400' },
              { label: 'Inconclusive',  value: result.evidence_summary.inconclusive_count,  color: 'text-amber-400' },
              { label: 'Not Relevant',  value: result.evidence_summary.not_relevant_count,  color: 'text-slate-500' },
            ].map((s) => (
              <div key={s.label} className="text-center">
                <div className={cn('text-2xl font-bold', s.color)}>{s.value}</div>
                <div className="text-xs text-slate-500 mt-0.5">{s.label}</div>
              </div>
            ))}
          </div>
          {result.evidence_summary.providers_used.length > 0 && (
            <div className="text-xs text-slate-600">
              Providers: {result.evidence_summary.providers_used.join(', ')}
            </div>
          )}
          {result.evidence_summary.all_providers_failed && (
            <div className="mt-2 text-xs text-amber-500">
              ⚠ All evidence providers were unavailable. Evidence assessment could not be performed.
            </div>
          )}
          {result.evidence_summary.evidence_limitations.map((lim, i) => (
            <div key={i} className="mt-1 text-xs text-slate-600">• {lim}</div>
          ))}
        </div>
      )}
    </div>
  )
}
