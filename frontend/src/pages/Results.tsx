/**
 * Results page
 *
 * Fetches from GET /api/v1/analysis/{id} (or uses router state for instant
 * display when navigating directly from the Analyze page).
 *
 * Sections rendered (only when the relevant data exists in the response):
 *  1.  Page header       — back link, analysis ID, input type, timestamps
 *  2.  Source metadata   — article title + URL (url submissions only)
 *  3.  Assessment Hero   — ML verdict + confidence gauge + evidence verdict
 *  4.  Model Comparison  — per-model probability bars + ensemble summary
 *  5.  Evidence Section  — sources grouped by relationship (supporting /
 *                          contradicting / inconclusive / not relevant)
 *                          with explicit "Insufficient Evidence" state
 *  6.  Claims Breakdown  — per-claim accordion: verdict, evidence, tokens,
 *                          source links, conflict warnings, limitations
 *  7.  Explainability    — token heatmaps (LIME / attention / tfidf),
 *                          signal-vs-evidence warning always visible
 *  8.  Report Footer     — timestamp, processing time, model versions,
 *                          limitations (dynamic + static)
 *
 * Data integrity rules enforced here:
 *  • No section is rendered when its data is absent/empty.
 *  • INSUFFICIENT_EVIDENCE is never displayed as "Fake" or "Unverified".
 *  • evidence_verdict=null renders "No Assessment", not a fabricated verdict.
 *  • All evidence sources come from the API; none are fabricated.
 */

import { useEffect } from 'react'
import { useParams, useLocation, Link } from 'react-router-dom'
import {
  ArrowLeft, ExternalLink, Clock, Hash,
  FileText, Link2, MessageSquareText,
} from 'lucide-react'
import { useAnalysis } from '@/hooks/useApi'
import { Spinner } from '@/components/ui/Spinner'
import { ErrorState } from '@/components/ui/ErrorState'
import { Badge } from '@/components/ui/Badge'
import { formatDateTime, formatMs } from '@/lib/utils'

import { AssessmentHero }       from '@/components/results/AssessmentHero'
import { ModelComparison }      from '@/components/results/ModelComparison'
import { EvidenceSection }      from '@/components/results/EvidenceSection'
import { ClaimsBreakdown }      from '@/components/results/ClaimsBreakdown'
import { ExplainabilitySection } from '@/components/results/ExplainabilitySection'
import { ReportFooter }         from '@/components/results/ReportFooter'

import type { AnalysisResponse, EvidenceSourceResult } from '@/types/api'

// ── Input type icon ───────────────────────────────────────────────────────────

function InputTypeIcon({ type }: { type: string }) {
  const cls = 'w-3.5 h-3.5'
  if (type === 'url')   return <Link2            className={cls} />
  if (type === 'claim') return <MessageSquareText className={cls} />
  return                        <FileText         className={cls} />
}

// ── Page header ───────────────────────────────────────────────────────────────

function PageHeader({ result }: { result: AnalysisResponse }) {
  const inputLabel =
    result.input_type === 'url'   ? 'URL'    :
    result.input_type === 'claim' ? 'Claim'  :
    'Text'

  return (
    <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
      <div>
        <Link
          to="/analyze"
          className="inline-flex items-center gap-1.5 text-sm text-slate-500
                     hover:text-slate-300 mb-4 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          New Analysis
        </Link>

        <h1 className="text-2xl font-bold text-slate-100 tracking-tight mb-2">
          Analysis Report
        </h1>

        {/* Meta chips */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 text-xs text-slate-600">
            <Hash className="w-3 h-3" />
            <span className="font-mono">{result.id}</span>
          </div>

          <Badge variant="info" className="flex items-center gap-1">
            <InputTypeIcon type={result.input_type} />
            {inputLabel}
          </Badge>

          <div className="flex items-center gap-1 text-xs text-slate-600">
            <Clock className="w-3 h-3" />
            <span>{formatDateTime(result.created_at)}</span>
          </div>

          {result.processing_time_ms !== undefined && (
            <div className="text-xs text-slate-700 font-mono">
              {formatMs(result.processing_time_ms)}
            </div>
          )}
        </div>
      </div>

      <Link
        to="/history"
        className="btn-secondary text-xs shrink-0 self-start sm:self-auto mt-1"
      >
        View History
      </Link>
    </div>
  )
}

// ── Source metadata card ──────────────────────────────────────────────────────

function SourceMeta({ result }: { result: AnalysisResponse }) {
  if (!result.article_title && !result.source_url) return null

  return (
    <div className="glass-card p-4 space-y-1.5">
      {result.article_title && (
        <p className="text-sm font-semibold text-slate-200 leading-snug">
          {result.article_title}
        </p>
      )}
      {result.source_url && (
        <a
          href={result.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-xs text-brand-400
                     hover:text-brand-300 transition-colors break-all"
        >
          <ExternalLink className="w-3 h-3 shrink-0" />
          {result.source_url}
        </a>
      )}
    </div>
  )
}

// ── Collect all evidence sources across all claims ────────────────────────────

function collectAllSources(result: AnalysisResponse): EvidenceSourceResult[] {
  const seen = new Set<string>()
  const out:  EvidenceSourceResult[] = []

  for (const claim of result.claims) {
    for (const src of claim.evidence_sources) {
      if (!seen.has(src.url)) {
        seen.add(src.url)
        out.push(src)
      }
    }
  }
  // Sort by relevance_score desc
  return out.sort((a, b) => b.relevance_score - a.relevance_score)
}

// ── Collect all evidence limitations ─────────────────────────────────────────

function collectLimitations(result: AnalysisResponse): string[] {
  const global = result.evidence_summary?.evidence_limitations ?? []
  const perClaim = result.claims.flatMap((c) => c.evidence_limitations ?? [])
  return [...new Set([...global, ...perClaim])]
}

// ── Check if any claim has a conflict ────────────────────────────────────────

function hasAnyConflict(result: AnalysisResponse): boolean {
  return result.claims.some((c) => c.assessment_conflict)
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Results() {
  const { id }       = useParams<{ id: string }>()
  const location     = useLocation()
  const numId        = id ? parseInt(id, 10) : null

  // If navigated from Analyze page, result is already in router state — use it
  // immediately (zero-latency display). If opened directly by URL, fetch it.
  const stateResult  = location.state?.result as AnalysisResponse | undefined
  const { data: fetched, isLoading, error, refetch } = useAnalysis(
    stateResult ? null : numId,
  )

  const result = stateResult ?? fetched

  // Scroll to top on mount
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' })
  }, [id])

  // ── Loading state ─────────────────────────────────────────────────────────

  if (!stateResult && isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-4">
        <Spinner size="lg" />
        <p className="text-slate-400 text-sm">Loading analysis…</p>
        <p className="text-slate-600 text-xs">Fetching from /api/v1/analysis/{id}</p>
      </div>
    )
  }

  // ── Error state ───────────────────────────────────────────────────────────

  if (!stateResult && error) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-24">
        <ErrorState
          title="Could not load analysis"
          message={(error as Error)?.message ?? 'Unknown error'}
          onRetry={() => refetch()}
        />
        <div className="mt-6 text-center">
          <Link to="/analyze" className="text-sm text-brand-400 hover:text-brand-300">
            ← Start a new analysis
          </Link>
        </div>
      </div>
    )
  }

  // ── No result (shouldn't happen but guards the render) ────────────────────

  if (!result) return null

  // ── Derived data ──────────────────────────────────────────────────────────

  const allSources  = collectAllSources(result)
  const limitations = collectLimitations(result)
  const anyConflict = hasAnyConflict(result)

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 animate-fade-in">

      {/* ① Page header */}
      <div className="mb-6">
        <PageHeader result={result} />
      </div>

      {/* ② Source metadata (URL / title) */}
      <div className="mb-6">
        <SourceMeta result={result} />
      </div>

      {/* ③ Assessment Hero — dual ML + evidence verdict */}
      <div className="mb-6">
        <AssessmentHero
          mlVerdict={result.ml_verdict}
          mlConfidence={result.ml_confidence}
          evidenceVerdict={result.evidence_verdict}
          evidenceExplanation={result.evidence_explanation}
          hasConflict={anyConflict}
        />
      </div>

      {/* ④ Model Comparison */}
      {result.model_predictions.length > 0 && (
        <div className="mb-6">
          <ModelComparison predictions={result.model_predictions} />
        </div>
      )}

      {/* ⑤ Evidence (grouped by relationship) */}
      <div className="mb-6">
        <EvidenceSection
          summary={result.evidence_summary}
          allSources={allSources}
        />
      </div>

      {/* ⑥ Claims breakdown */}
      {result.claims.length > 0 && (
        <div className="mb-6">
          <ClaimsBreakdown claims={result.claims} />
        </div>
      )}

      {/* ⑦ Explainability (lazy-fetches from /explanation/{id}) */}
      <div className="mb-6">
        <ExplainabilitySection analysisId={result.id} />
      </div>

      {/* ⑧ Report footer — metadata, model versions, limitations */}
      <div className="mb-6">
        <ReportFooter
          analysisId={result.id}
          createdAt={result.created_at}
          processingTimeMs={result.processing_time_ms}
          predictions={result.model_predictions}
          evidenceLimitations={limitations}
          analysisStatus={result.status}
        />
      </div>

    </div>
  )
}
