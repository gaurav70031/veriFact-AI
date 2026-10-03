/**
 * AnalysisProgress
 *
 * Displays real-time progress during an analysis submission.
 *
 * Stage design
 * ─────────────
 * The backend processes every request through the same pipeline stages:
 *   1. Extracting content    — URL only: fetch + trafilatura/newspaper3k
 *   2. Running NLP models    — TF-IDF baselines (LR, SVM, NB)
 *   3. Running transformer   — DistilBERT forward pass (chunk aggregation)
 *   4. Searching evidence    — NewsAPI + GNews + RSS + Search (concurrent)
 *   5. Comparing evidence    — lexical / semantic comparator per claim
 *   6. Generating explanation— LIME / attention attribution
 *   7. Finalising            — DB persist + response construction
 *
 * Because the API is a single POST (no streaming), the stage advances based on
 * `elapsed_ms` against per-stage expected durations. The final stage only
 * completes when the API promise resolves — it never auto-completes early.
 * If the API errors, the progress bar stops and the error is displayed.
 */

import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, Loader2, AlertCircle, XCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

// ── Stage definitions ─────────────────────────────────────────────────────────

interface Stage {
  id: string
  label: string
  detail: string
  /** expected wall-clock ms from submission start at which this stage begins */
  startsAt: number
}

const STAGES_URL: Stage[] = [
  { id: 'extract',     label: 'Extracting content',        detail: 'Fetching and parsing article from URL',            startsAt: 0 },
  { id: 'nlp',         label: 'Running NLP models',        detail: 'TF-IDF + Logistic Regression, SVM, Naive Bayes',   startsAt: 3_500 },
  { id: 'transformer', label: 'Running transformer',       detail: 'DistilBERT inference with long-article chunking',  startsAt: 5_000 },
  { id: 'evidence',    label: 'Searching current evidence', detail: 'Querying NewsAPI, GNews, RSS feeds and search',   startsAt: 7_000 },
  { id: 'compare',     label: 'Comparing evidence',        detail: 'Scoring source relevance and claim alignment',     startsAt: 12_000 },
  { id: 'explain',     label: 'Generating explanation',    detail: 'LIME attribution and attention weights',           startsAt: 15_000 },
  { id: 'finalise',    label: 'Finalising',                detail: 'Saving to database and building response',         startsAt: 17_000 },
]

const STAGES_TEXT: Stage[] = [
  { id: 'nlp',         label: 'Running NLP models',        detail: 'TF-IDF + Logistic Regression, SVM, Naive Bayes',   startsAt: 0 },
  { id: 'transformer', label: 'Running transformer',       detail: 'DistilBERT inference with long-article chunking',  startsAt: 1_500 },
  { id: 'evidence',    label: 'Searching current evidence', detail: 'Querying NewsAPI, GNews, RSS feeds and search',   startsAt: 3_500 },
  { id: 'compare',     label: 'Comparing evidence',        detail: 'Scoring source relevance and claim alignment',     startsAt: 8_000 },
  { id: 'explain',     label: 'Generating explanation',    detail: 'LIME attribution and attention weights',           startsAt: 11_000 },
  { id: 'finalise',    label: 'Finalising',                detail: 'Saving to database and building response',         startsAt: 13_000 },
]

const STAGES_CLAIM: Stage[] = [
  { id: 'nlp',         label: 'Running NLP models',        detail: 'TF-IDF + Logistic Regression, SVM, Naive Bayes',   startsAt: 0 },
  { id: 'transformer', label: 'Running transformer',       detail: 'DistilBERT inference on claim text',               startsAt: 1_000 },
  { id: 'evidence',    label: 'Searching current evidence', detail: 'Querying NewsAPI, GNews, RSS feeds and search',   startsAt: 2_500 },
  { id: 'compare',     label: 'Comparing evidence',        detail: 'Scoring source relevance and claim alignment',     startsAt: 7_000 },
  { id: 'explain',     label: 'Generating explanation',    detail: 'LIME attribution and attention weights',           startsAt: 9_500 },
  { id: 'finalise',    label: 'Finalising',                detail: 'Saving to database and building response',         startsAt: 11_000 },
]

function stagesFor(mode: 'claim' | 'article' | 'url'): Stage[] {
  if (mode === 'url')     return STAGES_URL
  if (mode === 'claim')   return STAGES_CLAIM
  return STAGES_TEXT
}

// ── Status types ──────────────────────────────────────────────────────────────

type StageStatus = 'pending' | 'active' | 'done'

interface StageState {
  status: StageStatus
  startedAt?: number
}

// ── Component ─────────────────────────────────────────────────────────────────

interface AnalysisProgressProps {
  mode: 'claim' | 'article' | 'url'
  /** true while the API call is in flight */
  isRunning: boolean
  /** set when the API call completed (success) */
  isDone: boolean
  /** set when the API call failed */
  error?: string | null
}

export function AnalysisProgress({ mode, isRunning, isDone, error }: AnalysisProgressProps) {
  const stages = stagesFor(mode)
  const [activeIndex, setActiveIndex] = useState(0)
  const [stageStates, setStageStates] = useState<StageState[]>(
    stages.map(() => ({ status: 'pending' as StageStatus }))
  )
  const startRef   = useRef<number>(Date.now())
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Reset when a new submission starts
  useEffect(() => {
    if (!isRunning && !isDone && !error) return

    startRef.current = Date.now()
    setActiveIndex(0)
    setStageStates(stages.map(() => ({ status: 'pending' })))
  }, [isRunning]) // eslint-disable-line react-hooks/exhaustive-deps

  // Advance stages based on elapsed time
  useEffect(() => {
    if (!isRunning) {
      if (intervalRef.current) clearInterval(intervalRef.current)
      return
    }

    intervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startRef.current

      setStageStates((prev) => {
        const next = [...prev]
        let newActive = 0

        // Find which stage we're currently in based on elapsed time
        for (let i = 0; i < stages.length; i++) {
          const stage = stages[i]
          const nextStage = stages[i + 1]

          if (elapsed >= stage.startsAt) {
            // Mark all earlier stages as done
            if (i > 0) {
              for (let j = 0; j < i; j++) {
                next[j] = { status: 'done', startedAt: next[j].startedAt }
              }
            }
            // This stage is active unless it's the last one
            // (last stage only completes when isDone)
            const isLast = i === stages.length - 1
            if (!isLast) {
              // Check if we're past the next stage's start time
              if (nextStage && elapsed >= nextStage.startsAt) {
                next[i] = { status: 'done', startedAt: next[i].startedAt ?? Date.now() }
              } else {
                next[i] = { status: 'active', startedAt: next[i].startedAt ?? Date.now() }
                newActive = i
              }
            } else {
              // Last stage stays active until isDone
              next[i] = { status: 'active', startedAt: next[i].startedAt ?? Date.now() }
              newActive = i
            }
          }
        }

        setActiveIndex(newActive)
        return next
      })
    }, 200)

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [isRunning, stages])

  // When done: mark all stages complete
  useEffect(() => {
    if (!isDone) return
    if (intervalRef.current) clearInterval(intervalRef.current)
    setStageStates(stages.map(() => ({ status: 'done' })))
    setActiveIndex(stages.length - 1)
  }, [isDone, stages])

  // When error: stop the ticker
  useEffect(() => {
    if (!error) return
    if (intervalRef.current) clearInterval(intervalRef.current)
  }, [error])

  // Overall progress bar percentage
  const doneCount  = stageStates.filter((s) => s.status === 'done').length
  const activeCount = stageStates.filter((s) => s.status === 'active').length
  const progressPct = isDone
    ? 100
    : Math.round(((doneCount + activeCount * 0.5) / stages.length) * 100)

  return (
    <div className="glass-card p-7 animate-fade-in">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm font-semibold text-slate-200">
            {error ? 'Analysis failed' : isDone ? 'Analysis complete' : 'Analysing…'}
          </span>
          <span className={cn(
            'text-xs font-mono font-medium',
            error ? 'text-red-400' : isDone ? 'text-emerald-400' : 'text-brand-400'
          )}>
            {progressPct}%
          </span>
        </div>

        {/* Progress bar */}
        <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
          <div
            className={cn(
              'h-full rounded-full transition-all duration-500',
              error   ? 'bg-red-500' :
              isDone  ? 'bg-emerald-500' :
              'bg-brand-500'
            )}
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* Stage list */}
      <div className="space-y-1">
        {stages.map((stage, i) => {
          const state = stageStates[i]
          // If there's an error, the active stage shows as failed
          const isErrorStage = !!error && state.status === 'active'

          return (
            <div
              key={stage.id}
              className={cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-300',
                state.status === 'active' && !isErrorStage && 'bg-brand-500/8 border border-brand-500/20',
                state.status === 'done'   && 'opacity-60',
                state.status === 'pending'&& 'opacity-30',
                isErrorStage             && 'bg-red-500/8 border border-red-500/20',
              )}
            >
              {/* Icon */}
              <div className="w-5 h-5 shrink-0 flex items-center justify-center">
                {isErrorStage ? (
                  <XCircle className="w-4.5 h-4.5 text-red-400" />
                ) : state.status === 'done' ? (
                  <CheckCircle2 className="w-4.5 h-4.5 text-emerald-500" />
                ) : state.status === 'active' ? (
                  <Loader2 className="w-4 h-4 text-brand-400 animate-spin" />
                ) : (
                  <div className="w-2 h-2 rounded-full bg-slate-700" />
                )}
              </div>

              {/* Label + detail */}
              <div className="flex-1 min-w-0">
                <div className={cn(
                  'text-sm font-medium leading-tight',
                  isErrorStage             ? 'text-red-300' :
                  state.status === 'active'? 'text-slate-100' :
                  state.status === 'done'  ? 'text-slate-400' :
                  'text-slate-600'
                )}>
                  {stage.label}
                </div>
                {(state.status === 'active' || isErrorStage) && (
                  <div className={cn(
                    'text-xs mt-0.5',
                    isErrorStage ? 'text-red-400' : 'text-slate-500'
                  )}>
                    {isErrorStage ? error : stage.detail}
                  </div>
                )}
              </div>

              {/* Stage index badge */}
              <div className={cn(
                'text-xs font-mono shrink-0',
                state.status === 'done'   ? 'text-emerald-600' :
                state.status === 'active' ? 'text-brand-500'  :
                'text-slate-700'
              )}>
                {String(i + 1).padStart(2, '0')}
              </div>
            </div>
          )
        })}
      </div>

      {/* Bottom message */}
      {!error && !isDone && (
        <p className="mt-5 text-xs text-slate-600 text-center">
          This typically takes 10–30 seconds depending on article length and evidence availability.
        </p>
      )}
      {isDone && (
        <p className="mt-5 text-xs text-emerald-600 text-center font-medium">
          ✓ Redirecting to results…
        </p>
      )}
    </div>
  )
}
