/**
 * Analyze page
 *
 * Three input modes, each with a distinct form:
 *   Claim   — single large text input for a factual claim
 *   Article — headline + full article body
 *   URL     — URL input with safety notes
 *
 * During submission, AnalysisProgress shows real stage advancement
 * tied to the actual in-flight API call.  On success the page
 * navigates to Results with the full response in router state.
 */

import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  MessageSquareText,
  Newspaper,
  Link2,
  ArrowRight,
  AlertCircle,
  Info,
  RotateCcw,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAnalyzeText, useAnalyzeUrl, useAnalyzeClaim } from '@/hooks/useApi'
import { AnalysisProgress } from '@/components/analysis/AnalysisProgress'
import type { AnalysisResponse } from '@/types/api'

// ── Mode types ────────────────────────────────────────────────────────────────

type Mode = 'claim' | 'article' | 'url'

const MODES: { id: Mode; label: string; icon: React.ReactNode; shortDesc: string }[] = [
  {
    id:        'claim',
    label:     'Claim',
    icon:      <MessageSquareText className="w-5 h-5" />,
    shortDesc: 'A short, checkable statement',
  },
  {
    id:        'article',
    label:     'Article',
    icon:      <Newspaper className="w-5 h-5" />,
    shortDesc: 'Full headline + body text',
  },
  {
    id:        'url',
    label:     'URL',
    icon:      <Link2 className="w-5 h-5" />,
    shortDesc: 'Public article link',
  },
]

// ── Shared constraints ────────────────────────────────────────────────────────

const CLAIM_MIN   = 10
const CLAIM_MAX   = 500
const ARTICLE_MIN = 20
const ARTICLE_MAX = 50_000

// ── Sub-forms ─────────────────────────────────────────────────────────────────

interface ClaimFormProps {
  value: string
  onChange: (v: string) => void
  context: string
  onContextChange: (v: string) => void
}

function ClaimForm({ value, onChange, context, onContextChange }: ClaimFormProps) {
  const remaining = CLAIM_MAX - value.length
  return (
    <div className="space-y-5">
      <div>
        <label className="label-sm block mb-2">
          Factual claim <span className="text-slate-600 normal-case tracking-normal text-xs ml-1">(required)</span>
        </label>
        <textarea
          className="input-field resize-none text-base leading-relaxed"
          rows={5}
          maxLength={CLAIM_MAX}
          placeholder={`Type or paste a checkable factual claim.\n\nExamples:\n• The WHO declared COVID-19 a pandemic in March 2020.\n• The Eiffel Tower was built in 1889.\n• 5G towers caused the spread of COVID-19.`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          autoFocus
        />
        <div className="flex justify-between mt-1.5">
          <span className="text-xs text-slate-600">
            {value.length < CLAIM_MIN && value.length > 0
              ? `${CLAIM_MIN - value.length} more characters needed`
              : ''}
          </span>
          <span className={cn('text-xs font-mono', remaining < 50 ? 'text-amber-400' : 'text-slate-600')}>
            {remaining}
          </span>
        </div>
      </div>

      <div>
        <label className="label-sm block mb-2">
          Context <span className="text-slate-600 normal-case tracking-normal text-xs ml-1">(optional — helps evidence search)</span>
        </label>
        <textarea
          className="input-field resize-none"
          rows={3}
          maxLength={5_000}
          placeholder="Surrounding paragraph, article excerpt, or additional context…"
          value={context}
          onChange={(e) => onContextChange(e.target.value)}
        />
      </div>
    </div>
  )
}

interface ArticleFormProps {
  headline: string
  onHeadlineChange: (v: string) => void
  body: string
  onBodyChange: (v: string) => void
}

function ArticleForm({ headline, onHeadlineChange, body, onBodyChange }: ArticleFormProps) {
  const charCount = body.length
  const pct = Math.min(charCount / ARTICLE_MAX, 1)
  return (
    <div className="space-y-5">
      <div>
        <label className="label-sm block mb-2">
          Headline <span className="text-slate-600 normal-case tracking-normal text-xs ml-1">(optional)</span>
        </label>
        <input
          className="input-field text-base font-medium"
          maxLength={500}
          placeholder="Article headline or title…"
          value={headline}
          onChange={(e) => onHeadlineChange(e.target.value)}
        />
      </div>

      <div>
        <label className="label-sm block mb-2">
          Article text <span className="text-slate-600 normal-case tracking-normal text-xs ml-1">(required — min 20 chars)</span>
        </label>
        <textarea
          className="input-field resize-y min-h-[220px] leading-relaxed"
          rows={10}
          maxLength={ARTICLE_MAX}
          placeholder="Paste the full article body here…"
          value={body}
          onChange={(e) => onBodyChange(e.target.value)}
          autoFocus
        />
        <div className="flex items-center gap-3 mt-1.5">
          {/* mini char-count bar */}
          <div className="flex-1 h-1 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-brand-600/60 rounded-full transition-all"
              style={{ width: `${pct * 100}%` }}
            />
          </div>
          <span className="text-xs font-mono text-slate-600 shrink-0">
            {charCount.toLocaleString()} / {ARTICLE_MAX.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  )
}

interface UrlFormProps {
  value: string
  onChange: (v: string) => void
}

function UrlForm({ value, onChange }: UrlFormProps) {
  const isHttp = value.startsWith('http://') || value.startsWith('https://')
  const isValid = isHttp && value.length > 15

  return (
    <div className="space-y-5">
      <div>
        <label className="label-sm block mb-2">
          Article URL <span className="text-slate-600 normal-case tracking-normal text-xs ml-1">(required)</span>
        </label>
        <div className="relative">
          <Link2 className={cn(
            'absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 transition-colors',
            isValid ? 'text-brand-400' : 'text-slate-600'
          )} />
          <input
            className={cn(
              'input-field pl-10 text-base font-mono',
              value.length > 0 && !isValid && 'border-red-500/40 focus:border-red-500'
            )}
            type="url"
            placeholder="https://www.example.com/article/…"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            autoFocus
            spellCheck={false}
          />
        </div>
        {value.length > 0 && !isHttp && (
          <p className="text-xs text-red-400 mt-1.5">URL must start with https:// or http://</p>
        )}
      </div>

      {/* Info cards */}
      <div className="grid sm:grid-cols-2 gap-3">
        {[
          {
            icon: '✓',
            color: 'text-emerald-400',
            bg:   'bg-emerald-500/5 border-emerald-500/20',
            items: [
              'Public news articles',
              'Open-access pages',
              'Blog posts without login',
            ],
          },
          {
            icon: '✕',
            color: 'text-red-400',
            bg:   'bg-red-500/5 border-red-500/20',
            items: [
              'Paywalled content',
              'Login-required pages',
              'JavaScript-only SPAs',
            ],
          },
        ].map((card) => (
          <div key={card.icon} className={cn('rounded-lg p-3.5 border', card.bg)}>
            <div className={cn('text-xs font-semibold mb-2', card.color)}>
              {card.icon === '✓' ? 'Works with' : 'Does not work with'}
            </div>
            <ul className="space-y-1">
              {card.items.map((item) => (
                <li key={item} className="text-xs text-slate-500 flex items-center gap-1.5">
                  <span className={cn('text-xs', card.color)}>{card.icon}</span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="flex items-start gap-2 p-3 rounded-lg bg-slate-800/50 border border-slate-800">
        <Info className="w-3.5 h-3.5 text-slate-500 shrink-0 mt-0.5" />
        <p className="text-xs text-slate-500">
          The article is fetched server-side via a secure SSRF-protected proxy.
          Private IP ranges, localhost, and non-HTTP/HTTPS schemes are blocked.
        </p>
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Analyze() {
  const navigate = useNavigate()

  // Mode
  const [mode, setMode] = useState<Mode>('claim')

  // Claim fields
  const [claim, setClaim]     = useState('')
  const [context, setContext] = useState('')

  // Article fields
  const [headline, setHeadline] = useState('')
  const [body, setBody]         = useState('')

  // URL field
  const [url, setUrl] = useState('')

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isDone, setIsDone]             = useState(false)
  const [submitError, setSubmitError]   = useState<string | null>(null)

  const textMutation  = useAnalyzeText()
  const urlMutation   = useAnalyzeUrl()
  const claimMutation = useAnalyzeClaim()

  // Compute isValid per mode
  const isValid =
    mode === 'claim'   ? claim.trim().length >= CLAIM_MIN :
    mode === 'article' ? body.trim().length  >= ARTICLE_MIN :
    /* url */            url.trim().length   > 15 && (url.startsWith('http://') || url.startsWith('https://'))

  const canSubmit = isValid && !isSubmitting

  // Reset to idle (after an error or to try again)
  const handleReset = () => {
    setIsSubmitting(false)
    setIsDone(false)
    setSubmitError(null)
    textMutation.reset()
    urlMutation.reset()
    claimMutation.reset()
  }

  const handleModeChange = (m: Mode) => {
    setMode(m)
    handleReset()
  }

  // Submit
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!canSubmit) return

    setIsSubmitting(true)
    setIsDone(false)
    setSubmitError(null)

    try {
      let result: AnalysisResponse

      if (mode === 'claim') {
        result = await claimMutation.mutateAsync({
          claim:   claim.trim(),
          context: context.trim() || undefined,
        })
      } else if (mode === 'article') {
        result = await textMutation.mutateAsync({
          text:  body.trim(),
          title: headline.trim() || undefined,
        })
      } else {
        result = await urlMutation.mutateAsync({ url: url.trim() })
      }

      // Mark done — progress bar completes, then navigate
      setIsDone(true)
      setTimeout(() => {
        navigate(`/results/${result.id}`, { state: { result } })
      }, 800)
    } catch (err) {
      setIsSubmitting(false)
      setSubmitError((err as Error).message ?? 'An unexpected error occurred.')
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8 py-12 animate-fade-in">

      {/* Page title */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-100 mb-2">Analyze Content</h1>
        <p className="text-slate-400 text-sm leading-relaxed">
          Submit a claim, article, or URL. The system runs multiple ML models,
          retrieves current evidence, and generates an explainable assessment.
        </p>
      </div>

      {/* Mode selector */}
      <div className="grid grid-cols-3 gap-3 mb-7">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => handleModeChange(m.id)}
            disabled={isSubmitting}
            className={cn(
              'relative flex flex-col items-center gap-2.5 px-4 py-5 rounded-xl border-2 transition-all duration-200 group',
              'focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              mode === m.id
                ? 'border-brand-500 bg-brand-500/10 text-brand-300'
                : 'border-slate-700/60 bg-slate-900/40 text-slate-500 hover:border-slate-600 hover:text-slate-300 hover:bg-slate-800/50',
            )}
          >
            <div className={cn(
              'transition-colors',
              mode === m.id ? 'text-brand-400' : 'text-slate-600 group-hover:text-slate-400'
            )}>
              {m.icon}
            </div>
            <div className="text-center">
              <div className={cn(
                'text-sm font-semibold leading-tight',
                mode === m.id ? 'text-brand-300' : 'text-slate-400'
              )}>
                {m.label}
              </div>
              <div className="text-xs text-slate-600 mt-0.5">{m.shortDesc}</div>
            </div>
            {mode === m.id && (
              <span className="absolute top-2.5 right-2.5 w-1.5 h-1.5 rounded-full bg-brand-400" />
            )}
          </button>
        ))}
      </div>

      {/* Progress view (during / after submission) */}
      {(isSubmitting || isDone || submitError) && (
        <div className="mb-6">
          <AnalysisProgress
            mode={mode}
            isRunning={isSubmitting && !isDone}
            isDone={isDone}
            error={submitError}
          />

          {/* Error recovery */}
          {submitError && (
            <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center gap-3 p-4 rounded-xl bg-red-500/8 border border-red-500/20">
              <div className="flex items-start gap-2.5 flex-1">
                <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <p className="text-sm text-red-300 leading-relaxed">{submitError}</p>
              </div>
              <button
                type="button"
                onClick={handleReset}
                className="btn-secondary text-xs shrink-0"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Try again
              </button>
            </div>
          )}
        </div>
      )}

      {/* Form — hidden while progress is showing (but not on error) */}
      {!isSubmitting && !isDone && (
        <form onSubmit={handleSubmit} noValidate>
          <div className="glass-card p-6 mb-5">
            {mode === 'claim' && (
              <ClaimForm
                value={claim}
                onChange={setClaim}
                context={context}
                onContextChange={setContext}
              />
            )}
            {mode === 'article' && (
              <ArticleForm
                headline={headline}
                onHeadlineChange={setHeadline}
                body={body}
                onBodyChange={setBody}
              />
            )}
            {mode === 'url' && (
              <UrlForm value={url} onChange={setUrl} />
            )}
          </div>

          {/* Submit row */}
          <div className="flex items-center justify-between gap-4">
            <p className="text-xs text-slate-600">
              {mode === 'url'
                ? 'Extraction + ML + evidence: ~15–30s'
                : mode === 'article'
                ? 'ML + evidence: ~10–25s depending on length'
                : 'ML + evidence: ~8–20s'}
            </p>
            <button
              type="submit"
              disabled={!canSubmit}
              className={cn(
                'btn-primary text-sm px-6 py-2.5',
                'disabled:opacity-40 disabled:cursor-not-allowed',
              )}
            >
              Analyze
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </form>
      )}

      {/* Disclaimer */}
      {!isSubmitting && !isDone && (
        <div className="mt-8 p-4 rounded-xl bg-slate-900/40 border border-slate-800/60 flex gap-3">
          <Info className="w-4 h-4 text-slate-600 shrink-0 mt-0.5" />
          <div className="text-xs text-slate-600 leading-relaxed space-y-1">
            <p>
              <span className="text-slate-500 font-medium">ML predictions</span> reflect statistical
              patterns from training data — not factual determinations. High confidence ≠ factual truth.
            </p>
            <p>
              <span className="text-slate-500 font-medium">Evidence assessments</span> use real retrieved
              sources. <span className="text-slate-500">Insufficient evidence ≠ false.</span>
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
