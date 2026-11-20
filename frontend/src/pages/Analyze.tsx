/**
 * Analyze page — unified entry for Text Analysis, URL Verification, and Claim checks.
 * Route: /analyze/text  |  /analyze/url  |  /analyze/claim (all load this component,
 *   the mode is set from the URL segment passed via defaultMode prop or the path).
 *
 * This page is also reached via /analyze (falls back to 'claim' mode).
 */

import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  MessageSquareText, Newspaper, Link2, ArrowRight,
  AlertCircle, Info, RotateCcw, Shield,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAnalyzeText, useAnalyzeUrl, useAnalyzeClaim } from '@/hooks/useApi'
import { AnalysisProgress } from '@/components/analysis/AnalysisProgress'
import type { AnalysisResponse } from '@/types/api'

type Mode = 'claim' | 'article' | 'url'

const MODES: { id: Mode; label: string; icon: React.ReactNode; desc: string; path: string }[] = [
  { id: 'claim',   label: 'Claim',       icon: <MessageSquareText className="w-5 h-5" />, desc: 'Short checkable statement', path: '/analyze/text' },
  { id: 'article', label: 'Article',     icon: <Newspaper         className="w-5 h-5" />, desc: 'Full headline + body',      path: '/analyze/text' },
  { id: 'url',     label: 'URL',         icon: <Link2             className="w-5 h-5" />, desc: 'Public article link',       path: '/analyze/url'  },
]

const CLAIM_MIN = 10;  const CLAIM_MAX = 500
const ARTICLE_MIN = 20; const ARTICLE_MAX = 50_000

function ClaimForm({ value, onChange, context, onContextChange }: {
  value: string; onChange: (v: string) => void
  context: string; onContextChange: (v: string) => void
}) {
  const remaining = CLAIM_MAX - value.length
  return (
    <div className="space-y-5">
      <div>
        <label className="label-sm block mb-2">
          Factual claim
          <span className="ml-2 normal-case tracking-normal text-xs" style={{ color: 'var(--text-faint)' }}>required</span>
        </label>
        <textarea
          className="input-field resize-none text-sm leading-relaxed"
          rows={5}
          maxLength={CLAIM_MAX}
          placeholder={`A checkable factual statement.\n\nExamples:\n• The WHO declared COVID-19 a pandemic in March 2020.\n• 5G towers caused the spread of COVID-19.`}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          autoFocus
        />
        <div className="flex justify-between mt-1.5">
          <span className="text-xs" style={{ color: 'var(--text-faint)' }}>
            {value.length > 0 && value.length < CLAIM_MIN ? `${CLAIM_MIN - value.length} more chars needed` : ''}
          </span>
          <span className={cn('text-xs font-mono', remaining < 50 ? 'text-amber-400' : '')} style={{ color: remaining < 50 ? undefined : 'var(--text-faint)' }}>
            {remaining}
          </span>
        </div>
      </div>
      <div>
        <label className="label-sm block mb-2">
          Context
          <span className="ml-2 normal-case tracking-normal text-xs" style={{ color: 'var(--text-faint)' }}>optional</span>
        </label>
        <textarea
          className="input-field resize-none"
          rows={3}
          maxLength={5000}
          placeholder="Surrounding paragraph or additional context…"
          value={context}
          onChange={(e) => onContextChange(e.target.value)}
        />
      </div>
    </div>
  )
}

function ArticleForm({ headline, onHeadlineChange, body, onBodyChange }: {
  headline: string; onHeadlineChange: (v: string) => void
  body: string; onBodyChange: (v: string) => void
}) {
  const pct = Math.min(body.length / ARTICLE_MAX, 1)
  return (
    <div className="space-y-5">
      <div>
        <label className="label-sm block mb-2">
          Headline
          <span className="ml-2 normal-case tracking-normal text-xs" style={{ color: 'var(--text-faint)' }}>optional</span>
        </label>
        <input
          className="input-field text-sm font-medium"
          maxLength={500}
          placeholder="Article headline or title…"
          value={headline}
          onChange={(e) => onHeadlineChange(e.target.value)}
        />
      </div>
      <div>
        <label className="label-sm block mb-2">
          Article text
          <span className="ml-2 normal-case tracking-normal text-xs" style={{ color: 'var(--text-faint)' }}>required · min 20 chars</span>
        </label>
        <textarea
          className="input-field resize-y min-h-[200px] leading-relaxed text-sm"
          rows={10}
          maxLength={ARTICLE_MAX}
          placeholder="Paste the full article body here…"
          value={body}
          onChange={(e) => onBodyChange(e.target.value)}
          autoFocus
        />
        <div className="flex items-center gap-3 mt-1.5">
          <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
            <div className="h-full rounded-full transition-all" style={{ width: `${pct * 100}%`, backgroundColor: 'var(--brand)' }} />
          </div>
          <span className="text-xs font-mono shrink-0" style={{ color: 'var(--text-faint)' }}>
            {body.length.toLocaleString()} / {ARTICLE_MAX.toLocaleString()}
          </span>
        </div>
      </div>
    </div>
  )
}

function UrlForm({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const isHttp  = value.startsWith('http://') || value.startsWith('https://')
  const isValid = isHttp && value.length > 15
  return (
    <div className="space-y-5">
      <div>
        <label className="label-sm block mb-2">Article URL</label>
        <div className="relative">
          <Link2 className={cn(
            'absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 transition-colors',
            isValid ? 'text-brand-400' : ''
          )} style={{ color: isValid ? 'var(--brand)' : 'var(--text-faint)' }} />
          <input
            className={cn('input-field pl-10 text-sm font-mono', value.length > 0 && !isValid && 'border-red-500/40')}
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

      <div className="grid sm:grid-cols-2 gap-3">
        {[
          { icon: '✓', label: 'Works with',       color: 'text-emerald-400', bg: 'bg-emerald-500/5 border-emerald-500/20',
            items: ['Public news articles', 'Open-access pages', 'Blog posts (no login)'] },
          { icon: '✕', label: 'Does not work with', color: 'text-red-400', bg: 'bg-red-500/5 border-red-500/20',
            items: ['Paywalled content', 'Login-required pages', 'JavaScript-only SPAs'] },
        ].map((card) => (
          <div key={card.icon} className={cn('rounded-xl p-3.5 border', card.bg)}>
            <div className={cn('text-xs font-semibold mb-2', card.color)}>{card.label}</div>
            <ul className="space-y-1">
              {card.items.map((item) => (
                <li key={item} className="text-xs flex items-center gap-1.5" style={{ color: 'var(--text-subtle)' }}>
                  <span className={card.color}>{card.icon}</span>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="flex items-start gap-2 p-3 rounded-xl" style={{ backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)' }}>
        <Info className="w-3.5 h-3.5 shrink-0 mt-0.5" style={{ color: 'var(--text-subtle)' }} />
        <p className="text-xs" style={{ color: 'var(--text-subtle)' }}>
          Fetched server-side via a secure SSRF-protected proxy.
          Private IP ranges, localhost, and non-HTTP/HTTPS schemes are blocked.
        </p>
      </div>
    </div>
  )
}

// ── Page titles per mode ──────────────────────────────────────────────────────

const PAGE_META = {
  claim:   { title: 'Text Analysis',    sub: 'Submit a factual claim for ML + evidence verification.' },
  article: { title: 'Text Analysis',    sub: 'Submit an article for ML + evidence verification.' },
  url:     { title: 'URL Verification', sub: 'Paste a public article URL for extraction + ML + evidence analysis.' },
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Analyze() {
  const navigate  = useNavigate()
  const location  = useLocation()

  // Derive initial mode from URL path
  const initMode: Mode = location.pathname.includes('/url') ? 'url' : 'claim'
  const [mode, setMode] = useState<Mode>(initMode)

  // Keep mode in sync with URL navigation
  useEffect(() => {
    setMode(location.pathname.includes('/url') ? 'url' : 'claim')
  }, [location.pathname])

  const [claim,    setClaim]    = useState('')
  const [context,  setContext]  = useState('')
  const [headline, setHeadline] = useState('')
  const [body,     setBody]     = useState('')
  const [url,      setUrl]      = useState('')

  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isDone,       setIsDone]       = useState(false)
  const [submitError,  setSubmitError]  = useState<string | null>(null)

  const textMutation  = useAnalyzeText()
  const urlMutation   = useAnalyzeUrl()
  const claimMutation = useAnalyzeClaim()

  const isValid =
    mode === 'claim'   ? claim.trim().length >= CLAIM_MIN :
    mode === 'article' ? body.trim().length  >= ARTICLE_MIN :
    url.trim().length > 15 && (url.startsWith('http://') || url.startsWith('https://'))

  const canSubmit = isValid && !isSubmitting

  const handleReset = () => {
    setIsSubmitting(false); setIsDone(false); setSubmitError(null)
    textMutation.reset(); urlMutation.reset(); claimMutation.reset()
  }

  const handleModeChange = (m: Mode) => { setMode(m); handleReset() }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!canSubmit) return
    setIsSubmitting(true); setIsDone(false); setSubmitError(null)
    try {
      let result: AnalysisResponse
      if (mode === 'claim') {
        result = await claimMutation.mutateAsync({ claim: claim.trim(), context: context.trim() || undefined })
      } else if (mode === 'article') {
        result = await textMutation.mutateAsync({ text: body.trim(), title: headline.trim() || undefined })
      } else {
        result = await urlMutation.mutateAsync({ url: url.trim() })
      }
      setIsDone(true)
      setTimeout(() => navigate(`/results/${result.id}`, { state: { result } }), 800)
    } catch (err) {
      setIsSubmitting(false)
      setSubmitError((err as Error).message ?? 'An unexpected error occurred.')
    }
  }

  const meta = PAGE_META[mode]

  return (
    <div className="page-content max-w-3xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <h1 className="page-title">{meta.title}</h1>
        </div>
        <p className="text-sm ml-14" style={{ color: 'var(--text-subtle)' }}>{meta.sub}</p>
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
              'flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all duration-200',
              'disabled:opacity-40 disabled:cursor-not-allowed',
              mode === m.id
                ? 'border-brand-500/60 text-white'
                : 'hover:border-brand-500/30',
            )}
            style={{
              borderColor: mode === m.id ? 'var(--brand)' : 'var(--border)',
              backgroundColor: mode === m.id ? 'var(--brand-light)' : 'var(--bg-surface)',
            }}
          >
            <div style={{ color: mode === m.id ? 'var(--brand)' : 'var(--text-faint)' }}>{m.icon}</div>
            <div className="text-center">
              <div className="text-sm font-semibold" style={{ color: mode === m.id ? 'var(--brand)' : 'var(--text-secondary)' }}>
                {m.label}
              </div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-faint)' }}>{m.desc}</div>
            </div>
          </button>
        ))}
      </div>

      {/* Progress */}
      {(isSubmitting || isDone || submitError) && (
        <div className="mb-6">
          <AnalysisProgress mode={mode} isRunning={isSubmitting && !isDone} isDone={isDone} error={submitError} />
          {submitError && (
            <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center gap-3 p-4 rounded-xl" style={{ backgroundColor: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)' }}>
              <div className="flex items-start gap-2.5 flex-1">
                <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <p className="text-sm text-red-400">{submitError}</p>
              </div>
              <button type="button" onClick={handleReset} className="btn-secondary text-xs shrink-0">
                <RotateCcw className="w-3.5 h-3.5" /> Try again
              </button>
            </div>
          )}
        </div>
      )}

      {/* Form */}
      {!isSubmitting && !isDone && (
        <form onSubmit={handleSubmit} noValidate>
          <div className="card p-6 mb-5">
            {mode === 'claim'   && <ClaimForm value={claim} onChange={setClaim} context={context} onContextChange={setContext} />}
            {mode === 'article' && <ArticleForm headline={headline} onHeadlineChange={setHeadline} body={body} onBodyChange={setBody} />}
            {mode === 'url'     && <UrlForm value={url} onChange={setUrl} />}
          </div>

          <div className="flex items-center justify-between gap-4">
            <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
              {mode === 'url' ? 'Extraction + ML + evidence: ~15–30s' :
               mode === 'article' ? 'ML + evidence: ~10–25s' :
               'ML + evidence: ~8–20s'}
            </p>
            <button type="submit" disabled={!canSubmit} className="btn-primary">
              Analyze
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </form>
      )}

      {/* Disclaimer */}
      {!isSubmitting && !isDone && (
        <div className="mt-8 p-4 rounded-xl flex gap-3" style={{ backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)' }}>
          <Info className="w-4 h-4 shrink-0 mt-0.5" style={{ color: 'var(--text-faint)' }} />
          <div className="text-xs leading-relaxed space-y-1" style={{ color: 'var(--text-subtle)' }}>
            <p>
              <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>ML predictions</span>{' '}
              reflect statistical patterns from training data — not factual determinations.
            </p>
            <p>
              <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>Evidence assessments</span>{' '}
              use real retrieved sources. Insufficient evidence ≠ false.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
