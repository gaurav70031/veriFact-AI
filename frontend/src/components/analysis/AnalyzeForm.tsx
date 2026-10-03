import { useState, useRef } from 'react'
import { FileText, Link2, MessageSquare, Loader2, AlertCircle } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Tabs } from '@/components/ui/Tabs'
import { useAnalyzeText, useAnalyzeUrl, useAnalyzeClaim } from '@/hooks/useApi'
import type { AnalysisResponse } from '@/types/api'

interface AnalyzeFormProps {
  onResult: (result: AnalysisResponse) => void
}

const TABS = [
  { id: 'text',  label: 'Text / Article',  icon: <FileText className="w-4 h-4" /> },
  { id: 'url',   label: 'URL',              icon: <Link2 className="w-4 h-4" /> },
  { id: 'claim', label: 'Claim',            icon: <MessageSquare className="w-4 h-4" /> },
]

export function AnalyzeForm({ onResult }: AnalyzeFormProps) {
  const [tab, setTab] = useState('text')
  const [text, setText]   = useState('')
  const [title, setTitle] = useState('')
  const [url, setUrl]     = useState('')
  const [claim, setClaim] = useState('')
  const [context, setContext] = useState('')

  const textMutation  = useAnalyzeText()
  const urlMutation   = useAnalyzeUrl()
  const claimMutation = useAnalyzeClaim()

  const isPending =
    textMutation.isPending || urlMutation.isPending || claimMutation.isPending

  const errorMsg =
    (textMutation.error as Error)?.message ||
    (urlMutation.error as Error)?.message ||
    (claimMutation.error as Error)?.message

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      let result: AnalysisResponse
      if (tab === 'text') {
        result = await textMutation.mutateAsync({ text, title: title || undefined })
      } else if (tab === 'url') {
        result = await urlMutation.mutateAsync({ url })
      } else {
        result = await claimMutation.mutateAsync({ claim, context: context || undefined })
      }
      onResult(result)
    } catch {
      // error shown below
    }
  }

  const isValid = tab === 'text' ? text.trim().length >= 20 :
    tab === 'url' ? url.trim().length > 10 :
    claim.trim().length >= 10

  return (
    <div className="glass-card p-6">
      <Tabs tabs={TABS} active={tab} onChange={setTab} className="mb-6" />

      <form onSubmit={handleSubmit} className="space-y-4">
        {tab === 'text' && (
          <>
            <div>
              <label className="label-sm block mb-1.5">Article Headline (optional)</label>
              <input
                className="input-field"
                placeholder="e.g. Scientists discover miracle cure…"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>
            <div>
              <label className="label-sm block mb-1.5">Article Text *</label>
              <textarea
                className="input-field resize-none"
                rows={8}
                placeholder="Paste the full article text here (minimum 20 characters)…"
                value={text}
                onChange={(e) => setText(e.target.value)}
                required
              />
              <div className="text-xs text-slate-600 mt-1 text-right">{text.length.toLocaleString()} chars</div>
            </div>
          </>
        )}

        {tab === 'url' && (
          <div>
            <label className="label-sm block mb-1.5">Article URL *</label>
            <input
              className="input-field"
              type="url"
              placeholder="https://www.example.com/article/…"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              required
            />
            <p className="text-xs text-slate-600 mt-1">
              The article is fetched server-side. Only public, non-paywalled pages can be extracted.
            </p>
          </div>
        )}

        {tab === 'claim' && (
          <>
            <div>
              <label className="label-sm block mb-1.5">Claim *</label>
              <input
                className="input-field"
                placeholder="e.g. The ECB raised rates by 25 basis points in June 2023."
                value={claim}
                onChange={(e) => setClaim(e.target.value)}
                required
              />
            </div>
            <div>
              <label className="label-sm block mb-1.5">Context (optional)</label>
              <textarea
                className="input-field resize-none"
                rows={4}
                placeholder="Surrounding paragraph or article excerpt…"
                value={context}
                onChange={(e) => setContext(e.target.value)}
              />
            </div>
          </>
        )}

        {errorMsg && (
          <div className="flex items-start gap-2.5 p-3.5 rounded-lg bg-red-500/10 border border-red-500/20">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            <p className="text-sm text-red-300">{errorMsg}</p>
          </div>
        )}

        <div className="flex items-center justify-between pt-2">
          <p className="text-xs text-slate-600">
            Analysis takes 5–30 seconds depending on content length.
          </p>
          <button
            type="submit"
            disabled={isPending || !isValid}
            className="btn-primary"
          >
            {isPending ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Analysing…
              </>
            ) : (
              'Analyse'
            )}
          </button>
        </div>
      </form>
    </div>
  )
}
