import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Info } from 'lucide-react'
import { AnalyzeForm } from '@/components/analysis/AnalyzeForm'
import type { AnalysisResponse } from '@/types/api'

export default function Analyze() {
  const navigate = useNavigate()

  const handleResult = (result: AnalysisResponse) => {
    navigate(`/results/${result.id}`, { state: { result } })
  }

  return (
    <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-12 animate-fade-in">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-100 mb-2">Analyze Content</h1>
        <p className="text-slate-400">
          Submit an article, URL, or factual claim for fake-news analysis.
        </p>
      </div>

      <AnalyzeForm onResult={handleResult} />

      {/* Disclaimer */}
      <div className="mt-6 p-4 rounded-xl bg-slate-900/50 border border-slate-800 flex gap-3">
        <Info className="w-4 h-4 text-brand-400 shrink-0 mt-0.5" />
        <div className="text-xs text-slate-500 leading-relaxed space-y-1">
          <p>
            <strong className="text-slate-400">Model predictions</strong> are probabilistic estimates based on
            statistical patterns — not factual determinations. They reflect training data, not ground truth.
          </p>
          <p>
            <strong className="text-slate-400">Evidence assessments</strong> are derived from real retrieved sources.
            Insufficient evidence does not mean a claim is false.
          </p>
        </div>
      </div>
    </div>
  )
}
