import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import * as api from '@/lib/api'
import type {
  AnalyzeTextRequest,
  AnalyzeUrlRequest,
  AnalyzeClaimRequest,
} from '@/types/api'

// ── Query keys ────────────────────────────────────────────────────────────────

export const KEYS = {
  health:       ['health']           as const,
  stats:        ['stats']            as const,
  models:       ['models']           as const,
  performance:  ['model-performance'] as const,
  history:      (p: object) => ['history', p] as const,
  analysis:     (id: number) => ['analysis', id] as const,
  explanation:  (id: number) => ['explanation', id] as const,
  evidence:     (id: number) => ['evidence', id] as const,
}

// ── Health ────────────────────────────────────────────────────────────────────

export function useHealth() {
  return useQuery({
    queryKey: KEYS.health,
    queryFn: api.getHealth,
    staleTime: 10_000,
    refetchInterval: 30_000,
  })
}

// ── Stats ─────────────────────────────────────────────────────────────────────

export function useStats() {
  return useQuery({
    queryKey: KEYS.stats,
    queryFn: api.getStats,
    staleTime: 60_000,
  })
}

// ── Models ────────────────────────────────────────────────────────────────────

export function useModels() {
  return useQuery({
    queryKey: KEYS.models,
    queryFn: api.getModels,
    staleTime: 5 * 60_000,
  })
}

export function useModelPerformance() {
  return useQuery({
    queryKey: KEYS.performance,
    queryFn: api.getModelPerformance,
    staleTime: 5 * 60_000,
  })
}

// ── History ───────────────────────────────────────────────────────────────────

export function useHistory(params: {
  page?: number
  page_size?: number
  verdict?: string
  input_type?: string
}) {
  return useQuery({
    queryKey: KEYS.history(params),
    queryFn: () => api.getHistory(params),
    staleTime: 30_000,
    placeholderData: (prev) => prev,
  })
}

// ── Single analysis ───────────────────────────────────────────────────────────

export function useAnalysis(id: number | null) {
  return useQuery({
    queryKey: KEYS.analysis(id!),
    queryFn: () => api.getAnalysisById(id!),
    enabled: id !== null,
    staleTime: 60_000,
  })
}

// ── Explanation ───────────────────────────────────────────────────────────────

export function useExplanation(analysisId: number | null) {
  return useQuery({
    queryKey: KEYS.explanation(analysisId!),
    queryFn: () => api.getExplanation(analysisId!),
    enabled: analysisId !== null,
    staleTime: 5 * 60_000,
  })
}

// ── Mutations ─────────────────────────────────────────────────────────────────

export function useAnalyzeText() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: AnalyzeTextRequest) => api.analyzeText(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['history'] })
      qc.invalidateQueries({ queryKey: KEYS.stats })
    },
  })
}

export function useAnalyzeUrl() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: AnalyzeUrlRequest) => api.analyzeUrl(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['history'] })
      qc.invalidateQueries({ queryKey: KEYS.stats })
    },
  })
}

export function useAnalyzeClaim() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: AnalyzeClaimRequest) => api.analyzeClaim(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['history'] })
      qc.invalidateQueries({ queryKey: KEYS.stats })
    },
  })
}

export function useSearchEvidence() {
  return useMutation({
    mutationFn: api.searchEvidence,
  })
}
