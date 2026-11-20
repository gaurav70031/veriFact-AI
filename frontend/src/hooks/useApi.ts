import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import * as api from '@/lib/api'
import type {
  AnalyzeTextRequest,
  AnalyzeUrlRequest,
  AnalyzeClaimRequest,
  NewsSearchParams,
  RegisterRequest,
  LoginRequest,
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
  search?: string
  date_from?: string
  date_to?: string
  model_id?: string
  confidence_min?: number
  confidence_max?: number
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

export function useAnnotatedText(analysisId: number | null) {
  return useQuery({
    queryKey: ['annotated', analysisId],
    queryFn:  () => api.getAnnotatedText(analysisId!),
    enabled:  analysisId !== null,
    staleTime: 5 * 60_000,
  })
}

export function useExplanationSummary(analysisId: number | null) {
  return useQuery({
    queryKey: ['explanation-summary', analysisId],
    queryFn:  () => api.getExplanationSummary(analysisId!),
    enabled:  analysisId !== null,
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

// ── Live news search (GET — URL-shareable) ────────────────────────────────────

export function useNewsSearch(params: NewsSearchParams | null) {
  return useQuery({
    queryKey: ['news-search', params],
    queryFn:  () => api.searchNews(params!),
    enabled:  params !== null && params.q.trim().length >= 3,
    staleTime: 2 * 60_000,   // 2 min — news results go stale quickly
    gcTime:    5 * 60_000,
    retry:     1,
  })
}

// ── Auth mutations ─────────────────────────────────────────────────────────────
// These wrap the API calls but AuthContext is the source of truth for user state.
// Use useAuth() directly for login/logout in components.

export function useLoginMutation() {
  return useMutation({ mutationFn: (body: LoginRequest) => api.authLogin(body) })
}

export function useRegisterMutation() {
  return useMutation({ mutationFn: (body: RegisterRequest) => api.authRegister(body) })
}

export function useLogoutMutation() {
  return useMutation({ mutationFn: api.authLogout })
}

// ── Dataset hooks ─────────────────────────────────────────────────────────────

export function useDatasetCatalog() {
  return useQuery({
    queryKey: ['dataset-catalog'],
    queryFn:  api.getDatasetCatalog,
    staleTime: 60_000,
  })
}

export function useDatasetStats(name: string) {
  return useQuery({
    queryKey: ['dataset-stats', name],
    queryFn:  () => api.getDatasetStats(name),
    staleTime: 2 * 60_000,
  })
}

export function useDatasetAnalytics(name: string, split = 'train') {
  return useQuery({
    queryKey: ['dataset-analytics', name, split],
    queryFn:  () => api.getDatasetAnalytics(name, split),
    staleTime: 5 * 60_000,
  })
}

export function useDatasetSamples(
  name: string,
  split = 'train',
  n = 10,
  label?: number
) {
  return useQuery({
    queryKey: ['dataset-samples', name, split, n, label],
    queryFn:  () => api.getDatasetSamples(name, split, n, label),
    staleTime: 60_000,
  })
}

export function useDatasetAudit(name: string) {
  return useQuery({
    queryKey: ['dataset-audit', name],
    queryFn:  () => api.getDatasetAudit(name),
    staleTime: 5 * 60_000,
  })
}

export function useUploadDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.uploadDataset,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['dataset-catalog'] })
    },
  })
}

export function usePreprocessDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.preprocessDataset,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['dataset-catalog'] })
      qc.invalidateQueries({ queryKey: ['dataset-stats'] })
      qc.invalidateQueries({ queryKey: ['dataset-analytics'] })
      qc.invalidateQueries({ queryKey: ['dataset-audit'] })
    },
  })
}

// ── Verification hooks ────────────────────────────────────────────────────────

export function useVerify() {
  return useMutation({ mutationFn: api.verifyContent })
}

export function useProviderStatus() {
  return useQuery({
    queryKey: ['provider-status'],
    queryFn:  api.getProviderStatus,
    staleTime: 30_000,
  })
}

// ── Model Laboratory hooks ────────────────────────────────────────────────────

export function useLabModels() {
  return useQuery({
    queryKey: ['lab-models'],
    queryFn:  api.getLabModels,
    staleTime: 60_000,
  })
}

export function useLabModel(id: number | null) {
  return useQuery({
    queryKey: ['lab-model', id],
    queryFn:  () => api.getLabModel(id!),
    enabled:  id !== null,
    staleTime: 60_000,
  })
}

export function useLabCurves(id: number | null) {
  return useQuery({
    queryKey: ['lab-curves', id],
    queryFn:  () => api.getLabCurves(id!),
    enabled:  id !== null,
    staleTime: 5 * 60_000,
  })
}

export function useLabConfusion(id: number | null) {
  return useQuery({
    queryKey: ['lab-confusion', id],
    queryFn:  () => api.getLabConfusion(id!),
    enabled:  id !== null,
    staleTime: 5 * 60_000,
  })
}

export function useLabRoc(id: number | null) {
  return useQuery({
    queryKey: ['lab-roc', id],
    queryFn:  () => api.getLabRoc(id!),
    enabled:  id !== null,
    staleTime: 5 * 60_000,
  })
}

export function useLabComparison() {
  return useQuery({
    queryKey: ['lab-comparison'],
    queryFn:  api.getLabComparison,
    staleTime: 60_000,
  })
}

export function useLabReport() {
  return useQuery({
    queryKey: ['lab-report'],
    queryFn:  api.getLabReport,
    staleTime: 2 * 60_000,
  })
}

export function useStartTraining() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.startTraining,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['lab-models'] })
    },
  })
}

export function useJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ['job-status', jobId],
    queryFn:  () => api.getJobStatus(jobId!),
    enabled:  jobId !== null,
    refetchInterval: (query) => {
      const data = query.state.data as { status?: string } | undefined
      const done = data?.status === 'completed' || data?.status === 'failed'
      return done ? false : 2000
    },
  })
}

// ── Analytics hooks ───────────────────────────────────────────────────────────

export function usePatterns(filters: api.AnalyticsFilters = {}, n_topics = 5, n_clusters = 5) {
  return useQuery({
    queryKey: ['analytics-patterns', filters, n_topics, n_clusters],
    queryFn:  () => api.getPatterns(filters, n_topics, n_clusters),
    staleTime: 60_000,
  })
}

export function useWordFreq(filters: api.AnalyticsFilters = {}, top_n = 80) {
  return useQuery({
    queryKey: ['analytics-wordfreq', filters, top_n],
    queryFn:  () => api.getWordFreq(filters, top_n),
    staleTime: 30_000,
  })
}

export function useTemporalTrend(filters: api.AnalyticsFilters = {}, granularity = 'day') {
  return useQuery({
    queryKey: ['analytics-temporal', filters, granularity],
    queryFn:  () => api.getTemporal(filters, granularity),
    staleTime: 30_000,
  })
}

export function useSentiment(filters: api.AnalyticsFilters = {}) {
  return useQuery({
    queryKey: ['analytics-sentiment', filters],
    queryFn:  () => api.getSentiment(filters),
    staleTime: 30_000,
  })
}

export function useTopics(filters: api.AnalyticsFilters = {}, n_topics = 5) {
  return useQuery({
    queryKey: ['analytics-topics', filters, n_topics],
    queryFn:  () => api.getTopics(filters, n_topics),
    staleTime: 60_000,
  })
}

export function useInsights(filters: api.AnalyticsFilters = {}) {
  return useQuery({
    queryKey: ['analytics-insights', filters],
    queryFn:  () => api.getInsights(filters),
    staleTime: 60_000,
  })
}
