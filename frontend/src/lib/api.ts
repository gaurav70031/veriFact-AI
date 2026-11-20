import apiClient from './apiClient'
import type {
  AnalyzeTextRequest,
  AnalyzeUrlRequest,
  AnalyzeClaimRequest,
  AnalysisResponse,
  PaginatedHistory,
  ModelInfo,
  ModelPerformanceList,
  StatsResponse,
  HealthResponse,
  ExplanationResponse,
  EvidenceSearchRequest,
  EvidenceSearchResult,
  NewsSearchParams,
  NewsSearchResult,
  RegisterRequest,
  LoginRequest,
  AuthResponse,
  UserOut,
} from '@/types/api'

// ── Analysis ──────────────────────────────────────────────────────────────────

export const analyzeText = (body: AnalyzeTextRequest) =>
  apiClient.post<AnalysisResponse>('/analyze/text', body).then((r) => r.data)

export const analyzeUrl = (body: AnalyzeUrlRequest) =>
  apiClient.post<AnalysisResponse>('/analyze/url', body).then((r) => r.data)

export const analyzeClaim = (body: AnalyzeClaimRequest) =>
  apiClient.post<AnalysisResponse>('/analyze/claim', body).then((r) => r.data)

export const getAnalysisById = (id: number) =>
  apiClient.get<AnalysisResponse>(`/analysis/${id}`).then((r) => r.data)

// ── History ───────────────────────────────────────────────────────────────────

export const getHistory = (params: {
  page?: number
  page_size?: number
  verdict?: string
  input_type?: string
}) =>
  apiClient
    .get<PaginatedHistory>('/history', { params })
    .then((r) => r.data)

// ── Models & performance ──────────────────────────────────────────────────────

export const getModels = () =>
  apiClient.get<ModelInfo[]>('/models').then((r) => r.data)

export const getModelPerformance = () =>
  apiClient.get<ModelPerformanceList>('/model-performance').then((r) => r.data)

// ── Stats ─────────────────────────────────────────────────────────────────────

export const getStats = () =>
  apiClient.get<StatsResponse>('/stats').then((r) => r.data)

// ── Health ────────────────────────────────────────────────────────────────────

export const getHealth = () =>
  apiClient.get<HealthResponse>('/health').then((r) => r.data)

// ── Explanation ───────────────────────────────────────────────────────────────

export const getExplanation = (analysisId: number) =>
  apiClient
    .get<ExplanationResponse>(`/explanation/${analysisId}`)
    .then((r) => r.data)

export const getAnnotatedText = (analysisId: number) =>
  apiClient.get(`/explanation/${analysisId}/annotated`).then((r) => r.data)

export const getExplanationSummary = (analysisId: number) =>
  apiClient.get(`/explanation/${analysisId}/summary`).then((r) => r.data)

export const getExplanationReportUrl = (analysisId: number) =>
  `/api/v1/explanation/${analysisId}/report`

// ── Evidence ──────────────────────────────────────────────────────────────────

export const searchEvidence = (body: EvidenceSearchRequest) =>
  apiClient
    .post<EvidenceSearchResult>('/evidence/search', body)
    .then((r) => r.data)

export const getEvidenceForAnalysis = (analysisId: number) =>
  apiClient.get(`/evidence/${analysisId}`).then((r) => r.data)

// ── Live news (GET /api/v1/news/search) ───────────────────────────────────────
// Frontend never calls news APIs directly — all requests go through this proxy.

export const searchNews = (params: NewsSearchParams) =>
  apiClient
    .get<NewsSearchResult>('/news/search', { params })
    .then((r) => r.data)

// ── Authentication ────────────────────────────────────────────────────────────

export const authRegister = (body: RegisterRequest) =>
  apiClient.post<AuthResponse>('/auth/register', body).then((r) => r.data)

export const authLogin = (body: LoginRequest) =>
  apiClient.post<AuthResponse>('/auth/login', body).then((r) => r.data)

export const authLogout = () =>
  apiClient.post<{ message: string }>('/auth/logout').then((r) => r.data)

export const authMe = () =>
  apiClient.get<UserOut>('/auth/me').then((r) => r.data)

// ── Dataset management ────────────────────────────────────────────────────────

export const getDatasetCatalog = () =>
  apiClient.get('/datasets').then((r) => r.data)

export const getDatasetStats = (name: string) =>
  apiClient.get(`/datasets/${name}/stats`).then((r) => r.data)

export const getDatasetAnalytics = (name: string, split = 'train') =>
  apiClient.get(`/datasets/${name}/analytics`, { params: { split } }).then((r) => r.data)

export const getDatasetSamples = (
  name: string,
  split = 'train',
  n = 10,
  label?: number
) =>
  apiClient
    .get(`/datasets/${name}/samples`, { params: { split, n, ...(label !== undefined && { label }) } })
    .then((r) => r.data)

export const getDatasetAudit = (name: string) =>
  apiClient.get(`/datasets/${name}/audit`).then((r) => r.data)

export const uploadDataset = (file: File) => {
  const form = new FormData()
  form.append('file', file)
  return apiClient
    .post('/datasets/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } })
    .then((r) => r.data)
}

export const preprocessDataset = (body: {
  files: string[]
  cleaning_mode?: string
  lemmatize?: boolean
  test_size?: number
  val_size?: number
  force?: boolean
}) => apiClient.post('/datasets/preprocess', body).then((r) => r.data)

// ── Verification ──────────────────────────────────────────────────────────────

export const verifyContent = (body: {
  text?: string
  url?: string
  claim?: string
  max_results?: number
  from_days?: number
  max_claims?: number
}) => apiClient.post('/verify', body).then((r) => r.data)

export const getProviderStatus = () =>
  apiClient.get('/verify/providers').then((r) => r.data)

// ── Model Laboratory ──────────────────────────────────────────────────────────

export const getLabModels = () =>
  apiClient.get('/lab/models').then((r) => r.data)

export const getLabModel = (id: number) =>
  apiClient.get(`/lab/models/${id}`).then((r) => r.data)

export const getLabCurves = (id: number) =>
  apiClient.get(`/lab/models/${id}/curves`).then((r) => r.data)

export const getLabConfusion = (id: number) =>
  apiClient.get(`/lab/models/${id}/confusion`).then((r) => r.data)

export const getLabRoc = (id: number) =>
  apiClient.get(`/lab/models/${id}/roc`).then((r) => r.data)

export const getLabComparison = () =>
  apiClient.get('/lab/comparison').then((r) => r.data)

export const getLabReport = () =>
  apiClient.get('/lab/report').then((r) => r.data)

export const startTraining = (body: { model_id: string; data_dir?: string }) =>
  apiClient.post('/lab/train', body).then((r) => r.data)

export const getJobStatus = (jobId: string) =>
  apiClient.get(`/lab/train/${jobId}`).then((r) => r.data)

export const listJobs = () =>
  apiClient.get('/lab/train').then((r) => r.data)

// ── Misinformation Analytics ──────────────────────────────────────────────────

export interface AnalyticsFilters {
  date_from?:      string
  date_to?:        string
  verdict?:        string
  input_type?:     string
  confidence_min?: number
  confidence_max?: number
  limit?:          number
  source?:         string  // "live" | "dataset" | "both"
}

export const getPatterns = (filters: AnalyticsFilters = {}, n_topics = 5, n_clusters = 5) =>
  apiClient.get('/analytics/patterns', { params: { ...filters, n_topics, n_clusters } }).then((r) => r.data)

export const getWordFreq = (filters: AnalyticsFilters = {}, top_n = 80) =>
  apiClient.get('/analytics/wordfreq', { params: { ...filters, top_n } }).then((r) => r.data)

export const getTemporal = (filters: AnalyticsFilters = {}, granularity = 'day') =>
  apiClient.get('/analytics/temporal', { params: { ...filters, granularity } }).then((r) => r.data)

export const getSentiment = (filters: AnalyticsFilters = {}) =>
  apiClient.get('/analytics/sentiment', { params: filters }).then((r) => r.data)

export const getTopics = (filters: AnalyticsFilters = {}, n_topics = 5) =>
  apiClient.get('/analytics/topics', { params: { ...filters, n_topics } }).then((r) => r.data)

export const getInsights = (filters: AnalyticsFilters = {}) =>
  apiClient.get('/analytics/insights', { params: filters }).then((r) => r.data)

// ── History & Reports ────────────────────────────────────────────────────────

export const exportHistory = (params: Record<string, string | number | undefined>) =>
  apiClient.get('/history/export', { params, responseType: 'blob' }).then((r) => r.data)

export const compareAnalyses = (id1: number, id2: number) =>
  apiClient.get('/history/compare', { params: { id1, id2 } }).then((r) => r.data)

export const getReportPdfUrl = (analysisId: number) =>
  `/api/v1/report/pdf/${analysisId}`

export const getResearchReportPdfUrl = () =>
  '/api/v1/report/research'

export const getResearchPaperUrl = () =>
  '/api/v1/report/paper'
