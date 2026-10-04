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
