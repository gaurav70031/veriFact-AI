// ── Request types ─────────────────────────────────────────────────────────────

export interface AnalyzeTextRequest {
  text: string
  title?: string
}

export interface AnalyzeUrlRequest {
  url: string
}

export interface AnalyzeClaimRequest {
  claim: string
  context?: string
}

// ── Shared sub-types ──────────────────────────────────────────────────────────

export interface TokenWeight {
  token: string
  weight: number    // [-1, +1]  positive = toward FAKE
  position: number
}

export interface ModelExplanation {
  model_id: string
  model_name: string
  method: 'lime' | 'attention' | 'tfidf_weights' | 'unavailable'
  label: string
  top_tokens: TokenWeight[]
  plain_text: string
  disclaimer: string
  error?: string
}

export interface ModelPrediction {
  model_id: string
  model_name: string
  label: 'FAKE' | 'REAL'
  is_fake: boolean
  confidence: number
  fake_probability: number
  real_probability: number
  inference_time_ms: number
  explanation?: ModelExplanation
}

export interface EvidenceSourceResult {
  source_name: string
  title: string
  url: string
  snippet?: string
  source_type: string
  published_at?: string
  retrieved_at: string
  relevance_score: number
  comparison_score: number
  rank: number
  relationship_to_claim: 'supporting' | 'contradicting' | 'inconclusive' | 'not_relevant'
}

export interface ClaimResult {
  position: number
  claim_text: string
  ml_verdict?: string
  ml_confidence?: number
  evidence_verdict?: string
  evidence_explanation?: string
  top_tokens: TokenWeight[]
  evidence_sources: EvidenceSourceResult[]
  supporting_count: number
  contradicting_count: number
  inconclusive_count: number
  not_relevant_count: number
  total_evidence: number
  assessment_conflict: boolean
  evidence_limitations: string[]
}

export interface EvidenceSummary {
  total_evidence: number
  supporting_count: number
  contradicting_count: number
  inconclusive_count: number
  not_relevant_count: number
  providers_used: string[]
  providers_failed: string[]
  evidence_limitations: string[]
  all_providers_failed: boolean
}

// ── Analysis response ─────────────────────────────────────────────────────────

export type AnalysisStatus = 'pending' | 'processing' | 'completed' | 'failed'
export type MLVerdict = 'FAKE' | 'REAL' | 'UNVERIFIED' | 'MIXED'
export type EvidenceVerdict =
  | 'LIKELY_CREDIBLE'
  | 'LIKELY_MISLEADING'
  | 'CONTRADICTED'
  | 'UNVERIFIED'
  | 'INSUFFICIENT_EVIDENCE'

export interface AnalysisResponse {
  id: number
  input_type: 'text' | 'url' | 'claim'
  original_input: string
  source_url?: string
  article_title?: string
  ml_verdict: MLVerdict
  ml_confidence?: number
  evidence_verdict?: EvidenceVerdict
  evidence_explanation?: string
  model_predictions: ModelPrediction[]
  claims: ClaimResult[]
  evidence_summary?: EvidenceSummary
  summary?: string
  status: AnalysisStatus
  processing_time_ms?: number
  created_at: string
}

// ── History ───────────────────────────────────────────────────────────────────

export interface AnalysisListItem {
  id: number
  input_type: string
  original_input: string
  source_url?: string
  article_title?: string
  final_verdict?: string
  final_confidence?: number
  status: AnalysisStatus
  processing_time_ms?: number
  created_at: string
}

export interface PaginatedHistory {
  items: AnalysisListItem[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

// ── Models ────────────────────────────────────────────────────────────────────

export interface ModelInfo {
  id: number
  model_name: string
  version: string
  model_type: 'traditional' | 'transformer' | 'ensemble'
  algorithm: string
  is_active: boolean
  description?: string
  dataset_name?: string
  training_samples?: number
  created_at: string
}

export interface ModelPerformance {
  model_name: string
  version: string
  algorithm: string
  accuracy?: number
  precision?: number
  recall?: number
  f1_score?: number
  roc_auc?: number
  test_samples?: number
}

export interface ModelPerformanceList {
  models: ModelPerformance[]
}

// ── Stats ─────────────────────────────────────────────────────────────────────

export interface VerdictCounts {
  FAKE: number
  REAL: number
  UNVERIFIED: number
  MIXED: number
}

export interface StatsResponse {
  total_analyses: number
  completed: number
  failed: number
  pending: number
  verdict_counts: VerdictCounts
  fake_percentage: number
  real_percentage: number
  avg_confidence: number
  avg_processing_ms: number
}

// ── Health ────────────────────────────────────────────────────────────────────

export interface ComponentStatus {
  status: 'ok' | 'degraded' | 'unavailable'
  message?: string
  latency_ms?: number
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'unavailable'
  version: string
  environment: string
  components: Record<string, ComponentStatus>
}

// ── Explanation ───────────────────────────────────────────────────────────────

export interface PerModelExplanation {
  model_id: string
  model_name: string
  method: string
  label: string
  confidence: number
  tokens: TokenWeight[]
  top_tokens: TokenWeight[]
  plain_text: string
  disclaimer: string
  error?: string
}

export interface AggregateTokenWeight {
  token: string
  weight: number
}

export interface ClaimExplanation {
  position: number
  claim_text: string
  aggregate_tokens: AggregateTokenWeight[]
}

export interface ExplanationResponse {
  analysis_id: number
  ml_verdict: string
  ml_confidence?: number
  evidence_verdict?: string
  model_explanations: PerModelExplanation[]
  claim_explanations: ClaimExplanation[]
  signal_vs_evidence_warning: string
}

// ── Evidence search ───────────────────────────────────────────────────────────

export interface EvidenceSearchRequest {
  claim: string
  max_results?: number
  from_days?: number
}

export interface EvidenceItem {
  source_name: string
  title: string
  url: string
  source_type: string
  description?: string
  published_at?: string
  retrieved_at: string
  provider_name: string
  relevance_score: number
}

export interface EvidenceSearchResult {
  claim: string
  query: string
  status: 'found' | 'INSUFFICIENT_EVIDENCE' | 'all_providers_failed'
  total_found: number
  after_dedup: number
  search_time_ms: number
  providers_used: string[]
  providers_failed: string[]
  items: EvidenceItem[]
}

// ── API error ─────────────────────────────────────────────────────────────────

export interface ApiError {
  error: string
  message: string
  detail?: unknown
}
