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
  input_type: 'text' | 'url' | 'query'
  original_input: string
  source_url?: string
  article_title?: string
  ml_verdict: MLVerdict
  ml_confidence?: number
  gemini_verdict?: 'TRUE' | 'FALSE' | 'UNCERTAIN' | 'MISLEADING'
  gemini_confidence?: number
  gemini_explanation?: string
  gemini_available?: boolean
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

// ── Live news search (GET /api/v1/news/search) ────────────────────────────────
// Same response shape as EvidenceSearchResult — aliased for clarity.
export type NewsSearchResult = EvidenceSearchResult

export interface NewsSearchParams {
  q: string
  from_days?: number
  max_results?: number
}

// ── API error ─────────────────────────────────────────────────────────────────

export interface ApiError {
  error: string
  message: string
  detail?: unknown
}

// ── Authentication ────────────────────────────────────────────────────────────

export interface RegisterRequest {
  email:     string
  username:  string
  password:  string
  full_name?: string
}

export interface LoginRequest {
  email:    string
  password: string
}

export interface UserOut {
  id:        number
  email:     string
  username:  string
  full_name: string | null
  role:      'user' | 'analyst' | 'admin'
  is_active: boolean
  created_at: string
}

export interface AuthResponse {
  user:    UserOut
  message: string
}

// ── Dataset management ────────────────────────────────────────────────────────

export interface DatasetLicense {
  type:              string
  summary:           string
  commercial_ok:     boolean
  redistribution_ok: boolean
  source_url:        string
}

export interface DatasetCatalogEntry {
  name:             string
  display_name:     string
  description:      string
  source_url:       string
  paper_url:        string | null
  citation:         string
  license:          DatasetLicense
  domain:           string
  language:         string
  total_articles:   number | null
  fake_articles:    number | null
  real_articles:    number | null
  expected_files:   string[]
  is_downloaded:    boolean
  is_processed:     boolean
  download_command: string
}

export interface DatasetCatalogResponse {
  datasets: DatasetCatalogEntry[]
  total:    number
}

export interface SplitStats {
  split:       string
  total_rows:  number
  fake_count:  number
  real_count:  number
  fake_pct:    number
  real_pct:    number
  text_length: Record<string, number>
  token_stats: Record<string, number>
  empty_texts: number
}

export interface DatasetStats {
  dataset_name:   string
  total_rows:     number
  splits:         Record<string, SplitStats>
  deduplication:  Record<string, number>
  leakage:        Record<string, number>
  last_processed: string | null
}

export interface WordFreqEntry {
  word:  string
  count: number
  pct:   number
}

export interface DatasetAnalytics {
  dataset_name:        string
  split:               string
  computed_at:         string
  class_distribution:  Record<string, unknown>
  length_distribution: Record<string, unknown>
  missing_values:      Record<string, unknown>
  duplicates:          Record<string, unknown>
  quality_report:      Record<string, unknown>
  word_frequency: {
    all:  WordFreqEntry[]
    fake: WordFreqEntry[]
    real: WordFreqEntry[]
  }
}

export interface DatasetSamples {
  dataset_name:   string
  split:          string
  label_filter:   number | null
  total_in_split: number
  samples:        Record<string, unknown>[]
}

export interface AuditStep {
  step:        number
  operation:   string
  timestamp:   string
  rows_before: number
  rows_after:  number
  dropped:     number
  params?:     Record<string, unknown>
  warnings?:   string[]
  notes?:      string
}

export interface PreprocessingAudit {
  dataset_name:  string
  source_files:  string[]
  cleaning_mode: string
  started_at:    string
  completed_at:  string | null
  total_steps:   number
  total_dropped: number
  steps:         AuditStep[]
}

export interface DatasetUploadResponse {
  filename:           string
  size_bytes:         number
  rows_detected:      number | null
  columns:            string[]
  detected_text_col:  string | null
  detected_label_col: string | null
  sample_labels:      Record<string, number>
  message:            string
  next_step:          string
}

// ── Verification ──────────────────────────────────────────────────────────────

export interface EvidenceSourceOut {
  source_name:           string
  title:                 string
  url:                   string
  snippet:               string | null
  source_type:           string
  provider_name:         string
  published_at:          string | null
  relevance_score:       number
  relationship_to_claim: string
  comparison_score:      number
  trust_tier:            number
}

export interface ClaimVerificationOut {
  claim_text:            string
  position:              number
  evidence_verdict:      string
  evidence_explanation:  string
  supporting_count:      number
  contradicting_count:   number
  inconclusive_count:    number
  not_relevant_count:    number
  total_evidence:        number
  evidence_sources:      EvidenceSourceOut[]
  evidence_limitations:  string[]
}

export interface VerificationResponse {
  input_type:          string
  article_title:       string | null
  source_url:          string | null
  overall_verdict:     string
  overall_explanation: string
  claims_checked:      ClaimVerificationOut[]
  total_evidence:      number
  supporting:          number
  contradicting:       number
  inconclusive:        number
  providers_used:      string[]
  providers_failed:    string[]
  processing_time_ms:  number
  limitations:         string[]
}

export interface ProviderStatus {
  name:         string
  configured:   boolean
  description:  string
  requires_key: boolean
  env_var:      string | null
}

// ── Model Laboratory ──────────────────────────────────────────────────────────

export interface LossCurvePoint {
  epoch:      number
  train_loss: number
  val_loss:   number | null
  val_acc:    number | null
}

export interface PerClassDetail {
  precision: number | null
  recall:    number | null
  f1:        number | null
  support:   number | null
}

export interface ModelDetailResponse {
  id:               number
  model_name:       string
  version:          string
  model_type:       string
  algorithm:        string
  is_active:        boolean
  description:      string | null
  dataset_name:     string | null
  training_samples: number | null
  test_samples:     number | null
  random_seed:      number | null
  train_time_s:     number | null
  inference_time_ms: number | null
  model_size_bytes: number | null
  accuracy:         number | null
  precision:        number | null
  recall:           number | null
  f1_score:         number | null
  roc_auc:          number | null
  confusion_matrix:  number[][] | null
  training_history:  LossCurvePoint[] | null
  per_class_metrics: Record<string, PerClassDetail> | null
  hyperparameters:  Record<string, unknown> | null
  created_at:       string
}

export interface ModelComparisonRow {
  model_name:        string
  model_type:        string
  version:           string
  accuracy:          number | null
  precision:         number | null
  recall:            number | null
  f1_score:          number | null
  roc_auc:           number | null
  train_time_s:      number | null
  inference_time_ms: number | null
  model_size_bytes:  number | null
  test_samples:      number | null
}

export interface ComparisonResponse {
  models:          ModelComparisonRow[]
  dataset_name:    string | null
  test_split_pct:  number
  evaluation_note: string
}

export interface RocPoint {
  fpr: number
  tpr: number
}

export interface RocResponse {
  model_id:   number
  model_name: string
  roc_auc:    number | null
  points:     RocPoint[]
  note:       string
}

export interface ConfusionResponse {
  model_id:    number
  model_name:  string
  matrix:      number[][]
  labels:      string[]
  test_samples: number | null
}

export interface BestModelEntry {
  metric:     string
  model_name: string
  value:      number | null
}

export interface ModelGroupSummary {
  group:          string
  models:         string[]
  avg_accuracy:   number | null
  avg_f1:         number | null
  avg_train_time: number | null
}

export interface ResearchReport {
  generated_at:        string
  dataset:             string
  test_split:          string
  total_models:        number
  best_by_accuracy:    BestModelEntry
  best_by_precision:   BestModelEntry
  best_by_recall:      BestModelEntry
  best_by_f1:          BestModelEntry
  best_by_roc_auc:     BestModelEntry
  fastest_training:    BestModelEntry
  fastest_inference:   BestModelEntry
  traditional_summary: ModelGroupSummary
  transformer_summary: ModelGroupSummary
  narrative:           string[]
  limitations:         string[]
  evaluation_conditions: Record<string, unknown>
}

export interface TrainJobStatus {
  job_id:      string
  model_id:    string
  status:      string
  started_at:  string | null
  finished_at: string | null
  error:       string | null
  result_model_version_id: number | null
  log_lines:   string[]
}

// ── Misinformation Analytics ──────────────────────────────────────────────────

export interface WordFreqEntry {
  word:  string
  count: number
  pct:   number
}

export interface WordFreqResponse {
  all:            WordFreqEntry[]
  FAKE:           WordFreqEntry[]
  REAL:           WordFreqEntry[]
  document_count: number
}

export interface TemporalPoint {
  date:       string
  total:      number
  fake:       number
  real:       number
  mixed:      number
  unverified: number
}

export interface TemporalResponse {
  series:         TemporalPoint[]
  granularity:    string
  document_count: number
}

export interface SentimentDistribution {
  label: string
  count: number
  pct:   number
}

export interface SentimentResult {
  distribution:  SentimentDistribution[]
  avg_compound:  number
  method:        string
  note:          string
}

export interface SentimentResponse {
  all:             SentimentResult
  FAKE:            SentimentResult
  REAL:            SentimentResult
  document_count:  number
}

export interface LDAWord {
  word:   string
  weight: number
}

export interface LDATopic {
  topic_id:   number
  words:      LDAWord[]
  label:      string
  doc_count:  number
}

export interface LDAResponse {
  topics:         LDATopic[]
  method:         string
  perplexity:     number | null
  note:           string
  document_count: number
}

export interface Insight {
  category:    string
  observation: string
  caveat:      string
  data_basis:  string
}

export interface InsightsResponse {
  insights:       Insight[]
  document_count:  number
  computed_at:     string
  source:          string
}

export interface VerdictDistribution {
  total:  number
  counts: Record<string, number>
  fake:   { count: number; pct: number }
  real:   { count: number; pct: number }
}

export interface NgramEntry {
  ngram: string
  count: number
}

export interface LengthStats {
  min:    number
  max:    number
  mean:   number
  median: number
  p25:    number
  p75:    number
}

export interface LengthBucket {
  bucket: string
  count:  number
}

export interface ClusterEntry {
  cluster_id:    number
  size:          number
  top_terms:     string[]
  sample_ids:    number[]
  representative: string
}

export interface RepeatedClaim {
  group_id:       number
  representative: string
  count:          number
  ids:            number[]
  similarity:     number
}

export interface PatternsResponse {
  document_count:        number
  error?:                string
  message?:              string
  verdict_distribution:  VerdictDistribution
  word_frequency:        { all: WordFreqEntry[]; FAKE: WordFreqEntry[]; REAL: WordFreqEntry[] }
  ngrams:                { bigrams: NgramEntry[]; trigrams: NgramEntry[] }
  tfidf_keywords:        Record<string, Array<{ keyword: string; score: number }>>
  sentiment:            SentimentResult
  sentiment_by_label:   Record<string, SentimentResult>
  text_length:          {
    by_label:    Record<string, { char: LengthStats; word: LengthStats; count: number }>
    histogram:   LengthBucket[]
    observation: string
  }
  linguistic_features:  { all: Record<string, number>; by_label: Record<string, Record<string, number>> }
  temporal_distribution: TemporalPoint[]
  source_distribution:  Array<{ source: string; count: number; pct: number }>
  evidence_relationships: Array<{ relationship: string; count: number }>
  lda_topics:            { topics: LDATopic[]; method: string; perplexity: number | null; note: string }
  clusters:              { clusters: ClusterEntry[]; method: string; note: string }
  repeated_claims:      RepeatedClaim[]
  insights:              Insight[]
  computed_at:           string
  source:                string
  filters_applied:       Record<string, unknown>
}

// ── XAI / Explanation ─────────────────────────────────────────────────────────

export interface AnnotatedWord {
  word:     string
  weight:   number
  is_space: boolean
}

export interface CrossModelToken {
  token:      string
  weights:    Record<string, number>
  avg_weight: number
  direction:  'fake' | 'real' | 'mixed'
}

export interface ExplanationSummary {
  analysis_id: number
  tokens:      CrossModelToken[]
  models:      string[]
  methods:     Record<string, string>
}
