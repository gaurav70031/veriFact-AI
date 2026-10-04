/**
 * MSW request handlers — intercept real HTTP calls in tests.
 * All handlers return realistic response shapes matching the backend API.
 * No hardcoded predictions — all mock data follows the real schema.
 */
import { http, HttpResponse } from 'msw'

const BASE = 'http://localhost:8000/api/v1'

// ── Shared fixtures ────────────────────────────────────────────────────────────

export const mockUser = {
  id:        1,
  email:     'test@example.com',
  username:  'testuser',
  full_name: 'Test User',
  role:      'user',
  is_active: true,
  created_at: '2024-01-01T00:00:00Z',
}

export const mockAnalysis = {
  id:               42,
  input_type:       'text',
  original_input:   'The WHO declared COVID-19 a pandemic in March 2020.',
  source_url:       null,
  article_title:    null,
  ml_verdict:       'REAL',
  ml_confidence:    0.91,
  evidence_verdict: 'LIKELY_CREDIBLE',
  evidence_explanation: 'Multiple credible sources corroborate this claim.',
  model_predictions: [
    {
      model_id:          'logistic_regression',
      model_name:        'Logistic Regression',
      label:             'REAL',
      is_fake:           false,
      confidence:        0.91,
      fake_probability:  0.09,
      real_probability:  0.91,
      inference_time_ms: 12.1,
      explanation:       null,
    },
  ],
  claims:           [],
  evidence_summary: {
    total_evidence:      2,
    supporting_count:    2,
    contradicting_count: 0,
    inconclusive_count:  0,
    not_relevant_count:  0,
    providers_used:      ['newsapi'],
    providers_failed:    [],
    evidence_limitations: [],
    all_providers_failed: false,
  },
  summary:            'ML: REAL (91%) — Evidence: LIKELY_CREDIBLE.',
  status:             'completed',
  processing_time_ms: 1842,
  created_at:         '2024-05-01T10:00:00Z',
}

export const mockHistoryItem = {
  id:                42,
  input_type:        'text',
  original_input:    'The WHO declared COVID-19 a pandemic in March 2020.',
  source_url:        null,
  article_title:     null,
  final_verdict:     'REAL',
  final_confidence:  0.91,
  status:            'completed',
  processing_time_ms: 1842,
  created_at:         '2024-05-01T10:00:00Z',
}

// ── Handlers ──────────────────────────────────────────────────────────────────

export const handlers = [
  // Health
  http.get(`${BASE}/health`, () =>
    HttpResponse.json({
      status: 'ok', version: '1.0.0', environment: 'development',
      components: {
        database:  { status: 'ok', latency_ms: 2.1 },
        ml_models: { status: 'ok', message: 'Available: logistic_regression, linear_svm' },
      },
    })
  ),

  // Auth — me (not authenticated by default)
  http.get(`${BASE}/auth/me`, () =>
    HttpResponse.json({ error: 'UNAUTHORIZED', message: 'Authentication required.' }, { status: 401 })
  ),

  // Auth — register
  http.post(`${BASE}/auth/register`, async ({ request }) => {
    const body = await request.json() as Record<string, unknown>
    if ((body.email as string)?.includes('duplicate')) {
      return HttpResponse.json(
        { error: 'CONFLICT', message: 'An account with this email address already exists.' },
        { status: 409 }
      )
    }
    return HttpResponse.json({ user: mockUser, message: 'Authentication successful.' }, { status: 201 })
  }),

  // Auth — login
  http.post(`${BASE}/auth/login`, async ({ request }) => {
    const body = await request.json() as Record<string, unknown>
    if (body.password === 'wrongpassword') {
      return HttpResponse.json(
        { error: 'UNAUTHORIZED', message: 'Invalid email or password.' },
        { status: 401 }
      )
    }
    return HttpResponse.json({ user: mockUser, message: 'Login successful.' })
  }),

  // Auth — logout
  http.post(`${BASE}/auth/logout`, () =>
    HttpResponse.json({ message: 'Logged out successfully.' })
  ),

  // Analyze text
  http.post(`${BASE}/analyze/text`, async ({ request }) => {
    const body = await request.json() as Record<string, unknown>
    const text = (body.text as string) ?? ''
    if (text.trim().length < 20) {
      return HttpResponse.json(
        { error: 'VALIDATION_ERROR', message: 'text must have at least 20 characters' },
        { status: 422 }
      )
    }
    return HttpResponse.json(mockAnalysis)
  }),

  // Analyze URL
  http.post(`${BASE}/analyze/url`, async ({ request }) => {
    const body = await request.json() as Record<string, unknown>
    const url = (body.url as string) ?? ''
    if (url.includes('192.168') || url.includes('localhost') || url.includes('127.0')) {
      return HttpResponse.json(
        { error: 'BAD_REQUEST', message: 'Requests to private or loopback IP addresses are not permitted.' },
        { status: 400 }
      )
    }
    if (url.includes('inaccessible')) {
      return HttpResponse.json(
        { error: 'ARTICLE_EXTRACTION_FAILED', message: 'HTTP 404 when fetching URL.' },
        { status: 422 }
      )
    }
    return HttpResponse.json({ ...mockAnalysis, input_type: 'url', source_url: url })
  }),

  // Analyze claim
  http.post(`${BASE}/analyze/claim`, async ({ request }) => {
    const body = await request.json() as Record<string, unknown>
    const claim = (body.claim as string) ?? ''
    if (claim.length < 10) {
      return HttpResponse.json(
        { error: 'VALIDATION_ERROR', message: 'claim must have at least 10 characters' },
        { status: 422 }
      )
    }
    return HttpResponse.json({ ...mockAnalysis, input_type: 'claim', original_input: claim })
  }),

  // Single analysis
  http.get(`${BASE}/analysis/:id`, ({ params }) => {
    if (params.id === '999') {
      return HttpResponse.json({ error: 'NOT_FOUND', message: 'Analysis 999 not found.' }, { status: 404 })
    }
    return HttpResponse.json(mockAnalysis)
  }),

  // History (requires auth)
  http.get(`${BASE}/history`, () =>
    HttpResponse.json({
      items:       [mockHistoryItem],
      total:       1,
      page:        1,
      page_size:   20,
      total_pages: 1,
    })
  ),

  // Stats
  http.get(`${BASE}/stats`, () =>
    HttpResponse.json({
      total_analyses:  1,
      completed:       1,
      failed:          0,
      pending:         0,
      verdict_counts:  { FAKE: 0, REAL: 1, UNVERIFIED: 0, MIXED: 0 },
      fake_percentage: 0.0,
      real_percentage: 100.0,
      avg_confidence:  0.91,
      avg_processing_ms: 1842,
    })
  ),

  // Models
  http.get(`${BASE}/models`, () => HttpResponse.json([])),
  http.get(`${BASE}/model-performance`, () => HttpResponse.json({ models: [] })),

  // Explanation
  http.get(`${BASE}/explanation/:id`, () =>
    HttpResponse.json({
      analysis_id:  42,
      ml_verdict:   'REAL',
      ml_confidence: 0.91,
      evidence_verdict: null,
      model_explanations: [],
      claim_explanations: [],
      signal_vs_evidence_warning: 'Token weights reflect model patterns, not factual correctness.',
    })
  ),

  // News search
  http.get(`${BASE}/news/search`, ({ request }) => {
    const url   = new URL(request.url)
    const query = url.searchParams.get('q') ?? ''
    if (!query || query.length < 3) {
      return HttpResponse.json(
        { error: 'VALIDATION_ERROR', message: 'q must have at least 3 characters' },
        { status: 422 }
      )
    }
    return HttpResponse.json({
      claim:    query,
      query:    query,
      status:   'found',
      total_found: 1,
      after_dedup: 1,
      search_time_ms: 312,
      providers_used: ['newsapi'],
      providers_failed: [],
      items: [{
        source_name:  'Reuters',
        title:        `News article about: ${query}`,
        url:          'https://reuters.com/test',
        source_type:  'news_api',
        description:  'A relevant article excerpt.',
        published_at: '2024-05-01T08:00:00Z',
        retrieved_at: '2024-05-01T10:00:00Z',
        provider_name: 'newsapi',
        relevance_score: 0.82,
      }],
    })
  }),
]
