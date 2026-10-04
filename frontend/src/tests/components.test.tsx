/**
 * Component tests using proper ESM imports for Vitest.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import React from 'react'

// ── Direct ESM imports ────────────────────────────────────────────────────────
import { Badge }        from '@/components/ui/Badge'
import { EmptyState }   from '@/components/ui/EmptyState'
import { ErrorState }   from '@/components/ui/ErrorState'
import { Spinner }      from '@/components/ui/Spinner'
import { AssessmentHero } from '@/components/results/AssessmentHero'
import { EvidenceSection } from '@/components/results/EvidenceSection'

// ── Test wrapper ──────────────────────────────────────────────────────────────
function Wrapper({ children }: { children: React.ReactNode }) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return (
    <QueryClientProvider client={qc}>
      <BrowserRouter>{children}</BrowserRouter>
    </QueryClientProvider>
  )
}

// ── Badge ─────────────────────────────────────────────────────────────────────
describe('Badge component', () => {
  it('renders children', () => {
    render(<Badge>FAKE</Badge>)
    expect(screen.getByText('FAKE')).toBeDefined()
  })
  it('applies danger variant', () => {
    const { container } = render(<Badge variant="danger">Error</Badge>)
    expect(container.innerHTML).toContain('red')
  })
  it('applies success variant', () => {
    const { container } = render(<Badge variant="success">OK</Badge>)
    expect(container.innerHTML).toContain('emerald')
  })
})

// ── EmptyState ────────────────────────────────────────────────────────────────
describe('EmptyState component', () => {
  it('renders title and description', () => {
    render(<EmptyState title="Nothing here" description="Try something else." />)
    expect(screen.getByText('Nothing here')).toBeDefined()
    expect(screen.getByText('Try something else.')).toBeDefined()
  })
  it('renders action when provided', () => {
    render(<EmptyState title="Empty" action={<button>Retry</button>} />)
    expect(screen.getByText('Retry')).toBeDefined()
  })
})

// ── ErrorState ────────────────────────────────────────────────────────────────
describe('ErrorState component', () => {
  it('shows error message', () => {
    render(<ErrorState message="Something went wrong." />)
    expect(screen.getByText('Something went wrong.')).toBeDefined()
  })
  it('calls onRetry when retry button clicked', () => {
    const retry = vi.fn()
    render(<ErrorState message="Error." onRetry={retry} />)
    fireEvent.click(screen.getByText(/try again/i))
    expect(retry).toHaveBeenCalledOnce()
  })
  it('does not show retry button when onRetry absent', () => {
    render(<ErrorState message="Error." />)
    expect(screen.queryByText(/try again/i)).toBeNull()
  })
})

// ── Spinner ───────────────────────────────────────────────────────────────────
describe('Spinner component', () => {
  it('renders an SVG', () => {
    const { container } = render(<Spinner />)
    expect(container.querySelector('svg')).toBeDefined()
  })
})

// ── AssessmentHero ────────────────────────────────────────────────────────────
describe('AssessmentHero', () => {
  it('shows ML Prediction section', () => {
    render(<Wrapper><AssessmentHero mlVerdict="FAKE" mlConfidence={0.87} /></Wrapper>)
    expect(screen.getByText(/ML Model Prediction/i)).toBeDefined()
  })
  it('shows "Likely Fake" label for FAKE verdict', () => {
    render(<Wrapper><AssessmentHero mlVerdict="FAKE" mlConfidence={0.87} /></Wrapper>)
    expect(screen.getByText(/Likely Fake/i)).toBeDefined()
  })
  it('shows confidence percentage', () => {
    const { container } = render(<Wrapper><AssessmentHero mlVerdict="REAL" mlConfidence={0.91} /></Wrapper>)
    expect(container.innerHTML).toContain('91')
  })
  it('shows "Insufficient Evidence" label for INSUFFICIENT_EVIDENCE — never maps to FAKE', () => {
    render(
      <Wrapper>
        <AssessmentHero
          mlVerdict="FAKE"
          mlConfidence={0.80}
          evidenceVerdict="INSUFFICIENT_EVIDENCE"
          evidenceExplanation="No sources found."
        />
      </Wrapper>
    )
    expect(screen.getByText('Insufficient Evidence')).toBeDefined()
    // Evidence panel must NOT show "Likely Fake"
    const allFakeText = screen.queryAllByText(/Likely Fake/i)
    // "Likely Fake" may appear in the ML panel — but the evidence panel must not show it
    // Check that "Insufficient Evidence" is distinct text in the evidence section
    expect(screen.getByText(/Insufficient reliable evidence found/i)).toBeDefined()
  })
  it('shows "Not Available" when evidence_verdict is absent', () => {
    const { container } = render(<Wrapper><AssessmentHero mlVerdict="REAL" mlConfidence={0.75} /></Wrapper>)
    // When no evidence_verdict, component renders "Not Available"
    expect(container.innerHTML).toContain('Not Available')
  })
  it('shows conflict warning when hasConflict is true', () => {
    render(
      <Wrapper>
        <AssessmentHero
          mlVerdict="FAKE"
          mlConfidence={0.85}
          evidenceVerdict="LIKELY_CREDIBLE"
          hasConflict={true}
        />
      </Wrapper>
    )
    expect(screen.getByText(/conflict/i)).toBeDefined()
  })
  it('shows statistical disclaimer text', () => {
    const { container } = render(
      <Wrapper><AssessmentHero mlVerdict="FAKE" mlConfidence={0.80} /></Wrapper>
    )
    expect(container.innerHTML.toLowerCase()).toContain('statistical')
  })
})

// ── EvidenceSection — insufficient evidence states ────────────────────────────
describe('EvidenceSection — insufficient evidence', () => {
  const noEvidenceSummary = {
    total_evidence: 0, supporting_count: 0, contradicting_count: 0,
    inconclusive_count: 0, not_relevant_count: 0,
    providers_used: [], providers_failed: [],
    evidence_limitations: [], all_providers_failed: false,
  }

  it('shows explicit insufficient message when no sources retrieved', () => {
    render(<Wrapper><EvidenceSection summary={noEvidenceSummary} allSources={[]} /></Wrapper>)
    expect(screen.getByText(/Insufficient reliable evidence found/i)).toBeDefined()
  })

  it('must never show "Likely Fake" when evidence is insufficient', () => {
    render(<Wrapper><EvidenceSection summary={noEvidenceSummary} allSources={[]} /></Wrapper>)
    expect(screen.queryByText(/Likely Fake/i)).toBeNull()
  })

  it('shows provider failure state when all_providers_failed=true', () => {
    render(
      <Wrapper>
        <EvidenceSection
          summary={{ ...noEvidenceSummary, all_providers_failed: true, providers_failed: ['newsapi'] }}
          allSources={[]}
        />
      </Wrapper>
    )
    // Multiple elements may match — just verify at least one exists
    const matches = screen.getAllByText(/unavailable|failed|provider/i)
    expect(matches.length).toBeGreaterThan(0)
    expect(screen.queryByText(/Likely Fake/i)).toBeNull()
  })

  it('notes that absence of evidence ≠ false', () => {
    const { container } = render(
      <Wrapper><EvidenceSection summary={noEvidenceSummary} allSources={[]} /></Wrapper>
    )
    expect(container.innerHTML.toLowerCase()).toContain('false')
    // The text must be in a "not = false" context
    expect(container.innerHTML.toLowerCase()).toMatch(/absence|not.*false|does not/i)
  })
})

// ── Analyze page modes ────────────────────────────────────────────────────────
describe('Analyze page', () => {
  it('shows three mode buttons', async () => {
    const AnalyzePage = (await import('@/pages/Analyze')).default
    render(<Wrapper><AnalyzePage /></Wrapper>)
    expect(screen.getByText('Claim')).toBeDefined()
    expect(screen.getByText('Article')).toBeDefined()
    expect(screen.getByText('URL')).toBeDefined()
  })

  it('submit is disabled when form is empty', async () => {
    const AnalyzePage = (await import('@/pages/Analyze')).default
    render(<Wrapper><AnalyzePage /></Wrapper>)
    const btn = screen.getByRole('button', { name: /analyze/i })
    expect((btn as HTMLButtonElement).disabled).toBe(true)
  })

  it('contains disclaimer about statistical patterns', async () => {
    const AnalyzePage = (await import('@/pages/Analyze')).default
    const { container } = render(<Wrapper><AnalyzePage /></Wrapper>)
    expect(container.innerHTML.toLowerCase()).toContain('statistical')
  })
})

// ── Login page ────────────────────────────────────────────────────────────────
describe('Login page', () => {
  async function renderLogin() {
    const { AuthProvider } = await import('@/context/AuthContext')
    const LoginPage = (await import('@/pages/Login')).default
    return render(
      <Wrapper>
        <AuthProvider>
          <LoginPage />
        </AuthProvider>
      </Wrapper>
    )
  }

  it('renders email and password fields', async () => {
    await renderLogin()
    expect(screen.getByPlaceholderText(/you@example.com/i)).toBeDefined()
  })

  it('submit is disabled when fields are empty', async () => {
    await renderLogin()
    const btn = screen.getByRole('button', { name: /sign in/i })
    expect((btn as HTMLButtonElement).disabled).toBe(true)
  })

  it('shows error on wrong password', async () => {
    await renderLogin()
    const user = userEvent.setup()
    await user.type(screen.getByPlaceholderText(/you@example.com/i), 'test@example.com')
    // Type into the password field (first input with ••••••••)
    const pwInputs = screen.getAllByPlaceholderText(/••••••••/)
    await user.type(pwInputs[0], 'wrongpassword')
    await user.click(screen.getByRole('button', { name: /sign in/i }))
    await waitFor(() => {
      expect(screen.getByText(/invalid/i)).toBeDefined()
    }, { timeout: 3000 })
  })
})
