/**
 * Frontend API client + hook tests.
 *
 * Uses MSW to intercept HTTP calls — no real backend needed.
 * Tests:
 *   - Correct endpoints are called
 *   - Error responses are propagated (not swallowed)
 *   - 401 dispatches auth:unauthorized event
 *   - Rate limit / 429 surfaces as an error
 *   - withCredentials is set (cookie auth)
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from './mocks/server'
import apiClient from '@/lib/apiClient'
import * as api from '@/lib/api'

describe('apiClient configuration', () => {
  it('has withCredentials set to true', () => {
    expect(apiClient.defaults.withCredentials).toBe(true)
  })

  it('has correct base URL', () => {
    expect(apiClient.defaults.baseURL).toContain('api/v1')
  })
})

describe('api.authLogin', () => {
  it('returns user on success', async () => {
    const result = await api.authLogin({ email: 'test@example.com', password: 'correct' })
    expect(result.user.email).toBe('test@example.com')
  })

  it('throws on wrong password (401)', async () => {
    await expect(
      api.authLogin({ email: 'test@example.com', password: 'wrongpassword' })
    ).rejects.toThrow()
  })
})

describe('api.authRegister', () => {
  it('returns user on success', async () => {
    const result = await api.authRegister({
      email: 'new@example.com', username: 'newuser', password: 'StrongP1!',
    })
    expect(result.user.username).toBe('testuser')
  })

  it('throws on duplicate email (409)', async () => {
    await expect(
      api.authRegister({
        email: 'duplicate@example.com', username: 'u', password: 'P1!pass',
      })
    ).rejects.toThrow()
  })
})

describe('api.analyzeText', () => {
  it('returns analysis with ml_verdict', async () => {
    const result = await api.analyzeText({
      text: 'This is a long enough article text for analysis here.',
    })
    expect(result.id).toBe(42)
    expect(result.ml_verdict).toMatch(/FAKE|REAL|UNVERIFIED|MIXED/)
  })

  it('throws on short text (422)', async () => {
    server.use(
      http.post('http://localhost:8000/api/v1/analyze/text', () =>
        HttpResponse.json({ error: 'VALIDATION_ERROR', message: 'too short' }, { status: 422 })
      )
    )
    await expect(api.analyzeText({ text: 'short' })).rejects.toThrow()
  })
})

describe('api.analyzeUrl SSRF prevention', () => {
  it('throws on private IP URL', async () => {
    server.use(
      http.post('http://localhost:8000/api/v1/analyze/url', () =>
        HttpResponse.json({ error: 'BAD_REQUEST', message: 'private IP' }, { status: 400 })
      )
    )
    await expect(api.analyzeUrl({ url: 'http://192.168.1.1/page' })).rejects.toThrow()
  })

  it('throws on inaccessible URL (422)', async () => {
    await expect(
      api.analyzeUrl({ url: 'https://inaccessible.example.com/page' })
    ).rejects.toThrow()
  })
})

describe('api.searchNews', () => {
  it('returns items for valid query', async () => {
    const result = await api.searchNews({ q: 'ECB interest rates' })
    expect(result.items.length).toBeGreaterThan(0)
    expect(result.status).toBe('found')
  })

  it('throws for query shorter than 3 chars', async () => {
    server.use(
      http.get('http://localhost:8000/api/v1/news/search', () =>
        HttpResponse.json({ error: 'VALIDATION_ERROR', message: 'too short' }, { status: 422 })
      )
    )
    await expect(api.searchNews({ q: 'ab' })).rejects.toThrow()
  })
})

describe('401 event dispatch', () => {
  it('dispatches auth:unauthorized DOM event on 401 response', async () => {
    server.use(
      http.get('http://localhost:8000/api/v1/auth/me', () =>
        HttpResponse.json({ error: 'UNAUTHORIZED', message: 'not auth' }, { status: 401 })
      )
    )

    const events: Event[] = []
    window.addEventListener('auth:unauthorized', (e) => events.push(e))

    try {
      await api.authMe()
    } catch {
      // expected
    }

    expect(events.length).toBe(1)
    window.removeEventListener('auth:unauthorized', () => {})
  })
})

describe('api graceful degradation', () => {
  it('provider failure surfaces as error, not silent empty result', async () => {
    server.use(
      http.post('http://localhost:8000/api/v1/analyze/text', () =>
        HttpResponse.json({ error: 'MODEL_UNAVAILABLE', message: 'No models available' }, { status: 503 })
      )
    )
    await expect(
      api.analyzeText({ text: 'Some long enough article text for testing.' })
    ).rejects.toThrow('No models available')
  })

  it('rate limit (429) surfaces as error', async () => {
    server.use(
      http.get('http://localhost:8000/api/v1/news/search', () =>
        HttpResponse.json({ error: 'RATE_LIMITED', message: 'Rate limit exceeded' }, { status: 429 })
      )
    )
    await expect(api.searchNews({ q: 'test query here' })).rejects.toThrow()
  })
})
