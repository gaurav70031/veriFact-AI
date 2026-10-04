/**
 * Frontend utility unit tests.
 * No DOM rendering — pure logic.
 */
import { describe, it, expect } from 'vitest'
import {
  cn, formatPct, formatMs, truncate,
  getVerdictConfig, verdictConfig,
} from '@/lib/utils'

describe('cn (class merging)', () => {
  it('merges class names', () => {
    expect(cn('a', 'b')).toBe('a b')
  })
  it('resolves Tailwind conflicts', () => {
    // twMerge keeps the later class when there is a conflict
    expect(cn('text-red-400', 'text-emerald-400')).toBe('text-emerald-400')
  })
  it('handles falsy values', () => {
    expect(cn('a', false && 'b', undefined, 'c')).toBe('a c')
  })
})

describe('formatPct', () => {
  it('formats 0.0 as 0.0%', () => expect(formatPct(0)).toBe('0.0%'))
  it('formats 1.0 as 100.0%', () => expect(formatPct(1)).toBe('100.0%'))
  it('formats 0.823 as 82.3%', () => expect(formatPct(0.823)).toBe('82.3%'))
})

describe('formatMs', () => {
  it('shows ms for < 1 s', () => expect(formatMs(450)).toBe('450ms'))
  it('shows s for >= 1 s', () => expect(formatMs(2500)).toBe('2.5s'))
})

describe('truncate', () => {
  it('returns short string unchanged', () => {
    expect(truncate('hello', 10)).toBe('hello')
  })
  it('truncates long strings with ellipsis', () => {
    const result = truncate('a'.repeat(200), 100)
    expect(result).toHaveLength(101)   // 100 + '…'
    expect(result.endsWith('…')).toBe(true)
  })
})

describe('getVerdictConfig', () => {
  it('returns FAKE config for FAKE', () => {
    const cfg = getVerdictConfig('FAKE')
    expect(cfg.color).toContain('red')
    expect(cfg.label).toBe('Likely Fake')
  })
  it('returns REAL config for REAL', () => {
    const cfg = getVerdictConfig('REAL')
    expect(cfg.color).toContain('emerald')
  })
  it('returns UNVERIFIED as fallback for unknown verdict', () => {
    const cfg = getVerdictConfig('SOME_UNKNOWN_VALUE')
    expect(cfg).toEqual(verdictConfig.UNVERIFIED)
  })
  it('handles undefined gracefully', () => {
    const cfg = getVerdictConfig(undefined)
    expect(cfg).toEqual(verdictConfig.UNVERIFIED)
  })
  it('INSUFFICIENT_EVIDENCE uses slate color', () => {
    const cfg = getVerdictConfig('INSUFFICIENT_EVIDENCE')
    expect(cfg.color).toContain('slate')
  })
})
