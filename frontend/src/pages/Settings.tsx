/**
 * Settings
 *
 * User preferences, theme, display options.
 * Persisted to localStorage. No backend calls.
 */

import { useState, useEffect } from 'react'
import {
  Settings as SettingsIcon, Sun, Moon, Monitor,
  Eye, Bell, Shield, Info, ChevronRight,
  Save, RotateCcw,
} from 'lucide-react'
import { useTheme } from '@/context/ThemeContext'
import { cn } from '@/lib/utils'

// ── Preference store ──────────────────────────────────────────────────────────

interface Prefs {
  compactTables: boolean
  showConfidenceBars: boolean
  defaultAnalysisMode: 'claim' | 'article' | 'url'
  showProcessingTime: boolean
  animationsEnabled: boolean
  tooltipsEnabled: boolean
}

const DEFAULT_PREFS: Prefs = {
  compactTables:        false,
  showConfidenceBars:   true,
  defaultAnalysisMode:  'claim',
  showProcessingTime:   true,
  animationsEnabled:    true,
  tooltipsEnabled:      true,
}

function loadPrefs(): Prefs {
  try {
    const raw = localStorage.getItem('vf_prefs')
    return raw ? { ...DEFAULT_PREFS, ...JSON.parse(raw) } : DEFAULT_PREFS
  } catch {
    return DEFAULT_PREFS
  }
}

function savePrefs(p: Prefs) {
  localStorage.setItem('vf_prefs', JSON.stringify(p))
}

// ── Toggle row ────────────────────────────────────────────────────────────────

function ToggleRow({ label, desc, checked, onChange }: {
  label: string; desc?: string; checked: boolean; onChange: (v: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between py-3.5" style={{ borderBottom: '1px solid var(--border-subtle)' }}>
      <div className="flex-1 min-w-0 mr-4">
        <div className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>{label}</div>
        {desc && <p className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>{desc}</p>}
      </div>
      <button
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          'relative w-11 h-6 rounded-full transition-colors duration-200 shrink-0',
          checked ? 'bg-brand-500' : 'bg-surface-3',
        )}
        style={{ backgroundColor: checked ? 'var(--brand)' : 'var(--bg-surface-3)' }}
      >
        <span className={cn(
          'absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform duration-200',
          checked ? 'translate-x-5' : 'translate-x-0',
        )} />
      </button>
    </div>
  )
}

// ── Section ───────────────────────────────────────────────────────────────────

function Section({ icon, title, children }: {
  icon: React.ReactNode; title: string; children: React.ReactNode
}) {
  return (
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-4">
        <div className="p-1.5 rounded-lg" style={{ backgroundColor: 'var(--brand-light)' }}>
          <span style={{ color: 'var(--brand)' }}>{icon}</span>
        </div>
        <h2 className="section-title">{title}</h2>
      </div>
      {children}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function Settings() {
  const { isDark, toggleTheme } = useTheme()
  const [prefs, setPrefs]       = useState<Prefs>(loadPrefs)
  const [saved,  setSaved]      = useState(false)

  const updatePref = <K extends keyof Prefs>(key: K, value: Prefs[K]) => {
    setPrefs((p) => ({ ...p, [key]: value }))
  }

  const handleSave = () => {
    savePrefs(prefs)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const handleReset = () => {
    setPrefs(DEFAULT_PREFS)
    savePrefs(DEFAULT_PREFS)
  }

  // Theme options
  const themes = [
    { id: 'dark',  label: 'Dark',   icon: <Moon  className="w-4 h-4" />, active: isDark },
    { id: 'light', label: 'Light',  icon: <Sun   className="w-4 h-4" />, active: !isDark },
  ]

  const analysisModes = [
    { id: 'claim',   label: 'Claim'   },
    { id: 'article', label: 'Article' },
    { id: 'url',     label: 'URL'     },
  ]

  return (
    <div className="page-content max-w-2xl">
      {/* Header */}
      <div className="flex items-start gap-4 mb-8">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-slate-600 to-slate-500 flex items-center justify-center shrink-0">
          <SettingsIcon className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="page-title mb-1">Settings</h1>
          <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>Preferences saved to your browser</p>
        </div>
      </div>

      <div className="space-y-5">
        {/* Appearance */}
        <Section icon={<Eye className="w-4 h-4" />} title="Appearance">
          <div className="mb-4">
            <div className="text-sm font-medium mb-2.5" style={{ color: 'var(--text-secondary)' }}>Theme</div>
            <div className="grid grid-cols-2 gap-3">
              {themes.map((t) => (
                <button
                  key={t.id}
                  onClick={toggleTheme}
                  className={cn(
                    'flex items-center gap-3 px-4 py-3 rounded-xl border-2 transition-all',
                    t.active ? 'border-brand-500' : 'hover:border-brand-500/30',
                  )}
                  style={{
                    borderColor: t.active ? 'var(--brand)' : 'var(--border)',
                    backgroundColor: t.active ? 'var(--brand-light)' : 'var(--bg-surface-2)',
                  }}
                >
                  <span style={{ color: t.active ? 'var(--brand)' : 'var(--text-subtle)' }}>{t.icon}</span>
                  <span className="text-sm font-medium" style={{ color: t.active ? 'var(--brand)' : 'var(--text-secondary)' }}>
                    {t.label}
                  </span>
                  {t.active && (
                    <span className="ml-auto w-2 h-2 rounded-full" style={{ backgroundColor: 'var(--brand)' }} />
                  )}
                </button>
              ))}
            </div>
          </div>

          <ToggleRow
            label="Animations"
            desc="Enable page transitions and micro-animations"
            checked={prefs.animationsEnabled}
            onChange={(v) => updatePref('animationsEnabled', v)}
          />
          <ToggleRow
            label="Tooltips"
            desc="Show explanatory tooltips on hover"
            checked={prefs.tooltipsEnabled}
            onChange={(v) => updatePref('tooltipsEnabled', v)}
          />
        </Section>

        {/* Analysis */}
        <Section icon={<Shield className="w-4 h-4" />} title="Analysis">
          <div className="mb-4">
            <div className="text-sm font-medium mb-2.5" style={{ color: 'var(--text-secondary)' }}>Default Input Mode</div>
            <div className="flex gap-2">
              {analysisModes.map((m) => (
                <button
                  key={m.id}
                  onClick={() => updatePref('defaultAnalysisMode', m.id as Prefs['defaultAnalysisMode'])}
                  className={cn(
                    'px-4 py-1.5 rounded-lg text-sm font-medium transition-all border',
                    prefs.defaultAnalysisMode === m.id ? 'text-white border-brand-500' : 'border-border',
                  )}
                  style={prefs.defaultAnalysisMode === m.id
                    ? { background: 'linear-gradient(135deg, var(--brand), var(--accent))', borderColor: 'var(--brand)' }
                    : { backgroundColor: 'var(--bg-surface-2)', color: 'var(--text-secondary)', borderColor: 'var(--border)' }
                  }
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          <ToggleRow
            label="Show Confidence Bars"
            desc="Display progress bar indicators for model confidence"
            checked={prefs.showConfidenceBars}
            onChange={(v) => updatePref('showConfidenceBars', v)}
          />
          <ToggleRow
            label="Show Processing Time"
            desc="Display analysis processing duration in history"
            checked={prefs.showProcessingTime}
            onChange={(v) => updatePref('showProcessingTime', v)}
          />
        </Section>

        {/* Table preferences */}
        <Section icon={<Bell className="w-4 h-4" />} title="Display">
          <ToggleRow
            label="Compact Tables"
            desc="Reduce row padding in history and performance tables"
            checked={prefs.compactTables}
            onChange={(v) => updatePref('compactTables', v)}
          />
        </Section>

        {/* About / info */}
        <div className="card p-5" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-4 h-4" style={{ color: 'var(--text-subtle)' }} />
            <h3 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Data & Privacy</h3>
          </div>
          <p className="text-xs leading-relaxed" style={{ color: 'var(--text-subtle)' }}>
            Preferences are stored locally in your browser (localStorage) and never sent to the server.
            Analysis history is stored in the backend database associated with your account.
            No personal data is collected beyond what you submit for analysis.
          </p>
        </div>

        {/* Action row */}
        <div className="flex items-center justify-between gap-4 pt-2">
          <button onClick={handleReset} className="btn-secondary">
            <RotateCcw className="w-3.5 h-3.5" />
            Reset to defaults
          </button>
          <button
            onClick={handleSave}
            className="btn-primary"
          >
            {saved ? (
              <><span className="text-emerald-400">✓</span> Saved!</>
            ) : (
              <><Save className="w-3.5 h-3.5" /> Save preferences</>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
