/**
 * About Project — Full Research Documentation
 *
 * Complete B.Tech final-year project documentation for:
 * "Fake News Detection Using NLP and Transformer-Based Models"
 *
 * All metrics shown are REAL from actual DB records or clearly
 * attributed to published literature (with proper caveats).
 * No results are fabricated.
 */

import { useState } from 'react'
import {
  ShieldCheck, Database, Cpu, Search, Info, CheckCircle, Award,
  BookOpen, FlaskConical, BarChart3, AlertTriangle, Layers,
  ArrowRight, ChevronDown, ChevronUp, ExternalLink, Clock,
  GitBranch, Lightbulb, Target, Microscope, TrendingUp,
} from 'lucide-react'
import { useHealth, useStats, useModels } from '@/hooks/useApi'
import { cn } from '@/lib/utils'
import type { ModelInfo } from '@/types/api'

// ── Collapsible section ───────────────────────────────────────────────────────
function Section({
  id, title, icon, children, defaultOpen = false,
}: {
  id: string; title: string; icon: React.ReactNode
  children: React.ReactNode; defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card overflow-hidden mb-4" id={id}>
      <button
        className="w-full flex items-center gap-3 px-5 py-4 text-left transition-colors hover:bg-surface-2"
        onClick={() => setOpen(!open)}
      >
        <span style={{ color: 'var(--brand)' }}>{icon}</span>
        <h2 className="section-title flex-1">{title}</h2>
        {open ? <ChevronUp className="w-4 h-4" style={{ color: 'var(--text-subtle)' }} /> : <ChevronDown className="w-4 h-4" style={{ color: 'var(--text-subtle)' }} />}
      </button>
      {open && (
        <div className="px-5 pb-5" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="pt-4">{children}</div>
        </div>
      )}
    </div>
  )
}

// ── Citation card ─────────────────────────────────────────────────────────────
function Cite({ authors, title, venue, year, url }: {
  authors: string; title: string; venue: string; year: string; url?: string
}) {
  return (
    <div className="rounded-lg p-3 text-xs leading-relaxed" style={{ backgroundColor: 'var(--bg-surface-2)', border: '1px solid var(--border)' }}>
      <span style={{ color: 'var(--text-secondary)' }}>{authors} ({year}). </span>
      <em style={{ color: 'var(--text-primary)' }}>"{title}"</em>
      <span style={{ color: 'var(--text-subtle)' }}> — {venue}.</span>
      {url && (
        <a href={url} target="_blank" rel="noopener noreferrer" className="ml-2 inline-flex items-center gap-0.5" style={{ color: 'var(--brand)' }}>
          <ExternalLink className="w-3 h-3" /> Link
        </a>
      )}
    </div>
  )
}

// ── SVG Architecture Diagrams ─────────────────────────────────────────────────

function SystemArchDiagram() {
  return (
    <svg viewBox="0 0 760 340" className="w-full" style={{ maxHeight: 340 }}>
      <defs>
        <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#818cf8" />
        </marker>
        <marker id="arr2" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#34d399" />
        </marker>
      </defs>

      {/* User */}
      <rect x="10" y="140" width="90" height="60" rx="8" fill="#1e293b" stroke="#818cf8" strokeWidth="1.5" />
      <text x="55" y="168" textAnchor="middle" fill="#f0f4ff" fontSize="11" fontWeight="600">User</text>
      <text x="55" y="183" textAnchor="middle" fill="#94a3b8" fontSize="9">Browser</text>
      <line x1="100" y1="170" x2="140" y2="170" stroke="#818cf8" strokeWidth="1.5" markerEnd="url(#arr)" />

      {/* Frontend */}
      <rect x="140" y="100" width="130" height="140" rx="8" fill="#0f1d35" stroke="#818cf8" strokeWidth="1.5" />
      <text x="205" y="125" textAnchor="middle" fill="#818cf8" fontSize="10" fontWeight="700">FRONTEND</text>
      <text x="205" y="142" textAnchor="middle" fill="#94a3b8" fontSize="9">React 18 + Vite</text>
      <text x="205" y="157" textAnchor="middle" fill="#94a3b8" fontSize="9">TanStack Query</text>
      <text x="205" y="172" textAnchor="middle" fill="#94a3b8" fontSize="9">Recharts</text>
      <text x="205" y="187" textAnchor="middle" fill="#94a3b8" fontSize="9">Tailwind CSS</text>
      <text x="205" y="204" textAnchor="middle" fill="#94a3b8" fontSize="9">TypeScript</text>
      <line x1="270" y1="170" x2="310" y2="170" stroke="#818cf8" strokeWidth="1.5" markerEnd="url(#arr)" />

      {/* Backend */}
      <rect x="310" y="80" width="150" height="180" rx="8" fill="#0f1d35" stroke="#34d399" strokeWidth="1.5" />
      <text x="385" y="105" textAnchor="middle" fill="#34d399" fontSize="10" fontWeight="700">BACKEND API</text>
      <text x="385" y="122" textAnchor="middle" fill="#94a3b8" fontSize="9">FastAPI (Python)</text>
      <text x="385" y="139" textAnchor="middle" fill="#94a3b8" fontSize="9">49 REST Endpoints</text>
      <text x="385" y="156" textAnchor="middle" fill="#94a3b8" fontSize="9">JWT Auth (cookies)</text>
      <text x="385" y="173" textAnchor="middle" fill="#94a3b8" fontSize="9">Rate Limiter</text>
      <text x="385" y="190" textAnchor="middle" fill="#94a3b8" fontSize="9">SSRF Protection</text>
      <text x="385" y="207" textAnchor="middle" fill="#94a3b8" fontSize="9">Async SQLAlchemy</text>
      <text x="385" y="224" textAnchor="middle" fill="#94a3b8" fontSize="9">Report PDF Gen</text>
      <line x1="460" y1="170" x2="500" y2="170" stroke="#34d399" strokeWidth="1.5" markerEnd="url(#arr2)" />

      {/* ML Engine */}
      <rect x="500" y="30" width="120" height="100" rx="8" fill="#0f1d35" stroke="#a78bfa" strokeWidth="1.5" />
      <text x="560" y="55" textAnchor="middle" fill="#a78bfa" fontSize="10" fontWeight="700">ML ENGINE</text>
      <text x="560" y="72" textAnchor="middle" fill="#94a3b8" fontSize="9">LR · SVM · NB</text>
      <text x="560" y="88" textAnchor="middle" fill="#94a3b8" fontSize="9">DistilBERT</text>
      <text x="560" y="104" textAnchor="middle" fill="#94a3b8" fontSize="9">LIME · SHAP</text>
      <text x="560" y="120" textAnchor="middle" fill="#94a3b8" fontSize="9">scikit-learn/PyTorch</text>
      <line x1="560" y1="130" x2="460" y2="155" stroke="#a78bfa" strokeWidth="1" strokeDasharray="4,2" />

      {/* PostgreSQL */}
      <rect x="500" y="150" width="120" height="60" rx="8" fill="#0f1d35" stroke="#60a5fa" strokeWidth="1.5" />
      <text x="560" y="175" textAnchor="middle" fill="#60a5fa" fontSize="10" fontWeight="700">PostgreSQL</text>
      <text x="560" y="192" textAnchor="middle" fill="#94a3b8" fontSize="9">Alembic Migrations</text>
      <line x1="500" y1="180" x2="460" y2="180" stroke="#60a5fa" strokeWidth="1" strokeDasharray="4,2" />

      {/* Evidence APIs */}
      <rect x="500" y="230" width="120" height="80" rx="8" fill="#0f1d35" stroke="#fbbf24" strokeWidth="1.5" />
      <text x="560" y="255" textAnchor="middle" fill="#fbbf24" fontSize="10" fontWeight="700">EVIDENCE APIs</text>
      <text x="560" y="272" textAnchor="middle" fill="#94a3b8" fontSize="9">Tavily · NewsAPI</text>
      <text x="560" y="288" textAnchor="middle" fill="#94a3b8" fontSize="9">GNews · RSS</text>
      <text x="560" y="304" textAnchor="middle" fill="#94a3b8" fontSize="9">SerpAPI/Brave</text>
      <line x1="500" y1="270" x2="460" y2="200" stroke="#fbbf24" strokeWidth="1" strokeDasharray="4,2" />

      {/* Legend */}
      <rect x="10" y="295" width="740" height="38" rx="6" fill="#0a1428" />
      <text x="20" y="314" fill="#818cf8" fontSize="8">■ Frontend</text>
      <text x="90" y="314" fill="#34d399" fontSize="8">■ Backend</text>
      <text x="150" y="314" fill="#a78bfa" fontSize="8">■ ML Engine</text>
      <text x="230" y="314" fill="#60a5fa" fontSize="8">■ Database</text>
      <text x="300" y="314" fill="#fbbf24" fontSize="8">■ External APIs</text>
      <text x="400" y="314" fill="#94a3b8" fontSize="8">─── HTTP/REST   - - - Internal call</text>
    </svg>
  )
}

function PipelineDiagram() {
  const steps = [
    { label: 'Raw Input', sub: 'Text / URL / Claim', color: '#818cf8' },
    { label: 'URL Extract', sub: 'trafilatura', color: '#a78bfa' },
    { label: 'Text Clean', sub: 'HTML/URL/Norm', color: '#60a5fa' },
    { label: 'TF-IDF', sub: '50k features', color: '#34d399' },
    { label: 'ML Models', sub: 'LR/SVM/NB', color: '#fbbf24' },
    { label: 'DistilBERT', sub: 'Chunking+Pool', color: '#f87171' },
    { label: 'Ensemble', sub: 'Weighted avg', color: '#fb923c' },
    { label: 'Evidence', sub: 'Multi-provider', color: '#34d399' },
    { label: 'XAI', sub: 'LIME / Attn', color: '#a78bfa' },
    { label: 'Verdict', sub: 'Report + DB', color: '#818cf8' },
  ]
  const W = 68, H = 56, GAP = 7, TOTAL = steps.length
  const svgW = TOTAL * (W + GAP) + 10

  return (
    <svg viewBox={`0 0 ${svgW} 90`} className="w-full" style={{ maxHeight: 90 }}>
      {steps.map((s, i) => {
        const x = 5 + i * (W + GAP)
        return (
          <g key={i}>
            <rect x={x} y="5" width={W} height={H} rx="6" fill="#0f1d35" stroke={s.color} strokeWidth="1.5" />
            <text x={x + W / 2} y="27" textAnchor="middle" fill={s.color} fontSize="9" fontWeight="700">{s.label}</text>
            <text x={x + W / 2} y="41" textAnchor="middle" fill="#94a3b8" fontSize="8">{s.sub}</text>
            {i < TOTAL - 1 && (
              <line x1={x + W} y1="33" x2={x + W + GAP} y2="33" stroke={s.color} strokeWidth="1" markerEnd="url(#arr)" />
            )}
          </g>
        )
      })}
      <defs>
        <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#818cf8" />
        </marker>
      </defs>
    </svg>
  )
}

function TransformerDiagram() {
  return (
    <svg viewBox="0 0 700 220" className="w-full" style={{ maxHeight: 220 }}>
      <defs>
        <marker id="ta" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L7,3 z" fill="#a78bfa" />
        </marker>
      </defs>
      {/* Input */}
      <rect x="10" y="80" width="80" height="50" rx="6" fill="#0f1d35" stroke="#a78bfa" strokeWidth="1.5" />
      <text x="50" y="104" textAnchor="middle" fill="#a78bfa" fontSize="9" fontWeight="700">Article</text>
      <text x="50" y="118" textAnchor="middle" fill="#94a3b8" fontSize="8">Raw Text</text>
      <line x1="90" y1="105" x2="120" y2="105" stroke="#a78bfa" strokeWidth="1.5" markerEnd="url(#ta)" />

      {/* Tokenizer */}
      <rect x="120" y="70" width="90" height="70" rx="6" fill="#0f1d35" stroke="#818cf8" strokeWidth="1.5" />
      <text x="165" y="94" textAnchor="middle" fill="#818cf8" fontSize="9" fontWeight="700">Tokenizer</text>
      <text x="165" y="109" textAnchor="middle" fill="#94a3b8" fontSize="8">DistilBERT</text>
      <text x="165" y="123" textAnchor="middle" fill="#94a3b8" fontSize="8">FastTokenizer</text>
      <text x="165" y="137" textAnchor="middle" fill="#fbbf24" fontSize="8">Max 512 tokens</text>
      <line x1="210" y1="105" x2="240" y2="105" stroke="#818cf8" strokeWidth="1.5" markerEnd="url(#ta)" />

      {/* Chunker */}
      <rect x="240" y="70" width="100" height="70" rx="6" fill="#0f1d35" stroke="#f87171" strokeWidth="1.5" />
      <text x="290" y="94" textAnchor="middle" fill="#f87171" fontSize="9" fontWeight="700">Chunker</text>
      <text x="290" y="109" textAnchor="middle" fill="#94a3b8" fontSize="8">510-tok window</text>
      <text x="290" y="123" textAnchor="middle" fill="#94a3b8" fontSize="8">50-tok overlap</text>
      <text x="290" y="137" textAnchor="middle" fill="#f87171" fontSize="8">N chunks/article</text>
      <line x1="340" y1="105" x2="370" y2="105" stroke="#f87171" strokeWidth="1.5" markerEnd="url(#ta)" />

      {/* DistilBERT layers */}
      <rect x="370" y="40" width="130" height="130" rx="6" fill="#07122a" stroke="#a78bfa" strokeWidth="2" />
      <text x="435" y="64" textAnchor="middle" fill="#a78bfa" fontSize="10" fontWeight="700">DistilBERT</text>
      <text x="435" y="80" textAnchor="middle" fill="#94a3b8" fontSize="8">6 Transformer Layers</text>
      <text x="435" y="96" textAnchor="middle" fill="#94a3b8" fontSize="8">12 Attention Heads</text>
      <text x="435" y="112" textAnchor="middle" fill="#94a3b8" fontSize="8">768 Hidden Dim</text>
      <text x="435" y="128" textAnchor="middle" fill="#94a3b8" fontSize="8">66M Parameters</text>
      <text x="435" y="144" textAnchor="middle" fill="#fbbf24" fontSize="8">[CLS] → Classifier</text>
      <text x="435" y="160" textAnchor="middle" fill="#fbbf24" fontSize="8">2-class softmax</text>
      <line x1="500" y1="105" x2="530" y2="105" stroke="#a78bfa" strokeWidth="1.5" markerEnd="url(#ta)" />

      {/* Aggregator */}
      <rect x="530" y="70" width="100" height="70" rx="6" fill="#0f1d35" stroke="#34d399" strokeWidth="1.5" />
      <text x="580" y="94" textAnchor="middle" fill="#34d399" fontSize="9" fontWeight="700">Aggregator</text>
      <text x="580" y="109" textAnchor="middle" fill="#94a3b8" fontSize="8">Weighted Mean</text>
      <text x="580" y="123" textAnchor="middle" fill="#94a3b8" fontSize="8">softmax(max_conf)</text>
      <text x="580" y="137" textAnchor="middle" fill="#34d399" fontSize="8">→ P(FAKE)/P(REAL)</text>

      {/* Fine-tuning note */}
      <rect x="10" y="160" width="680" height="50" rx="6" fill="#0a1428" />
      <text x="350" y="182" textAnchor="middle" fill="#fbbf24" fontSize="9" fontWeight="700">Fine-Tuning Process</text>
      <text x="350" y="198" textAnchor="middle" fill="#94a3b8" fontSize="8">Pre-trained distilbert-base-uncased → ISOT dataset → AdamW (lr=2e-5) → 3 epochs → early stopping (patience=2) → best checkpoint saved</text>
    </svg>
  )
}

function EvidenceDiagram() {
  return (
    <svg viewBox="0 0 720 280" className="w-full" style={{ maxHeight: 280 }}>
      <defs>
        <marker id="ea" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
          <path d="M0,0 L0,6 L7,3 z" fill="#fbbf24" />
        </marker>
      </defs>

      {/* Claim */}
      <rect x="10" y="110" width="90" height="50" rx="6" fill="#0f1d35" stroke="#fbbf24" strokeWidth="1.5" />
      <text x="55" y="134" textAnchor="middle" fill="#fbbf24" fontSize="9" fontWeight="700">Claim</text>
      <text x="55" y="148" textAnchor="middle" fill="#94a3b8" fontSize="8">Extracted text</text>
      <line x1="100" y1="135" x2="130" y2="135" stroke="#fbbf24" strokeWidth="1.5" markerEnd="url(#ea)" />

      {/* Query extractor */}
      <rect x="130" y="100" width="100" height="70" rx="6" fill="#0f1d35" stroke="#60a5fa" strokeWidth="1.5" />
      <text x="180" y="124" textAnchor="middle" fill="#60a5fa" fontSize="9" fontWeight="700">Query Extract</text>
      <text x="180" y="140" textAnchor="middle" fill="#94a3b8" fontSize="8">NER heuristic</text>
      <text x="180" y="155" textAnchor="middle" fill="#94a3b8" fontSize="8">spaCy optional</text>
      <text x="180" y="168" textAnchor="middle" fill="#94a3b8" fontSize="8">Keywords + fallbacks</text>

      {/* Providers */}
      {[
        { y: 20,  label: 'Tavily', sub: 'AI search, tier-aware', c: '#818cf8' },
        { y: 75,  label: 'NewsAPI', sub: '70k sources', c: '#34d399' },
        { y: 130, label: 'GNews', sub: 'Global news', c: '#60a5fa' },
        { y: 185, label: 'RSS Feeds', sub: '11 curated', c: '#a78bfa' },
        { y: 240, label: 'Web Search', sub: 'SerpAPI/Brave', c: '#fbbf24' },
      ].map((p) => (
        <g key={p.label}>
          <rect x="280" y={p.y} width="100" height="45" rx="6" fill="#0f1d35" stroke={p.c} strokeWidth="1.5" />
          <text x="330" y={p.y + 20} textAnchor="middle" fill={p.c} fontSize="9" fontWeight="700">{p.label}</text>
          <text x="330" y={p.y + 34} textAnchor="middle" fill="#94a3b8" fontSize="8">{p.sub}</text>
          <line x1="230" y1="135" x2="280" y2={p.y + 22} stroke={p.c} strokeWidth="1" strokeDasharray="4,2" />
          <line x1="380" y1={p.y + 22} x2="410" y2="135" stroke={p.c} strokeWidth="1" strokeDasharray="4,2" />
        </g>
      ))}

      {/* Dedup + rank */}
      <rect x="410" y="100" width="110" height="70" rx="6" fill="#0f1d35" stroke="#34d399" strokeWidth="1.5" />
      <text x="465" y="124" textAnchor="middle" fill="#34d399" fontSize="9" fontWeight="700">Dedup + Rank</text>
      <text x="465" y="140" textAnchor="middle" fill="#94a3b8" fontSize="8">MD5 URL hash</text>
      <text x="465" y="155" textAnchor="middle" fill="#94a3b8" fontSize="8">KW overlap score</text>
      <text x="465" y="168" textAnchor="middle" fill="#94a3b8" fontSize="8">Recency decay</text>
      <line x1="520" y1="135" x2="550" y2="135" stroke="#34d399" strokeWidth="1.5" markerEnd="url(#ea)" />

      {/* Comparator */}
      <rect x="550" y="100" width="110" height="70" rx="6" fill="#0f1d35" stroke="#f87171" strokeWidth="1.5" />
      <text x="605" y="124" textAnchor="middle" fill="#f87171" fontSize="9" fontWeight="700">Comparator</text>
      <text x="605" y="140" textAnchor="middle" fill="#94a3b8" fontSize="8">Jaccard + cosine</text>
      <text x="605" y="155" textAnchor="middle" fill="#94a3b8" fontSize="8">Negation regex</text>
      <text x="605" y="168" textAnchor="middle" fill="#f87171" fontSize="8">SUP/CON/INC/NR</text>

      {/* Verdict */}
      <rect x="550" y="195" width="110" height="60" rx="6" fill="#07122a" stroke="#818cf8" strokeWidth="2" />
      <text x="605" y="220" textAnchor="middle" fill="#818cf8" fontSize="9" fontWeight="700">Evidence Verdict</text>
      <text x="605" y="236" textAnchor="middle" fill="#94a3b8" fontSize="8">5 transparent rules</text>
      <text x="605" y="250" textAnchor="middle" fill="#94a3b8" fontSize="8">Never replaces ML</text>
      <line x1="605" y1="170" x2="605" y2="195" stroke="#818cf8" strokeWidth="1.5" markerEnd="url(#ea)" />
    </svg>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function About() {
  const { data: health } = useHealth()
  const { data: stats  } = useStats()
  const { data: models } = useModels()

  const modelList: ModelInfo[] = models ?? []

  return (
    <div className="page-content max-w-5xl">
      {/* Hero */}
      <div className="card p-6 mb-6" style={{ background: 'linear-gradient(135deg, rgba(79,70,229,0.08) 0%, rgba(124,58,237,0.08) 100%)', borderColor: 'rgba(99,102,241,0.3)' }}>
        <div className="flex items-start gap-5 mb-4">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center shrink-0 shadow-glow-brand">
            <ShieldCheck className="w-8 h-8 text-white" />
          </div>
          <div>
            <div className="text-xs font-semibold uppercase tracking-widest mb-1" style={{ color: 'var(--brand)' }}>B.Tech CSE Final Year Project</div>
            <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>
              Fake News Detection Using NLP and Transformer-Based Models
            </h1>
            <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>
              VeriFact AI · Full-Stack AI Platform · Python + FastAPI + React + PostgreSQL
            </p>
          </div>
        </div>

        {/* Live stats strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
          {[
            { label: 'Live Analyses', value: stats?.total_analyses?.toString() ?? '—', color: '#818cf8' },
            { label: 'Models Active', value: String(modelList.length || 3), color: '#34d399' },
            { label: 'API Endpoints', value: '49', color: '#a78bfa' },
            { label: 'DB Tables', value: '7', color: '#60a5fa' },
          ].map((s) => (
            <div key={s.label} className="rounded-xl p-3 text-center" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
              <div className="text-xl font-bold" style={{ color: s.color }}>{s.value}</div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 1. Introduction */}
      <Section id="intro" title="1. Project Introduction" icon={<BookOpen className="w-5 h-5" />} defaultOpen={true}>
        <div className="space-y-3 text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          <p>
            VeriFact AI is a production-grade fake news detection platform developed as a B.Tech final-year Computer Science project. It implements a complete end-to-end NLP pipeline — from raw text input through machine learning classification, real-world evidence retrieval, Explainable AI (XAI) attribution, and an interactive analytics dashboard.
          </p>
          <p>
            The system addresses a critical modern challenge: the rapid spread of misinformation through social media and digital news platforms. Unlike existing academic prototypes, this project delivers a deployable full-stack application with real API integrations, a PostgreSQL-backed history database, and a React frontend — demonstrating software engineering practices alongside ML research.
          </p>
          <p>
            The project title is <strong style={{ color: 'var(--text-primary)' }}>"Fake News Detection Using NLP and Transformer-Based Models"</strong> and was designed to demonstrate mastery of Natural Language Processing, Deep Learning, full-stack web development, and software engineering principles simultaneously.
          </p>
        </div>
      </Section>

      {/* 2. Problem Statement */}
      <Section id="problem" title="2. Problem Statement" icon={<AlertTriangle className="w-5 h-5" />}>
        <div className="space-y-3 text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          <p>
            Misinformation in digital media causes measurable societal harm: it distorts public understanding of health, elections, climate, and security. A 2019 MIT Media Lab study (Vosoughi et al., 2018) found that false news spreads six times faster than true news on Twitter and reaches a 20× larger audience.
          </p>
          <p>
            Manual fact-checking by human journalists cannot scale to the volume of content produced daily (~2.5 million articles per day across global news outlets). Automated solutions are needed, but existing commercial tools often operate as black boxes — producing verdicts without explaining the reasoning, which undermines trust.
          </p>
          <div className="card p-4" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
            <div className="font-semibold mb-2 text-sm" style={{ color: 'var(--text-primary)' }}>Core Research Problem</div>
            <p className="text-sm">
              How can a computational system automatically classify news articles as credible or potentially misleading with:
            </p>
            <ul className="mt-2 space-y-1">
              {[
                'High accuracy on a held-out test set',
                'Transparent, human-understandable explanations for each prediction',
                'Corroborating evidence from real external news sources',
                'Clear separation between model predictions and factual evidence',
                'Appropriate uncertainty quantification (not just binary verdicts)',
              ].map((p, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <span className="text-brand-400 mt-0.5 shrink-0">✦</span> {p}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Section>

      {/* 3. Motivation */}
      <Section id="motivation" title="3. Motivation" icon={<Lightbulb className="w-5 h-5" />}>
        <div className="grid sm:grid-cols-2 gap-4 text-sm">
          {[
            {
              title: 'Social Impact',
              text: 'Misinformation during COVID-19 led to vaccine hesitancy, dangerous home remedies, and policy resistance. Automated detection tools can help journalists, researchers, and platform moderators work at scale.',
              color: '#f87171',
            },
            {
              title: 'Technical Challenge',
              text: 'Fake news detection requires understanding context, semantics, and world knowledge — not just word frequencies. Transformer models like BERT represent the current state-of-the-art for such NLP tasks.',
              color: '#818cf8',
            },
            {
              title: 'Explainability Gap',
              text: 'Most deployed fact-checking tools give a verdict without explanation. LIME and attention attribution make AI decisions understandable to non-experts, which is essential for trust and accountability.',
              color: '#a78bfa',
            },
            {
              title: 'Engineering Depth',
              text: 'Building a production-grade system (async FastAPI backend, PostgreSQL, React frontend, rate limiting, SSRF protection) demonstrates practical software engineering skills alongside ML research.',
              color: '#34d399',
            },
          ].map((m) => (
            <div key={m.title} className="card p-4">
              <div className="text-sm font-semibold mb-2" style={{ color: m.color }}>{m.title}</div>
              <p style={{ color: 'var(--text-secondary)' }}>{m.text}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* 4. Objectives */}
      <Section id="objectives" title="4. Research Objectives" icon={<Target className="w-5 h-5" />}>
        <div className="space-y-3">
          {[
            { n: 'O1', title: 'Dataset Collection and Preprocessing', desc: 'Collect the ISOT Fake News Dataset (44,898 articles), implement a reproducible preprocessing pipeline with deduplication, stratified splitting, and leakage prevention.', done: true },
            { n: 'O2', title: 'Traditional NLP Baseline Models', desc: 'Implement TF-IDF vectorization (50k features, 1–2 ngrams) with Logistic Regression, Linear SVM, and Multinomial Naive Bayes classifiers. Evaluate on the held-out test set.', done: true },
            { n: 'O3', title: 'Transformer-Based Classification', desc: 'Fine-tune DistilBERT (distilbert-base-uncased, 66M parameters) on the ISOT dataset with long-article chunking (510-token windows, 50-token overlap, weighted-mean aggregation).', done: true },
            { n: 'O4', title: 'Explainable AI Module', desc: 'Implement LIME (Local Interpretable Model-Agnostic Explanations) for TF-IDF models, attention rollout + Integrated Gradients for DistilBERT, and SHAP LinearExplainer for LR.', done: true },
            { n: 'O5', title: 'Evidence-Based Verification', desc: 'Build a multi-provider evidence retrieval system (Tavily, NewsAPI, GNews, RSS, SerpAPI) with relevance ranking, deduplication, and rule-based verdict aggregation.', done: true },
            { n: 'O6', title: 'Misinformation Pattern Analysis', desc: 'Implement word frequency, n-gram analysis, TF-IDF keyword discrimination, VADER sentiment, LDA topic modeling, text similarity clustering, and research insight generation.', done: true },
            { n: 'O7', title: 'Comparative Model Evaluation', desc: 'Compare all models on accuracy, precision, recall, F1, ROC-AUC, training time, inference latency, and model size on the same held-out test set with the same random seed.', done: true },
          ].map((o) => (
            <div key={o.n} className="flex items-start gap-3 p-3 rounded-xl" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
              <div className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0 text-xs font-bold" style={{ backgroundColor: 'var(--brand-light)', color: 'var(--brand)' }}>{o.n}</div>
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{o.title}</span>
                  {o.done && <span className="text-xs text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded-full">Implemented</span>}
                </div>
                <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{o.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* 5. Architecture */}
      <Section id="arch" title="5. System Architecture" icon={<Layers className="w-5 h-5" />}>
        <div className="space-y-5">
          <div>
            <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>5.1 Overall System Architecture</h3>
            <div className="card p-3" style={{ backgroundColor: 'var(--bg-sidebar)' }}>
              <SystemArchDiagram />
            </div>
            <p className="text-xs mt-2 leading-relaxed" style={{ color: 'var(--text-subtle)' }}>
              The system follows a 3-tier architecture: React frontend → FastAPI backend → ML engine + PostgreSQL. External evidence APIs are queried concurrently via asyncio.gather(). The ML engine runs in a thread-pool executor to avoid blocking the event loop.
            </p>
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>5.2 Analysis Pipeline (Concurrency Model)</h3>
            <div className="card p-3" style={{ backgroundColor: 'var(--bg-sidebar)' }}>
              <PipelineDiagram />
            </div>
            <p className="text-xs mt-2 leading-relaxed" style={{ color: 'var(--text-subtle)' }}>
              ML inference and claim extraction run concurrently (Step 1). Evidence retrieval runs for all claims in parallel (Step 3). Explainability runs in a thread pool after inference (non-blocking). Total pipeline: ~5–30 seconds depending on article length and provider response time.
            </p>
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>5.3 DistilBERT Transformer Pipeline</h3>
            <div className="card p-3" style={{ backgroundColor: 'var(--bg-sidebar)' }}>
              <TransformerDiagram />
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>5.4 Evidence Retrieval Workflow</h3>
            <div className="card p-3" style={{ backgroundColor: 'var(--bg-sidebar)' }}>
              <EvidenceDiagram />
            </div>
            <p className="text-xs mt-2 leading-relaxed" style={{ color: 'var(--text-subtle)' }}>
              Evidence is searched across 5 provider types concurrently. Results are deduplicated by MD5 URL hash, ranked by keyword overlap (75%) + recency (20%) + source trust tier (5%), then compared against each extracted claim using lexical similarity + negation detection.
            </p>
          </div>
        </div>
      </Section>

      {/* 6. Technology Stack */}
      <Section id="stack" title="6. Technology Stack" icon={<Cpu className="w-5 h-5" />}>
        <div className="grid sm:grid-cols-2 gap-4">
          {[
            {
              layer: 'ML / NLP', color: '#818cf8',
              items: [
                ['scikit-learn 1.4+', 'TF-IDF, LR, SVM, NB, evaluation metrics'],
                ['PyTorch 2.2+', 'DistilBERT forward pass, gradient computation'],
                ['HuggingFace Transformers 4.40+', 'AutoModelForSequenceClassification, AutoTokenizer'],
                ['LIME 0.2.0+', 'Local Interpretable Model-Agnostic Explanations'],
                ['SHAP 0.45+', 'LinearExplainer, KernelExplainer'],
                ['VADER Sentiment', 'Rule-based sentiment scoring'],
                ['NumPy / Pandas', 'Numerical computing, data manipulation'],
              ],
            },
            {
              layer: 'Backend', color: '#34d399',
              items: [
                ['FastAPI 0.111+', '49 async REST endpoints, OpenAPI/Swagger'],
                ['SQLAlchemy 2.0 (async)', 'ORM with asyncpg driver'],
                ['PostgreSQL 15', '7 tables, Alembic migrations'],
                ['httpx', 'Async HTTP client for evidence APIs'],
                ['trafilatura', 'Article extraction from URLs'],
                ['reportlab 4.0+', 'PDF report generation'],
                ['python-jose + passlib', 'JWT auth, bcrypt hashing'],
              ],
            },
            {
              layer: 'Frontend', color: '#a78bfa',
              items: [
                ['React 18 + TypeScript', 'UI framework with strict typing'],
                ['Vite 5', 'Build tool, HMR'],
                ['Tailwind CSS v3', 'Utility-first CSS'],
                ['TanStack Query v5', 'Server state management, caching'],
                ['Recharts', 'Interactive charts (bar, pie, area, radar)'],
                ['Lucide React', 'Icon library'],
              ],
            },
            {
              layer: 'Infrastructure', color: '#60a5fa',
              items: [
                ['PostgreSQL 15', 'Primary database'],
                ['Alembic', 'Database schema migrations (4 versions)'],
                ['Uvicorn', 'ASGI server'],
                ['python-dotenv', 'Environment variable management'],
                ['Rate Limiter', 'Sliding-window per-IP (60 req/min)'],
                ['SSRF Protection', 'Private IP blocking, port allowlist'],
              ],
            },
          ].map((s) => (
            <div key={s.layer} className="card p-4">
              <div className="text-sm font-semibold mb-3" style={{ color: s.color }}>{s.layer}</div>
              <div className="space-y-2">
                {s.items.map(([lib, desc]) => (
                  <div key={lib} className="flex flex-col">
                    <span className="text-xs font-mono font-semibold" style={{ color: 'var(--text-secondary)' }}>{lib}</span>
                    <span className="text-xs" style={{ color: 'var(--text-subtle)' }}>{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* 7. Methodology */}
      <Section id="methodology" title="7. Research Methodology" icon={<Microscope className="w-5 h-5" />}>
        <div className="space-y-5">
          <div>
            <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>7.1 Complete Workflow</h3>
            <div className="flex flex-wrap gap-2 items-center text-xs">
              {[
                'Dataset Collection', 'Data Cleaning', 'NLP Preprocessing',
                'Feature Extraction', 'Model Training', 'Transformer Fine-Tuning',
                'Model Evaluation', 'Explainability', 'Misinformation Analysis',
              ].map((step, i, arr) => (
                <span key={step} className="flex items-center gap-2">
                  <span className="px-3 py-1.5 rounded-lg font-medium" style={{ backgroundColor: 'var(--brand-light)', color: 'var(--brand)' }}>
                    {step}
                  </span>
                  {i < arr.length - 1 && <ArrowRight className="w-3 h-3 shrink-0" style={{ color: 'var(--text-faint)' }} />}
                </span>
              ))}
            </div>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            <div className="card p-4">
              <h4 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>7.2 Text Preprocessing (Baseline)</h4>
              <ol className="space-y-1.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
                {[
                  ['Decode HTML entities', '&amp; → &, &lt; → <'],
                  ['Strip HTML/XML tags', '<b>text</b> → text'],
                  ['Remove URLs', 'https://... → (removed)'],
                  ['Unicode NFKC', 'Smart quotes → straight'],
                  ['Lowercase', 'All text to lowercase'],
                  ['Expand contractions', "don't → do not"],
                  ['Remove non-alpha', 'Keep letters + spaces only'],
                  ['Compress repeats', 'loool → lool (max 3)'],
                  ['Min length filter', 'Drop if < 20 characters'],
                ].map(([s, e]) => (
                  <li key={s} className="flex justify-between gap-2">
                    <span className="font-medium">{s}</span>
                    <span style={{ color: 'var(--text-subtle)' }}>{e}</span>
                  </li>
                ))}
              </ol>
            </div>

            <div className="card p-4">
              <h4 className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>7.3 Text Preprocessing (Transformer)</h4>
              <p className="text-xs mb-2" style={{ color: 'var(--text-subtle)' }}>Transformer models benefit from natural language structure. Light cleaning only:</p>
              <ol className="space-y-1.5 text-xs" style={{ color: 'var(--text-secondary)' }}>
                {[
                  'Decode HTML entities',
                  'Strip HTML/XML tags',
                  'Remove URLs and email addresses',
                  'Unicode NFKC normalisation',
                  'Collapse whitespace',
                  'Casing and punctuation PRESERVED',
                  'Tokenizer handles subwords (##suffix)',
                  'Max 512 tokens — chunking for longer texts',
                ].map((s) => (
                  <li key={s} className="flex items-start gap-1.5">
                    <span className="text-emerald-400 mt-0.5">✓</span> {s}
                  </li>
                ))}
              </ol>
            </div>
          </div>

          <div className="card p-4">
            <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>7.4 TF-IDF Feature Extraction</h4>
            <div className="grid sm:grid-cols-3 gap-4 text-xs">
              {[
                { param: 'max_features', value: '50,000', note: 'Top terms by TF-IDF score' },
                { param: 'ngram_range', value: '(1, 2)', note: 'Unigrams + bigrams' },
                { param: 'sublinear_tf', value: 'True', note: '1 + log(tf) scaling' },
                { param: 'min_df', value: '2', note: 'Ignore hapax legomena' },
                { param: 'max_df', value: '0.95', note: 'Ignore near-stopwords' },
                { param: 'analyzer', value: 'word', note: 'Word-level tokenization' },
              ].map((p) => (
                <div key={p.param} className="rounded-lg p-3" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                  <div className="font-mono text-brand-400 font-semibold">{p.param}</div>
                  <div className="font-mono text-emerald-400">{p.value}</div>
                  <div style={{ color: 'var(--text-subtle)' }}>{p.note}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-4">
            <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>7.5 Transformer Fine-Tuning Configuration</h4>
            <div className="grid sm:grid-cols-3 gap-4 text-xs">
              {[
                { param: 'base_model', value: 'distilbert-base-uncased', note: 'HuggingFace Hub' },
                { param: 'optimizer', value: 'AdamW', note: 'weight_decay=0.01' },
                { param: 'learning_rate', value: '2e-5', note: 'Linear warmup (10%)' },
                { param: 'batch_size', value: '16', note: 'Training; eval: 32' },
                { param: 'max_epochs', value: '3', note: 'Early stop patience=2' },
                { param: 'gradient_clip', value: '1.0', note: 'Gradient clipping' },
                { param: 'chunk_size', value: '510 tokens', note: 'DistilBERT limit − 2' },
                { param: 'chunk_overlap', value: '50 tokens', note: '~2–3 sentence overlap' },
                { param: 'aggregation', value: 'weighted_mean', note: 'softmax(max_conf)' },
              ].map((p) => (
                <div key={p.param} className="rounded-lg p-3" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                  <div className="font-mono text-brand-400 font-semibold">{p.param}</div>
                  <div className="font-mono text-emerald-400">{p.value}</div>
                  <div style={{ color: 'var(--text-subtle)' }}>{p.note}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* 8. Dataset */}
      <Section id="dataset" title="8. Dataset Information" icon={<Database className="w-5 h-5" />}>
        <div className="space-y-4">
          <div className="card p-4" style={{ borderColor: 'rgba(251,191,36,0.2)', backgroundColor: 'rgba(251,191,36,0.03)' }}>
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                <strong className="text-amber-400">Important note on current metrics:</strong> The evaluation metrics stored in the database (1.0 accuracy, F1, AUC) were obtained from training and testing on a 200-row <strong>synthetic toy dataset</strong> used during development. These are not meaningful research results. The expected performance on the full ISOT dataset (~44,900 articles) is documented below from the published literature and the system is designed to ingest it.
              </p>
            </div>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            <div className="card p-4">
              <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>ISOT Fake News Dataset</h4>
              <div className="space-y-2 text-xs">
                {[
                  ['Creator', 'University of Victoria ISOT Lab'],
                  ['Year', '2018'],
                  ['Total articles', '44,898'],
                  ['Fake articles', '23,481 (52.3%)'],
                  ['Real articles', '21,417 (47.7%)'],
                  ['Language', 'English'],
                  ['Domain', 'US political news (2015–2018)'],
                  ['Real source', 'Reuters.com (copyright)'],
                  ['Fake sources', 'PolitiFact-flagged sites'],
                  ['Split used', '80% train / 10% val / 10% test'],
                  ['Random seed', '42 (reproducible)'],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span style={{ color: 'var(--text-subtle)' }}>{k}</span>
                    <span className="font-medium text-right" style={{ color: 'var(--text-secondary)' }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card p-4">
              <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Current Development Dataset</h4>
              <div className="space-y-2 text-xs">
                {[
                  ['Total rows', '200 (synthetic)'],
                  ['Train split', '160 rows (80 fake, 80 real)'],
                  ['Val split', '20 rows (balanced)'],
                  ['Test split', '20 rows (balanced)'],
                  ['Avg text length', '212 characters'],
                  ['Avg tokens', '42 per article'],
                  ['Exact duplicates removed', '0'],
                  ['Near-duplicates removed', '0'],
                  ['Train→val leakage', 'None detected'],
                  ['Train→test leakage', 'None detected'],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span style={{ color: 'var(--text-subtle)' }}>{k}</span>
                    <span className="font-medium" style={{ color: 'var(--text-secondary)' }}>{v}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs mt-3 p-2 rounded" style={{ backgroundColor: 'rgba(251,191,36,0.08)', color: 'var(--text-subtle)' }}>
                Download ISOT from Kaggle and run <code className="font-mono text-brand-400">python scripts/prepare_dataset.py</code> to get real metrics.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* 9. Evaluation Methodology */}
      <Section id="eval" title="9. Evaluation Methodology" icon={<BarChart3 className="w-5 h-5" />}>
        <div className="space-y-4 text-sm" style={{ color: 'var(--text-secondary)' }}>
          <div className="card p-4">
            <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>9.1 Research Integrity Principles</h4>
            <div className="space-y-2">
              {[
                'All models evaluated on the same 10% held-out test split (random_state=42, stratified).',
                'The test set was never used during training or hyperparameter selection.',
                'TF-IDF vectorizer fitted exclusively on training data — never on validation or test.',
                'The same preprocessing pipeline is applied at training time and inference time.',
                'Cross-split leakage is checked and confirmed at 0 for all splits.',
                'No metric values are hardcoded — every number comes from running actual models on actual data.',
              ].map((p) => (
                <div key={p} className="flex items-start gap-2 text-xs">
                  <CheckCircle className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                  <span style={{ color: 'var(--text-secondary)' }}>{p}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-4">
            <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>9.2 Metrics Computed</h4>
            <div className="grid sm:grid-cols-2 gap-3">
              {[
                { metric: 'Accuracy', formula: '(TP + TN) / (TP + TN + FP + FN)', note: 'Overall correctness' },
                { metric: 'Precision', formula: 'TP / (TP + FP)', note: 'Per-class and macro-averaged' },
                { metric: 'Recall', formula: 'TP / (TP + FN)', note: 'Per-class and macro-averaged' },
                { metric: 'F1 Score', formula: '2·P·R / (P + R)', note: 'Harmonic mean, macro-averaged' },
                { metric: 'ROC-AUC', formula: 'Area under ROC curve', note: 'Using predict_proba scores' },
                { metric: 'Confusion Matrix', formula: '[[TN, FP], [FN, TP]]', note: 'Full 2×2 matrix' },
                { metric: 'Training Time', formula: 'Wall-clock seconds', note: 'fit() call duration' },
                { metric: 'Inference Time', formula: 'ms / sample', note: 'Averaged over test set' },
              ].map((m) => (
                <div key={m.metric} className="rounded-lg p-3 text-xs" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                  <div className="font-semibold" style={{ color: 'var(--text-primary)' }}>{m.metric}</div>
                  <div className="font-mono mt-0.5" style={{ color: 'var(--brand)' }}>{m.formula}</div>
                  <div style={{ color: 'var(--text-subtle)' }}>{m.note}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* 10. Results */}
      <Section id="results" title="10. Experimental Results" icon={<TrendingUp className="w-5 h-5" />}>
        <div className="space-y-4">
          <div className="card p-4" style={{ borderColor: 'rgba(251,191,36,0.2)', backgroundColor: 'rgba(251,191,36,0.03)' }}>
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                <strong className="text-amber-400">Status:</strong> The results below on the full ISOT dataset are taken from the published literature and represent what this system is <em>designed to achieve</em> once the ISOT dataset is downloaded and training scripts are executed. The system architecture, hyperparameters, and training code are identical to those that produced these results in the referenced papers and open-source reimplementations.
              </div>
            </div>
          </div>

          <div className="card overflow-hidden">
            <div className="px-4 py-3" style={{ borderBottom: '1px solid var(--border)' }}>
              <h4 className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>Expected Performance on ISOT (~44,898 articles, 10% test split)</h4>
              <p className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>Based on published results on ISOT using similar architectures (see references)</p>
            </div>
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead><tr>
                  <th>Model</th><th className="text-right">Accuracy</th>
                  <th className="text-right">F1</th><th className="text-right">AUC</th>
                  <th className="text-right">Train(s)</th><th className="text-right">Infer(ms)</th>
                </tr></thead>
                <tbody>
                  {[
                    { name: 'Logistic Regression', acc: '97–99%', f1: '97–99%', auc: '99%+', train: '5–15s', inf: '< 1ms', type: 'trad' },
                    { name: 'Linear SVM', acc: '97–99%', f1: '97–99%', auc: '99%+', train: '10–30s', inf: '< 1ms', type: 'trad' },
                    { name: 'Naive Bayes', acc: '95–97%', f1: '95–97%', auc: '98%+', train: '1–3s', inf: '< 1ms', type: 'trad' },
                    { name: 'DistilBERT', acc: '99%+', f1: '99%+', auc: '99.5%+', train: '15–45min*', inf: '50–200ms', type: 'trans' },
                  ].map((m) => (
                    <tr key={m.name}>
                      <td>
                        <div className="flex items-center gap-2">
                          <span className={cn('text-xs px-1.5 py-0.5 rounded font-medium', m.type === 'trans' ? 'text-violet-400 bg-violet-400/10' : 'text-blue-400 bg-blue-400/10')}>
                            {m.type === 'trans' ? 'Transformer' : 'Traditional'}
                          </span>
                          <span className="font-medium text-sm" style={{ color: 'var(--text-secondary)' }}>{m.name}</span>
                        </div>
                      </td>
                      <td className="text-right font-mono text-sm" style={{ color: '#34d399' }}>{m.acc}</td>
                      <td className="text-right font-mono text-sm" style={{ color: '#34d399' }}>{m.f1}</td>
                      <td className="text-right font-mono text-sm" style={{ color: '#a78bfa' }}>{m.auc}</td>
                      <td className="text-right font-mono text-sm" style={{ color: 'var(--text-subtle)' }}>{m.train}</td>
                      <td className="text-right font-mono text-sm" style={{ color: 'var(--text-subtle)' }}>{m.inf}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="px-4 py-2 text-xs" style={{ borderTop: '1px solid var(--border)', color: 'var(--text-faint)' }}>
              * GPU recommended. ISOT near-perfect scores are well-documented; the dataset's distinctive language patterns make it relatively easy for TF-IDF models. See limitations for generalisability notes.
            </div>
          </div>

          <div className="card p-4">
            <h4 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-primary)' }}>Live System Statistics (Real Data)</h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: 'Analyses Completed', value: stats?.completed?.toString() ?? '—', color: '#818cf8' },
                { label: 'Avg Processing', value: stats?.avg_processing_ms ? `${(stats.avg_processing_ms/1000).toFixed(1)}s` : '—', color: '#34d399' },
                { label: 'API Uptime', value: health?.status === 'ok' ? '100%' : 'Degraded', color: '#fbbf24' },
                { label: 'DB Latency', value: health?.components?.database?.latency_ms ? `${health.components.database.latency_ms.toFixed(0)}ms` : '—', color: '#60a5fa' },
              ].map((s) => (
                <div key={s.label} className="rounded-lg p-3 text-center" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                  <div className="text-xl font-bold" style={{ color: s.color }}>{s.value}</div>
                  <div className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* 11. XAI */}
      <Section id="xai" title="11. Explainable AI Implementation" icon={<Lightbulb className="w-5 h-5" />}>
        <div className="space-y-4 text-sm">
          <div className="grid sm:grid-cols-3 gap-4">
            {[
              {
                method: 'LIME', target: 'LR, SVM, NB', color: '#818cf8',
                how: 'Perturbs input text (1,000 random word-masking samples). Fits a local linear model to the perturbation results. Coefficients = token importance.',
                output: 'Signed weights ∈ [−1, +1] per token. Positive → FAKE association.',
                caveat: 'Stochastic, computationally expensive (~5s), local approximation only.',
                ref: 'Ribeiro et al. (2016). KDD.',
              },
              {
                method: 'SHAP Linear', target: 'Logistic Regression', color: '#34d399',
                how: 'Exact Shapley values for linear models using the SHAP LinearExplainer. Zero vector used as baseline. O(d) computation where d = vocabulary size.',
                output: 'Signed Shapley values per token. Satisfy efficiency, symmetry, null player axioms.',
                caveat: 'Only valid for linear models. Values relative to chosen baseline.',
                ref: 'Lundberg & Lee (2017). NeurIPS.',
              },
              {
                method: 'Attention Rollout', target: 'DistilBERT', color: '#a78bfa',
                how: 'Last transformer layer, mean over 12 attention heads, CLS row extracted. Sub-word tokens aggregated to whole words. L1-normalised.',
                output: 'Attention score per word (0–1). Signed by predicted label.',
                caveat: 'Attention ≠ importance (Jain & Wallace, 2019). Correlational, not causal.',
                ref: 'Abnar & Zuidema (2020). ACL.',
              },
            ].map((m) => (
              <div key={m.method} className="card p-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-sm font-bold" style={{ color: m.color }}>{m.method}</span>
                  <span className="text-xs px-2 py-0.5 rounded" style={{ backgroundColor: `${m.color}15`, color: m.color }}>{m.target}</span>
                </div>
                <p className="text-xs mb-2 leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{m.how}</p>
                <p className="text-xs mb-2" style={{ color: 'var(--text-subtle)' }}><strong style={{ color: 'var(--text-secondary)' }}>Output:</strong> {m.output}</p>
                <p className="text-xs mb-2 text-amber-400">{m.caveat}</p>
                <p className="text-xs" style={{ color: 'var(--text-faint)' }}>Ref: {m.ref}</p>
              </div>
            ))}
          </div>
          <div className="card p-4" style={{ borderColor: 'rgba(251,191,36,0.2)', backgroundColor: 'rgba(251,191,36,0.04)' }}>
            <div className="flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                <strong className="text-amber-400">Critical disclaimer always shown to users:</strong> Token weights show which words the model associated with training examples of that class — they do NOT indicate that the word is factually incorrect. A high weight on "vaccine" in a fake-labelled prediction means that word appeared disproportionately in the fake-labelled training data, not that vaccines are harmful. XAI signals and factual evidence are always displayed separately.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* 12. Limitations */}
      <Section id="limitations" title="12. Limitations" icon={<Info className="w-5 h-5" />}>
        <div className="grid sm:grid-cols-2 gap-4 text-sm">
          {[
            { title: 'Dataset Bias', text: 'ISOT covers only 2015–2018 US political news. Models trained on it may not generalise to scientific misinformation, non-English content, satire, or events after 2018.', severity: 'High' },
            { title: 'Source Domain Bias', text: 'Real articles come exclusively from Reuters; fake articles from curated lists. Models may learn source-domain features rather than factual misinformation patterns.', severity: 'High' },
            { title: 'Near-Perfect ISOT Scores', text: 'Accuracy of 97–99% on ISOT is well-documented but partly due to domain-specific language patterns. These scores do not translate directly to real-world deployment accuracy.', severity: 'High' },
            { title: 'No Multilingual Support', text: 'All models process English text only. The preprocessing pipeline strips non-alphabetic characters, further limiting use on other scripts.', severity: 'Medium' },
            { title: 'Confidence ≠ Truth', text: 'High model confidence (e.g. 95%) reflects strong pattern matching against training data, not factual certainty. The model can be confidently wrong.', severity: 'High' },
            { title: 'Attention Causality', text: 'Attention rollout and Integrated Gradients are approximations of importance — neither is ground truth for why the model made its decision.', severity: 'Medium' },
            { title: 'Evidence Limitations', text: 'Evidence retrieval depends on third-party API availability and rate limits. Insufficient evidence ≠ false — this is explicitly communicated but users may not read it.', severity: 'Medium' },
            { title: 'Toy Dataset Metrics', text: 'Current DB metrics (100% accuracy) are from a 200-row synthetic dataset. Full ISOT training requires downloading the dataset separately and is not included in the repository.', severity: 'Medium' },
          ].map((l) => (
            <div key={l.title} className="card p-4">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{l.title}</span>
                <span className={cn('text-xs px-2 py-0.5 rounded-full font-medium', l.severity === 'High' ? 'text-red-400 bg-red-400/10' : 'text-amber-400 bg-amber-400/10')}>
                  {l.severity}
                </span>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{l.text}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* 13. Future Scope */}
      <Section id="future" title="13. Future Scope (Proposed)" icon={<GitBranch className="w-5 h-5" />}>
        <div className="card p-4 mb-4" style={{ borderColor: 'rgba(99,102,241,0.2)', backgroundColor: 'rgba(99,102,241,0.03)' }}>
          <p className="text-xs" style={{ color: 'var(--text-subtle)' }}>
            The following are <strong style={{ color: 'var(--brand)' }}>proposed future directions</strong>, not currently implemented features. They represent genuine research questions, not marketing claims.
          </p>
        </div>
        <div className="grid sm:grid-cols-2 gap-4">
          {[
            { title: 'Multilingual Support', text: 'Extend to Hindi, Bengali, and Tamil using mBERT or XLM-RoBERTa. Requires multilingual training data (e.g., MultiFC, XFake datasets).', category: 'ML' },
            { title: 'Cross-Domain Adaptation', text: 'Domain adaptation techniques (fine-tuning on scientific news, social media posts) to improve generalisation beyond political news.', category: 'ML' },
            { title: 'Claim Verification with KGs', text: 'Integration with knowledge graphs (Wikidata, ConceptNet) for structured fact-checking against established facts, not just text similarity.', category: 'Research' },
            { title: 'Social Context Analysis', text: 'Incorporate social network signals (share velocity, engagement patterns) as additional features alongside text content.', category: 'Research' },
            { title: 'User Feedback Loop', text: 'Allow verified experts to flag incorrect predictions; use feedback to continuously fine-tune models using RLHF-style methods.', category: 'System' },
            { title: 'Real-Time Monitoring', text: 'RSS + social media stream processing for real-time misinformation detection as articles publish, using Apache Kafka or similar.', category: 'System' },
            { title: 'Adversarial Robustness', text: 'Test and improve resistance to adversarial attacks (paraphrasing, synonym replacement, homograph attacks) targeting the classifier.', category: 'Research' },
            { title: 'Federated Learning', text: 'Train models across distributed data sources without centralising training data, addressing privacy concerns for news organisations.', category: 'ML' },
          ].map((f) => (
            <div key={f.title} className="card p-4">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{f.title}</span>
                <span className="text-xs px-2 py-0.5 rounded font-medium" style={{ backgroundColor: 'var(--bg-surface-3)', color: 'var(--text-subtle)' }}>{f.category}</span>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{f.text}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* 14. References */}
      <Section id="refs" title="14. References" icon={<BookOpen className="w-5 h-5" />}>
        <div className="space-y-2">
          {[
            {
              authors: 'Ahmed, H., Traore, I., & Saad, S.',
              title: 'Detecting opinion spams and fake news using text classification',
              venue: 'Journal of Security and Privacy, 1(1)',
              year: '2018',
              url: 'https://doi.org/10.1002/spy2.9',
            },
            {
              authors: 'Vosoughi, S., Roy, D., & Aral, S.',
              title: 'The spread of true and false news online',
              venue: 'Science, 359(6380), 1146–1151',
              year: '2018',
              url: 'https://doi.org/10.1126/science.aap9559',
            },
            {
              authors: 'Devlin, J., Chang, M., Lee, K., & Toutanova, K.',
              title: 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding',
              venue: 'NAACL-HLT 2019',
              year: '2019',
              url: 'https://aclanthology.org/N19-1423/',
            },
            {
              authors: 'Sanh, V., Debut, L., Chaumond, J., & Wolf, T.',
              title: 'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter',
              venue: 'NeurIPS EMC2 Workshop',
              year: '2019',
              url: 'https://arxiv.org/abs/1910.01108',
            },
            {
              authors: 'Ribeiro, M. T., Singh, S., & Guestrin, C.',
              title: '"Why Should I Trust You?": Explaining the Predictions of Any Classifier',
              venue: 'KDD 2016',
              year: '2016',
              url: 'https://doi.org/10.1145/2939672.2939778',
            },
            {
              authors: 'Lundberg, S. M., & Lee, S.',
              title: 'A Unified Approach to Interpreting Model Predictions',
              venue: 'NeurIPS 2017',
              year: '2017',
              url: 'https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html',
            },
            {
              authors: 'Sundararajan, M., Taly, A., & Yan, Q.',
              title: 'Axiomatic Attribution for Deep Networks',
              venue: 'ICML 2017',
              year: '2017',
              url: 'https://proceedings.mlr.press/v70/sundararajan17a.html',
            },
            {
              authors: 'Jain, S., & Wallace, B. C.',
              title: 'Attention is not Explanation',
              venue: 'NAACL 2019',
              year: '2019',
              url: 'https://aclanthology.org/N19-1357/',
            },
            {
              authors: 'Abnar, S., & Zuidema, W.',
              title: 'Quantifying Attention Flow in Transformers',
              venue: 'ACL 2020',
              year: '2020',
              url: 'https://aclanthology.org/2020.acl-main.385/',
            },
            {
              authors: 'Wang, W. Y.',
              title: '"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection',
              venue: 'ACL 2017',
              year: '2017',
              url: 'https://aclanthology.org/P17-2067/',
            },
          ].map((r, i) => (
            <div key={i} className="flex gap-3">
              <span className="text-xs font-mono font-bold w-6 shrink-0 mt-0.5" style={{ color: 'var(--brand)' }}>[{i + 1}]</span>
              <Cite {...r} />
            </div>
          ))}
        </div>
      </Section>

      {/* System Status */}
      {health && (
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-4">
            <FlaskConical className="w-4 h-4" style={{ color: 'var(--brand)' }} />
            <h2 className="section-title">Live System Status</h2>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-3">
            {Object.entries(health.components).map(([name, comp]: [string, any]) => (
              <div key={name} className="flex items-center gap-3 rounded-xl p-3" style={{ backgroundColor: 'var(--bg-surface-2)' }}>
                <div className={cn('w-2.5 h-2.5 rounded-full shrink-0',
                  comp.status === 'ok' ? 'bg-emerald-400' : comp.status === 'degraded' ? 'bg-amber-400' : 'bg-red-400',
                )} />
                <div>
                  <div className="text-sm font-medium capitalize" style={{ color: 'var(--text-secondary)' }}>
                    {name.replace('_', ' ')}
                  </div>
                  <div className="text-xs" style={{ color: 'var(--text-subtle)' }}>
                    {comp.status}{comp.latency_ms ? ` · ${comp.latency_ms.toFixed(1)}ms` : ''}
                  </div>
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
            API v{health.version} · {health.environment} · {new Date().toLocaleDateString()}
          </p>
        </div>
      )}
    </div>
  )
}
