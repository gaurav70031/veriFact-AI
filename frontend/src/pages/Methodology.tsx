import { ArrowRight } from 'lucide-react'

const steps = [
  {
    n: '01',
    title: 'Input',
    color: 'text-brand-400',
    items: [
      'Raw text / article body',
      'Public HTTPS URL (server-side extraction)',
      'Short factual claim with optional context',
    ],
  },
  {
    n: '02',
    title: 'ML Inference',
    color: 'text-violet-400',
    items: [
      'TF-IDF vectorisation (50k features, 1–2 ngrams)',
      'Logistic Regression (C=1.0)',
      'LinearSVC (calibrated for probabilities)',
      'Multinomial Naive Bayes (α=0.1)',
      'DistilBERT fine-tuned on ISOT dataset',
      'Weighted ensemble (LR 20%, SVM 30%, NB 10%, BERT 40%)',
    ],
  },
  {
    n: '03',
    title: 'Claim Extraction',
    color: 'text-emerald-400',
    items: [
      'Sentence boundary detection',
      'Heuristic scoring (named entities, numbers, years)',
      'Top-N claims extracted per article',
    ],
  },
  {
    n: '04',
    title: 'Evidence Retrieval',
    color: 'text-amber-400',
    items: [
      'Query generation from claim keywords',
      'Concurrent search across NewsAPI, GNews, RSS, Web Search',
      'URL deduplication (MD5 fingerprint)',
      'Keyword overlap + recency relevance ranking',
    ],
  },
  {
    n: '05',
    title: 'Evidence Comparison',
    color: 'text-rose-400',
    items: [
      'Lexical similarity (Jaccard + keyword overlap)',
      'Contradiction pattern detection (regex-based)',
      'Optional: sentence-transformers cosine similarity',
      'Per-item classification: SUPPORTING / CONTRADICTING / INCONCLUSIVE / NOT_RELEVANT',
    ],
  },
  {
    n: '06',
    title: 'Verdict Aggregation',
    color: 'text-sky-400',
    items: [
      'Rule-based evidence assessment (5 transparent rules)',
      'LIKELY_CREDIBLE: ≥2 supporting, 0 contradicting',
      'CONTRADICTED: ≥2 contradicting, contradicting ≥ supporting',
      'LIKELY_MISLEADING: ML=FAKE + contradicting evidence',
      'INSUFFICIENT_EVIDENCE: no meaningful evidence found',
      'Assessment conflict flagged when ML and evidence disagree',
    ],
  },
  {
    n: '07',
    title: 'Explainability',
    color: 'text-orange-400',
    items: [
      'LIME for TF-IDF models (1000 perturbation samples)',
      'TF-IDF feature weight fallback when LIME unavailable',
      'Attention rollout for DistilBERT (last-layer CLS row)',
      'Sub-word aggregation to whole tokens',
      'Weights normalised to [-1, +1]',
      'Signal clearly distinguished from factual evidence',
    ],
  },
]

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <div className="mb-8">
    <h2 className="text-lg font-semibold text-slate-200 mb-3">{title}</h2>
    {children}
  </div>
)

export default function Methodology() {
  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 animate-fade-in">
      <div className="mb-10">
        <h1 className="text-3xl font-bold text-slate-100 mb-3">Methodology</h1>
        <p className="text-slate-400 leading-relaxed">
          A transparent description of how VeritasAI analyses content —
          from raw input to final verdict.
        </p>
      </div>

      {/* Pipeline steps */}
      <Section title="Analysis Pipeline">
        <div className="space-y-4">
          {steps.map((step, i) => (
            <div key={step.n} className="glass-card p-5">
              <div className="flex items-center gap-3 mb-3">
                <span className={`text-2xl font-bold font-mono ${step.color}`}>{step.n}</span>
                <h3 className="text-base font-semibold text-slate-200">{step.title}</h3>
                {i < steps.length - 1 && <ArrowRight className="w-4 h-4 text-slate-700 ml-auto" />}
              </div>
              <ul className="space-y-1.5">
                {step.items.map((item) => (
                  <li key={item} className="text-sm text-slate-500 flex items-center gap-2">
                    <span className={`w-1.5 h-1.5 rounded-full ${step.color} opacity-60 shrink-0`} />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </Section>

      {/* Key limitations */}
      <Section title="Limitations & Caveats">
        <div className="glass-card p-5 space-y-4">
          {[
            ['Model confidence ≠ factual correctness', 'A 95% confidence score means the model strongly matches patterns from training data — not that the article is definitively fake or real.'],
            ['Training data bias', 'Models trained on ISOT (2015–2018 political news) may not generalise well to newer topics, scientific claims, or non-English content.'],
            ['Evidence absence', 'If no evidence is found, the verdict is INSUFFICIENT_EVIDENCE — never interpreted as proof of falsehood.'],
            ['URL extraction limits', 'Paywalled, JavaScript-rendered, or robots.txt-restricted pages cannot be extracted.'],
            ['Explainability caveats', 'Token weights show which words the model attended to — not which words are factually incorrect.'],
          ].map(([title, desc]) => (
            <div key={title as string}>
              <div className="text-sm font-medium text-slate-300 mb-1">{title}</div>
              <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Training dataset */}
      <Section title="Training Dataset">
        <div className="glass-card p-5">
          <div className="text-sm font-semibold text-slate-300 mb-2">ISOT Fake News Dataset</div>
          <p className="text-xs text-slate-500 leading-relaxed mb-3">
            Ahmed, H., Traore, I., Saad, S. (2018). "Detecting opinion spams and fake news using text classification."
            Journal of Security and Privacy.
          </p>
          <div className="grid grid-cols-3 gap-3 text-center">
            {[
              { label: 'Total Articles', value: '~44,900' },
              { label: 'Fake Articles',  value: '~23,500' },
              { label: 'Real Articles',  value: '~21,400' },
            ].map((s) => (
              <div key={s.label} className="rounded-lg bg-slate-800/50 p-3">
                <div className="text-lg font-bold text-slate-200">{s.value}</div>
                <div className="text-xs text-slate-600 mt-0.5">{s.label}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-600 mt-3">
            Dataset used for academic research purposes. Full articles not redistributed.
          </p>
        </div>
      </Section>
    </div>
  )
}
