/**
 * WordCloud — lightweight SVG word cloud (no external wordcloud library).
 * Scales font size by count, colours by label (fake=red, real=green, all=brand).
 */

interface WordCloudProps {
  words: Array<{ word: string; count: number; pct?: number }>
  label?: 'all' | 'FAKE' | 'REAL'
  maxWords?: number
  width?: number
  height?: number
}

export function WordCloud({ words, label = 'all', maxWords = 50, width = 600, height = 300 }: WordCloudProps) {
  if (!words?.length) return null

  const top = words.slice(0, maxWords)
  const maxCount = Math.max(...top.map((w) => w.count), 1)
  const minCount = Math.min(...top.map((w) => w.count), 1)

  // Font size range: 12px (smallest) to 36px (largest)
  const minFont = 12
  const maxFont = 36

  const colors: Record<string, string> = {
    FAKE: '#f87171',
    REAL: '#34d399',
    all:  '#818cf8',
  }
  const baseColor = colors[label] ?? colors.all

  // Spiral placement (simple deterministic layout)
  const placed: Array<{ word: string; size: number; x: number; y: number; opacity: number }> = []
  let cx = width / 2
  let cy = height / 2
  let angle = 0
  let radius = 0

  for (const w of top) {
    const ratio = maxCount > minCount ? (w.count - minCount) / (maxCount - minCount) : 0.5
    const fontSize = minFont + ratio * (maxFont - minFont)
    const opacity = 0.4 + ratio * 0.6

    // Spiral outwards from centre
    const x = cx + radius * Math.cos(angle) - (w.word.length * fontSize) / 4
    const y = cy + radius * Math.sin(angle) + fontSize / 3
    angle += 0.5
    radius += fontSize * 0.8

    // Wrap around if off-screen
    const rx = ((x % width) + width) % width
    const ry = Math.max(fontSize, Math.min(y, height - fontSize))

    placed.push({ word: w.word, size: fontSize, x: rx, y: ry, opacity })
  }

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: 'visible' }}>
      {placed.map((p, i) => (
        <text
          key={i}
          x={p.x}
          y={p.y}
          fontSize={p.size}
          fill={baseColor}
          opacity={p.opacity}
          fontFamily="monospace"
          fontWeight={p.size > 24 ? 700 : 400}
          textAnchor="start"
        >
          {p.word}
        </text>
      ))}
    </svg>
  )
}
