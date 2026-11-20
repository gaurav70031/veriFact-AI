/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Brand — indigo/violet
        brand: {
          50:  '#eef2ff',
          100: '#e0e7ff',
          200: '#c7d2fe',
          300: '#a5b4fc',
          400: '#818cf8',
          500: '#6366f1',
          600: '#4f46e5',
          700: '#4338ca',
          800: '#3730a3',
          900: '#312e81',
          950: '#1e1b4b',
        },
        // Navy — deep dark backgrounds
        navy: {
          950: '#020917',
          900: '#060e1f',
          850: '#0a1428',
          800: '#0e1a30',
          750: '#132038',
          700: '#192742',
          600: '#1e3054',
          500: '#243a68',
        },
        // Violet accent
        violet: {
          50:  '#f5f3ff',
          100: '#ede9fe',
          200: '#ddd6fe',
          300: '#c4b5fd',
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed',
          700: '#6d28d9',
          800: '#5b21b6',
          900: '#4c1d95',
          950: '#2e1065',
        },
        // Semantic token aliases
        surface: {
          DEFAULT: 'var(--bg-surface)',
          2:       'var(--bg-surface-2)',
          3:       'var(--bg-surface-3)',
        },
        border:  { DEFAULT: 'var(--border)' },
        text: {
          primary:   'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          muted:     'var(--text-muted)',
          subtle:    'var(--text-subtle)',
          faint:     'var(--text-faint)',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'glow-brand':   '0 0 20px rgba(99,102,241,0.25)',
        'glow-violet':  '0 0 20px rgba(139,92,246,0.25)',
        'glow-emerald': '0 0 20px rgba(52,211,153,0.2)',
        'card':         '0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2)',
        'card-hover':   '0 4px 20px rgba(0,0,0,0.4), 0 2px 4px rgba(0,0,0,0.3)',
        'sidebar':      '1px 0 0 rgba(255,255,255,0.04)',
      },
      animation: {
        'fade-in':       'fadeIn 0.25s ease-out',
        'fade-in-up':    'fadeInUp 0.3s ease-out',
        'slide-in-left': 'slideInLeft 0.3s ease-out',
        'slide-up':      'slideUp 0.4s ease-out',
        'pulse-slow':    'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
        'shimmer':       'shimmer 1.6s infinite',
      },
      keyframes: {
        fadeIn: {
          '0%':   { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeInUp: {
          '0%':   { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideInLeft: {
          '0%':   { opacity: '0', transform: 'translateX(-16px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        slideUp: {
          '0%':   { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
      backgroundImage: {
        'gradient-brand':   'linear-gradient(135deg, #6366f1, #8b5cf6)',
        'gradient-dark':    'linear-gradient(180deg, #060e1f 0%, #020917 100%)',
        'shimmer-gradient': 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.05) 50%, transparent 100%)',
      },
      backgroundColor: {
        page:        'var(--bg-page)',
        surface:     'var(--bg-surface)',
        'surface-2': 'var(--bg-surface-2)',
        'surface-3': 'var(--bg-surface-3)',
        input:       'var(--bg-input)',
      },
      textColor: {
        primary:   'var(--text-primary)',
        secondary: 'var(--text-secondary)',
        muted:     'var(--text-muted)',
        subtle:    'var(--text-subtle)',
        faint:     'var(--text-faint)',
      },
      borderColor: {
        DEFAULT: 'var(--border)',
        subtle:  'var(--border-subtle)',
      },
    },
  },
  plugins: [],
}
