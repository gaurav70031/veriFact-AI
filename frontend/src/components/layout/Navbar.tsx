import { useState } from 'react'
import { NavLink, Link } from 'react-router-dom'
import { Menu, X, ShieldCheck } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useHealth } from '@/hooks/useApi'

const navItems = [
  { to: '/',              label: 'Home' },
  { to: '/analyze',       label: 'Analyze' },
  { to: '/live-news',     label: 'Live News' },
  { to: '/history',       label: 'History' },
  { to: '/analytics',     label: 'Analytics' },
  { to: '/performance',   label: 'Models' },
  { to: '/methodology',   label: 'Methodology' },
  { to: '/about',         label: 'About' },
]

export function Navbar() {
  const [open, setOpen] = useState(false)
  const { data: health } = useHealth()
  const isHealthy = health?.status === 'ok'

  return (
    <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/60">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group">
            <div className="p-1.5 rounded-lg bg-brand-600/20 border border-brand-500/30 group-hover:border-brand-400/50 transition-colors">
              <ShieldCheck className="w-5 h-5 text-brand-400" />
            </div>
            <span className="font-semibold text-slate-100 tracking-tight">
              Veritas<span className="text-brand-400">AI</span>
            </span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden lg:flex items-center gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                    isActive
                      ? 'text-brand-400 bg-brand-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>

          {/* Status dot + mobile toggle */}
          <div className="flex items-center gap-3">
            <Tooltip content={`API ${isHealthy ? 'Online' : 'Offline'}`}>
              <div className={cn(
                'w-2 h-2 rounded-full',
                health === undefined ? 'bg-slate-600' :
                isHealthy ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-red-400'
              )} />
            </Tooltip>
            <button
              onClick={() => setOpen(!open)}
              className="lg:hidden p-2 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800"
            >
              {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {open && (
          <div className="lg:hidden py-3 pb-4 border-t border-slate-800 space-y-1 animate-fade-in">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  cn(
                    'block px-4 py-2.5 rounded-md text-sm font-medium',
                    isActive
                      ? 'text-brand-400 bg-brand-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        )}
      </nav>
    </header>
  )
}

function Tooltip({ content, children }: { content: string; children: React.ReactNode }) {
  const [v, setV] = useState(false)
  return (
    <span className="relative" onMouseEnter={() => setV(true)} onMouseLeave={() => setV(false)}>
      {children}
      {v && (
        <span className="absolute right-0 top-full mt-1.5 whitespace-nowrap px-2.5 py-1.5 rounded-lg text-xs bg-slate-800 text-slate-200 border border-slate-700 shadow-xl">
          {content}
        </span>
      )}
    </span>
  )
}
