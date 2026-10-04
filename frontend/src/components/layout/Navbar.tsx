import { useState, useRef, useEffect } from 'react'
import { NavLink, Link, useNavigate } from 'react-router-dom'
import { Menu, X, ShieldCheck, User, LogOut, LogIn, UserPlus, ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useHealth } from '@/hooks/useApi'
import { useAuth } from '@/context/AuthContext'

const navItems = [
  { to: '/',            label: 'Home' },
  { to: '/analyze',     label: 'Analyze' },
  { to: '/live-news',   label: 'Live News' },
  { to: '/history',     label: 'History' },
  { to: '/analytics',   label: 'Analytics' },
  { to: '/performance', label: 'Models' },
  { to: '/methodology', label: 'Methodology' },
  { to: '/about',       label: 'About' },
]

// ── User menu (desktop) ───────────────────────────────────────────────────────

function UserMenu() {
  const { user, logout } = useAuth()
  const navigate          = useNavigate()
  const [open, setOpen]   = useState(false)
  const ref               = useRef<HTMLDivElement>(null)

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleLogout = async () => {
    setOpen(false)
    await logout()
    navigate('/')
  }

  if (!user) {
    return (
      <div className="flex items-center gap-1">
        <Link
          to="/login"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
        >
          <LogIn className="w-3.5 h-3.5" />
          Sign in
        </Link>
        <Link
          to="/register"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm bg-brand-600 hover:bg-brand-500 text-white font-medium transition-colors"
        >
          <UserPlus className="w-3.5 h-3.5" />
          Register
        </Link>
      </div>
    )
  }

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm text-slate-300 hover:bg-slate-800 transition-colors"
      >
        <div className="w-6 h-6 rounded-full bg-brand-600 flex items-center justify-center text-xs font-semibold text-white">
          {user.username[0].toUpperCase()}
        </div>
        <span className="hidden md:block max-w-[100px] truncate">{user.username}</span>
        <ChevronDown className={cn('w-3.5 h-3.5 transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-48 bg-slate-900 border border-slate-700 rounded-xl shadow-xl z-50 overflow-hidden animate-fade-in">
          <div className="px-3 py-2.5 border-b border-slate-800">
            <div className="text-sm font-medium text-slate-200 truncate">{user.username}</div>
            <div className="text-xs text-slate-500 truncate">{user.email}</div>
            <div className="text-xs text-brand-500 mt-0.5 capitalize">{user.role}</div>
          </div>
          <Link
            to="/history"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 px-3 py-2.5 text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <User className="w-3.5 h-3.5" />
            My History
          </Link>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-2 px-3 py-2.5 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
          >
            <LogOut className="w-3.5 h-3.5" />
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}

// ── Main Navbar ───────────────────────────────────────────────────────────────

export function Navbar() {
  const [open, setOpen]   = useState(false)
  const { data: health }  = useHealth()
  const { user, logout }  = useAuth()
  const navigate          = useNavigate()
  const isHealthy         = health?.status === 'ok'

  const handleMobileLogout = async () => {
    setOpen(false)
    await logout()
    navigate('/')
  }

  return (
    <header className="sticky top-0 z-50 bg-slate-950/80 backdrop-blur-xl border-b border-slate-800/60">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2.5 group shrink-0">
            <div className="p-1.5 rounded-lg bg-brand-600/20 border border-brand-500/30 group-hover:border-brand-400/50 transition-colors">
              <ShieldCheck className="w-5 h-5 text-brand-400" />
            </div>
            <span className="font-semibold text-slate-100 tracking-tight">
              Veritas<span className="text-brand-400">AI</span>
            </span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden lg:flex items-center gap-0.5">
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

          {/* Right side: API status + auth */}
          <div className="flex items-center gap-3">
            {/* API status dot */}
            <ApiStatusDot isHealthy={isHealthy} loading={health === undefined} />

            {/* User menu (desktop) */}
            <div className="hidden lg:block">
              <UserMenu />
            </div>

            {/* Mobile toggle */}
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

            {/* Mobile auth */}
            <div className="pt-2 mt-2 border-t border-slate-800">
              {user ? (
                <>
                  <div className="px-4 py-2 text-xs text-slate-500">
                    Signed in as <span className="text-slate-300">{user.username}</span>
                  </div>
                  <button
                    onClick={handleMobileLogout}
                    className="flex items-center gap-2 w-full px-4 py-2.5 rounded-md text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                  >
                    <LogOut className="w-4 h-4" />
                    Sign out
                  </button>
                </>
              ) : (
                <div className="flex flex-col gap-1 px-2">
                  <Link
                    to="/login"
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-2 px-4 py-2.5 rounded-md text-sm text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                  >
                    <LogIn className="w-4 h-4" />
                    Sign in
                  </Link>
                  <Link
                    to="/register"
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-2 px-4 py-2.5 rounded-md text-sm bg-brand-600/20 text-brand-400 hover:bg-brand-600/30"
                  >
                    <UserPlus className="w-4 h-4" />
                    Create account
                  </Link>
                </div>
              )}
            </div>
          </div>
        )}
      </nav>
    </header>
  )
}

// ── API status dot ────────────────────────────────────────────────────────────

function ApiStatusDot({ isHealthy, loading }: { isHealthy: boolean; loading: boolean }) {
  const [v, setV] = useState(false)
  return (
    <span className="relative" onMouseEnter={() => setV(true)} onMouseLeave={() => setV(false)}>
      <div className={cn(
        'w-2 h-2 rounded-full cursor-default',
        loading ? 'bg-slate-600' :
        isHealthy ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-red-400',
      )} />
      {v && (
        <span className="absolute right-0 top-full mt-1.5 whitespace-nowrap px-2.5 py-1.5 rounded-lg text-xs bg-slate-800 text-slate-200 border border-slate-700 shadow-xl z-50">
          API {loading ? 'checking…' : isHealthy ? 'Online' : 'Offline'}
        </span>
      )}
    </span>
  )
}
