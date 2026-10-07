import { useState, useRef, useEffect } from 'react'
import { NavLink, Link, useNavigate } from 'react-router-dom'
import {
  Menu, X, User, LogOut, LogIn,
  UserPlus, ChevronDown, MoreHorizontal,
  Sun, Moon,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useHealth } from '@/hooks/useApi'
import { useAuth } from '@/context/AuthContext'
import { useTheme } from '@/context/ThemeContext'
import { LogoIcon } from '@/components/ui/LogoIcon'

// Primary nav — visible to everyone at a glance
const primaryNav = [
  { to: '/',         label: 'Home'          },
  { to: '/analyze',  label: 'Check a Story' },
  { to: '/history',  label: 'My History'    },
]

// Advanced nav — tucked under "More"
const moreNav = [
  { to: '/live-news',   label: 'Live News'        },
  { to: '/analytics',   label: 'Analytics'        },
  { to: '/performance', label: 'Model Performance' },
  { to: '/methodology', label: 'Methodology'       },
  { to: '/about',       label: 'About'             },
]

// ── "More" dropdown ───────────────────────────────────────────────────────────

function MoreMenu() {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          'flex items-center gap-1 px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
          open
            ? 'text-brand-400 bg-brand-500/10'
            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800',
        )}
      >
        <MoreHorizontal className="w-4 h-4" />
        More
        <ChevronDown className={cn('w-3 h-3 transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-52 rounded-xl shadow-xl z-50
                        overflow-hidden animate-fade-in py-1"
             style={{ backgroundColor: 'var(--bg-surface)', border: '1px solid var(--border)' }}>
          {moreNav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                cn(
                  'block px-4 py-2.5 text-sm transition-colors',
                  isActive
                    ? 'text-brand-500 bg-brand-500/10'
                    : 'hover:bg-surface-2',
                )
              }
              style={({ isActive }) => ({ color: isActive ? undefined : 'var(--text-muted)' })}
            >
              {item.label}
            </NavLink>
          ))}
        </div>
      )}
    </div>
  )
}

// ── User menu ─────────────────────────────────────────────────────────────────

function UserMenu() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

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
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm
                     text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
        >
          <LogIn className="w-3.5 h-3.5" />
          Sign in
        </Link>
        <Link
          to="/register"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm
                     bg-brand-600 hover:bg-brand-500 text-white font-medium transition-colors"
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
        className="flex items-center gap-2 px-3 py-1.5 rounded-md text-sm
                   text-slate-300 hover:bg-slate-800 transition-colors"
      >
        <div className="w-6 h-6 rounded-full bg-brand-600 flex items-center justify-center
                        text-xs font-semibold text-white">
          {user.username[0].toUpperCase()}
        </div>
        <span className="hidden md:block max-w-[100px] truncate">{user.username}</span>
        <ChevronDown className={cn('w-3.5 h-3.5 transition-transform', open && 'rotate-180')} />
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-2 w-48 rounded-xl shadow-xl z-50
                        overflow-hidden animate-fade-in"
             style={{ backgroundColor: 'var(--bg-surface)', border: '1px solid var(--border)' }}>
          <div className="px-3 py-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
            <div className="text-sm font-medium truncate" style={{ color: 'var(--text-primary)' }}>
              {user.username}
            </div>
            <div className="text-xs truncate" style={{ color: 'var(--text-subtle)' }}>
              {user.email}
            </div>
          </div>
          <Link
            to="/history"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 px-3 py-2.5 text-sm text-slate-400
                       hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <User className="w-3.5 h-3.5" />
            My History
          </Link>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-2 px-3 py-2.5 text-sm
                       text-red-400 hover:bg-red-500/10 transition-colors"
          >
            <LogOut className="w-3.5 h-3.5" />
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}

// ── API status dot ────────────────────────────────────────────────────────────

function ApiStatusDot({ isHealthy, loading }: { isHealthy: boolean; loading: boolean }) {
  const [v, setV] = useState(false)
  return (
    <span
      className="relative"
      onMouseEnter={() => setV(true)}
      onMouseLeave={() => setV(false)}
    >
      <div className={cn(
        'w-2 h-2 rounded-full cursor-default',
        loading    ? 'bg-slate-600' :
        isHealthy  ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' :
                     'bg-red-400',
      )} />
      {v && (
        <span className="absolute right-0 top-full mt-1.5 whitespace-nowrap px-2.5 py-1.5
                         rounded-lg text-xs shadow-xl z-50"
              style={{
                backgroundColor: 'var(--bg-surface-2)',
                color: 'var(--text-secondary)',
                border: '1px solid var(--border)',
              }}>
          API {loading ? 'checking…' : isHealthy ? 'Online' : 'Offline'}
        </span>
      )}
    </span>
  )
}

// ── Main Navbar ───────────────────────────────────────────────────────────────

export function Navbar() {
  const [open, setOpen]      = useState(false)
  const { data: health }     = useHealth()
  const { user, logout }     = useAuth()
  const { isDark, toggleTheme } = useTheme()
  const navigate             = useNavigate()
  const isHealthy            = health?.status === 'ok'

  const handleMobileLogout = async () => {
    setOpen(false)
    await logout()
    navigate('/')
  }

  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl transition-colors duration-300"
            style={{
              backgroundColor: 'color-mix(in srgb, var(--bg-page) 85%, transparent)',
              borderBottom: '1px solid var(--border-subtle)',
            }}>
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group shrink-0">
            <LogoIcon size={34} />
            <span className="font-bold text-slate-100 tracking-tight text-lg leading-none">
              Veri<span className="text-brand-400">Fact</span>
              <span className="text-emerald-400"> AI</span>
            </span>
          </Link>

          {/* Desktop primary nav */}
          <div className="hidden lg:flex items-center gap-0.5">
            {primaryNav.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                    isActive
                      ? 'text-brand-500 bg-brand-500/10'
                      : 'text-slate-500 hover:text-slate-800 hover:bg-gray-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
            <MoreMenu />
          </div>

          {/* Right: status + theme toggle + auth */}
          <div className="flex items-center gap-2">
            <ApiStatusDot isHealthy={isHealthy} loading={health === undefined} />

            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
              className="p-2 rounded-lg transition-colors duration-200
                         text-slate-500 hover:text-slate-700 hover:bg-gray-100
                         dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800"
            >
              {isDark
                ? <Sun  className="w-4 h-4 text-amber-400" />
                : <Moon className="w-4 h-4 text-brand-600" />
              }
            </button>

            <div className="hidden lg:block">
              <UserMenu />
            </div>
            <button
              onClick={() => setOpen(!open)}
              className="lg:hidden p-2 rounded-md transition-colors
                         text-slate-500 hover:text-slate-800 hover:bg-gray-100
                         dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800"
            >
              {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {open && (
          <div className="lg:hidden py-3 pb-4 border-t border-gray-200 dark:border-slate-800
                          space-y-1 animate-fade-in">
            {primaryNav.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                onClick={() => setOpen(false)}
                className={({ isActive }) =>
                  cn(
                    'block px-4 py-2.5 rounded-md text-sm font-medium',
                    isActive
                      ? 'text-brand-500 bg-brand-500/10'
                      : 'text-slate-500 hover:text-slate-900 hover:bg-gray-100 dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}

            <div className="pt-2 mt-1 border-t border-gray-200/60 dark:border-slate-800/60">
              <div className="px-4 py-1.5 text-xs text-slate-400 dark:text-slate-600
                              font-medium uppercase tracking-wider">
                More
              </div>
              {moreNav.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  onClick={() => setOpen(false)}
                  className={({ isActive }) =>
                    cn(
                      'block px-4 py-2 rounded-md text-sm',
                      isActive
                        ? 'text-brand-500 bg-brand-500/10'
                        : 'text-slate-500 hover:text-slate-900 hover:bg-gray-100 dark:text-slate-500 dark:hover:text-slate-300 dark:hover:bg-slate-800',
                    )
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </div>

            {/* Mobile auth */}
            <div className="pt-2 mt-2 border-t border-gray-200 dark:border-slate-800">
              {user ? (
                <>
                  <div className="px-4 py-2 text-xs text-slate-400 dark:text-slate-500">
                    Signed in as{' '}
                    <span className="text-slate-700 dark:text-slate-300">{user.username}</span>
                  </div>
                  <button
                    onClick={handleMobileLogout}
                    className="flex items-center gap-2 w-full px-4 py-2.5 rounded-md
                               text-sm text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 transition-colors"
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
                    className="flex items-center gap-2 px-4 py-2.5 rounded-md text-sm
                               text-slate-500 hover:text-slate-900 hover:bg-gray-100
                               dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-800"
                  >
                    <LogIn className="w-4 h-4" />
                    Sign in
                  </Link>
                  <Link
                    to="/register"
                    onClick={() => setOpen(false)}
                    className="flex items-center gap-2 px-4 py-2.5 rounded-md text-sm
                               bg-brand-50 text-brand-600 hover:bg-brand-100
                               dark:bg-brand-600/20 dark:text-brand-400 dark:hover:bg-brand-600/30"
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
