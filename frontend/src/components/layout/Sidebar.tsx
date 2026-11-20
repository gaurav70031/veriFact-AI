import { useState, useRef, useEffect } from 'react'
import { NavLink, Link, useNavigate } from 'react-router-dom'
import {
  LayoutDashboard,
  Newspaper,
  ScanText,
  Link2,
  Database,
  FlaskConical,
  GitCompare,
  Lightbulb,
  BarChart3,
  History,
  Info,
  Settings,
  LogIn,
  LogOut,
  UserPlus,
  User,
  Sun,
  Moon,
  ChevronDown,
  Menu,
  X,
  Cpu,
  Shield,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useHealth } from '@/hooks/useApi'
import { useAuth } from '@/context/AuthContext'
import { useTheme } from '@/context/ThemeContext'

// ── Navigation definition ─────────────────────────────────────────────────────

const NAV_SECTIONS = [
  {
    label: 'Main',
    items: [
      { to: '/',           icon: LayoutDashboard, label: 'Overview Dashboard', end: true  },
      { to: '/live-news',  icon: Newspaper,       label: 'News Verification'              },
    ],
  },
  {
    label: 'Analysis',
    items: [
      { to: '/analyze/text', icon: ScanText,     label: 'Text Analysis'    },
      { to: '/analyze/url',  icon: Link2,         label: 'URL Verification' },
    ],
  },
  {
    label: 'Research',
    items: [
      { to: '/dataset',     icon: Database,     label: 'Dataset Explorer'       },
      { to: '/lab',         icon: FlaskConical, label: 'Model Laboratory'       },
      { to: '/comparison',  icon: GitCompare,   label: 'Model Comparison'       },
      { to: '/explainable', icon: Lightbulb,    label: 'Explainable AI'         },
    ],
  },
  {
    label: 'Insights',
    items: [
      { to: '/analytics', icon: BarChart3, label: 'Misinformation Analytics' },
      { to: '/history',   icon: History,   label: 'Verification History'      },
    ],
  },
  {
    label: 'Project',
    items: [
      { to: '/about',    icon: Info,     label: 'About Project' },
      { to: '/settings', icon: Settings, label: 'Settings'      },
    ],
  },
]

// ── API status dot ────────────────────────────────────────────────────────────

function ApiDot({ collapsed }: { collapsed: boolean }) {
  const { data: health } = useHealth()
  const isHealthy = health?.status === 'ok'
  const isLoading = health === undefined

  return (
    <div
      className={cn('flex items-center gap-2', collapsed ? 'justify-center' : '')}
      title={`API ${isLoading ? 'checking…' : isHealthy ? 'Online' : 'Offline'}`}
    >
      <span className={cn(
        'w-2 h-2 rounded-full shrink-0',
        isLoading   ? 'bg-slate-600' :
        isHealthy   ? 'bg-emerald-400 shadow-[0_0_6px_#34d399]' :
                      'bg-red-400',
      )} />
      {!collapsed && (
        <span className="text-xs" style={{ color: 'var(--sidebar-text)' }}>
          API {isLoading ? 'checking' : isHealthy ? 'Online' : 'Offline'}
        </span>
      )}
    </div>
  )
}

// ── User profile strip ────────────────────────────────────────────────────────

function UserStrip({
  collapsed,
  onLogout,
}: {
  collapsed: boolean
  onLogout: () => void
}) {
  const { user } = useAuth()
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const h = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', h)
    return () => document.removeEventListener('mousedown', h)
  }, [])

  if (!user) {
    return (
      <div className={cn('space-y-1', collapsed && 'flex flex-col items-center gap-1 space-y-0')}>
        <Link
          to="/login"
          className="nav-item w-full"
          title="Sign in"
        >
          <LogIn className="w-4 h-4 shrink-0" />
          {!collapsed && <span>Sign in</span>}
        </Link>
        <Link
          to="/register"
          className={cn(
            'nav-item w-full',
            collapsed && 'justify-center',
          )}
          title="Register"
        >
          <UserPlus className="w-4 h-4 shrink-0 text-brand-400" />
          {!collapsed && <span className="text-brand-400">Register</span>}
        </Link>
      </div>
    )
  }

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          'w-full flex items-center rounded-lg px-3 py-2.5 transition-colors',
          'hover:bg-white/5',
          collapsed ? 'justify-center' : 'gap-3',
        )}
      >
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center text-xs font-bold text-white shrink-0">
          {user.username[0].toUpperCase()}
        </div>
        {!collapsed && (
          <>
            <div className="flex-1 min-w-0 text-left">
              <div className="text-xs font-semibold truncate" style={{ color: 'var(--sidebar-text-active)' }}>
                {user.username}
              </div>
              <div className="text-xs truncate" style={{ color: 'var(--sidebar-text)', opacity: 0.6 }}>
                {user.email}
              </div>
            </div>
            <ChevronDown className={cn('w-3.5 h-3.5 transition-transform shrink-0', open && 'rotate-180')} style={{ color: 'var(--sidebar-text)' }} />
          </>
        )}
      </button>

      {open && (
        <div
          className="absolute left-full bottom-0 ml-2 w-52 rounded-xl shadow-2xl z-50 overflow-hidden animate-fade-in py-1"
          style={{ backgroundColor: 'var(--bg-surface)', border: '1px solid var(--border)' }}
        >
          <div className="px-3 py-2.5" style={{ borderBottom: '1px solid var(--border)' }}>
            <div className="text-sm font-semibold" style={{ color: 'var(--text-primary)' }}>{user.username}</div>
            <div className="text-xs mt-0.5" style={{ color: 'var(--text-subtle)' }}>{user.email}</div>
          </div>
          <Link
            to="/history"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 px-3 py-2.5 text-sm transition-colors hover:bg-white/5"
            style={{ color: 'var(--text-muted)' }}
          >
            <User className="w-3.5 h-3.5" />
            My History
          </Link>
          <Link
            to="/settings"
            onClick={() => setOpen(false)}
            className="flex items-center gap-2 px-3 py-2.5 text-sm transition-colors hover:bg-white/5"
            style={{ color: 'var(--text-muted)' }}
          >
            <Settings className="w-3.5 h-3.5" />
            Settings
          </Link>
          <button
            onClick={() => { setOpen(false); onLogout() }}
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

// ── Desktop sidebar ───────────────────────────────────────────────────────────

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false)
  const { isDark, toggleTheme }   = useTheme()
  const { logout }                = useAuth()
  const navigate                  = useNavigate()

  const handleLogout = async () => {
    await logout()
    navigate('/')
  }

  return (
    <aside
      className={cn(
        'hidden lg:flex flex-col h-screen sticky top-0',
        'transition-all duration-300 ease-in-out shrink-0',
        'sidebar-scroll overflow-y-auto overflow-x-hidden',
      )}
      style={{
        width: collapsed ? 'var(--sidebar-collapsed)' : 'var(--sidebar-width)',
        backgroundColor: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--sidebar-border)',
      }}
    >
      {/* ── Logo + collapse toggle ── */}
      <div
        className={cn(
          'flex items-center h-16 shrink-0 px-3',
          collapsed ? 'justify-center' : 'justify-between',
        )}
        style={{ borderBottom: '1px solid var(--sidebar-border)' }}
      >
        {!collapsed && (
          <Link to="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center shadow-glow-brand">
              <Shield className="w-4 h-4 text-white" />
            </div>
            <div className="leading-none">
              <div className="text-sm font-bold tracking-tight" style={{ color: 'var(--sidebar-text-active)' }}>
                Veri<span style={{ color: '#818cf8' }}>Fact</span>
              </div>
              <div className="text-xs mt-0.5" style={{ color: 'var(--sidebar-text)', opacity: 0.6 }}>
                AI Platform
              </div>
            </div>
          </Link>
        )}

        {collapsed && (
          <Link to="/" className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center">
            <Shield className="w-4 h-4 text-white" />
          </Link>
        )}

        <button
          onClick={() => setCollapsed(!collapsed)}
          className={cn(
            'p-1.5 rounded-lg transition-colors',
            'hover:bg-white/8',
            collapsed && 'hidden',
          )}
          style={{ color: 'var(--sidebar-text)' }}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <Menu className="w-4 h-4" />
        </button>
      </div>

      {/* ── Nav sections ── */}
      <nav className="flex-1 py-4 space-y-5 px-2">
        {NAV_SECTIONS.map((section) => (
          <div key={section.label}>
            {!collapsed && (
              <div
                className="px-2 mb-1.5 text-xs font-semibold uppercase tracking-widest"
                style={{ color: 'var(--sidebar-text)', opacity: 0.4 }}
              >
                {section.label}
              </div>
            )}
            {collapsed && <div className="border-t mx-1 mb-2" style={{ borderColor: 'var(--sidebar-border)' }} />}
            <ul className="space-y-0.5">
              {section.items.map(({ to, icon: Icon, label, end }) => (
                <li key={to}>
                  <NavLink
                    to={to}
                    end={end}
                    title={collapsed ? label : undefined}
                    className={({ isActive }) =>
                      cn(
                        'flex items-center rounded-lg text-sm font-medium transition-all duration-150',
                        'px-3 py-2.5',
                        collapsed ? 'justify-center' : 'gap-3',
                        isActive
                          ? 'active'
                          : 'hover:bg-white/5',
                      )
                    }
                    style={({ isActive }) => ({
                      color: isActive ? 'var(--sidebar-text-active)' : 'var(--sidebar-text)',
                      backgroundColor: isActive ? 'var(--sidebar-bg-active)' : undefined,
                      boxShadow: isActive ? 'inset 2px 0 0 var(--brand)' : undefined,
                    })}
                  >
                    {({ isActive }) => (
                      <>
                        <Icon
                          className="w-4 h-4 shrink-0"
                          style={{ color: isActive ? 'var(--brand)' : undefined }}
                        />
                        {!collapsed && (
                          <span className="text-sm leading-none">{label}</span>
                        )}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      {/* ── Bottom strip ── */}
      <div
        className="shrink-0 p-3 space-y-2"
        style={{ borderTop: '1px solid var(--sidebar-border)' }}
      >
        {/* API status */}
        <div className={cn('px-1 py-1', collapsed && 'flex justify-center')}>
          <ApiDot collapsed={collapsed} />
        </div>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className={cn(
            'w-full flex items-center rounded-lg px-3 py-2.5 transition-colors hover:bg-white/5',
            collapsed ? 'justify-center' : 'gap-3',
          )}
          style={{ color: 'var(--sidebar-text)' }}
          title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {isDark
            ? <Sun  className="w-4 h-4 shrink-0 text-amber-400" />
            : <Moon className="w-4 h-4 shrink-0 text-brand-500" />
          }
          {!collapsed && (
            <span className="text-sm">{isDark ? 'Light Mode' : 'Dark Mode'}</span>
          )}
        </button>

        {/* Collapse toggle (when collapsed, show expand) */}
        {collapsed && (
          <button
            onClick={() => setCollapsed(false)}
            className="w-full flex justify-center px-3 py-2 rounded-lg hover:bg-white/5 transition-colors"
            style={{ color: 'var(--sidebar-text)' }}
            aria-label="Expand sidebar"
          >
            <Menu className="w-4 h-4" />
          </button>
        )}

        {/* User */}
        <UserStrip collapsed={collapsed} onLogout={handleLogout} />
      </div>
    </aside>
  )
}

// ── Mobile top bar ────────────────────────────────────────────────────────────

export function MobileTopBar() {
  const [open, setOpen] = useState(false)
  const { isDark, toggleTheme } = useTheme()
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = async () => {
    setOpen(false)
    await logout()
    navigate('/')
  }

  return (
    <>
      <header
        className="lg:hidden sticky top-0 z-50 flex items-center justify-between px-4 h-14"
        style={{
          backgroundColor: 'color-mix(in srgb, var(--bg-sidebar) 92%, transparent)',
          backdropFilter: 'blur(12px)',
          borderBottom: '1px solid var(--sidebar-border)',
        }}
      >
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center">
            <Shield className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="text-sm font-bold" style={{ color: 'var(--sidebar-text-active)' }}>
            Veri<span style={{ color: '#818cf8' }}>Fact</span>
          </span>
        </Link>

        <div className="flex items-center gap-2">
          <button onClick={toggleTheme} className="p-2 rounded-lg" style={{ color: 'var(--sidebar-text)' }}>
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setOpen(!open)}
            className="p-2 rounded-lg"
            style={{ color: 'var(--sidebar-text-active)' }}
          >
            {open ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </header>

      {/* Mobile drawer */}
      {open && (
        <div
          className="lg:hidden fixed inset-0 z-40 animate-fade-in"
          style={{ backgroundColor: 'rgba(2,9,23,0.8)', backdropFilter: 'blur(4px)' }}
          onClick={() => setOpen(false)}
        >
          <div
            className="absolute left-0 top-14 bottom-0 w-72 overflow-y-auto animate-slide-in-left"
            style={{ backgroundColor: 'var(--bg-sidebar)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <nav className="p-3 space-y-4">
              {NAV_SECTIONS.map((section) => (
                <div key={section.label}>
                  <div className="px-2 mb-1 text-xs font-semibold uppercase tracking-widest" style={{ color: 'var(--sidebar-text)', opacity: 0.4 }}>
                    {section.label}
                  </div>
                  <ul className="space-y-0.5">
                    {section.items.map(({ to, icon: Icon, label, end }) => (
                      <li key={to}>
                        <NavLink
                          to={to}
                          end={end}
                          onClick={() => setOpen(false)}
                          className={({ isActive }) => cn(
                            'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors',
                            isActive
                              ? 'text-white'
                              : 'hover:bg-white/5',
                          )}
                          style={({ isActive }) => ({
                            color: isActive ? 'var(--sidebar-text-active)' : 'var(--sidebar-text)',
                            backgroundColor: isActive ? 'var(--sidebar-bg-active)' : undefined,
                            boxShadow: isActive ? 'inset 2px 0 0 var(--brand)' : undefined,
                          })}
                        >
                          {({ isActive }) => (
                            <>
                              <Icon className="w-4 h-4 shrink-0" style={{ color: isActive ? 'var(--brand)' : undefined }} />
                              {label}
                            </>
                          )}
                        </NavLink>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}

              {/* Auth in mobile */}
              <div className="border-t pt-3" style={{ borderColor: 'var(--sidebar-border)' }}>
                {user ? (
                  <div className="space-y-1">
                    <div className="px-3 py-2 text-xs" style={{ color: 'var(--sidebar-text)' }}>
                      Signed in as <span style={{ color: 'var(--sidebar-text-active)' }}>{user.username}</span>
                    </div>
                    <button
                      onClick={handleLogout}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-red-400 hover:bg-red-500/10 transition-colors"
                    >
                      <LogOut className="w-4 h-4" /> Sign out
                    </button>
                  </div>
                ) : (
                  <div className="space-y-1">
                    <Link to="/login" onClick={() => setOpen(false)} className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors hover:bg-white/5" style={{ color: 'var(--sidebar-text)' }}>
                      <LogIn className="w-4 h-4" /> Sign in
                    </Link>
                    <Link to="/register" onClick={() => setOpen(false)} className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors" style={{ color: '#818cf8', backgroundColor: 'rgba(99,102,241,0.1)' }}>
                      <UserPlus className="w-4 h-4" /> Create account
                    </Link>
                  </div>
                )}
              </div>
            </nav>
          </div>
        </div>
      )}
    </>
  )
}
