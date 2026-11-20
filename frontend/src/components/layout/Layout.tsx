import { Outlet, useLocation } from 'react-router-dom'
import { Sidebar, MobileTopBar } from './Sidebar'

/** Pages that use a full-bleed "auth" layout (no sidebar) */
const AUTH_PATHS = ['/login', '/register']

export function Layout() {
  const { pathname } = useLocation()
  const isAuth = AUTH_PATHS.includes(pathname)

  if (isAuth) {
    return (
      <div
        className="min-h-screen flex items-center justify-center p-4"
        style={{ backgroundColor: 'var(--bg-page)' }}
      >
        <Outlet />
      </div>
    )
  }

  return (
    <div
      className="min-h-screen flex"
      style={{ backgroundColor: 'var(--bg-page)' }}
    >
      {/* Desktop sidebar */}
      <Sidebar />

      {/* Main column */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile top bar */}
        <MobileTopBar />

        {/* Page content */}
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>

        {/* Footer */}
        <footer
          className="shrink-0 px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-2"
          style={{ borderTop: '1px solid var(--border)', backgroundColor: 'var(--bg-surface)' }}
        >
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
            VeriFact AI — Misinformation Detection Platform &copy; {new Date().getFullYear()}
          </p>
          <p className="text-xs" style={{ color: 'var(--text-faint)' }}>
            ML predictions are probabilistic — not factual determinations.
          </p>
        </footer>
      </div>
    </div>
  )
}
