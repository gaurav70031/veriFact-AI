import { Outlet } from 'react-router-dom'
import { Navbar } from './Navbar'

export function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-slate-800/60 py-6 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-600">
            VeritasAI — Fake News Detection Platform &copy; {new Date().getFullYear()}
          </p>
          <p className="text-xs text-slate-700">
            Model predictions are probabilistic, not factual determinations.
          </p>
        </div>
      </footer>
    </div>
  )
}
