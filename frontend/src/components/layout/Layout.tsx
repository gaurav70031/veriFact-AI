import { Outlet } from 'react-router-dom'
import { Navbar } from './Navbar'

export function Layout() {
  return (
    <div className="min-h-screen flex flex-col
                    bg-gray-50 text-slate-900
                    dark:bg-slate-950 dark:text-slate-100
                    transition-colors duration-300">
      <Navbar />
      <main className="flex-1">
        <Outlet />
      </main>
      <footer className="border-t border-gray-200 dark:border-slate-800/60 py-6 mt-16
                         transition-colors duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8
                        flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-400 dark:text-slate-600">
            VeriFact AI — Fake News Detection Platform &copy; {new Date().getFullYear()}
          </p>
          <p className="text-xs text-slate-400 dark:text-slate-700">
            Model predictions are probabilistic, not factual determinations.
          </p>
        </div>
      </footer>
    </div>
  )
}
