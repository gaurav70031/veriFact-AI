import { useState } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { Shield, Mail, Lock, Loader2, AlertCircle, Eye, EyeOff, ArrowRight } from 'lucide-react'
import { useAuth } from '@/context/AuthContext'

export default function Login() {
  const { login, isLoggedIn } = useAuth()
  const navigate              = useNavigate()
  const location              = useLocation()
  const from                  = (location.state as { from?: Location })?.from?.pathname ?? '/'

  const [email,     setEmail]     = useState('')
  const [password,  setPassword]  = useState('')
  const [showPwd,   setShowPwd]   = useState(false)
  const [isPending, setIsPending] = useState(false)
  const [error,     setError]     = useState<string | null>(null)

  if (isLoggedIn) { navigate(from, { replace: true }); return null }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setIsPending(true)
    try {
      await login({ email: email.trim(), password })
      navigate(from, { replace: true })
    } catch (err) {
      setError((err as Error).message ?? 'Login failed.')
    } finally {
      setIsPending(false)
    }
  }

  return (
    <div className="w-full max-w-sm animate-fade-in-up">
      {/* Logo */}
      <div className="flex flex-col items-center mb-8">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center shadow-glow-brand mb-4">
          <Shield className="w-7 h-7 text-white" />
        </div>
        <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>Welcome back</h1>
        <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>Sign in to your VeriFact AI account</p>
      </div>

      <div className="card p-6 space-y-4">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="label-sm block mb-1.5">Email</label>
            <div className="relative">
              <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-faint)' }} />
              <input type="email" className="input-field pl-10" placeholder="you@example.com" value={email}
                onChange={(e) => setEmail(e.target.value)} required autoComplete="email" autoFocus />
            </div>
          </div>

          <div>
            <label className="label-sm block mb-1.5">Password</label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-faint)' }} />
              <input type={showPwd ? 'text' : 'password'} className="input-field pl-10 pr-10"
                placeholder="••••••••" value={password} onChange={(e) => setPassword(e.target.value)}
                required autoComplete="current-password" />
              <button type="button" onClick={() => setShowPwd(!showPwd)}
                className="absolute right-3 top-1/2 -translate-y-1/2 transition-colors" tabIndex={-1}
                style={{ color: 'var(--text-faint)' }}>
                {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {error && (
            <div className="flex items-start gap-2.5 p-3 rounded-xl" style={{ backgroundColor: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)' }}>
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          <button type="submit" disabled={isPending || !email || !password} className="btn-primary w-full justify-center py-2.5">
            {isPending ? <><Loader2 className="w-4 h-4 animate-spin" />Signing in…</> : <><span>Sign in</span><ArrowRight className="w-4 h-4" /></>}
          </button>
        </form>
      </div>

      <p className="text-center text-sm mt-5" style={{ color: 'var(--text-subtle)' }}>
        Don't have an account?{' '}
        <Link to="/register" className="font-medium transition-colors" style={{ color: 'var(--brand)' }}>Create one</Link>
      </p>
      <p className="text-center text-xs mt-3">
        <Link to="/analyze/text" className="transition-colors" style={{ color: 'var(--text-faint)' }}>
          Continue without an account →
        </Link>
      </p>
    </div>
  )
}
