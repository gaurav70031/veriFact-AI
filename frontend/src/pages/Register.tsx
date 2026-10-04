import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ShieldCheck, Mail, Lock, User, Loader2, AlertCircle, Eye, EyeOff, CheckCircle } from 'lucide-react'
import { useAuth } from '@/context/AuthContext'
import { cn } from '@/lib/utils'

const PASSWORD_RULES = [
  { test: (p: string) => p.length >= 8,          label: 'At least 8 characters' },
  { test: (p: string) => /[A-Z]/.test(p),        label: 'One uppercase letter' },
  { test: (p: string) => /[0-9]/.test(p),        label: 'One number' },
]

export default function Register() {
  const { register, isLoggedIn } = useAuth()
  const navigate                  = useNavigate()

  const [email,     setEmail]     = useState('')
  const [username,  setUsername]  = useState('')
  const [password,  setPassword]  = useState('')
  const [fullName,  setFullName]  = useState('')
  const [showPwd,   setShowPwd]   = useState(false)
  const [isPending, setIsPending] = useState(false)
  const [error,     setError]     = useState<string | null>(null)

  if (isLoggedIn) { navigate('/', { replace: true }); return null }

  const pwdStrength = PASSWORD_RULES.filter((r) => r.test(password)).length
  const showStrength = password.length > 0

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setIsPending(true)
    try {
      await register({
        email:     email.trim(),
        username:  username.trim(),
        password,
        full_name: fullName.trim() || undefined,
      })
      navigate('/', { replace: true })
    } catch (err) {
      setError((err as Error).message ?? 'Registration failed.')
    } finally {
      setIsPending(false)
    }
  }

  return (
    <div className="min-h-[80vh] flex items-center justify-center px-4 py-12 animate-fade-in">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="p-3 rounded-2xl bg-brand-600/20 border border-brand-500/30 mb-4">
            <ShieldCheck className="w-7 h-7 text-brand-400" />
          </div>
          <h1 className="text-2xl font-bold text-slate-100">Create account</h1>
          <p className="text-sm text-slate-500 mt-1">Join VeritasAI to save your analysis history</p>
        </div>

        <form onSubmit={handleSubmit} className="glass-card p-6 space-y-4">
          {/* Full name */}
          <div>
            <label className="label-sm block mb-1.5">Full name <span className="text-slate-600 normal-case">(optional)</span></label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
              <input
                type="text"
                className="input-field pl-9"
                placeholder="Jane Smith"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                autoComplete="name"
                autoFocus
              />
            </div>
          </div>

          {/* Email */}
          <div>
            <label className="label-sm block mb-1.5">Email *</label>
            <div className="relative">
              <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
              <input
                type="email"
                className="input-field pl-9"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoComplete="email"
              />
            </div>
          </div>

          {/* Username */}
          <div>
            <label className="label-sm block mb-1.5">Username *</label>
            <div className="relative">
              <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-600 text-sm font-mono">@</span>
              <input
                type="text"
                className="input-field pl-8 font-mono"
                placeholder="yourname"
                value={username}
                onChange={(e) => setUsername(e.target.value.replace(/[^a-zA-Z0-9_.-]/g, ''))}
                required
                minLength={3}
                maxLength={50}
                autoComplete="username"
              />
            </div>
            <p className="text-xs text-slate-700 mt-1">Letters, numbers, underscores, dots, hyphens</p>
          </div>

          {/* Password */}
          <div>
            <label className="label-sm block mb-1.5">Password *</label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-600" />
              <input
                type={showPwd ? 'text' : 'password'}
                className="input-field pl-9 pr-10"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={8}
                autoComplete="new-password"
              />
              <button
                type="button"
                onClick={() => setShowPwd(!showPwd)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-600 hover:text-slate-400"
                tabIndex={-1}
              >
                {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>

            {/* Strength checklist */}
            {showStrength && (
              <ul className="mt-2 space-y-1">
                {PASSWORD_RULES.map((rule) => (
                  <li key={rule.label} className={cn(
                    'flex items-center gap-1.5 text-xs',
                    rule.test(password) ? 'text-emerald-400' : 'text-slate-600',
                  )}>
                    <CheckCircle className="w-3 h-3 shrink-0" />
                    {rule.label}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2.5 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}

          {/* Submit */}
          <button
            type="submit"
            disabled={isPending || !email || !username || !password}
            className="btn-primary w-full justify-center py-2.5"
          >
            {isPending ? (
              <><Loader2 className="w-4 h-4 animate-spin" />Creating account…</>
            ) : 'Create account'}
          </button>
        </form>

        <p className="text-center text-sm text-slate-500 mt-5">
          Already have an account?{' '}
          <Link to="/login" className="text-brand-400 hover:text-brand-300 font-medium transition-colors">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  )
}
