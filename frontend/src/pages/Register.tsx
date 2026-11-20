import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Shield, Mail, Lock, User, Loader2, AlertCircle, Eye, EyeOff, CheckCircle, ArrowRight } from 'lucide-react'
import { useAuth } from '@/context/AuthContext'
import { cn } from '@/lib/utils'

const PASSWORD_RULES = [
  { test: (p: string) => p.length >= 8,   label: 'At least 8 characters'  },
  { test: (p: string) => /[A-Z]/.test(p), label: 'One uppercase letter'   },
  { test: (p: string) => /[0-9]/.test(p), label: 'One number'             },
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setIsPending(true)
    try {
      await register({ email: email.trim(), username: username.trim(), password, full_name: fullName.trim() || undefined })
      navigate('/', { replace: true })
    } catch (err) {
      setError((err as Error).message ?? 'Registration failed.')
    } finally {
      setIsPending(false)
    }
  }

  return (
    <div className="w-full max-w-sm animate-fade-in-up">
      <div className="flex flex-col items-center mb-8">
        <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-brand-500 to-violet-600 flex items-center justify-center shadow-glow-brand mb-4">
          <Shield className="w-7 h-7 text-white" />
        </div>
        <h1 className="text-2xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>Create account</h1>
        <p className="text-sm" style={{ color: 'var(--text-subtle)' }}>Join VeriFact AI to save your analysis history</p>
      </div>

      <div className="card p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="label-sm block mb-1.5">Full name <span className="normal-case tracking-normal text-xs" style={{ color: 'var(--text-faint)' }}>(optional)</span></label>
            <div className="relative">
              <User className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-faint)' }} />
              <input type="text" className="input-field pl-10" placeholder="Jane Smith" value={fullName}
                onChange={(e) => setFullName(e.target.value)} autoComplete="name" autoFocus />
            </div>
          </div>

          <div>
            <label className="label-sm block mb-1.5">Email</label>
            <div className="relative">
              <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-faint)' }} />
              <input type="email" className="input-field pl-10" placeholder="you@example.com" value={email}
                onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
            </div>
          </div>

          <div>
            <label className="label-sm block mb-1.5">Username</label>
            <div className="relative">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-sm font-mono" style={{ color: 'var(--text-faint)' }}>@</span>
              <input type="text" className="input-field pl-9 font-mono" placeholder="yourname" value={username}
                onChange={(e) => setUsername(e.target.value.replace(/[^a-zA-Z0-9_.-]/g, ''))}
                required minLength={3} maxLength={50} autoComplete="username" />
            </div>
          </div>

          <div>
            <label className="label-sm block mb-1.5">Password</label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-faint)' }} />
              <input type={showPwd ? 'text' : 'password'} className="input-field pl-10 pr-10"
                placeholder="••••••••" value={password} onChange={(e) => setPassword(e.target.value)}
                required minLength={8} autoComplete="new-password" />
              <button type="button" onClick={() => setShowPwd(!showPwd)} className="absolute right-3 top-1/2 -translate-y-1/2" tabIndex={-1} style={{ color: 'var(--text-faint)' }}>
                {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            {password.length > 0 && (
              <ul className="mt-2.5 space-y-1">
                {PASSWORD_RULES.map((rule) => (
                  <li key={rule.label} className={cn('flex items-center gap-1.5 text-xs', rule.test(password) ? 'text-emerald-400' : '')} style={{ color: rule.test(password) ? undefined : 'var(--text-faint)' }}>
                    <CheckCircle className="w-3 h-3 shrink-0" />
                    {rule.label}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {error && (
            <div className="flex items-start gap-2.5 p-3 rounded-xl" style={{ backgroundColor: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)' }}>
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          <button type="submit" disabled={isPending || !email || !username || !password} className="btn-primary w-full justify-center py-2.5">
            {isPending ? <><Loader2 className="w-4 h-4 animate-spin" />Creating account…</> : <><span>Create account</span><ArrowRight className="w-4 h-4" /></>}
          </button>
        </form>
      </div>

      <p className="text-center text-sm mt-5" style={{ color: 'var(--text-subtle)' }}>
        Already have an account?{' '}
        <Link to="/login" className="font-medium" style={{ color: 'var(--brand)' }}>Sign in</Link>
      </p>
    </div>
  )
}
