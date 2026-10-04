/**
 * AuthContext
 *
 * Provides the current user and auth actions to the entire app.
 *
 * Session restoration
 * -------------------
 * On every page load, GET /auth/me is called.  The browser sends the
 * httpOnly cookie automatically.  If the cookie is valid the user is
 * restored silently; if expired the user is treated as anonymous.
 *
 * 401 handling
 * ------------
 * A custom DOM event "auth:unauthorized" is dispatched by apiClient
 * whenever the server returns 401.  AuthContext listens for this event
 * and clears the local user state so the UI updates correctly.
 *
 * The user is redirected to /login only when they try to access a
 * protected route (handled by ProtectedRoute), not automatically on
 * every 401 — this prevents redirect loops on the public pages.
 */

import {
  createContext, useContext, useEffect, useRef,
  useState, useCallback, type ReactNode,
} from 'react'
import { useQueryClient } from '@tanstack/react-query'
import * as api from '@/lib/api'
import type { UserOut, RegisterRequest, LoginRequest } from '@/types/api'

// ── Context shape ─────────────────────────────────────────────────────────────

interface AuthContextValue {
  user:         UserOut | null
  isLoading:    boolean   // true while /auth/me is in-flight on first load
  isLoggedIn:   boolean
  login:        (body: LoginRequest)    => Promise<UserOut>
  register:     (body: RegisterRequest) => Promise<UserOut>
  logout:       () => Promise<void>
  refreshUser:  () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

// ── Provider ──────────────────────────────────────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user,      setUser]      = useState<UserOut | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const queryClient               = useQueryClient()
  const mounted                   = useRef(true)

  // ── Restore session on mount ──────────────────────────────────────────────
  const refreshUser = useCallback(async () => {
    try {
      const u = await api.authMe()
      if (mounted.current) setUser(u)
    } catch {
      // Not authenticated — treat as anonymous
      if (mounted.current) setUser(null)
    } finally {
      if (mounted.current) setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    mounted.current = true
    refreshUser()
    return () => { mounted.current = false }
  }, [refreshUser])

  // ── Listen for 401 events from apiClient ─────────────────────────────────
  useEffect(() => {
    const handleUnauthorized = () => {
      setUser(null)
      // Invalidate all user-specific queries
      queryClient.invalidateQueries({ queryKey: ['history'] })
    }
    window.addEventListener('auth:unauthorized', handleUnauthorized)
    return () => window.removeEventListener('auth:unauthorized', handleUnauthorized)
  }, [queryClient])

  // ── Auth actions ──────────────────────────────────────────────────────────

  const login = useCallback(async (body: LoginRequest): Promise<UserOut> => {
    const resp = await api.authLogin(body)
    setUser(resp.user)
    queryClient.invalidateQueries({ queryKey: ['history'] })
    return resp.user
  }, [queryClient])

  const register = useCallback(async (body: RegisterRequest): Promise<UserOut> => {
    const resp = await api.authRegister(body)
    setUser(resp.user)
    return resp.user
  }, [])

  const logout = useCallback(async () => {
    await api.authLogout()
    setUser(null)
    queryClient.clear()   // wipe all cached queries on logout
  }, [queryClient])

  return (
    <AuthContext.Provider value={{
      user,
      isLoading,
      isLoggedIn: user !== null,
      login,
      register,
      logout,
      refreshUser,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
