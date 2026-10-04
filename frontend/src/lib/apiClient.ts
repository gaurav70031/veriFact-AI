/**
 * Axios client configuration.
 *
 * withCredentials: true  — required so the browser sends the httpOnly JWT
 * cookie automatically on every request to the backend.  Without this,
 * cookies are NOT sent for cross-origin requests (localhost:5173 → :8000).
 *
 * 401 interceptor — clears the auth state and redirects to /login so the
 * user can re-authenticate without seeing a broken page.
 */

import axios, { AxiosError } from 'axios'
import type { ApiError } from '@/types/api'

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1'

export const apiClient = axios.create({
  baseURL:         BASE_URL,
  timeout:         60_000,
  withCredentials: true,   // send httpOnly cookie on every request
  headers: {
    'Content-Type': 'application/json',
  },
})

// ── Response interceptor: normalise errors + handle 401 ───────────────────────

apiClient.interceptors.response.use(
  (res) => res,
  (error: AxiosError<ApiError>) => {
    // When the server returns 401, the JWT is missing or expired.
    // Redirect to /login so the user can re-authenticate.
    // We do a soft redirect (not window.location.replace) so React Router
    // can handle it via the auth context.
    if (error.response?.status === 401) {
      // Fire a custom DOM event that AuthContext listens to.
      // This avoids a circular dependency between apiClient and the context.
      window.dispatchEvent(new CustomEvent('auth:unauthorized'))
    }

    const message =
      error.response?.data?.message ??
      error.response?.data?.error   ??
      error.message                 ??
      'An unexpected error occurred.'

    return Promise.reject(new Error(message))
  },
)

export default apiClient
