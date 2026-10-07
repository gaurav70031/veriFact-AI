# VeriFact AI — Deployment Guide

**Frontend → Vercel** (free)  
**Backend + Database → Render** (free tier)

---

## Overview

```
Browser  →  Vercel (React/Vite)  →  Render (FastAPI)  →  Render PostgreSQL
```

---

## Step 1 — Push to GitHub

You must have the code on GitHub for both platforms to pull from.

```powershell
cd "c:\Users\gaura\Desktop\fake news detection project\fake-news-detection"
git add .
git commit -m "prepare for deployment"
git remote add origin https://github.com/YOUR_USERNAME/verifact-ai.git
git push -u origin main
```

---

## Step 2 — Deploy Backend on Render

### 2a. Create a Render account
Go to **https://render.com** → Sign up (free, use GitHub login).

### 2b. Deploy via Blueprint (easiest)
1. Render Dashboard → **New** → **Blueprint**
2. Connect your GitHub repo
3. Render finds `render.yaml` automatically and creates:
   - A **PostgreSQL** database (`verifact-postgres`)
   - A **Web Service** (`verifact-backend`) running the Docker container

### 2c. Set secret env vars in Render dashboard
After the blueprint creates the service, go to:
**verifact-backend → Environment → Add Environment Variable**

| Key | Value |
|-----|-------|
| `SECRET_KEY` | Run `python -c "import secrets; print(secrets.token_hex(32))"` and paste |
| `NEWSAPI_KEY` | `fa773fd3d0c0472f8e53a2261fd95844` |
| `GNEWS_API_KEY` | `ebb9cdf4d43422541a87fc21fa7b2838` |

### 2d. Note your backend URL
After deploy finishes, Render gives you a URL like:
```
https://verifact-backend.onrender.com
```
**Save this URL — you need it for Step 3.**

### 2e. Run migrations + create admin
Render doesn't auto-run scripts. Go to:
**verifact-backend → Shell** (or use Render's one-off jobs)

```bash
alembic upgrade head
python scripts/create_admin.py
```

---

## Step 3 — Deploy Frontend on Vercel

### 3a. Create a Vercel account
Go to **https://vercel.com** → Sign up (free, use GitHub login).

### 3b. Import your project
1. Vercel Dashboard → **Add New → Project**
2. Select your GitHub repo
3. Vercel auto-detects Vite
4. Set **Root Directory** to `frontend`

### 3c. Set the backend URL environment variable
In Vercel project settings → **Environment Variables**:

| Key | Value |
|-----|-------|
| `VITE_API_BASE_URL` | `https://verifact-backend.onrender.com/api/v1` |

> Replace with your actual Render URL from Step 2d.

### 3d. Deploy
Click **Deploy**. Vercel builds and gives you a URL like:
```
https://verifact-ai.vercel.app
```

---

## Step 4 — Wire CORS

Go back to **Render → verifact-backend → Environment**:

| Key | Value |
|-----|-------|
| `ALLOWED_ORIGINS` | `https://verifact-ai.vercel.app` |

Replace with your actual Vercel URL. Then click **Save** — Render redeploys automatically.

---

## Step 5 — Test the deployment

1. Open your Vercel URL in the browser
2. The green API dot in the navbar should light up (backend online)
3. Try checking a story — should work end-to-end
4. Log in with:
   - Email: `admin@verifasai.com`
   - Password: `Admin@1234`

---

## Free tier limitations

| Limitation | Impact |
|-----------|--------|
| Backend spins down after 15 min idle | First request after idle takes ~30s to wake up |
| 512 MB RAM on Render | ML models (baseline only) fit fine; DistilBERT won't load |
| 100 req/day NewsAPI | Sufficient for a demo |
| 100 req/day GNews | Sufficient for a demo |
| Vercel: 100 GB bandwidth/month | More than enough |

### Wake-up tip
Add a free uptime monitor at **https://uptimerobot.com** pinging
`https://verifact-backend.onrender.com/api/v1/health` every 14 minutes
to prevent the backend from sleeping.

---

## Environment variable summary

### Render (backend)
| Variable | Where to get it |
|----------|----------------|
| `SECRET_KEY` | Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `POSTGRES_*` | Auto-filled by Render from the database |
| `ALLOWED_ORIGINS` | Your Vercel URL |
| `NEWSAPI_KEY` | Already have it |
| `GNEWS_API_KEY` | Already have it |
| `COOKIE_SECURE` | `true` (Render uses HTTPS) |

### Vercel (frontend)
| Variable | Value |
|----------|-------|
| `VITE_API_BASE_URL` | `https://<your-render-service>.onrender.com/api/v1` |

---

## Custom domain (optional)

Both Vercel and Render support custom domains on free plans:
- **Vercel**: Project Settings → Domains → Add `verifactai.com`
- **Render**: Service Settings → Custom Domains → Add domain

---

## Troubleshooting

**CORS error in browser console**
→ Check `ALLOWED_ORIGINS` in Render matches your exact Vercel URL (no trailing slash)

**502 / service unavailable**
→ Backend is waking up — wait 30s and refresh

**Login returns 401 immediately**
→ Run `python scripts/create_admin.py` in the Render shell

**Evidence search returns no results**
→ Check `NEWSAPI_KEY` and `GNEWS_API_KEY` are set in Render env vars
