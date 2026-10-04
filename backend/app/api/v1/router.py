"""
Master API router for v1.
Mounts all sub-routers under /api/v1.
"""

from fastapi import APIRouter

from app.api.v1 import analyze, history, models, health, evidence, explanation, news, auth

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)        # POST /auth/register|login|logout, GET /auth/me
api_router.include_router(analyze.router)     # POST /analyze/text|url|claim
api_router.include_router(history.router)     # GET  /analysis/{id}, /history
api_router.include_router(models.router)      # GET  /models, /model-performance, /stats
api_router.include_router(health.router)      # GET  /health
api_router.include_router(evidence.router)    # GET  /evidence/{id}, POST /evidence/search
api_router.include_router(explanation.router) # GET  /explanation/{analysis_id}
api_router.include_router(news.router)        # GET  /news/search
