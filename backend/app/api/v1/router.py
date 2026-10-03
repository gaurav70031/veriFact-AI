"""
Master API router for v1.
Mounts all sub-routers under /api/v1.
"""

from fastapi import APIRouter

from app.api.v1 import analyze, history, models, health

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(analyze.router)   # /api/v1/analyze/*
api_router.include_router(history.router)   # /api/v1/analysis/{id}, /api/v1/history
api_router.include_router(models.router)    # /api/v1/models, /api/v1/model-performance, /api/v1/stats
api_router.include_router(health.router)    # /api/v1/health
