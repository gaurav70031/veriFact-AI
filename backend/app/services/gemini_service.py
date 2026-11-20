"""
Gemini verdict service.

Uses Google Gemini as the primary fact-checking layer. Given a claim and
the evidence already retrieved by Tavily/NewsAPI, Gemini reasons about
whether the claim is true, false, or uncertain — with a plain-English
explanation suitable for the UI.

Why Gemini instead of (or alongside) ML models
-----------------------------------------------
- The ML models were trained on 2015-2018 US political news.  They classify
  writing *style*, not facts.  They have no knowledge of Indian politics,
  post-2018 events, or domain-specific topics.
- Gemini has broad world knowledge up to its training cutoff and can reason
  about claims in any language/context (India, sports, science, etc.)
- Gemini reads the actual evidence snippets retrieved from Tavily/NewsAPI
  and grounds its answer in them — reducing hallucination.

Verdict mapping
---------------
Gemini returns one of:
  TRUE        → the claim is supported by evidence / world knowledge
  FALSE       → the claim contradicts evidence / world knowledge
  UNCERTAIN   → not enough information to decide
  MISLEADING  → partially true but missing important context

These map to the existing FinalVerdict enum:
  TRUE      → REAL
  FALSE     → FAKE
  UNCERTAIN → MIXED
  MISLEADING→ MIXED  (with explanation)

Free tier limits (gemini-1.5-flash)
------------------------------------
  1,500 requests / day
  1,000,000 tokens / minute
  Sufficient for a development / demo deployment.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from dataclasses import dataclass
from typing import Optional

# Module-level thread pool — lives for the duration of the process,
# avoiding the context-manager teardown issue with asyncio.wait_for.
_GEMINI_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GeminiVerdict:
    verdict: str            # "TRUE" | "FALSE" | "UNCERTAIN" | "MISLEADING"
    confidence: float       # 0.0 – 1.0
    explanation: str        # plain-English explanation for the UI
    reasoning: str          # internal chain-of-thought (not shown to user)
    model_used: str         # e.g. "gemini-1.5-flash"
    tokens_used: int        # prompt + completion tokens
    error: Optional[str]    # set if the call failed

    @property
    def final_verdict_label(self) -> str:
        """Map Gemini verdict to the app's FinalVerdict enum value."""
        return {
            "TRUE":       "REAL",
            "FALSE":      "FAKE",
            "UNCERTAIN":  "MIXED",
            "MISLEADING": "MIXED",
        }.get(self.verdict, "MIXED")

    @property
    def is_available(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(claim: str, evidence_snippets: list[dict]) -> str:
    """
    Build the fact-checking prompt sent to Gemini.

    evidence_snippets: list of dicts with keys: title, source, snippet, relationship
    """
    evidence_block = ""
    if evidence_snippets:
        lines = []
        for i, ev in enumerate(evidence_snippets[:8], 1):   # max 8 sources
            rel  = ev.get("relationship", "").upper()
            src  = ev.get("source", "unknown")
            title = ev.get("title", "")[:120]
            snip  = ev.get("snippet", "")[:300]
            lines.append(
                f"[Source {i}] {src} — {title}\n"
                f"  Relationship: {rel}\n"
                f"  Snippet: {snip}"
            )
        evidence_block = "\n\n".join(lines)
    else:
        evidence_block = "No external evidence was retrieved."

    prompt = f"""You are a professional fact-checker with expertise in global and Indian current affairs.

Your task is to evaluate whether the following claim is TRUE, FALSE, UNCERTAIN, or MISLEADING.

CLAIM:
\"{claim}\"

RETRIEVED EVIDENCE:
{evidence_block}

INSTRUCTIONS:
1. Carefully read the claim and all evidence snippets.
2. Use both the evidence AND your own knowledge to assess the claim.
3. Pay special attention to Indian context — Indian politics, events, people, and institutions.
4. If the claim is about a recent event, rely more on the evidence snippets.
5. If the evidence is insufficient, use your knowledge but be explicit about uncertainty.

Respond in EXACTLY this format (do not add extra text outside this format):

VERDICT: [TRUE / FALSE / UNCERTAIN / MISLEADING]
CONFIDENCE: [a number between 0.0 and 1.0]
EXPLANATION: [2-3 sentences explaining your verdict in simple English, suitable for a general audience. Mention key evidence sources if available.]
REASONING: [your internal step-by-step reasoning, 3-5 sentences]

Definitions:
- TRUE: The claim is factually correct and supported by evidence or well-established knowledge.
- FALSE: The claim is factually incorrect and contradicted by evidence or well-established knowledge.
- UNCERTAIN: There is not enough evidence to confirm or deny the claim.
- MISLEADING: The claim contains some truth but omits important context or is worded to mislead.
"""
    return prompt


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> tuple[str, float, str, str]:
    """
    Parse Gemini's structured response into (verdict, confidence, explanation, reasoning).
    Falls back to UNCERTAIN on parse failure.
    """
    verdict     = "UNCERTAIN"
    confidence  = 0.5
    explanation = text[:500]   # fallback: raw text
    reasoning   = ""

    v_match = re.search(r"VERDICT:\s*(TRUE|FALSE|UNCERTAIN|MISLEADING)", text, re.IGNORECASE)
    c_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", text, re.IGNORECASE)
    e_match = re.search(r"EXPLANATION:\s*(.+?)(?=REASONING:|$)", text, re.IGNORECASE | re.DOTALL)
    r_match = re.search(r"REASONING:\s*(.+?)$", text, re.IGNORECASE | re.DOTALL)

    if v_match:
        verdict = v_match.group(1).upper()
    if c_match:
        try:
            confidence = max(0.0, min(1.0, float(c_match.group(1))))
        except ValueError:
            pass
    if e_match:
        explanation = e_match.group(1).strip()
    if r_match:
        reasoning = r_match.group(1).strip()

    return verdict, confidence, explanation, reasoning


async def get_gemini_verdict(
    claim: str,
    evidence_snippets: list[dict],
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.5-flash-lite",
) -> GeminiVerdict:
    """
    Call Gemini to fact-check a single claim with supporting evidence.

    Parameters
    ----------
    claim             : The claim text to fact-check.
    evidence_snippets : List of dicts from the evidence retrieval layer.
    api_key           : Gemini API key (falls back to config if not provided).
    model_name        : Gemini model to use. gemini-1.5-flash is fast and free.

    Returns
    -------
    GeminiVerdict with verdict, confidence, and explanation.
    """
    # Resolve API key
    if not api_key:
        try:
            from app.core.config import get_settings
            get_settings.cache_clear()
            api_key = get_settings().gemini_api_key
        except Exception:
            pass

    if not api_key:
        return GeminiVerdict(
            verdict="UNCERTAIN", confidence=0.5,
            explanation="Gemini API key not configured.",
            reasoning="", model_used=model_name, tokens_used=0,
            error="GEMINI_API_KEY not set",
        )

    prompt = _build_prompt(claim, evidence_snippets)

    # Call Gemini REST API directly with httpx (native async, no SDK overhead).
    # This avoids all google.genai event-loop conflicts with uvicorn.
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512,
        },
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.post(url, json=payload)

        if response.status_code == 429:
            raise ValueError("Gemini rate limit exceeded")
        if response.status_code == 503:
            raise ValueError("Gemini service unavailable (high demand)")
        if response.status_code != 200:
            raise ValueError(f"Gemini HTTP {response.status_code}: {response.text[:200]}")

        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini returned no candidates")

        raw_text = ""
        try:
            raw_text = candidates[0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            finish = candidates[0].get("finishReason", "unknown")
            raise ValueError(f"Gemini candidate has no text (finishReason={finish})")

        if not raw_text:
            raise ValueError("Gemini returned empty text")

        tokens_used = 0
        try:
            meta = data.get("usageMetadata", {})
            tokens_used = meta.get("promptTokenCount", 0) + meta.get("candidatesTokenCount", 0)
        except Exception:
            pass

        verdict, confidence, explanation, reasoning = _parse_response(raw_text)

        logger.info(
            "[gemini] claim=%r  verdict=%s  confidence=%.2f  tokens=%d",
            claim[:60], verdict, confidence, tokens_used,
        )

        return GeminiVerdict(
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
            reasoning=reasoning,
            model_used=model_name,
            tokens_used=tokens_used,
            error=None,
        )

    except Exception as exc:
        logger.warning("[gemini] Call failed (%s): %s", type(exc).__name__, exc)
        return GeminiVerdict(
            verdict="UNCERTAIN", confidence=0.5,
            explanation="Gemini fact-check temporarily unavailable — falling back to ML + evidence signals.",
            reasoning="", model_used=model_name, tokens_used=0,
            error=str(exc),
        )


def is_gemini_configured() -> bool:
    """Return True if a Gemini API key is available."""
    try:
        from app.core.config import get_settings
        # Clear the LRU cache to pick up .env changes at runtime
        get_settings.cache_clear()
        return bool(get_settings().gemini_api_key)
    except Exception:
        return False
