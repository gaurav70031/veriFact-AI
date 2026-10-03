"""
app.analysis
~~~~~~~~~~~~
Claim extraction, evidence comparison, and verdict aggregation engine.

Three modules work in a pipeline:
  1. claim_extractor.py   — split article text into discrete factual claims
  2. evidence_comparator.py — classify each (claim, evidence) pair as
                              SUPPORTING / CONTRADICTING / INCONCLUSIVE / NOT_RELEVANT
  3. verdict_engine.py    — aggregate relationships into a final EvidenceAssessment

These modules are kept separate from the service layer so they can be
tested, configured, and upgraded independently.
"""
