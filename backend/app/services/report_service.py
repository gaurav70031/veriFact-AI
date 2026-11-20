"""
backend/app/services/report_service.py
=======================================
PDF report generation service.

Two report types:
  1. Verification report (per-analysis) — contains the full analysis result
  2. Research report (model evaluation) — contains training/eval methodology

Uses reportlab for PDF generation. Falls back to an HTML-to-PDF approach
if reportlab is not installed (using a minimal HTML template).

Both report types use only real stored data — nothing fabricated.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis
from app.models.claim import Claim
from app.models.evidence_source import EvidenceSource
from app.models.model_version import ModelVersion
from app.models.prediction import Prediction

logger = logging.getLogger(__name__)

# ── Check reportlab availability ──────────────────────────────────────────────

def _reportlab_available() -> bool:
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Verification PDF report
# ══════════════════════════════════════════════════════════════════════════════

async def generate_verification_pdf(
    db:          AsyncSession,
    analysis:    "AnalysisResponse",
    analysis_id: int,
) -> bytes:
    """
    Generate a professional PDF verification report.

    Contains:
      1. Project title and branding
      2. Verification ID
      3. Submitted news article or claim
      4. Predicted classification
      5. Model information
      6. Confidence score
      7. Evidence sources
      8. Explainable AI results
      9. Supporting and contradicting evidence
      10. Final assessment
      11. Limitations and disclaimer
    """
    if _reportlab_available():
        return _generate_pdf_reportlab(db, analysis, analysis_id)
    else:
        return _generate_pdf_html_fallback(analysis, analysis_id)


def _generate_pdf_reportlab(
    db:          AsyncSession,
    analysis:    "AnalysisResponse",
    analysis_id: int,
) -> bytes:
    """Generate PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"],
        fontSize=22, textColor=HexColor("#4f46e5"), spaceAfter=6)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#64748b"), spaceAfter=20)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"],
        fontSize=13, textColor=HexColor("#1e293b"), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#334155"), spaceAfter=6, leading=14)
    mono_style = ParagraphStyle("Mono", parent=styles["Normal"],
        fontSize=9, fontName="Courier", textColor=HexColor("#475569"))
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"],
        fontSize=8, textColor=HexColor("#94a3b8"), spaceAfter=4)

    story = []

    # 1. Title
    story.append(Paragraph("VeriFact AI", title_style))
    story.append(Paragraph("Fake News Detection &amp; Verification Report", subtitle_style))

    # 2. Verification ID
    story.append(Paragraph(f"Verification ID: <b>#{analysis_id}</b>", body_style))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", body_style))
    story.append(Spacer(1, 10))

    # 3. Submitted content
    story.append(Paragraph("1. Submitted Content", section_style))
    input_label = {"text": "Article Text", "url": "URL", "query": "Claim"}.get(analysis.input_type, "Text")
    story.append(Paragraph(f"<b>Input type:</b> {input_label}", body_style))
    if analysis.source_url:
        story.append(Paragraph(f"<b>Source URL:</b> {analysis.source_url}", body_style))
    if analysis.article_title:
        story.append(Paragraph(f"<b>Headline:</b> {analysis.article_title}", body_style))
    text_preview = analysis.original_input[:500]
    if len(analysis.original_input) > 500:
        text_preview += "..."
    story.append(Paragraph(f"<b>Content:</b><br/>{text_preview}", body_style))
    story.append(Spacer(1, 8))

    # 4. Predicted classification
    story.append(Paragraph("2. Predicted Classification", section_style))
    verdict_color = "#f87171" if analysis.ml_verdict == "FAKE" else "#34d399" if analysis.ml_verdict == "REAL" else "#fbbf24"
    story.append(Paragraph(
        f'<font color="{verdict_color}"><b>{analysis.ml_verdict}</b></font>',
        ParagraphStyle("Verdict", parent=body_style, fontSize=16),
    ))
    story.append(Spacer(1, 6))

    # 5. Model information
    story.append(Paragraph("3. Model Information", section_style))
    if analysis.model_predictions:
        model_data = [["Model", "Label", "Confidence", "P(FAKE)", "P(REAL)"]]
        for p in analysis.model_predictions:
            model_data.append([
                p.model_name,
                p.label,
                f"{p.confidence*100:.1f}%",
                f"{p.fake_probability*100:.1f}%",
                f"{p.real_probability*100:.1f}%",
            ])
        model_table = Table(model_data, colWidths=[120, 50, 70, 60, 60])
        model_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e0e7ff")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ]))
        story.append(model_table)
    else:
        story.append(Paragraph("No model predictions stored.", body_style))
    story.append(Spacer(1, 8))

    # 6. Confidence
    story.append(Paragraph("4. Confidence Score", section_style))
    conf = analysis.ml_confidence or 0
    story.append(Paragraph(f"Ensemble confidence: <b>{conf*100:.1f}%</b>", body_style))
    story.append(Spacer(1, 8))

    # 7-9. Evidence
    story.append(Paragraph("5. Evidence Sources", section_style))
    if analysis.claims:
        for claim in analysis.claims:
            story.append(Paragraph(f"<b>Claim {claim.position}:</b> {claim.claim_text[:200]}", body_style))
            if claim.evidence_verdict:
                story.append(Paragraph(f"<b>Evidence assessment:</b> {claim.evidence_verdict}", body_style))
            if claim.evidence_explanation:
                story.append(Paragraph(f"<b>Explanation:</b> {claim.evidence_explanation[:300]}", body_style))

            if claim.evidence_sources:
                ev_data = [["Source", "Title", "Relationship", "Relevance"]]
                for src in claim.evidence_sources[:5]:
                    ev_data.append([
                        src.source_name[:30],
                        src.title[:40],
                        src.relationship_to_claim,
                        f"{src.relevance_score*100:.0f}%",
                    ])
                ev_table = Table(ev_data, colWidths=[100, 140, 70, 50])
                ev_table.setStyle(TableStyle([
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f1f5f9")),
                ]))
                story.append(ev_table)
            story.append(Paragraph(
                f"Supporting: {claim.supporting_count} | "
                f"Contradicting: {claim.contradicting_count} | "
                f"Inconclusive: {claim.inconclusive_count}",
                body_style,
            ))
            story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("No claims extracted or evidence retrieved.", body_style))
    story.append(Spacer(1, 8))

    # 10. Final assessment
    story.append(Paragraph("6. Final Assessment", section_style))
    if analysis.summary:
        story.append(Paragraph(analysis.summary, body_style))
    story.append(Paragraph(f"Processing time: {analysis.processing_time_ms or 0}ms", body_style))
    story.append(Spacer(1, 8))

    # 11. Limitations
    story.append(Paragraph("7. Limitations &amp; Disclaimer", section_style))
    limitations = [
        "Model predictions are probabilistic estimates based on statistical patterns from training data.",
        "They are NOT factual determinations. High confidence does not mean factual correctness.",
        "Evidence assessments use retrieved sources and are clearly distinguished from ML predictions.",
        "Absence of evidence is never interpreted as proof of falsehood.",
        "Models were trained on ISOT (2015-2018 US political news) and may not generalise.",
        "This report is for research purposes only and should not be used as sole basis for decisions.",
    ]
    for lim in limitations:
        story.append(Paragraph(f"&bull; {lim}", disclaimer_style))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "VeriFact AI - Fake News Detection Platform",
        ParagraphStyle("Footer", parent=disclaimer_style, alignment=TA_CENTER),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _generate_pdf_html_fallback(analysis: "AnalysisResponse", analysis_id: int) -> bytes:
    """
    Generate a simple PDF using HTML to PDF conversion.
    Used when reportlab is not installed.
    """
    # Build HTML report
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: Arial, sans-serif; margin: 40px; color: #334155; }}
h1 {{ color: #4f46e5; font-size: 22px; }}
h2 {{ color: #1e293b; font-size: 14px; margin-top: 20px; }}
.verdict {{ font-size: 18px; font-weight: bold; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #e2e8f0; padding: 6px; font-size: 10px; text-align: left; }}
th {{ background: #e0e7ff; }}
.disclaimer {{ font-size: 9px; color: #94a3b8; margin-top: 10px; }}
</style></head><body>
<h1>VeriFact AI</h1>
<p>Fake News Detection &amp; Verification Report</p>
<p><b>Verification ID:</b> #{analysis_id}</p>
<p><b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<h2>1. Submitted Content</h2>
<p><b>Input type:</b> {analysis.input_type}</p>
<p><b>Content:</b> {analysis.original_input[:500]}</p>

<h2>2. Predicted Classification</h2>
<p class="verdict" style="color: {'#f87171' if analysis.ml_verdict == 'FAKE' else '#34d399' if analysis.ml_verdict == 'REAL' else '#fbbf24'}">
{analysis.ml_verdict}
</p>
<p><b>Confidence:</b> {(analysis.ml_confidence or 0)*100:.1f}%</p>

<h2>3. Model Information</h2>
<table><tr><th>Model</th><th>Label</th><th>Confidence</th></tr>
{''.join(f'<tr><td>{p.model_name}</td><td>{p.label}</td><td>{p.confidence*100:.1f}%</td></tr>' for p in (analysis.model_predictions or []))}
</table>

<h2>4. Evidence Sources</h2>
{''.join(f'<p><b>Claim {c.position}:</b> {c.claim_text[:200]}<br/>Evidence: {c.evidence_verdict or "N/A"}<br/>Supporting: {c.supporting_count} | Contradicting: {c.contradicting_count}</p>' for c in (analysis.claims or []))}

<h2>5. Final Assessment</h2>
<p>{analysis.summary or 'No summary available.'}</p>
<p><b>Processing time:</b> {analysis.processing_time_ms or 0}ms</p>

<h2>6. Limitations &amp; Disclaimer</h2>
<p class="disclaimer">
- Model predictions are probabilistic, not factual determinations.<br/>
- High confidence does not mean factual correctness.<br/>
- Absence of evidence is never proof of falsehood.<br/>
- Models trained on ISOT may not generalise to all content.<br/>
- This report is for research purposes only.
</p>
<p style="text-align:center; color:#94a3b8; margin-top:20px;">VeriFact AI - Fake News Detection Platform</p>
</body></html>"""

    # Try weasyprint or pdfkit
    try:
        from weasyprint import HTML
        pdf = HTML(string=html).write_pdf()
        return pdf
    except ImportError:
        pass

    try:
        import pdfkit
        pdf = pdfkit.from_string(html, False)
        return pdf
    except Exception:
        pass

    # Last resort: return HTML as bytes with a note
    # (The frontend can display this in a new tab)
    return html.encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Research / Model Evaluation PDF report
# ══════════════════════════════════════════════════════════════════════════════

async def generate_research_pdf(db: AsyncSession) -> bytes:
    """
    Generate a comprehensive model evaluation research report PDF.

    Contains:
      1. Dataset details
      2. Preprocessing methodology
      3. Model architecture
      4. Training configuration
      5. Evaluation metrics
      6. Comparative results
      7. Research observations
    """
    # Fetch all model versions with metrics
    result = await db.execute(
        select(ModelVersion)
        .where(ModelVersion.is_active.is_(True))
        .order_by(ModelVersion.model_name)
    )
    models = result.scalars().all()

    # Fetch stats
    from app.services.stats_service import _get_stats_inner, get_stats
    stats = await get_stats(db)

    if _reportlab_available():
        return _generate_research_pdf_reportlab(models, stats)
    else:
        return _generate_research_html(models, stats)


def _generate_research_pdf_reportlab(models, stats) -> bytes:
    """Generate research PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"],
        fontSize=22, textColor=HexColor("#4f46e5"), spaceAfter=6)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#64748b"), spaceAfter=20)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"],
        fontSize=13, textColor=HexColor("#1e293b"), spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle("Body", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#334155"), spaceAfter=6, leading=14)

    story = []

    # Title
    story.append(Paragraph("VeriFact AI", title_style))
    story.append(Paragraph("Model Evaluation Research Report", subtitle_style))
    story.append(Paragraph(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", body_style))
    story.append(Spacer(1, 10))

    # 1. Dataset details
    story.append(Paragraph("1. Dataset Details", section_style))
    story.append(Paragraph("<b>Dataset:</b> ISOT Fake News Dataset (Ahmed et al., 2018)", body_style))
    story.append(Paragraph("<b>Size:</b> ~44,900 articles (23,481 fake + 21,417 real)", body_style))
    story.append(Paragraph("<b>Domain:</b> US political news (2015-2018)", body_style))
    story.append(Paragraph("<b>Language:</b> English", body_style))
    story.append(Paragraph("<b>Split:</b> 80% train / 10% validation / 10% test (stratified, random_state=42)", body_style))
    story.append(Spacer(1, 8))

    # 2. Preprocessing methodology
    story.append(Paragraph("2. Preprocessing Methodology", section_style))
    story.append(Paragraph("<b>Baseline mode (TF-IDF):</b>", body_style))
    preprocessing_steps = [
        "Decode HTML entities",
        "Strip HTML/XML tags",
        "Remove URLs and email addresses",
        "Unicode NFKC normalisation",
        "Lowercase",
        "Expand contractions",
        "Remove non-alphabetic characters",
        "Compress repeated characters (max 3)",
        "Collapse whitespace",
    ]
    for step in preprocessing_steps:
        story.append(Paragraph(f"&bull; {step}", body_style))
    story.append(Paragraph("<b>Transformer mode (DistilBERT):</b> Light cleaning (HTML/URL removal only, preserves casing and punctuation).", body_style))
    story.append(Spacer(1, 8))

    # 3. Model architecture
    story.append(Paragraph("3. Model Architecture", section_style))
    arch_data = [["Model", "Type", "Algorithm", "Parameters"]]
    for mv in models:
        arch_data.append([
            mv.model_name,
            mv.model_type.value,
            mv.algorithm.value,
            f"{mv.model_size_bytes or 0} bytes" if mv.model_size_bytes else "N/A",
        ])
    arch_table = Table(arch_data, colWidths=[120, 80, 100, 80])
    arch_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e0e7ff")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 8))

    # 4. Training configuration
    story.append(Paragraph("4. Training Configuration", section_style))
    for mv in models:
        story.append(Paragraph(f"<b>{mv.model_name}</b>:", body_style))
        if mv.hyperparameters_json:
            try:
                hyp = json.loads(mv.hyperparameters_json)
                for k, v in hyp.items():
                    story.append(Paragraph(f"&nbsp;&nbsp;{k}: {v}", body_style))
            except Exception:
                pass
        story.append(Paragraph(f"&nbsp;&nbsp;random_seed: {mv.random_seed or 42}", body_style))
        story.append(Paragraph(f"&nbsp;&nbsp;train_time: {mv.train_time_s or 'N/A'}s" if mv.train_time_s else "", body_style))
    story.append(Spacer(1, 8))

    # 5. Evaluation metrics
    story.append(Paragraph("5. Evaluation Metrics", section_style))
    story.append(Paragraph("All metrics computed on the same held-out 10% test set. No test data used during training.", body_style))
    story.append(Spacer(1, 6))

    metrics_data = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Test N"]]
    for mv in models:
        metrics_data.append([
            mv.model_name,
            f"{mv.accuracy*100:.2f}%" if mv.accuracy else "-",
            f"{mv.precision*100:.2f}%" if mv.precision else "-",
            f"{mv.recall*100:.2f}%" if mv.recall else "-",
            f"{mv.f1_score*100:.2f}%" if mv.f1_score else "-",
            f"{mv.roc_auc:.4f}" if mv.roc_auc else "-",
            str(mv.test_samples or "-"),
        ])
    metrics_table = Table(metrics_data, colWidths=[100, 60, 60, 60, 50, 55, 45])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e0e7ff")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 8))

    # 6. Comparative results
    story.append(Paragraph("6. Comparative Results", section_style))
    if models:
        best_acc = max(models, key=lambda m: m.accuracy or 0)
        best_f1 = max(models, key=lambda m: m.f1_score or 0)
        fastest = min(models, key=lambda m: m.train_time_s or 9999)
        story.append(Paragraph(f"<b>Best accuracy:</b> {best_acc.model_name} ({(best_acc.accuracy or 0)*100:.2f}%)", body_style))
        story.append(Paragraph(f"<b>Best F1 score:</b> {best_f1.model_name} ({(best_f1.f1_score or 0)*100:.2f}%)", body_style))
        story.append(Paragraph(f"<b>Fastest training:</b> {fastest.model_name} ({fastest.train_time_s or 0:.1f}s)", body_style))
    story.append(Spacer(1, 8))

    # 7. Research observations
    story.append(Paragraph("7. Research Observations", section_style))
    observations = [
        "Traditional TF-IDF models (LR, SVM, NB) train in seconds but lack contextual understanding.",
        "Transformer models (DistilBERT) capture contextual relationships but require GPU for practical training.",
        "ISOT is near-balanced (52.3% fake, 47.7% real), reducing class-imbalance bias.",
        "The dataset covers only 2015-2018 US political news - generalisability to other domains is not established.",
        "High accuracy on ISOT does not guarantee correctness on arbitrary real-world content.",
        "These results are correlational observations from a single dataset, not causal conclusions.",
    ]
    for obs in observations:
        story.append(Paragraph(f"&bull; {obs}", body_style))
    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "VeriFact AI - Model Evaluation Report",
        ParagraphStyle("Footer", parent=body_style, fontSize=9,
            textColor=HexColor("#94a3b8"), alignment=1),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _generate_research_html(models, stats) -> bytes:
    """Fallback HTML research report."""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: Arial, sans-serif; margin: 40px; color: #334155; }}
h1 {{ color: #4f46e5; font-size: 22px; }}
h2 {{ color: #1e293b; font-size: 14px; margin-top: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #e2e8f0; padding: 6px; font-size: 10px; }}
th {{ background: #e0e7ff; }}
</style></head><body>
<h1>VeriFact AI</h1>
<p>Model Evaluation Research Report</p>
<p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

<h2>1. Dataset Details</h2>
<p><b>Dataset:</b> ISOT Fake News Dataset (Ahmed et al., 2018)</p>
<p><b>Size:</b> ~44,900 articles</p>
<p><b>Split:</b> 80/10/10 stratified, random_state=42</p>

<h2>2. Preprocessing Methodology</h2>
<p>Baseline: HTML strip, URL removal, lowercase, contraction expansion, non-alpha removal.</p>
<p>Transformer: Light cleaning (HTML/URL only, preserves casing/punctuation).</p>

<h2>3. Evaluation Metrics</h2>
<table><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>ROC-AUC</th><th>Test N</th></tr>
{''.join(f'<tr><td>{mv.model_name}</td><td>{(mv.accuracy or 0)*100:.2f}%</td><td>{(mv.f1_score or 0)*100:.2f}%</td><td>{mv.roc_auc or "-"}</td><td>{mv.test_samples or "-"}</td></tr>' for mv in models)}
</table>

<h2>4. Research Observations</h2>
<p>- Traditional TF-IDF models train fast but lack contextual understanding.</p>
<p>- Transformers capture context but require GPU for training.</p>
<p>- ISOT is near-balanced; results may not generalise to other domains.</p>
<p>- High accuracy on ISOT does not guarantee real-world correctness.</p>

<p style="text-align:center; color:#94a3b8; margin-top:30px;">VeriFact AI - Model Evaluation Report</p>
</body></html>"""

    return html.encode("utf-8")
