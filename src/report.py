"""Generate the Part 3 Markdown and four-page PDF technical report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "analysis"
REPORTS = ROOT / "reports"
LABELS = ["macro", "meso", "micro"]


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def metric_block(payload: dict, key: str) -> dict:
    value = payload[key]
    return value.get("metrics", value)


def excerpts() -> list[dict]:
    with (ANALYSIS / "qualitative_examples.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def short(text: str, limit: int = 190) -> str:
    value = " ".join(text.split()).replace("&", "and").replace("<", "(").replace(">", ")")
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def markdown() -> str:
    normal = load_json("in_domain_metrics.json")
    controlled = load_json("length_controlled_metrics.json")
    cross_domain = load_json("cross_domain_biomedicine_metrics.json")
    n_probe, n_majority, n_section = (
        metric_block(normal, "embedding_probe"),
        metric_block(normal, "majority_baseline"),
        metric_block(normal, "section_name_baseline"),
    )
    c_probe = metric_block(controlled, "embedding_probe")
    x_probe = metric_block(cross_domain, "embedding_probe")
    x_majority = metric_block(cross_domain, "majority_baseline")
    x_section = metric_block(cross_domain, "section_name_baseline")
    rows = excerpts()
    lines = [
        "# Is Semantic Level of Detail Linearly Decodable from Frozen SciBERT Embeddings?",
        "",
        "## Introduction",
        "",
        "Semantic Level of Detail (SLoD) distinguishes high-level scientific statements (macro), section-level framing (meso), and implementation or result details (micro). For the Temporal Knowledge Hypergraph project, a reliable SLoD signal could route retrieval toward summaries, section overviews, or fine-grained evidence. This study asks whether SLoD is linearly decodable from frozen scientific-language embeddings.",
        "",
        "## Methodology",
        "",
        "Weak supervision used rule-based labels derived from QASPER document structure: titles, abstracts, the first two introduction paragraphs, and conclusions were macro; the first sentence of eligible section leads was meso; and non-lead paragraphs in methods, experiments, results, evaluation, implementation, and approach sections were micro. The balanced dataset contains 5,292 spans (1,764 per class) from 877 NLP papers. Splits were made by paper ID (70/10/20), preventing paper overlap. Normal spans were filtered to 30–300 tokens, with a configurable per-paper/per-label cap.",
        "",
        "Frozen `allenai/scibert_scivocab_uncased` representations were attention-mask-aware mean-pooled token embeddings. Model weights were never updated. A logistic-regression probe was trained on the frozen vectors. Baselines were (1) the majority training class and (2) logistic regression using only `section_name`. The length control created a separate dataset by truncating every span to exactly 30 tokens before re-embedding and repeating the paper-grouped in-domain evaluation.",
        "",
        "## Results",
        "",
        "| Condition/model | Accuracy | Macro-F1 | Status |",
        "|---|---:|---:|---|",
        f"| In-domain embedding probe | {n_probe['accuracy']:.3f} | {n_probe['macro_f1']:.3f} | Complete |",
        f"| In-domain majority baseline | {n_majority['accuracy']:.3f} | {n_majority['macro_f1']:.3f} | Complete |",
        f"| In-domain section-name baseline | {n_section['accuracy']:.3f} | {n_section['macro_f1']:.3f} | Complete |",
        f"| Length-controlled embedding probe | {c_probe['accuracy']:.3f} | {c_probe['macro_f1']:.3f} | Complete |",
        f"| QASPER NLP → PMC biomedicine embedding probe | {x_probe['accuracy']:.3f} | {x_probe['macro_f1']:.3f} | Complete external test |",
        f"| Cross-domain majority baseline | {x_majority['accuracy']:.3f} | {x_majority['macro_f1']:.3f} | Complete |",
        f"| Cross-domain section-name baseline | {x_section['accuracy']:.3f} | {x_section['macro_f1']:.3f} | Complete |",
        "",
        "The in-domain probe substantially outperformed the majority baseline. Exact-length control reduced macro-F1 by 0.105 (10.5 points), but performance remained far above majority. The QASPER NLP → PMC biomedicine probe reached 0.824 macro-F1, 0.062 below in-domain and far above its 0.167 majority baseline. It was approximately tied with and slightly below the 0.827 section-name baseline. This is substantial external transfer, but it does not establish embedding-specific abstraction beyond structural cues. Because both domain and source corpus change, this condition measures combined domain-and-corpus transfer.",
        "",
        "### Confusion matrices (rows=true, columns=predicted; macro/meso/micro)",
        "",
        "- In-domain: `[[333, 15, 23], [6, 314, 17], [38, 23, 299]]`",
        "- Length-controlled: `[[285, 23, 63], [24, 280, 33], [60, 33, 267]]`",
        "- Cross-domain biomedical: `[[407, 17, 76], [13, 448, 39], [76, 43, 381]]`",
        "",
        "The main normal-condition error was micro→macro (38). Under length control, macro↔micro confusion became dominant (63 and 60), consistent with truncation removing cues that distinguish summaries from detailed evidence. Meso was not the hardest class: it had the best normal F1 (0.912); micro was hardest (0.856 normal, 0.739 controlled). The t-SNE visualization in `results/analysis/tsne_embeddings.png` shows partial class structure but substantial overlap; it is descriptive rather than evidence of linear separability.",
        "",
        "## Qualitative error analysis",
        "",
    ]
    for outcome, title in [("correct_high_confidence", "High-confidence correct examples"), ("failed_high_confidence", "High-confidence failures")]:
        lines += [f"### {title}", ""]
        for label in LABELS:
            subset = [r for r in rows if r["outcome"] == outcome and r["true_label"] == label]
            lines.append(f"**{label}**")
            lines.append("")
            for r in subset:
                lines.append(f"- `{r['true_label']}→{r['predicted_label']}` ({float(r['confidence']):.3f}): {short(r['text'], 280)}")
            lines.append("")
    lines += [
        "## Discussion and future work",
        "",
        "SLoD is strongly linearly decodable in-domain from frozen SciBERT embeddings under these weak labels, but the result is not yet evidence for a pure semantic hierarchy representation. The length drop and strong section-name baseline show that structural proxies contribute heavily. In a RAG system, the calibrated probe should therefore be an auxiliary routing signal: macro for overview retrieval, meso for section-level navigation, and micro for implementation, numerical, or evidential passages—combined with section metadata and uncertainty thresholds rather than used as ground truth.",
        "",
        "The strongest next tests are: repeat cross-domain evaluation with NLP and biomedicine drawn from the same source corpus; collect a small human-labelled SLoD set; mask section-title expressions; match content and length more tightly; compare SciBERT layers and embedding models; add random-label/selectivity controls; and test calibration and nonlinear probes. These experiments would separate semantic abstraction from document-template artifacts.",
    ]
    return "\n".join(lines) + "\n"


def build_pdf(path: Path) -> None:
    normal = load_json("in_domain_metrics.json")
    controlled = load_json("length_controlled_metrics.json")
    cross_domain = load_json("cross_domain_biomedicine_metrics.json")
    n_probe = metric_block(normal, "embedding_probe")
    n_majority = metric_block(normal, "majority_baseline")
    n_section = metric_block(normal, "section_name_baseline")
    c_probe = metric_block(controlled, "embedding_probe")
    x_probe = metric_block(cross_domain, "embedding_probe")
    examples = excerpts()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Title2", parent=styles["Title"], fontSize=18, leading=21, spaceAfter=8))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=13, leading=15, spaceBefore=5, spaceAfter=4, textColor=colors.HexColor("#183153")))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=10.5, leading=12, spaceBefore=4, spaceAfter=2, textColor=colors.HexColor("#2A5C8A")))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=8.5, leading=11, spaceAfter=4))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.2, leading=8.6, spaceAfter=2))
    styles.add(ParagraphStyle(name="Tiny", parent=styles["BodyText"], fontSize=6.6, leading=7.8, spaceAfter=1.5))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=7, leading=8, alignment=TA_CENTER, textColor=colors.grey))

    doc = SimpleDocTemplate(str(path), pagesize=A4, rightMargin=14*mm, leftMargin=14*mm, topMargin=12*mm, bottomMargin=12*mm,
                            title="SLoD Probe Technical Report", author="SLoD Probe Project")
    story = []
    P = lambda text, style="Bodyx": story.append(Paragraph(text, styles[style]))
    P("Is Semantic Level of Detail Linearly Decodable from Frozen SciBERT Embeddings?", "Title2")
    P("Part 3 — Evaluation & Technical Report", "Caption")
    P("Introduction", "H1x")
    P("Semantic Level of Detail (SLoD) distinguishes high-level scientific statements (<b>macro</b>), section-level framing (<b>meso</b>), and implementation or result details (<b>micro</b>). For the Temporal Knowledge Hypergraph, SLoD could route retrieval toward summaries, section overviews, or fine-grained evidence. The research question is whether SLoD is linearly encoded in frozen scientific-text embeddings.")
    P("Methodology", "H1x")
    P("<b>Weak labels.</b> Rule-based weak supervision mapped QASPER structure to labels: title/abstract/first two introduction paragraphs/conclusion → macro; first sentence of eligible section leads → meso; non-lead methods, experiments, results, evaluation, implementation, and approach text → micro. The balanced dataset has 5,292 spans (1,764/class) from 877 NLP papers. Paper-ID splits (70/10/20) prevent leakage; normal spans contain 30–300 tokens and use per-paper/per-label caps.")
    P("<b>Representations and probe.</b> Frozen <i>allenai/scibert_scivocab_uncased</i> token states were attention-mask-aware mean pooled; weights were never updated. Logistic regression was trained on cached vectors. Baselines predict the majority training class or use only section_name. The separate length-control dataset truncates every span to exactly 30 tokens before re-embedding and repeats the same paper-grouped evaluation.")
    P("Experimental conditions", "H2x")
    data = [["Condition", "Train/test", "Purpose"], ["In-domain", "NLP → held-out NLP papers", "Primary linear decoding test"], ["Cross-domain", "QASPER NLP → PMC biomedicine", "External domain-and-corpus transfer"], ["Length-controlled", "NLP → held-out NLP papers", "Remove span-length variation"]]
    t = Table(data, colWidths=[34*mm, 61*mm, 78*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DCE9F5")),("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONT",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4)]))
    story += [t, Spacer(1, 3*mm), PageBreak()]

    P("Quantitative results", "H1x")
    data = [["Condition / model", "Accuracy", "Macro-F1", "Status"],
            ["In-domain embedding", f"{n_probe['accuracy']:.3f}", f"{n_probe['macro_f1']:.3f}", "Complete"],
            ["Majority baseline", f"{n_majority['accuracy']:.3f}", f"{n_majority['macro_f1']:.3f}", "Complete"],
            ["Section-name baseline", f"{n_section['accuracy']:.3f}", f"{n_section['macro_f1']:.3f}", "Complete"],
            ["Length-controlled embedding", f"{c_probe['accuracy']:.3f}", f"{c_probe['macro_f1']:.3f}", "Complete"],
            ["Cross-domain biomedical", f"{x_probe['accuracy']:.3f}", f"{x_probe['macro_f1']:.3f}", "Complete external test"]]
    t = Table(data, colWidths=[64*mm, 25*mm, 25*mm, 59*mm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#183153")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.35,colors.grey),("FONT",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),7.5),("ALIGN",(1,1),(2,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story += [t, Spacer(1, 2*mm)]
    P("Length control reduced macro-F1 by <b>0.105</b> (10.5 points), yet controlled performance remained far above majority. Length is therefore a substantial confound. The section-name baseline (0.785) slightly exceeds the controlled embedding probe (0.780), so the present evidence does not isolate embedding-specific semantic abstraction from structural/lexical section cues.")
    P("Confusion matrices", "H2x")
    matrices = [["", "Pred macro", "Pred meso", "Pred micro"], ["True macro",333,15,23],["True meso",6,314,17],["True micro",38,23,299]]
    matrices2 = [["", "Pred macro", "Pred meso", "Pred micro"], ["True macro",285,23,63],["True meso",24,280,33],["True micro",60,33,267]]
    pair = Table([[Table(matrices, colWidths=[24*mm]*4), Table(matrices2, colWidths=[24*mm]*4)]], colWidths=[87*mm,87*mm])
    for inner in pair._cellvalues[0]:
        inner.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DCE9F5")),("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),6.8),("ALIGN",(1,1),(-1,-1),"CENTER")]))
    story += [Paragraph("In-domain", styles["Caption"]), pair, Paragraph("Left: in-domain. Right: exact-length controlled.", styles["Caption"])]
    P("The main normal error was micro→macro (38). Under control, macro↔micro confusion dominated (63/60). In the biomedical external test, macro↔micro confusion was symmetric (76/76), and the probe reached 0.824 macro-F1. Meso was the easiest class across conditions; micro remained hardest. Because source corpus also changes, the external result is combined domain-and-corpus transfer.")
    img = Image(str(ANALYSIS / "tsne_embeddings.png"), width=112*mm, height=78*mm)
    story += [img, Paragraph("t-SNE: partial class structure with substantial overlap; descriptive, not a linear-separability test.", styles["Caption"]), PageBreak()]

    P("High-confidence correct examples", "H1x")
    P("Three examples per true class; confidence is the probe's maximum predicted probability.", "Small")
    for label in LABELS:
        P(label.upper(), "H2x")
        subset = [r for r in examples if r["outcome"] == "correct_high_confidence" and r["true_label"] == label]
        for i, r in enumerate(subset, 1):
            P(f"<b>{i}. {r['true_label']}→{r['predicted_label']} ({float(r['confidence']):.3f})</b> — {short(r['text'])}", "Small")
    P("Interpretation", "H2x")
    P("Correct macro examples use contribution/conclusion language; meso examples frame section contents or processing choices; micro examples contain equations, model parameters, or fine-grained result comparisons. These cues are compatible with SLoD, but also with document genre conventions.", "Bodyx")
    story.append(PageBreak())

    P("High-confidence failures", "H1x")
    for label in LABELS:
        P(label.upper(), "H2x")
        subset = [r for r in examples if r["outcome"] == "failed_high_confidence" and r["true_label"] == label]
        for i, r in enumerate(subset, 1):
            P(f"<b>{i}. {r['true_label']}→{r['predicted_label']} ({float(r['confidence']):.3f})</b> — {short(r['text'], 170)}", "Tiny")
    P("Discussion and future work", "H1x")
    P("SLoD is <b>strongly linearly decodable in-domain under the current weak labels</b>, but this is not yet proof of a pure semantic hierarchy representation. Length and section-derived lexical structure explain substantial signal. In RAG, use the calibrated probe as an auxiliary router—macro for overviews, meso for section navigation, micro for implementation/numerical evidence—combined with metadata and uncertainty thresholds.", "Small")
    P("Next experiments: repeat NLP and biomedical evaluation within one source corpus; build a small human-labelled SLoD test set; mask section-title expressions; perform tighter content/length matching; compare layers and embedding models; add random-label/selectivity controls; and evaluate calibration and nonlinear probes. These would distinguish semantic abstraction from document-template artifacts.", "Small")

    doc.build(story)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "technical_report.md").write_text(markdown(), encoding="utf-8")
    build_pdf(REPORTS / "technical_report.pdf")
    print(f"Saved reports to {REPORTS}")


if __name__ == "__main__":
    main()
