"""Build the two-page write-up exclusively from exported experiment evidence."""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission"
metrics = json.loads((ROOT / "results/metrics.json").read_text())
summary = json.loads((ROOT / "results/dataset_summary.json").read_text())
selection = json.loads((ROOT / "results/selection.json").read_text())
provenance = json.loads((ROOT / "results/provenance.json").read_text())
names = {"majority": "Majority baseline", "tfidf": "TF-IDF + LR", "lda": "LDA + LR", "doc2vec": "Doc2Vec + LR"}
winner = selection["winner"]
styles = {
 "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=22, leading=25, textColor=colors.HexColor("#183746"), spaceAfter=8),
 "sub": ParagraphStyle("Sub", fontName="Helvetica", fontSize=9, leading=12, textColor=colors.HexColor("#52616B"), spaceAfter=12),
 "head": ParagraphStyle("Head", fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#183746"), spaceBefore=10, spaceAfter=5),
 "body": ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=13.5, spaceAfter=6),
 "small": ParagraphStyle("Small", fontName="Helvetica", fontSize=8, leading=10.5, spaceAfter=4),
}
story = []
def p(text, style="body"):
    story.append(Paragraph(text, styles[style]))
def table(values, widths):
    t = Table(values, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#183746")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"), ("FONTSIZE", (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7), ("TOPPADDING", (0,0), (-1,-1), 7),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F0F4F5"), colors.white]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ALIGN", (1,0), (-1,-1), "RIGHT")]))
    story.append(t)
    story.append(Spacer(1, 5))

p("Classification of Legal Text", "title")
p("UE24CS352A Machine Learning mini-project / Team 9 / Problem 5<br/>"
  "Sai Sathvik Gullipalli (PES2UG24CS430)<br/>Priyanka Mallikarjuna Reddy (PES2UG24CS378)", "sub")
p("Problem statement", "head")
p("Classify an English US Supreme Court opinion into its broad legal issue area. "
  "We compare lexical, topic, and document-embedding representations and demonstrate "
  "similar-opinion retrieval. This project adapts Iyer's Stanford study [1], with an "
  "independent dataset cleaning and evaluation procedure.")
p("Dataset and preparation", "head")
p(f"The original textacy release [2] contains <b>{summary['raw_records']:,} opinions</b>, spanning "
  f"{summary['date_range'][0]} to {summary['date_range'][1]}. The official mapping [3] has "
  f"14 named issue areas. After cleaning, this run retains <b>{summary['retained_records']:,} opinions "
  f"in {len(summary['present_labels'])} supported categories</b>, across {summary['retained_groups']:,} case groups.")
removed = summary['raw_records'] - summary['retained_records']
p(f"We exclude empty text and missing/unknown labels, remove identical normalized text, "
  f"and join records through shared citations or text hashes. Groups with conflicting labels "
  f"are excluded. Categories require at least five independent case groups. Overall, "
  f"{removed:,} records are excluded, including {summary.get('conflicting_group_records_removed',0):,} "
  f"records in conflicting groups and {summary.get('exact_duplicates_removed',0):,} duplicate copies. "
  f"Rare-category exclusions: {summary['excluded_rare_categories'] or 'none'}.")
table([["Partition", "Opinions", "Case groups"]] + [[n.title(), f"{s['records']:,}", f"{s['groups']:,}"] for n,s in summary['splits'].items()], [230,120,150])
p("A fixed, approximately 60/20/20 grouped, stratified split uses seed 42. Case groups "
  "never cross partitions. Every vocabulary, topic model, and embedding learns from "
  "training opinions only. Validation chooses configurations. Test opinions evaluate "
  "the frozen models once.")
p("Approach and implementation", "head")
p("<b>TF-IDF + LR:</b> sparse unigram/bigram features, sublinear term frequency, English "
  "stop words, and at most 30,000 features.<br/>"
  "<b>LDA + LR:</b> count features with at most 15,000 words and 20 or 40 topics. Online "
  "LDA uses five passes. Counts replace the reference's stated TF-IDF input because LDA "
  "models word counts.<br/>"
  "<b>Doc2Vec + LR:</b> 100-dimensional PV-DBOW embeddings, 30 epochs, one worker. "
  "All classification partitions use inferred vectors, with stable text-specific seeds.")
p("Each Logistic Regression compares C = 0.1, 1, 10. Validation macro-F1 chooses the "
  "configuration, with accuracy then inference latency as tie-breaks. Python modules "
  "implement training and inference. Separate Jupyter/Colab notebooks provide experiments "
  "and a saved-model demo. Exported artifacts preserve provenance, partitions, and metrics.")

story.append(PageBreak())
p("Results and conclusions", "title")
p(f"Frozen test comparison / {summary['splits']['test']['records']:,} held-out opinions", "sub")
table([["Model", "Accuracy", "Macro-F1", "Weighted F1"]] + [[names[n], f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}", f"{m['weighted_f1']:.3f}"] for n,m in metrics.items()], [200,100,100,100])
story.append(Image(str(ROOT / "results/model_comparison.png"), width=400, height=225))
p(f"Validation selects <b>{names[winner]}</b> with macro-F1 {selection['models'][winner]['macro_f1']:.3f}. "
  f"Its held-out accuracy is {metrics[winner]['accuracy']:.3f} and macro-F1 is "
  f"{metrics[winner]['macro_f1']:.3f}. The majority baseline reaches accuracy "
  f"{metrics['majority']['accuracy']:.3f}. Macro-F1 gives each category equal weight, "
  "revealing performance differences that accuracy can conceal in an imbalanced corpus.")
latencies = ", ".join(f"{names[n]}: {metrics[n]['test_ms_per_document']:.1f} ms" for n in ("tfidf","lda","doc2vec"))
p(f"Measured per-opinion inference on the execution CPU: {latencies}. These timings "
  "include feature transformation and classification and depend on hardware and opinion length.")
p("Demonstration and limitations", "head")
p("The demo loads saved models, displays a category and top-three uncalibrated probabilities, "
  "and shows influential terms for TF-IDF. Doc2Vec cosine similarity retrieves five distinct "
  "training-case citations with metadata and excerpts. This retrieval is qualitative: "
  "textual resemblance does not establish binding authority or a valid legal argument.")
p("Results apply to historical US Supreme Court opinions. Random grouped splits do not "
  "measure future-case generalization. Imbalanced categories, possible source metadata "
  "mismatches, and boilerplate affect reliability. Short user passages differ from full "
  "training opinions. No claim of generalization to Indian law is made. The reference "
  "paper's prose and results table conflict, so its scores are not reused as our results.")
p("References and reproducibility", "head")
p('[1] K. Iyer. <i>Classification of Legal Text</i>, Stanford CS229, 2020. '
  '<link href="https://cs229.stanford.edu/proj2020spr/report/Iyer.pdf">cs229.stanford.edu/proj2020spr/report/Iyer.pdf</link><br/>'
  '[2] textacy-data. Supreme Court corpus, Python 3 release v1.0. '
  '<link href="https://github.com/bdewilde/textacy-data/releases/tag/supreme_court_py3_v1.0">github.com/bdewilde/textacy-data, supreme_court_py3_v1.0</link><br/>'
  '[3] textacy SupremeCourt mapping. Metadata: Spaeth, Epstein et al., Supreme Court Database, '
  '2016 Release 1. CC BY-NC 3.0 US attribution, per corpus maintainer.', "small")
p(f"Dataset SHA256: {provenance['sha256']}<br/>"
  "Reproduction: README, pinned requirements, saved split manifest, validation candidates, "
  "test predictions, and per-category reports in the private repository.", "small")

def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#52616B"))
    canvas.drawString(45, 25, "Team 9 / Classification of Legal Text")
    canvas.drawRightString(A4[0]-45, 25, str(doc.page))

OUT.mkdir(exist_ok=True)
doc = SimpleDocTemplate(str(OUT / "team9_writeup.pdf"), pagesize=A4, rightMargin=45,
    leftMargin=45, topMargin=38, bottomMargin=38, title="Classification of Legal Text - Team 9",
    author="Sai Sathvik Gullipalli; Priyanka Mallikarjuna Reddy")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("Created", OUT / "team9_writeup.pdf")
