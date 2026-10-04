"""Create portable training and demo notebooks from the documented workflow."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip().splitlines(True)}


BOOTSTRAP = '''
import os, sys, json, subprocess, zipfile
from pathlib import Path

try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
if IS_COLAB:
    from google.colab import output
    output.enable_custom_widget_manager()

def extract_trusted_zip(path, target):
    target = Path(target).resolve()
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            if not (target / member.filename).resolve().is_relative_to(target):
                raise ValueError("Archive contains an unsafe path")
        z.extractall(target)

def find_project():
    for parent in [Path.cwd(), *Path.cwd().parents]:
        for candidate in [parent, parent / "legal_text_classification"]:
            if (candidate / "src" / "legal_text").is_dir():
                return candidate.resolve()
    return None

ROOT = find_project()
if ROOT is None and IS_COLAB:
    print("Upload team9-project.zip from the private GitHub release.")
    from google.colab import files
    for filename in files.upload():
        if filename.endswith(".zip"):
            extract_trusted_zip(filename, Path.cwd())
    ROOT = find_project()
if ROOT is None:
    raise FileNotFoundError("Open this notebook inside the extracted legal_text_classification project.")
os.chdir(ROOT)
if IS_COLAB:
    # Run this setup cell before importing NumPy/scikit-learn/gensim. If those
    # libraries were previously imported, restart the session after installation.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".build" / "matplotlib"))
print("Project:", ROOT)
print("Python:", sys.version.split()[0])
'''


def save(name, cells):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3 (Team 9)", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"},
                       "colab": {"name": name}}, "cells": cells}
    for i, cell in enumerate(cells):
        cell["id"] = f"team9-{i:03d}"
    (ROOT / "notebooks" / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")


save("01_train_and_evaluate.ipynb", [
    md('''# Classification of Legal Text: experiments
    **Team 9:** Sai Sathvik Gullipalli (PES2UG24CS430) and Priyanka Mallikarjuna Reddy (PES2UG24CS378).

    This notebook reproduces our workflow on the original textacy Supreme Court corpus.
    It trains TF-IDF, LDA, and Doc2Vec classifiers, selects configurations on validation
    macro-F1, and evaluates the frozen models on a held-out test partition.

    **Local:** install `requirements.txt` in the project's virtual environment and select its kernel.
    **Colab:** upload this notebook, run setup, then upload `team9-project.zip` when prompted.
    Use a Python 3.11 or 3.12 CPU runtime. No paid service or GPU is needed.
    A saved run loads its results without evaluating the test set again.'''),
    code(BOOTSTRAP),
    md('''## Train or load an existing run
    Feature learning uses training opinions only. The subprocess starts with fixed
    hashing and thread settings for reproducibility. Training can take tens of minutes
    on a CPU, depending on hardware. Use the saved demo bundle during the review.'''),
    code('''
    metrics_path = ROOT / "results" / "metrics.json"
    if metrics_path.exists():
        print("Loading completed results. No models are retrained and the test set is not reevaluated.")
    else:
        env = dict(os.environ, PYTHONHASHSEED="42", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env["PYTHONPATH"] = str(ROOT / "src")
        subprocess.run([sys.executable, "-m", "legal_text.train", "--project-root", str(ROOT)], env=env, check=True)
    '''),
    md("## Dataset, cleaning, and fixed partitions"),
    code('''
    import pandas as pd
    from IPython.display import display, Image
    summary = json.loads((ROOT / "results/dataset_summary.json").read_text())
    provenance = json.loads((ROOT / "results/provenance.json").read_text())
    print("Dataset SHA256:", provenance["sha256"])
    print("Source:", provenance["source_page"])
    print("Raw records:", summary["raw_records"], "Retained:", summary["retained_records"])
    print("Excluded rare categories:", summary["excluded_rare_categories"])
    display(pd.DataFrame({n: v["class_counts"] for n, v in summary["splits"].items()}).fillna(0).astype(int))
    manifest = pd.read_csv(ROOT / "results/split_manifest.csv", keep_default_na=False)
    assert manifest.groupby("group")["split"].nunique().max() == 1
    assert manifest.groupby("text_hash")["split"].nunique().max() == 1
    print("Verified: case groups and identical texts never cross partitions.")
    '''),
    md("## Validation selection\nMacro-F1 gives each category equal weight. Test scores do not participate in this choice."),
    code('''
    candidates = pd.read_csv(ROOT / "results/validation_candidates.csv")
    display(candidates[["model", "topics", "C", "accuracy", "macro_f1", "weighted_f1"]])
    selection = json.loads((ROOT / "results/selection.json").read_text())
    print("Validation-selected winner:", selection["winner"])
    print("Tie-breaks:", selection["selection_metric"])
    '''),
    md("## Frozen test results"),
    code('''
    metrics = json.loads(metrics_path.read_text())
    display(pd.DataFrame(metrics).T)
    display(Image(filename=str(ROOT / "results/model_comparison.png")))
    for name in ("tfidf", "lda", "doc2vec"):
        print(name)
        display(pd.read_csv(ROOT / f"results/{name}_per_category.csv"))
    display(Image(filename=str(ROOT / f"results/{selection['winner']}_confusion_matrix.png")))
    '''),
    md('''## Interpretation
    LDA uses count features because its generative model describes word counts.
    This differs from the reference report's stated TF-IDF input. Doc2Vec can discover
    similarity between opinions, but that similarity does not establish legal authority.
    Random grouped splits measure performance on this historical corpus, not generalization
    to future cases. Small categories and possible source metadata mismatches limit conclusions.

    See `docs/viva.md` for explanations and `results/` for all exported evidence.
    Open `02_live_demo.ipynb` for interactive inference without retraining.'''),
    code('''
    if IS_COLAB:
        import shutil
        archive = shutil.make_archive(str(ROOT / "team9-model-bundle"), "zip", root_dir=ROOT, base_dir="artifacts")
        from google.colab import files
        files.download(archive)
    else:
        print("Saved artifacts:", ROOT / "artifacts")
    '''),
])

save("02_live_demo.ipynb", [
    md('''# Classification of Legal Text: live demo
    **Team 9:** Sai Sathvik Gullipalli (PES2UG24CS430) and Priyanka Mallikarjuna Reddy (PES2UG24CS378).

    Classify an English legal opinion and find similar opinions using saved models.
    This notebook does **not** retrain models. Use `team9-project.zip` and
    `team9-model-bundle.zip` from the private release, or the complete offline ZIP.
    Load only the team's trusted model bundle: joblib files contain executable Python objects.'''),
    code(BOOTSTRAP),
    code('''
    if not (ROOT / "artifacts/manifest.json").exists():
        if IS_COLAB:
            print("Upload team9-model-bundle.zip.")
            from google.colab import files
            for filename in files.upload():
                if filename.endswith(".zip"):
                    extract_trusted_zip(filename, ROOT)
        else:
            raise FileNotFoundError("Extract team9-model-bundle.zip here, then rerun this cell.")
    from legal_text.inference import load, predict, retrieve_similar
    demo = load(ROOT / "artifacts")
    print("Loaded models:", list(demo.models))
    print("Validation-selected model:", demo.manifest["winner"])
    '''),
    md('''## Saved inference example
    This cell provides a static backup if the widget interface is unavailable.
    Example passages come from the held-out test partition and may be truncated
    for display. The reported full-test metrics come from complete opinions.'''),
    code('''
    import pandas as pd
    from IPython.display import display
    example = next(e for e in demo.examples if e["true_category"] == "Criminal Procedure")
    print("Case:", example["case_name"])
    print("True category:", example["true_category"])
    for model in demo.models:
        result = predict(example["text"], model)
        print(model, "predicted:", result["category"])
        display(pd.DataFrame(result["top_three"]))
    neighbors = retrieve_similar(example["text"], 5)
    display(pd.DataFrame(neighbors)[["case_name", "decision_date", "category", "similarity"]])
    '''),
    md('''## Interactive demonstration
    Choose an example or paste a passage, select a model, then press the prediction
    button. TF-IDF also displays influential terms. Probabilities are uncalibrated.
    Short passages are less representative of the full opinions used for training.'''),
    code('''
    from legal_text.widgets import launch
    interactive_demo = launch(ROOT / "artifacts")
    '''),
    md('''## Results and limitations
    Model probabilities measure model preference, not verified legal correctness.
    Similarity scores measure textual resemblance and do not establish binding precedent.
    The corpus covers historical US Supreme Court opinions, rather than Indian law
    or all types of legal documents.'''),
    code('''
    if (ROOT / "results/metrics.json").exists():
        display(pd.DataFrame(json.loads((ROOT / "results/metrics.json").read_text())).T)
    else:
        display(pd.DataFrame(json.loads((ROOT / "artifacts/metrics.json").read_text())).T)
    '''),
])
print("Created training and demo notebooks")
