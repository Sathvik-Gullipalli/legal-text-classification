# Classification of Legal Text

Machine Learning mini-project, UE24CS352A. **Team 9 / Problem 5**.

| Team member | SRN |
| --- | --- |
| Sai Sathvik Gullipalli | PES2UG24CS430 |
| Priyanka Mallikarjuna Reddy | PES2UG24CS378 |

We classify historical US Supreme Court opinions into broad issue areas and retrieve
similar training opinions. We compare TF-IDF, LDA, and Doc2Vec features, each followed
by Logistic Regression. Our dataset, partitions, and evaluation differ from the
Stanford reference, so scores are an independent experiment rather than an exact replication.

## Quick start: saved-model demo

Use Python **3.11 or 3.12**. The original full corpus is not required to run the demo.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e .
```

Download `team9-model-bundle.zip` from the repository's **private v1.0.0 release** and
extract it into this folder. The archive creates `artifacts/`. Load only this trusted
bundle because joblib deserialization executes Python objects.

```bash
python -m jupyterlab notebooks/02_live_demo.ipynb
```

Select the virtual environment's Python kernel and run the notebook cells in order.
Choose a held-out example or paste English legal text. Select a model, then press
**Classify and find similar opinions**. A static inference example is included before
the widgets so the demonstration can continue if widgets are unavailable.

Alternatively, extract `team9-offline-kit.zip` and open its included project. It already
contains the models, submission files, and executed notebooks. Dependencies still need
to be installed before running code. Once installed, the demo works without internet.

## Free Google Colab

1. Download `team9-project.zip` and `team9-model-bundle.zip` from the private release
   while signed into GitHub. Do not paste access tokens into notebooks.
2. Upload `notebooks/02_live_demo.ipynb` to Colab and select a CPU runtime using
   Python 3.11 or 3.12.
3. Run setup. Upload `team9-project.zip` when prompted, followed by the model ZIP.
4. Run the static example and interactive cells. If you imported NumPy or scikit-learn
   before setup, restart the session after installing dependencies.

For training, use `01_train_and_evaluate.ipynb` instead. An uploaded project containing
completed results displays the saved run. To deliberately start a new experiment,
use a clean project/results folder or the explicit `--force` CLI option.

## Reproduce experiments

```bash
PYTHONHASHSEED=42 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m legal_text.train --project-root .
python -m pytest -q
```

The script downloads the original ~111 MB gzip and verifies the official category mapping.
Training uses complete opinions and can take tens of minutes on a CPU. It saves model
checkpoints, validation candidates, selected configurations, test predictions,
classification reports, confusion matrices, and a retrieval index. Seed 42 and one worker
control stochastic inference and training. Tiny numerical/timing differences across
platforms remain possible. `requirements-lock.txt` records the environment used for the
delivered run; `requirements.txt` pins the principal cross-platform dependencies.

If `results/metrics.json` already exists, training refuses to evaluate the test set again
unless `--force` is provided. Preserve the original results before a deliberate rerun.

## Project structure

- `src/legal_text/data.py`: source verification, cleaning, duplicate grouping, fixed splits.
- `src/legal_text/features.py`: training-only TF-IDF, count-based LDA, and Doc2Vec features.
- `src/legal_text/train.py`: validation selection, frozen test evaluation, exports and plots.
- `src/legal_text/inference.py`: classification and similar-opinion retrieval.
- `src/legal_text/widgets.py`: notebook interaction.
- `notebooks/`: training, interactive demo, and executed backups.
- `results/`: measured scores, dataset summary, provenance, and split manifest.
- `submission/`: two-page write-up, eight-slide editable deck, and slides PDF.
- `docs/`: methodology, live-demo script, viva explanations, and contribution template.

## Python interface

```python
from legal_text.inference import load, predict, retrieve_similar
load("artifacts")
prediction = predict("An accused challenges evidence obtained during a search.", "tfidf")
neighbors = retrieve_similar("An accused challenges evidence obtained during a search.", k=5)
```

`predict` returns model name, category/label, top-three probabilities, input warnings,
and TF-IDF term contributions. `retrieve_similar` returns case metadata, an excerpt,
and cosine similarity for up to five distinct citations from training cases.

## Evaluation and interpretation

See `results/metrics.json`, `results/selection.json`, and the write-up for measured scores.
The winner is selected by **validation macro-F1**, then validation accuracy, then
end-to-end inference latency. Test results compare frozen models and never choose a winner.
All splits share a saved manifest. Citation/text groups cannot cross partitions.

Probabilities are uncalibrated. Short passages differ from full training opinions.
Cosine similarity indicates textual resemblance rather than legal authority. The
historical US corpus does not establish performance on Indian law, new cases, or other
document types. Source metadata may contain a small number of opinion-label mismatches.

## Sources and attribution

- Krithika Iyer, *Classification of Legal Text*, Stanford CS229 (2020):
  https://cs229.stanford.edu/proj2020spr/report/Iyer.pdf
- Original corpus release: https://github.com/bdewilde/textacy-data/releases/tag/supreme_court_py3_v1.0
- Official mapping: https://github.com/chartbeat-labs/textacy/blob/main/src/textacy/datasets/supreme_court.py
- Metadata: Spaeth, Epstein et al., Supreme Court Database, 2016 Release 1.
  Dataset maintainer identifies the metadata license as CC BY-NC 3.0 US:
  https://creativecommons.org/licenses/by-nc/3.0/us/
- Opinion text derives from FindLaw as distributed in the textacy dataset. Preserve
  upstream attribution and review the source terms before redistributing the full corpus.

The Stanford paper's table and prose disagree about label-count results. We do not
reuse those scores as our results. We use counts for LDA and inferred Doc2Vec vectors
consistently for all classification partitions; these choices are documented adaptations.

## Submission and repository access

Submission deadline: **10 October 2026, 11:59 PM**. Reviews: **5-9 October 2026**.
The guidelines call the write-up one page in the heading and two pages in the body.
The delivered PDF follows the explicit two-page body instruction.

Keep this repository private. Add faculty and assigned TAs using **Settings > Collaborators**
after obtaining their exact GitHub usernames. The repository and release are private;
the review panel needs access before opening links. Fill `docs/contributions.csv` with
actual work and confirm the demo/review schedule with faculty.

