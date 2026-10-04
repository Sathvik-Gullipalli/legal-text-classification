# Methodology

## Problem and target

Input is the full English text of a US Supreme Court opinion. Output is one broad issue
area from the official textacy mapping. The study compares representations rather
than predicting case outcomes or generating legal advice. Fine-grained issue codes,
BERT, and public hosting are outside this implementation.

## Dataset integrity

The original compressed JSONL release is cached without changes. `provenance.json`
records the download URL, byte count, SHA256, official mapping source checksum, and
license attribution. The official mapping has 14 named categories and an unknown code.
Observed supported categories, rather than the reference's label count, define this run.

Whitespace-normalized, lowercase text hashes identify duplicates. Union-find joins
records connected through shared citations or identical normalized text. This includes
transitive links. Any group with conflicting category labels is excluded conservatively.
Within a consistent group, only one copy of each identical text remains. Nonidentical
opinions may remain in the same group. Categories with fewer than five distinct groups
are excluded and counted. Missing/unknown labels and empty opinions are excluded.

Two fixed StratifiedGroupKFold partitions reserve approximately 20% for test and 20%
for validation, leaving 60% for training. Ratios are approximate because entire groups
move together. Category coverage and zero overlap are asserted before training.
No temporal holdout is used. This design estimates historical corpus performance.

## Models

**TF-IDF + LR:** English stop words, unigrams and bigrams, max 30,000 features,
min_df=3, max_df=0.95, sublinear term frequency. Features are sparse.

**LDA + LR:** CountVectorizer with max 15,000 words, min_df=5, max_df=0.95 and
English stop words. Online LDA compares 20 and 40 topics, uses five passes,
batch_size=128 and max_doc_update_iter=30. Counts fit the multinomial topic model.
This is a documented change from the reference's stated TF-IDF representation.

**Doc2Vec + LR:** PV-DBOW, 100 dimensions, 30 epochs, min_count=5, five negative
samples, sample=1e-4, one worker, seed 42, max_vocab_size=100,000. A stable hash
replaces platform-dependent hashing. Training uses only training opinions. Classification
uses inferred vectors for training, validation, test, and demo inputs. Text-specific RNG
seeding makes repeated inference stable within a kernel. Gensim's inferred-vector
initialization also uses Python hashing, so the delivered training and executed
notebook kernels start with PYTHONHASHSEED=42. Exact reproduction requires the same
startup hash seed; Colab kernels may differ. This avoids using learned document vectors
for training and inferred vectors only for evaluation.

All LR models compare C values 0.1, 1, 10 using lbfgs, max_iter=1500, with no class
weighting. Validation macro-F1 chooses configurations. Accuracy and inference latency
break ties. The fitted models remain trained on the training partition for the final
comparison; they are not refitted on train plus validation before testing.

## Metrics and retrieval

Accuracy weights every document equally. Macro-F1 weights every supported category
equally. Weighted F1 weights category F1 by test support. Reports provide per-category
precision, recall, F1, support, raw confusion matrices and normalized images. Latency
includes feature transformation and classification. Recorded feature runtime includes
feature fitting plus training/validation transformation. It is not pure model fitting time.

The majority baseline always predicts the largest training category. The test set
evaluates each frozen model once, after saving validation selection. `test_predictions.csv`
allows independent metric reconciliation without running model evaluation again.

Retrieval normalizes the inferred training-opinion Doc2Vec vectors and query vector,
then ranks dot products (cosine similarity). It returns one result per citation.
No retrieval-quality labels exist, so similar-case search is a qualitative demonstration,
not a benchmark of valid precedents.

## Limitations

The corpus is historical and limited to one US court. Categories are imbalanced.
Full text is supplied to all models, but Gensim 4.3.3's Doc2Vec backend caps each
pass at 10,000 retained tokens after vocabulary filtering and downsampling. Very
long opinions can therefore lose later context in Doc2Vec. TF-IDF and LDA use
the complete input text without this backend cap.
Metadata-text matching in the upstream dataset may be imperfect. Very long legal
texts contain boilerplate and citations. Short user input has a different distribution.
LR probabilities are uncalibrated. Grouped random partitions do not establish
future-case generalization. Model comparisons use the same partitions, but limited
hyperparameter search does not establish the best possible implementation of each method.
