# Viva preparation

**What is the project actually predicting?** A broad issue area from the opinion's
text. It does not predict the judgment, guilt, ideology, or applicable Indian law.

**Why classify legal text?** Opinion catalogs can support search and organization.
Legal wording is long, specialized, and often contains repeated citations, making
ordinary short-text assumptions less reliable.

**How does TF-IDF work?** Term frequency measures occurrence in an opinion. Inverse
document frequency reduces the importance of words appearing across many training
opinions. Unigrams/bigrams preserve useful lexical phrases in a sparse vector.

**What does LDA do?** It treats each document as a mixture of topics and each topic
as a distribution over words. It learns topics without using category labels. LR
then learns how topic proportions relate to the supplied issue-area labels.

**Why counts for LDA?** LDA's generative model describes word counts. We deliberately
use CountVectorizer rather than the reference paper's stated TF-IDF representation
and disclose the difference.

**What does Doc2Vec learn?** A fixed-length document vector. PV-DBOW predicts words
from a document representation. New documents need inference because they were not
present during training.

**Why infer training vectors too?** Learned document vectors and newly inferred
vectors may have different distributions. We use the same inference procedure for
training and evaluation so the classifier receives consistent feature types.

**Is Logistic Regression regression here?** Despite its name it is a classifier.
Multiclass LR transforms class scores into probabilities using softmax. Training
minimizes log loss with regularization. C is inverse regularization strength.

**Why macro-F1 instead of accuracy alone?** Majority categories can dominate accuracy.
Macro-F1 averages category F1 equally, exposing weaknesses in small categories.
F1 is the harmonic mean of precision and recall. Weighted F1 uses category support.

**What is precision versus recall?** Precision asks how many predictions for a category
were correct. Recall asks how many true cases of that category the model found.

**What is leakage?** Evaluation information influences training or selection. We fit
all vocabularies, topics, and embeddings on training documents only, keep case groups
together, choose parameters on validation, and use test only after freezing choices.

**Why group by citation and text?** Duplicate or linked opinions could otherwise be
memorized in training and scored in test. Union-find catches transitive links too.
Conflicting-label groups are excluded, which is conservative and disclosed.

**What happens to rare categories?** Categories need at least five independent case
groups for the grouped split procedure. Excluded categories and records are logged.
The delivered dataset summary reports the actual supported categories.

**Can the best test score choose the model?** No. That would tune against the test set.
We select by validation macro-F1, then accuracy, then end-to-end latency. Test scores
describe the frozen comparison.

**Why no BERT?** The approved scope compares three CPU-friendly representations.
The reference encountered transformer memory problems. A transformer experiment
would add complexity without being required for this adapted mini-project.

**How does retrieval work?** Normalize a query Doc2Vec vector and training vectors.
Their dot products are cosine similarities. Return five opinions with distinct
citations, sorted from highest similarity. Similarity is not a probability.

**Do similar cases prove a legal argument?** No. Textual similarity does not establish
jurisdiction, binding authority, current legal validity, or relevance to specific facts.

**Are 90% model probabilities calibrated?** We did not calibrate them. They express
the classifier's relative preference, not a verified 90% chance of legal correctness.

**What are the limitations?** Historical US Supreme Court data, imbalance, noisy
source alignment, boilerplate, limited tuning, and mismatch between short input and
full opinions. Random grouped splits do not estimate future-case performance.

**Can you reproduce the result exactly?** The fixed dataset checksum, splits,
dependency versions, one-worker Doc2Vec, stable hashing, and seeded inference make
the run reproducible. Floating-point and runtime differences across hardware remain.

**What differs from the reference?** We add a TF-IDF baseline, use counts for LDA,
group duplicate/citation records before splitting, infer Doc2Vec features consistently,
focus on supported broad issue areas, and report an independent evaluation. The
reference table and prose conflict, so we do not copy its reported scores.

