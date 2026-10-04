# Live demo script (about five minutes)

## Before the review

Install dependencies, extract the model bundle, and run the demo notebook once.
Keep the executed backup notebook, slide PDF, write-up, and results locally. Confirm
faculty/TA access to the private repository. Do not train during the live review.

## Walkthrough

1. **Problem (30 seconds):** "Our input is a Supreme Court opinion. We predict its
   broad issue area, such as Criminal Procedure or Civil Rights. We compare lexical
   TF-IDF features with LDA topics and Doc2Vec embeddings."
2. **Data and split (45 seconds):** Show the dataset summary and split manifest in the
   training notebook. Explain why citations and duplicate texts remain in a single
   partition. State the actual retained opinions and categories shown in the results.
3. **Results (45 seconds):** Show the model-comparison plot and validation selection.
   Explain macro-F1 and the difference between selecting on validation and reporting
   on test. Read the delivered values rather than estimating them.
4. **Classification (90 seconds):** In the live notebook, choose the held-out Criminal
   Procedure example, select TF-IDF, and press the prediction button. Show the predicted
   category, top-three probabilities, and influential terms. Switch to LDA or Doc2Vec
   on the same input and discuss any disagreement.
5. **Retrieval (45 seconds):** Show five similar training opinions. Explain cosine
   similarity and why a similar case is not automatically a valid precedent.
6. **Boundary and limitations (30 seconds):** Try an empty input and a short passage.
   Explain the input message and short-text warning. State the US historical corpus
   limitation and that model probabilities are uncalibrated.

## Fallback

If widgets fail, use the static example cell above the interface. If a notebook kernel
fails, show the executed backup notebook and slide PDF, then explain the saved artifacts.
The executed HTML backup also opens directly in a browser and contains static predictions.
Keep the terminal command ready: `python -m pytest -q`.

## Explanation ownership

Both members should be able to explain the data split, a representation method,
validation selection, and one failure case. Complete the contribution log with actual
work and practice a shared walkthrough rather than claiming unperformed work.
