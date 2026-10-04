"""Save the measured notebook example used in presentation materials."""
from pathlib import Path
from legal_text.inference import Demo
from legal_text.data import dump_json

ROOT = Path(__file__).resolve().parents[1]
demo = Demo(ROOT / "artifacts")
example = next(e for e in demo.examples if e["true_category"] == "Criminal Procedure")
dump_json(ROOT / "results/demo_snapshot.json", {
    "case_name": example["case_name"], "true_category": example["true_category"],
    "prediction": demo.predict(example["text"], "tfidf"),
    "neighbors": demo.retrieve_similar(example["text"], 5)})
print("Saved measured demonstration output")
