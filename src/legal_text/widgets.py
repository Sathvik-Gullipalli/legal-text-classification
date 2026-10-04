"""Colab/Jupyter interface for the saved-model demo."""
import html
import ipywidgets as widgets
from IPython.display import HTML, display
from .inference import Demo


def launch(artifact_dir="artifacts"):
    demo = Demo(artifact_dir)
    examples = widgets.Dropdown(options=[("Custom passage", -1)] + [(r["name"], i) for i, r in enumerate(demo.examples)],
                               description="Example:", layout=widgets.Layout(width="700px"))
    models = widgets.Dropdown(options=[("TF-IDF + Logistic Regression", "tfidf"), ("LDA + Logistic Regression", "lda"),
                                      ("Doc2Vec + Logistic Regression", "doc2vec")],
                             value=demo.manifest["winner"], description="Model:", layout=widgets.Layout(width="500px"))
    text = widgets.Textarea(placeholder="Paste an English legal opinion here, or choose a held-out example above.",
                           layout=widgets.Layout(width="100%", height="260px"))
    button = widgets.Button(description="Classify and find similar opinions", button_style="primary",
                            layout=widgets.Layout(width="300px"))
    output = widgets.Output()
    def choose(change):
        if change["new"] >= 0:
            text.value = demo.examples[change["new"]]["text"]
    examples.observe(choose, names="value")
    def run(_):
        output.clear_output(wait=True)
        with output:
            try:
                result = demo.predict(text.value, models.value)
                display(HTML(f"<h2>Predicted category: {html.escape(result['category'])}</h2>"))
                rows = "".join(f"<tr><td>{html.escape(r['category'])}</td><td>{r['probability']:.1%}</td></tr>" for r in result["top_three"])
                display(HTML(f"<table><tr><th>Category</th><th>Model probability</th></tr>{rows}</table>"))
                for warning in result["warnings"]:
                    print(warning)
                if result["influential_terms"]:
                    print("Terms supporting this TF-IDF prediction: " + ", ".join(t["term"] for t in result["influential_terms"]))
                display(HTML("<h3>Similar training opinions</h3>"))
                print("Cosine similarity measures textual resemblance, not legal authority or precedent validity.")
                for case in demo.retrieve_similar(text.value):
                    display(HTML(f"<h4>{html.escape(case['case_name'])}</h4><p>{html.escape(str(case['decision_date']))} / {html.escape(case['category'])} / cosine {case['similarity']:.3f}</p><p>{html.escape(case['excerpt'][:700])}...</p>"))
            except (ValueError, FileNotFoundError) as exc:
                print(str(exc))
    button.on_click(run)
    display(HTML("<h1>Classification of Legal Text</h1><p>Team 9 / US Supreme Court opinions / broad issue areas</p>"))
    display(widgets.VBox([examples, models, text, button, output]))
    return demo
