"""Execute clean kernels and save review backups without retraining a completed run."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ["JUPYTER_PATH"] = str(ROOT / ".venv/share/jupyter")
os.environ["IPYTHONDIR"] = str(ROOT / ".build/ipython")
os.environ["JUPYTER_CONFIG_DIR"] = str(ROOT / ".build/jupyter-config")
os.environ["JUPYTER_DATA_DIR"] = str(ROOT / ".build/jupyter-data")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".build/jupyter-runtime")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".build/matplotlib")
os.environ["PYTHONHASHSEED"] = "42"
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

for name in ("01_train_and_evaluate", "02_live_demo"):
    nb = nbformat.read(ROOT / f"notebooks/{name}.ipynb", as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}}, store_widget_state=True)
    client.execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors, errors
    nbformat.write(nb, ROOT / f"notebooks/{name}_executed.ipynb")
    print(f"Executed {name} successfully", flush=True)

