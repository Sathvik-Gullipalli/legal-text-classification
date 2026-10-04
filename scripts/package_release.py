"""Build private-release ZIPs without raw corpus data or environment files."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"
OUT.mkdir(exist_ok=True)
IGNORED = {".git", ".venv", ".build", ".pytest_cache", "__pycache__", ".ipynb_checkpoints", "data", "dist", "node_modules"}


def source_files():
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(p in IGNORED or p.endswith(".egg-info") for p in relative.parts):
            continue
        if "artifacts" in relative.parts:
            continue
        if path.is_file() and path.suffix != ".pyc" and path.name != ".DS_Store":
            yield path


sources = list(source_files())
models = [p for p in (ROOT / "artifacts").rglob("*") if p.is_file()]
packages = {
    "team9-project.zip": [(p, f"legal_text_classification/{p.relative_to(ROOT)}") for p in sources],
    "team9-model-bundle.zip": [(p, str(p.relative_to(ROOT))) for p in models],
    "team9-offline-kit.zip": [(p, f"legal_text_classification/{p.relative_to(ROOT)}") for p in sources + models],
}
for name, members in packages.items():
    with zipfile.ZipFile(OUT / name, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path, member in members:
            z.write(path, member)
        assert z.testzip() is None
    print(name, (OUT / name).stat().st_size, "bytes")
lines = []
for name in packages:
    digest = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
    lines.append(f"{digest}  {name}")
(OUT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

