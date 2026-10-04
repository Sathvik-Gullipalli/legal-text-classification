"""Download, clean, group, and split the original textacy Supreme Court corpus."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

DATA_URL = "https://github.com/bdewilde/textacy-data/releases/download/supreme_court_py3_v1.0/supreme-court-py3.json.gz"
MAPPING_URL = "https://raw.githubusercontent.com/chartbeat-labs/textacy/main/src/textacy/datasets/supreme_court.py"
SOURCE_PAGE = "https://github.com/bdewilde/textacy-data/releases/tag/supreme_court_py3_v1.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/3.0/us/"
# Official textacy mapping, checked against the maintainer's source. The source
# snapshot and source SHA256 are also saved with downloaded data.
ISSUE_AREAS = {
    1: "Criminal Procedure", 2: "Civil Rights", 3: "First Amendment",
    4: "Due Process", 5: "Privacy", 6: "Attorneys", 7: "Unions",
    8: "Economic Activity", 9: "Judicial Power", 10: "Federalism",
    11: "Interstate Relations", 12: "Federal Taxation",
    13: "Miscellaneous", 14: "Private Action",
}


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "supreme-court-py3.json.gz"
    if not target.exists():
        partial = target.with_suffix(".partial")
        try:
            req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Team9-ML-Project/1.0"})
            with urllib.request.urlopen(req, timeout=120) as response, partial.open("wb") as out:
                size = 0
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
                    size += len(chunk)
                    if size % (20 * 1024 * 1024) < 1024 * 1024:
                        print(f"Downloaded {size / 1024**2:.0f} MiB", flush=True)
            # Fully parse gzip before accepting the cached file.
            list(read_raw(partial))
            partial.replace(target)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Dataset download failed. Retry with internet access or copy the original gzip to {target}. Cause: {exc}") from exc
    snapshot = data_dir / "textacy_supreme_court_source.py.txt"
    if not snapshot.exists():
        try:
            with urllib.request.urlopen(MAPPING_URL, timeout=30) as response:
                snapshot.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError("Cannot verify the official category mapping. Retry with internet access.") from exc
    tree = ast.parse(snapshot.read_text())
    official = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "issue_area_codes" for t in targets):
                official = ast.literal_eval(node.value)
    if official is None:
        raise ValueError("Official source has changed: issue_area_codes was not found.")
    official = {int(k): v for k, v in official.items() if v is not None}
    if official != ISSUE_AREAS:
        raise ValueError("The official category mapping differs from the project mapping. Review before training.")
    metadata = {"url": DATA_URL, "source_page": SOURCE_PAGE,
                "sha256": sha256_file(target), "bytes": target.stat().st_size,
                "mapping_url": MAPPING_URL, "mapping_sha256": sha256_file(snapshot),
                "license": "CC BY-NC 3.0 US (metadata, per dataset maintainer)",
                "license_url": LICENSE_URL,
                "text_source": "FindLaw, as distributed by textacy-data",
                "label_mapping": official}
    dump_json(data_dir / "provenance.json", metadata)
    return target, metadata


def read_raw(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        # The original release contains one JSON object per line.
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict) or "text" not in obj:
                    raise ValueError("Expected a JSONL corpus with text and metadata.")
                yield obj


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def tokens(text):
    return re.findall(r"[a-z]{2,}", str(text).lower())


def clean_and_group(raw):
    rows = []
    stats = Counter(raw_records=len(raw))
    for i, r in enumerate(raw):
        text = r.get("text") or ""
        if not normalize(text):
            stats["empty_text_removed"] += 1
            continue
        try:
            label = int(r.get("issue_area"))
        except (TypeError, ValueError):
            label = -1
        if label not in ISSUE_AREAS:
            stats["missing_or_unknown_label_removed"] += 1
            continue
        rows.append({"id": f"scotus-{i:05d}", "text": text,
                     "label": label, "category": ISSUE_AREAS[label],
                     "case_name": r.get("case_name", "Unknown case"),
                     "decision_date": r.get("decision_date"),
                     "citation": str(r.get("us_cite_id") or "").strip(),
                     "text_hash": hashlib.sha256(normalize(text).encode()).hexdigest()})
    # Union-find prevents transitive duplicate/citation links from crossing splits.
    parent = list(range(len(rows)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    for key in ("text_hash", "citation"):
        seen = {}
        for i, row in enumerate(rows):
            val = row[key]
            if val:
                if val in seen:
                    union(i, seen[val])
                else:
                    seen[val] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    cleaned = []
    for rs in groups.values():
        if len({r["label"] for r in rs}) > 1:
            stats["conflicting_group_records_removed"] += len(rs)
            stats["conflicting_groups_removed"] += 1
            continue
        seen = set()
        group_id = min(r["id"] for r in rs)
        for r in rs:
            if r["text_hash"] in seen:
                stats["exact_duplicates_removed"] += 1
                continue
            seen.add(r["text_hash"])
            r["group"] = group_id
            cleaned.append(r)
    group_counts = defaultdict(set)
    for r in cleaned:
        group_counts[r["label"]].add(r["group"])
    excluded = {k: len(v) for k, v in group_counts.items() if len(v) < 5}
    stats["rare_category_records_removed"] = sum(r["label"] in excluded for r in cleaned)
    cleaned = [r for r in cleaned if r["label"] not in excluded]
    stats["retained_records"] = len(cleaned)
    stats["retained_groups"] = len({r["group"] for r in cleaned})
    return cleaned, {**dict(stats), "excluded_rare_categories": excluded,
                     "class_counts": dict(Counter(r["category"] for r in cleaned)),
                     "present_labels": sorted({r["label"] for r in cleaned})}


def make_splits(rows, seed=42):
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    x = np.zeros(len(rows))
    remaining, test = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y, groups))
    tr, va = next(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed).split(x[remaining], y[remaining], groups[remaining]))
    indices = {"train": remaining[tr], "validation": remaining[va], "test": test}
    splits = {name: [rows[int(i)] for i in idx] for name, idx in indices.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not ({r["group"] for r in splits[a]} & {r["group"] for r in splits[b]})
    train_labels = {r["label"] for r in splits["train"]}
    for name in ("validation", "test"):
        if {r["label"] for r in splits[name]} != train_labels:
            raise ValueError(f"{name} lacks a category or has an unseen category. Review group support before training.")
    return splits
