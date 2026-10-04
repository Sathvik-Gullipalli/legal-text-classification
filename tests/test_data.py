from legal_text.data import clean_and_group, make_splits


def record(text, label=1, cite=""):
    return {"text": text, "issue_area": label, "us_cite_id": cite, "case_name": "Example", "decision_date": "2000-01-01"}


def test_transitive_groups_conflicts_duplicates_and_missing():
    raw = [record(f"document {i} criminal evidence", 1, f"A{i}") for i in range(15)]
    raw += [record(f"document {i} taxation income", 12, f"B{i}") for i in range(15)]
    raw += [record("duplicate text", 1, "link1"), record(" DUPLICATE   TEXT ", 1, "link2"), record("linked other text", 1, "link2")]
    raw += [record("conflict one", 1, "bad"), record("conflict two", 12, "bad")]
    raw += [record("", 1), record("unknown", -1), record("rare", 5)]
    rows, stats = clean_and_group(raw)
    linked = [r for r in rows if "duplicate" in r["text"].lower() or "linked" in r["text"]]
    assert len(linked) == 2
    assert len({r["group"] for r in linked}) == 1
    assert stats["exact_duplicates_removed"] == 1
    assert stats["conflicting_group_records_removed"] == 2
    assert stats["empty_text_removed"] == 1
    assert stats["missing_or_unknown_label_removed"] == 1
    assert stats["rare_category_records_removed"] == 1


def test_splits_are_repeatable_balanced_and_group_disjoint():
    raw = [record(f"label {label} document {i} text", label, f"{label}-{i}") for label in (1, 12) for i in range(40)]
    raw += [record("a second opinion from same case", 1, "1-0")]
    rows, _ = clean_and_group(raw)
    first = make_splits(rows)
    second = make_splits(rows)
    assert {k: [r["id"] for r in rs] for k, rs in first.items()} == {k: [r["id"] for r in rs] for k, rs in second.items()}
    assert sum(map(len, first.values())) == len(rows)
    for a, b in (("train", "test"), ("train", "validation"), ("validation", "test")):
        assert not {r["group"] for r in first[a]} & {r["group"] for r in first[b]}
    assert all({r["label"] for r in rs} == {1, 12} for rs in first.values())

