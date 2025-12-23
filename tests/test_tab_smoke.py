from __future__ import annotations

from typing import List, Tuple

from dp.loaders.tab import TabDatasetAdapter
from dp.utils.selector.all_selector import AllUnit
from dp.utils.token_ledger import TokenLedger


def _whitespace_offsets(text: str) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    in_token = False
    start = 0
    for i, ch in enumerate(text):
        if ch.isspace():
            if in_token:
                offsets.append((start, i))
                in_token = False
            continue
        if not in_token:
            in_token = True
            start = i
    if in_token:
        offsets.append((start, len(text)))
    return offsets


def test_tab_first_sample_adapter_and_selector_smoke(tmp_path) -> None:
    tab_path = tmp_path / "tab.json"
    tab_path.write_text(
        '[\n'
        '  {"doc_id": "0", "text": "Alice Smith lives in Berlin.", "meta": {"countries": ["DE"], "year": 2020}},\n'
        '  {"doc_id": "1", "text": "Bob Jones moved to Paris.", "meta": {"countries": ["FR"], "year": 2021}}\n'
        ']\n',
        encoding="utf-8",
    )

    adapter = TabDatasetAdapter(data_in=str(tab_path), max_records=1)
    records = list(adapter.iter_records())
    assert len(records) == 1

    record = records[0]
    assert record.uid == "0"
    assert "Alice" in record.text

    offsets = _whitespace_offsets(record.text)
    assert offsets

    ledger = TokenLedger(record.text, offsets)

    def apply_fn(idx: int, ledger_obj: TokenLedger) -> None:
        ledger_obj.replace(idx, "[X]")

    unit = AllUnit()
    steps = list(unit.anonymize(record.text, offsets, apply_fn, ledger=ledger))
    assert len(steps) == 1
    assert "[X]" in steps[0].text
