import json
import tempfile
import unittest
from pathlib import Path

from dp.loaders.annotations import read_batch_textannotations_from_path
from dp.loaders.base import TokenEdit, TextAnnotations


class TestTextAnnotations(unittest.TestCase):
    def test_token_edit_from_mapping(self) -> None:
        edit = TokenEdit.from_mapping({"kind": "replaced", "span": [1, 3], "text": "X"})
        self.assertEqual(edit.kind, "replaced")
        self.assertEqual(edit.span, (1, 3))
        self.assertEqual(edit.text, "X")
        self.assertEqual(edit.to_dict(), {"kind": "replaced", "span": [1, 3], "text": "X"})

    def test_read_jsonl_textannotations_parses_token_edits(self) -> None:
        record = {
            "idx": 0,
            "text": "Hello Bob",
            "metadata": {
                "token_edits": [
                    {"kind": "replaced", "span": [6, 9], "text": "[PERSON]"},
                    {"kind": "added", "span": [9, 9], "text": "!"},
                ]
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            anns = read_batch_textannotations_from_path(str(path))

        self.assertEqual(len(anns), 1)
        self.assertIsInstance(anns[0], TextAnnotations)
        self.assertEqual([e.kind for e in anns[0].token_edits], ["replaced", "added"])
        self.assertEqual([e.span for e in anns[0].token_edits], [(6, 9), (9, 9)])
        self.assertEqual([e.text for e in anns[0].token_edits], ["[PERSON]", "!"])


if __name__ == "__main__":
    unittest.main()
