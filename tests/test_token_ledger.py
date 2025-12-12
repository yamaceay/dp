import unittest

from dp.utils.token_ledger import TokenLedger


def _spans_for_tokens(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = text.index(token, cursor)
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


class TestTokenLedger(unittest.TestCase):
    def test_len_and_entry(self) -> None:
        text = "Hi Alice."
        spans = _spans_for_tokens(text, ["Hi", "Alice"])
        ledger = TokenLedger(text, spans)
        self.assertEqual(len(ledger), 2)
        self.assertEqual(ledger.entry(0).original_text, "Hi")
        self.assertEqual(ledger.entry(1).original_text, "Alice")

    def test_replace_updates_metadata_and_iter(self) -> None:
        text = "Hello Bob!"
        spans = _spans_for_tokens(text, ["Hello", "Bob"])
        ledger = TokenLedger(text, spans)

        ledger.replace(1, "[PERSON]")

        md = ledger.edits_metadata()
        self.assertEqual(len(md), 1)
        self.assertEqual(md[0]["kind"], "replaced")
        self.assertEqual(md[0]["span"], spans[1])
        self.assertEqual(md[0]["text"], "[PERSON]")

        rendered = "".join(ledger.build_context())
        self.assertIn("[PERSON]", rendered)
        self.assertNotIn("Bob", rendered)

    def test_delete_removes_text_and_additions(self) -> None:
        text = "A B C"
        spans = _spans_for_tokens(text, ["A", "B", "C"])
        ledger = TokenLedger(text, spans)

        ledger.add_after(0, "X")
        ledger.delete(0)

        md = ledger.edits_metadata()
        self.assertEqual(len(md), 1)
        self.assertEqual(md[0]["kind"], "deleted")
        self.assertEqual(md[0]["span"], spans[0])
        self.assertEqual(md[0]["text"], "A")

        rendered = "".join(ledger.build_context())
        self.assertNotIn("A", rendered)
        self.assertNotIn("X", rendered)

    def test_add_after_records_added_edit(self) -> None:
        text = "A B"
        spans = _spans_for_tokens(text, ["A", "B"])
        ledger = TokenLedger(text, spans)

        ledger.add_after(0, "X")

        md = ledger.edits_metadata()
        self.assertEqual(len(md), 1)
        self.assertEqual(md[0]["kind"], "added")
        self.assertEqual(md[0]["span"], (spans[0][1], spans[0][1]))
        self.assertEqual(md[0]["text"], "X")

        context = ledger.build_context()
        self.assertEqual("".join(context), "AX B")

    def test_iter_tokens_preserves_gaps(self) -> None:
        text = "Hi, Bob."
        spans = _spans_for_tokens(text, ["Hi", "Bob"])
        ledger = TokenLedger(text, spans)

        pieces = list(ledger.iter_tokens())
        self.assertEqual("".join(pieces), text)

    def test_render_offsets_matches_apply_token_edits(self) -> None:
        text = "Hello Bob!"
        spans = _spans_for_tokens(text, ["Hello", "Bob"])
        ledger = TokenLedger(text, spans)
        ledger.replace(1, "[PERSON]")

        out = ledger.render_offsets(text)
        self.assertEqual(out, "Hello [PERSON]!")

    def test_replace_after_delete_is_noop(self) -> None:
        text = "X Y"
        spans = _spans_for_tokens(text, ["X", "Y"])
        ledger = TokenLedger(text, spans)

        ledger.delete(0)
        ledger.replace(0, "Z")

        md = ledger.edits_metadata()
        self.assertEqual(len(md), 1)
        self.assertEqual(md[0]["kind"], "deleted")

    def test_surviving_spans_excludes_deleted(self) -> None:
        text = "A B C"
        spans = _spans_for_tokens(text, ["A", "B", "C"])
        ledger = TokenLedger(text, spans)
        ledger.delete(1)
        self.assertEqual(list(ledger.surviving_spans()), [spans[0], spans[2]])

    def test_render_with_bert_tokenizer(self) -> None:
        try:
            from transformers import BertTokenizerFast
        except Exception as exc:
            self.skipTest(str(exc))

        try:
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        except Exception as exc:
            self.skipTest(str(exc))

        text = "Hello, Bob!"
        spans = _spans_for_tokens(text, ["Hello", ",", "Bob", "!"])
        ledger = TokenLedger(text, spans)

        detok = lambda parts: tokenizer.convert_tokens_to_string([p for p in parts if p.strip()])
        self.assertEqual(ledger.render(detok), tokenizer.convert_tokens_to_string(["Hello", ",", "Bob", "!"]))

        ledger.replace(2, tokenizer.mask_token)
        expected = tokenizer.convert_tokens_to_string(["Hello", ",", tokenizer.mask_token, "!"])
        self.assertEqual(ledger.render(detok), expected)


if __name__ == "__main__":
    unittest.main()
