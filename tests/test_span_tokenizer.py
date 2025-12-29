#!/usr/bin/env python3
"""Test the _SpanTokenizer decode logic."""

from dp.utils.explainer.shap import _SpanTokenizer


def test_full_decode():
    text = "Hello, World! How are you?"
    spans = [(0, 5), (7, 12), (14, 17), (18, 21), (22, 25)]
    tok = _SpanTokenizer(text, spans)

    print("Original text:", repr(text))
    print("Tokens:", tok._tokens)
    print("Gaps:", tok._gaps)

    full = tok.decode([0, 1, 2, 3, 4])
    print("Full decode:", repr(full))
    assert full == text, f"Expected {text!r}, got {full!r}"
    print("test_full_decode PASSED")


def test_mask_middle():
    text = "Hello, World! How are you?"
    spans = [(0, 5), (7, 12), (14, 17), (18, 21), (22, 25)]
    tok = _SpanTokenizer(text, spans)

    # Mask 'World' (index 1)
    # gaps: ['', ', ', '! ', ' ', ' ', '?']
    # With World masked: '' + 'Hello' + ', ' + '' + '! ' + 'How' + ' ' + 'are' + ' ' + 'you' + '?'
    masked = tok.decode([0, 2, 3, 4])
    print("Masked World:", repr(masked))
    expected = "Hello, ! How are you?"
    assert masked == expected, f"Expected {expected!r}, got {masked!r}"
    print("test_mask_middle PASSED")


def test_mask_edges():
    text = "Hello, World! How are you?"
    spans = [(0, 5), (7, 12), (14, 17), (18, 21), (22, 25)]
    tok = _SpanTokenizer(text, spans)

    # Mask first and last
    masked = tok.decode([1, 2, 3])
    print("Masked first+last:", repr(masked))
    expected = ", World! How are ?"
    assert masked == expected, f"Expected {expected!r}, got {masked!r}"
    print("test_mask_edges PASSED")


def test_special_chars():
    text = "foo\tbar\nbaz"
    spans = [(0, 3), (4, 7), (8, 11)]
    tok = _SpanTokenizer(text, spans)

    print("Text with tabs/newlines:", repr(text))
    print("Tokens:", tok._tokens)
    print("Gaps:", tok._gaps)

    full = tok.decode([0, 1, 2])
    assert full == text, f"Expected {text!r}, got {full!r}"

    # Mask middle
    masked = tok.decode([0, 2])
    expected = "foo\t\nbaz"
    assert masked == expected, f"Expected {expected!r}, got {masked!r}"
    print("test_special_chars PASSED")


def test_empty():
    text = "abc"
    spans = []
    tok = _SpanTokenizer(text, spans)
    result = tok.decode([])
    assert result == "abc", f"Expected 'abc', got {result!r}"
    print("test_empty PASSED")


if __name__ == "__main__":
    test_full_decode()
    test_mask_middle()
    test_mask_edges()
    test_special_chars()
    test_empty()
    print("\nAll _SpanTokenizer tests PASSED!")
