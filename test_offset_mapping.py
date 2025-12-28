#!/usr/bin/env python3
"""Quick test for offset mapping functions."""

from dp.utils.token_edits import (
    map_result_offset_to_original,
    map_original_offset_to_result,
    apply_token_edits,
)


def test_replacement():
    # Original: 'Hello World' -> Result: 'Hi World'
    original = "Hello World"
    edits = [{"kind": "replaced", "span": [0, 5], "text": "Hi"}]
    result = apply_token_edits(original, edits)
    assert result == "Hi World", f"Got {result!r}"

    # 'World' in result is at (3, 8), in original at (6, 11)
    o_start, o_end = map_result_offset_to_original(3, 8, edits)
    assert (o_start, o_end) == (6, 11), f"Got ({o_start}, {o_end})"
    assert original[o_start:o_end] == "World"
    print("test_replacement PASSED")


def test_deletion():
    # Original: 'Hello World' -> Result: ' World' (deleted 'Hello')
    original = "Hello World"
    edits = [{"kind": "deleted", "span": [0, 5], "text": ""}]
    result = apply_token_edits(original, edits)
    assert result == " World", f"Got {result!r}"

    # 'World' in result is at (1, 6), in original at (6, 11)
    o_start, o_end = map_result_offset_to_original(1, 6, edits)
    assert (o_start, o_end) == (6, 11), f"Got ({o_start}, {o_end})"
    assert original[o_start:o_end] == "World"
    print("test_deletion PASSED")


def test_no_edits():
    o_start, o_end = map_result_offset_to_original(5, 10, [])
    assert (o_start, o_end) == (5, 10)
    print("test_no_edits PASSED")


def test_forward_mapping():
    original = "Hello World"
    edits = [{"kind": "replaced", "span": [0, 5], "text": "Hi"}]
    
    # 'World' in original is at (6, 11), should map to (3, 8) in result
    r = map_original_offset_to_result(6, 11, edits)
    assert r == (3, 8), f"Got {r}"
    print("test_forward_mapping PASSED")


def test_forward_mapping_modified_span():
    original = "Hello World"
    edits = [{"kind": "replaced", "span": [0, 5], "text": "Hi"}]
    
    # 'Hello' in original was modified, should return None
    r = map_original_offset_to_result(0, 5, edits)
    assert r is None, f"Expected None, got {r}"
    print("test_forward_mapping_modified_span PASSED")


if __name__ == "__main__":
    test_replacement()
    test_deletion()
    test_no_edits()
    test_forward_mapping()
    test_forward_mapping_modified_span()
    print("\nAll tests PASSED!")
