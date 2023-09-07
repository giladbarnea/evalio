from llm_utils import english, semantics
import pytest


def test_is_correct_grammar_returns_true_for_correct_grammar():
    is_correct, correction = english.is_correct_grammar("I am a cat.")
    assert is_correct
    assert correction is None


def test_is_correct_grammar_returns_false_for_incorrect_grammar():
    is_correct, correction = english.is_correct_grammar("I are a cat")
    assert not is_correct
    expected = "I am a cat"
    max_allowed_semantic_distance = semantics.get_semantic_distance(expected, "I am a cat.")
    expected_embed = semantics.get_embedding(expected)
    correction_embed = semantics.get_embedding(correction)
    actual_semantic_distance = semantics.get_semantic_distance(expected_embed, correction_embed)
    practically_equal_semantic_distance = pytest.approx(max_allowed_semantic_distance) == pytest.approx(
        actual_semantic_distance, abs=1e-4
    )
    assert actual_semantic_distance < max_allowed_semantic_distance or practically_equal_semantic_distance
