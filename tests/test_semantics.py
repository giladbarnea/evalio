from llm_utils import semantics


def is_a_memoized_function(function) -> bool:
    def has_cache_attributes(obj):
        """See docs for functools.cache"""
        return hasattr(obj, 'cache') or hasattr(obj, 'cache_info')

    if has_cache_attributes(function):
        return True

    if function.__closure__:
        for closed in function.__closure__:
            if has_cache_attributes(closed.cell_contents):
                return True

    return False


def test_get_embedding_is_deterministic():
    assert not is_a_memoized_function(semantics.get_embedding)
    hello_embedding_1 = semantics.get_embedding("hello")
    hello_embedding_2 = semantics.get_embedding("hello")
    assert (hello_embedding_1 == hello_embedding_2).all()


def test_get_semantic_distance_returns_0_for_same_word():
    hello_semantic_distance = semantics.get_semantic_distance("hello", "hello")
    assert round(hello_semantic_distance, 5) == 0


def test_get_semantic_distance_is_less_than_006_for_begin_and_start():
    hello_semantic_distance = semantics.get_semantic_distance("begin", "start")
    assert hello_semantic_distance < 0.06


def test_get_semantic_distance_knows_greetings_and_electricity_are_far_away():
    semantically_close_pair = ("hello", "greetings")
    semantically_far_pair = ("hello", "electricity")
    small_semantic_distance = semantics.get_semantic_distance(*semantically_close_pair)
    large_semantic_distance = semantics.get_semantic_distance(*semantically_far_pair)
    assert small_semantic_distance < large_semantic_distance
