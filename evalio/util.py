import functools


def convert_string_to_numpy_array(series: "pandas.Series") -> "pandas.Series":
    from ast import literal_eval

    import numpy as np

    return series.apply(literal_eval).apply(np.array)


def unquote(string: str) -> str:
    return string.strip('"').strip("'")


@functools.cache
def get_tiktoken_encoding(encoding_name="cl100k_base"):
    import tiktoken

    return tiktoken.get_encoding(encoding_name)


def tokens_count(string: str) -> int:
    tiktoken_encoding = get_tiktoken_encoding()
    return len(tiktoken_encoding.encode(string))
