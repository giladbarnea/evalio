def convert_string_to_numpy_array(series: "pandas.Series") -> "pandas.Series":
    from ast import literal_eval
    import numpy as np

    return series.apply(literal_eval).apply(np.array)


def unquote(string: str) -> str:
    return string.strip('"').strip("'")
