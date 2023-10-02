from typing import Literal

import openai

from evalio import util

ANSWER_COMPARISON_PROMPT_TEMPLATE = '''Are these two answers {criterion}?

Answer 1:
"""
{answer_1}
"""

Answer 2:
"""
{answer_2}
"""

Are the two answers {criterion}?
'''

ANSWER_CHOICE_PROMPT_TEMPLATE = '''Which answer is {criterion}?

Answer 1:
"""
{answer_1}
"""

Answer 2:
"""
{answer_2}
"""

The answer that is {criterion} is the:
'''


def are_texts(text_1: str, text_2: str, *, criterion: str) -> bool:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI Assistant. "
                    "You have a deep understanding of any text you read, and you pay attention to detail. "
                    "You respond only with 'yes' or 'no', without quotes."
                ),
            },
            {
                "role": "user",
                "content": ANSWER_COMPARISON_PROMPT_TEMPLATE.format(
                    criterion=criterion, answer_1=text_1, answer_2=text_2
                ),
            },
        ],
        temperature=0,
    )
    response_text = util.unquote(response.choices[0].message["content"]).strip().lower()
    if response_text in ("yes", "yes."):
        return True
    if response_text in ("no", "no."):
        return False
    raise ValueError(f"Unexpected response: {response_text!r}")


def are_texts_factually_consistent(text_1: str, text_2: str) -> bool:
    return are_texts(text_1, text_2, criterion="factually consistent")


def which_text(text_1: str, text_2: str, *, criterion: str) -> Literal[1] | Literal[2]:
    """The models returns 1 if it chose the first text, 2 if it chose the second text."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI Assistant. "
                    "You have a deep understanding of any text you read, and you pay attention to detail. "
                    "You respond only with 'first' or 'second', without quotes."
                ),
            },
            {
                "role": "user",
                "content": ANSWER_CHOICE_PROMPT_TEMPLATE.format(criterion=criterion, answer_1=text_1, answer_2=text_2),
            },
        ],
        temperature=0,
    )
    response_text = util.unquote(response.choices[0].message["content"]).strip().lower()
    if response_text == "first":
        return 1
    if response_text == "second":
        return 2
    raise ValueError(f"Unexpected response: {response_text!r}")


def which_text_is_easier_to_read(text_1: str, text_2: str) -> Literal[1] | Literal[2]:
    return which_text(text_1, text_2, criterion="easier to read")
