import betterprompt
import pandas as pd


def compare_perplexity(text_1, text_2, *other_texts) -> pd.DataFrame:
    perplexities = {
        0: betterprompt.calculate_perplexity(text_1),
        1: betterprompt.calculate_perplexity(text_2),
        **{i + 2: betterprompt.calculate_perplexity(text) for i, text in enumerate(other_texts)},
    }
    sorted_by_perplexity = sorted(perplexities.keys(), key=lambda x: perplexities[x])
    ret = []
    for i in sorted_by_perplexity:
        text = (text_1, text_2, *other_texts)[i]
        perplexity = perplexities[i]
        relative_perplexity = perplexity // perplexities[0]
        ret.append({'text': text, 'perplexity': perplexity, 'relative_perplexity': relative_perplexity})
    return pd.DataFrame(ret)


compare_perplexity(
    "What is this article about?",
    "What is this piece of news regarding?",
    "What is the best way to describe this article?",
    "In what way would someone that is interested in this article describe its contents in a way"
    " that is most relevant to them?",
)
