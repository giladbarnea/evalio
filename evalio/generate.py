import openai

from llm_utils import util


def generate_prompt_test_cases(prompt_template, *, cases_num=3) -> list[str]:
    system_message = (
        "You are generating test cases for a prompt that will be given to a Large-Language-Model (LLM). "
        "Note that the prompt is not given to you, but you will be given the template for that prompt, "
        "in order to help test its quality and performance.\n\n"
        "The prompt template consists of "
        "parameterized instructions for the model, denoted by "
        "curly braces. For example: {action}.\n\n"
        "You will fill the parameters with values that would "
        "makes sense for the full prompt.\n\n"
        "Note that you will be given previously generated test cases, "
        "but you will generate only one test case each turn.\n\n"
        "For each turn, make the example slightly more complex than the last, while ensuring diversity.\n\n"
        "Make sure your test cases are unique and diverse, yet high-quality and complex enough to test a well-performing model."
    )
    prompt_template_message = "PROMPT TEMPLATE:\n\n" + "#" * len("PROMPT TEMPLATE:") + "\n\n" + prompt_template
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt_template_message}]
    cases_num = min(cases_num, 5)
    test_cases = []
    for i in range(cases_num):
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages, temperature=0)
        response_text = util.unquote(response.choices[0].message["content"]).strip()
        test_cases.append(response_text)
        messages.insert(-1, {"role": "user", "content": response_text})

    return test_cases
