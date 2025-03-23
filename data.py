import datasets
import numpy as np

def addition_problem_generator(size=1000, digits=3):
    add_prompt = """What is the sum of {} and {}? Provide the numerical answer strictly within <answer></answer> tags, answer in less than 10 words.
                    Example:
                    Input: What is the sum of 5 and 3?
                    Output: <answer>8</answer>
                    Input: What is the sum of 61 and 7?
                    Output: <answer>68</answer>
                    """
    for _ in range(size):

        a = np.random.randint(0, 10**digits)
        b = np.random.randint(0, 10**digits)
        prompt = add_prompt.format(a, b)
        right_answer = str(a + b)
        yield {"prompt": prompt, "right_answer": right_answer}

def calculator_problem_generator(size=1000, digits=3):
    calculator_prompt = """What is the result of the following expression {} {} {}?
    Provide the numerical answer strictly within <answer></answer> tags, answer in less than 10 words.
    Do not use python just put the expression in the answer tags.
    Example:
    Input: What is the result of 5 plus 3?
    Output: <answer>5 + 3</answer>
    Input: What is the result of 283 times 9?
    Output: <answer>283 * 9</answer>
    Input: What is the result of 12 minus 68?
    Output: <answer>12 - 68</answer>
    """
    for _ in range(size):
        a = np.random.randint(0, 10**digits)
        b = np.random.randint(0, 10**digits)
        choice = np.random.randint(0, 3)
        operators = ["+", "-", "*"]
        operator_names = ["plus", "minus", "times"]
        operator = operators[choice]
        operator_name = operator_names[choice]

        prompt = calculator_prompt.format(a, operator_name, b)
        right_answer = str(eval(f"{a} {operator} {b}"))
        yield {"prompt": prompt, "right_answer": right_answer}


def get_dataset(size=1000, digits=3, calculator=False):

    
    generator = lambda: addition_problem_generator(size, digits)
    if calculator:
        generator = lambda: calculator_problem_generator(size)

    dataset = datasets.Dataset.from_generator(generator)

    return dataset
