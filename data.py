import datasets
import numpy as np

no_calculator_prompt = """
Provide the numerical answer strictly within <answer></answer> tags, answer in less than 10 words.
Example:
Input: 'What is the sum of 5 and 3?'
Output: '<answer>8</answer>'
Input: 'What is the sum of 61 and 7?'
Output: '<answer>68</answer>'
Input: 'What is the result of {} {} {}?'
"""
calculator_prompt = """
Provide the numerical answer strictly within <answer></answer> tags, answer in less than 10 words.
Example:
Input: 'What is the result of 5 plus 3?'
Output: '<answer>calculator(5 + 3)</answer>'
Input: 'What is the result of {} {} {}?'
"""



def problem_generator(size=1000, digits=3, use_calculator=False):
    prompt = calculator_prompt if use_calculator else no_calculator_prompt
    for _ in range(size):
        a = np.random.randint(0, 10**digits)
        b = np.random.randint(0, 10**digits)
        choice = np.random.randint(0, 3)
        operators = ["+", "-", "*"]
        operator_names = ["plus", "minus", "times"]
        operator = operators[choice]
        operator_name = operator_names[choice]

        prompt = prompt.format(a, operator_name, b)
        right_answer = eval(f"{a} {operator} {b}")
        yield {"prompt": prompt, "right_answer": right_answer, "question": prompt}


def get_dataset(size=1000, digits=3, use_calculator=False):

    
    generator = lambda: problem_generator(size, digits, use_calculator)

    dataset = datasets.Dataset.from_generator(generator)

    return dataset
