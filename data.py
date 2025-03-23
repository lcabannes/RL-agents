import datasets
import numpy as np

def addition_problem_generator(size=1000, digits=3):
    add_prompt = """What is the sum of {} and {}? Provide the numerical answer strictly within <answer></answer> tags.
                    Example:
                    Input: What is the sum of 5 and 3?
                    Output: <answer>8</answer>"""
    for _ in range(size):

        a = np.random.randint(0, 10**digits)
        b = np.random.randint(0, 10**digits)
        prompt = add_prompt.format(a, b)
        right_answer = str(a + b)
        yield {"prompt": prompt, "right_answer": right_answer}


def get_dataset(size=1000, digits=3):

    add_generator = lambda: addition_problem_generator(size, digits)
    dataset = datasets.Dataset.from_generator(add_generator)

    return dataset
