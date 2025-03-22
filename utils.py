import torch
import re
pattern = "<answer>(.*?)</answer>"

def format_reward(output, target):
    extracted = re.findall(pattern, output)
    if len(extracted) == 0:
        return -1

def basic_reward(output, target):
    extracted = re.findall(pattern, output)
    if len(extracted) == 0:
        return -1

    extracted = extracted[0]

    return extracted == target

def compute_reward(output, target):
    cur_reward = 0
    cur_reward += format_reward(output, target)
    cur_reward += basic_reward(output, target)
    cur_reward += eval_reward(output, target)
    
    return cur_reward
   
def eval_reward(output, target):
    extracted = re.findall(pattern, output)
    if len(extracted) == 0:
        return 0

    extracted = extracted[0]
    target = target

    if extracted == target:
        return 1
    else:
        return 0






