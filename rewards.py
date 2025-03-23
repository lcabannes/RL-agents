import torch
import re
from torch.nn import functional as F

pattern = "<answer>(.*?)</answer>"
eps = 0.2
beta = 0.01

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

def compute_rewards(outputs, target):
    rewards = []

    targets = [target] * len(outputs)
    for output, target in zip(outputs, targets):
        cur_reward = 0
        cur_reward += format_reward(output, target)
        cur_reward += basic_reward(output, target)
        cur_reward += eval_reward(output, target)
        rewards.append(cur_reward)
    
    return torch.tensor(rewards).to(torch.float32)
   
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


def compute_log_probs(model, outputs, prompt_length):
    logits = model(outputs).logits

    # logits.shape = (prompt_length + answers_length + 1, batch_size * num_samples, vocab_size)

    # we only need the log probabilities for the new tokens
    # this introduces a shift: the logits for a position are the predictions for the next token
    logits = logits[prompt_length-1:-1, :, :]
    # logits.shape = (answers_length + 1, batch_size * num_samples, vocab_size)

    # convert raw logits into log probabilities along the vocabulary axis
    log_probs = F.log_softmax(logits, dim=-1)
    # log_probs.shape = (answers_length + 1, batch_size * num_samples, vocab_size)
    return log_probs


def calculate_grpo_advantages(rewards, num_samples):
    # reshape rewards to group by prompt
    # compute mean and standard deviation for each prompt group
    mean_rewards = rewards.view(-1, num_samples).mean(dim=1)
    std_rewards = rewards.view(-1, num_samples).std(dim=1)
    # print(f"mean rewards: {mean_rewards.mean()}")
    # expand the means and stds to match the original flat rewards tensor shape
    mean_rewards = mean_rewards.repeat_interleave(num_samples, dim=0)
    std_rewards = std_rewards.repeat_interleave(num_samples, dim=0)
    # normalize rewards to get advantages
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-5)
    return advantages

def compute_ppo_loss(advantages, log_probs, old_log_probs, ref_log_probs, responses):
    responses = responses.unsqueeze(-1)
    selected_log_probs = log_probs.gather(dim=-1, index=responses).squeeze(-1)
    old_selected_log_probs = old_log_probs.gather(dim=-1, index=responses).squeeze(-1)
    ref_selected_log_probs = ref_log_probs.gather(dim=-1, index=responses).squeeze(-1)

    ratios = torch.exp(selected_log_probs - old_selected_log_probs)
    advantages = advantages.unsqueeze(dim=0)

    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1-eps, 1+eps) * advantages


    clip_loss = - torch.min(surr1, surr2).mean()

    KL =  torch.exp(ref_selected_log_probs - selected_log_probs) - (ref_selected_log_probs - selected_log_probs)  - 1 
    KL = KL.mean()
    return clip_loss + beta * KL







