import time
import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
from rewards import compute_log_probs, compute_rewards, compute_ppo_loss, calculate_grpo_advantages
from data import get_dataset
from peft import PeftModelForCausalLM

def evaluate():
    model.eval()
    correct = 0.
    with torch.no_grad():
        for batch, i in enumerate(range(0, len(test_set) - 1, batch_size)):
            prompts, target_answers, prompt_length, answers_length, _, _ = get_batch("test", i, batch_size)
            prompts = prompts.to(device) 
            target_answers = target_answers.to(device) 
            output = generate(model, prompts, answers_length + 1)
            answers_tokens = output[prompt_length:, :]
            equality_test = answers_tokens == target_answers
            correct += torch.all(equality_test, axis=0).float().sum()
        accuracy = correct / len(test_set)
    return accuracy.item()

pad_token="[PAD]"
eos_token="[EOS]"
def pad(token_list, type_list="prompts"):
    max_length = max(len(x) for x in token_list)
    out = []
    
    for x in token_list:
        padding = [pad_token_id] * (max_length - len(x)) 
        if type_list == "prompts":
            padded = padding + x  
        elif type_list == "answers":
            padded = x + [eos_token_id] + padding 
        out.append(padded)
    return out, max_length

def get_batch(split, i, batch_size):
    data = train_set if split == 'train' else test_set
    end_index = min(i + batch_size, len(data))
    batches = data.select(range(i, end_index))
    prompts = batches["prompt"]
    encoded_prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    padded_prompts, prompt_length = pad(encoded_prompts, "prompts")
    answers = batches["right_answer"]
    encoded_answers = [tokenizer.encode(answer, add_special_tokens=False) for answer in answers]
    padded_answers, answers_length = pad(encoded_answers, "answers")
    X = torch.tensor(padded_prompts).T
    Y = torch.tensor(padded_answers).T

    return X, Y, prompt_length, answers_length, prompts, answers

def generate(model, prompts, new_tokens, mode="sampling", num_samples=1, temperature=1.0):
    input_tensor = torch.repeat_interleave(prompts, repeats = num_samples, dim = 1).to(device)
    # (prompt_length, batch_size * num_samples)
    for _ in range(new_tokens):
        output = model(input_tensor) # (prompt_length, batch_size * num_samples, ntokens)
        logits = output.logits
        logits = logits[-1, :, :]
        # logits = output[-1,:,:] # (batch_size * num_samples, ntokens)
        if mode == "greedy":
            tokens = torch.argmax(logits, -1).view((1,-1)) # (1, batch_size * num_samples)
        else: # mode == "sampling"
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs, num_samples = 1).view((1,-1)) # (1, batch_size * num_samples)
        input_tensor = torch.cat((input_tensor, tokens), 0)
    return input_tensor


train_set = get_dataset(size=100)
test_set = get_dataset(size=100)


temperature = 1.0
num_samples = 16 if torch.cuda.is_available() else 4
mu = 2
learning_rate = 1e-4
epochs = 10
batch_size = 2
log_interval = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# smollm 
model_id = "HuggingFaceTB/SmolLM-360m-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).to(torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")


pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)  
eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)  


def collate_fn(batch):
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False)
        for x in batch
        ]
    targets = [x["right_answer"] for x in batch]
    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    targets = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    return inputs, targets


train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_test_accuracy = None
test_accuracy = evaluate()
print('-' * 89)
print('| initialisation | test accuracy {:5.2f}'.format(test_accuracy))
print('-' * 89)

# switch eval for train model (enables dropout)
model.train()
ref_model = deepcopy(model) # reference model for KL divergence penalty
old_model = deepcopy(model) # old model for PPO ratio


for epoch in range(1, epochs+1):
    ref_model.load_state_dict(model.state_dict()) # update ref model every epoch like in the GRPO paper
    epoch_start_time = time.time()
    for i, batch in enumerate(train_loader): 

        start_time = time.time()
        # get a batch of prompts and answers
        prompts, answer = batch

        # decode prompts just to check
        decoded_prompts = [tokenizer.decode(prompt) for prompt in prompts["input_ids"]]
        # generate samples for each prompt
        with torch.no_grad():  
            outputs = model.generate(
                prompts["input_ids"].to(device), 
                max_new_tokens=50,
                num_return_sequences=num_samples,
                temperature=temperature,
                do_sample=True,
                
                )

        # outputs.shape = (prompt_length + answers_length + 1, batch_size * num_samples)
        prompt_length = prompts["input_ids"].shape[1]
        text_outputs = tokenizer.batch_decode(outputs[:, prompt_length:])
        questions = tokenizer.batch_decode(prompts["input_ids"])
        right_answers = tokenizer.batch_decode(answer["input_ids"])

        # print(f"text_outputs 0 {text_outputs[0]}")
        # print(f"prompts 0 {prompts[0]}")
        for i in range(num_samples):
            print(f"sample {i}: {text_outputs[i]}")
            break
        old_model.load_state_dict(model.state_dict()) # update the old model before the update steps like in the paper

        # compute old log probabilities for ratio and ref for KL divergence penalty
        with torch.no_grad():
            ref_log_probs = compute_log_probs(ref_model, outputs, prompt_length).detach()
            old_log_probs = compute_log_probs(old_model, outputs, prompt_length).detach()



        rewards = compute_rewards(text_outputs, right_answers).to(device)
        print(f"rewards 0: {rewards}")

        # compute advantages
        advantages = calculate_grpo_advantages(rewards, num_samples=num_samples)

        # compute loss
        responses = outputs[:, prompt_length:]

        for inner_iter in range(mu): # do mu iteration on the generated trajectories like in the paper

            new_log_probs = compute_log_probs(model, outputs, prompt_length)
            loss = compute_ppo_loss(advantages, new_log_probs, old_log_probs, ref_log_probs, responses)

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    

        if i % log_interval == 0:
            elapsed = time.time() - start_time
            print(f"elapsed: {elapsed} mean rewards: {rewards.mean()}")

    test_accuracy = evaluate()
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test accuracy {:5.2f}'.format(epoch, (time.time() - epoch_start_time), test_accuracy))
    print('-' * 89)
    # Save the model if the test accuracy is the best we've seen so far.
    if not best_test_accuracy or test_accuracy < best_test_accuracy:
        with open("arithmetic_vanilla_GRPO.pt", 'wb') as f:
            torch.save(model, f)
        best_test_accuracy = test_accuracy
    if test_accuracy > 0.99:
        print(f"achieved near perfect accuracy after {epoch} epochs")
        break