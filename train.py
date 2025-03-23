import time
import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
from rewards import compute_log_probs, compute_rewards, compute_ppo_loss, calculate_grpo_advantages
from data import get_dataset

def evaluate():
    pass

def get_batch(split, i, batch_size):
    pass

def generate(model, prompts, new_tokens, mode="sampling", num_samples=1, temperature=1.0):
    pass


train_set = get_dataset()
test_set = get_dataset()

print(f"train set 0: {train_set[0]}")
print(f"test set 0: {test_set[0]}")

temperature = 1.0
num_samples = 16
mu = 2
learning_rate = 1e-4
epochs = 10
batch_size = 16
log_interval = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# smollm 
model_id = "HuggingFaceTB/SmolLM-135m-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)


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
    start_time = time.time()
    for batch, i in enumerate(range(0, len(data_train) - 1, batch_size)):

        # get a batch of prompts and answers
        prompts, _, prompt_length, answers_length, questions, answers = get_batch("train", i, batch_size)
        prompts = prompts.to(device) # (prompt_length, batch_size)

        # generate samples for each prompt
        outputs = generate(model,
                            prompts,
                            new_tokens=answers_length + 1,
                            mode="sampling",
                            num_samples=num_samples,
                            temperature=temperature)
        # outputs.shape = (prompt_length + answers_length + 1, batch_size * num_samples)
        text_outputs = [tokenizer.decode(outputs[prompt_length:, i].tolist())
                        for i in range(outputs.size(1))]
        
        old_model.load_state_dict(model.state_dict()) # update the old model before the update steps like in the paper

        # compute old log probabilities for ratio and ref for KL divergence penalty
        with torch.no_grad():
            ref_log_probs = compute_log_probs(ref_model, outputs, prompt_length).detach()
            old_log_probs = compute_log_probs(old_model, outputs, prompt_length).detach()
        
        # compute rewards
        rewards = compute_rewards(text_outputs, answers)

        # compute advantages
        advantages = calculate_grpo_advantages(rewards)

        # compute loss
        responses = outputs[prompt_length:, :]

        for inner_iter in range(mu): # do mu iteration on the generated trajectories like in the paper

            new_log_probs = compute_log_probs(model, outputs, prompt_length)
            loss = compute_ppo_loss(advantages, new_log_probs, old_log_probs, ref_log_probs, responses)

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    

        if i % log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | ms/batch {:5.2f}'.format(batch, len(data_train) // batch_size, elapsed))

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