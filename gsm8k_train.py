import time
import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
from rewards import compute_log_probs, compute_calculator_rewards, compute_ppo_loss, calculate_grpo_advantages
from data import get_dataset
from peft import PeftModelForCausalLM
import matplotlib.pyplot as plt
import re
import datasets
from tools.calculator import calculator
import torch.nn.functional as F
from tqdm import tqdm


def evaluate(use_calculator=False):
    model.eval()
    correct = 0.0
    total = 0
    
    # Define patterns to extract answers
    if not use_calculator:
        pattern = "<answer>(.*?)</answer>"
    else:
       pattern = "<answer>calculator\\((.*?)\\)</answer>"
    
    # Set error tolerance
    tolerance = 1e-6  # Adjust as needed
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            prompts, right_answers = batch
            
            # Generate outputs
            outputs = model.generate(
                prompts["input_ids"].to(device),
                max_new_tokens=200,
                do_sample=False  # Use greedy decoding for evaluation
            )
            
            # Calculate prompt length to extract only the generated part
            prompt_length = prompts["input_ids"].shape[1]
            
            # Extract text outputs
            text_outputs = tokenizer.batch_decode(outputs[:, prompt_length:])
            right_answers = [answer for answer in right_answers for _ in range(num_samples)]

            
            # Check each output
            for output, target in zip(text_outputs, right_answers):
                # Try regular pattern first
                extracted = re.findall(pattern, output)
                
                # If not found, try use_calculator pattern
                try:
                    # Use the use_calculator function to evaluate the expression
                    if use_calculator:
                        result = float(calculator(extracted[0]))
                    else:
                        result = float(eval(extracted[0]))
                    if abs(result - target) <= tolerance:
                        correct += 1
                    else:
                        print(f"wrong results output: {output} target: {target} result: {result}")
                except:
                    print(f"exception output: {output} target: {target}")
                    pass
                       
                total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy





temperature = 0.8
num_samples = 12 if torch.cuda.is_available() else 4
mu = 2
learning_rate = 1e-4
epochs = 10
batch_size = 2
log_interval = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# smollm 
model_id = "HuggingFaceTB/SmolLM2-360m-Instruct"
# model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id).to(device).to(torch.bfloat16)
# get peft model 
from peft import LoraConfig, get_peft_model
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False, 
    r=128,  # LoRA rank
    lora_alpha=32,
    # Target the attention layers - may need adjustment based on model architecture
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Get PEFT model
model = get_peft_model(model, peft_config).to(torch.bfloat16)
model.print_trainable_parameters()  # Shows % of trainable parameters

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")



def collate_fn(batch):
    system_prompt = """Answer the question using a calculator in the following way:
    Input 'Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?'
    Output: 'In the beginning, Betty has only 100 / 2 .
    Betty's parents gave her 15 .
    Betty's grandparents gave her 15 * 2 .
    This means, Betty has (100 / 2) + (15 * 2) + 15.
    So, Betty needs 100 - ((100 / 2) + (15 * 2) + 15) to buy the wallet.'
    <answer>calculator( 100 - ((100 / 2) + (15 * 2) + 15) )</answer>'  
    """
    inputs = [
        tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            tokenize=False)
        for x in batch
        ]
    targets = [float(x["answer"].split("####")[-1].replace(",", "")) for x in batch]
    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    # targets = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    return inputs, targets


train_set = datasets.load_dataset("gsm8k", 'main', split="train[:100%]")
test_set = datasets.load_dataset("gsm8k", 'main', split="test[:100%]")

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=collate_fn)



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_test_accuracy = None
use_calculator = True
test_accuracy = evaluate(use_calculator=use_calculator)
print('-' * 89)
print('| initialisation | test accuracy {:5.2f}'.format(test_accuracy))
print('-' * 89)

# switch eval for train model (enables dropout)
model.train()
ref_model = deepcopy(model) # reference model for KL divergence penalty
old_model = deepcopy(model) # old model for PPO ratio


reward_hist = [] 
accuracies = []

for epoch in range(1, epochs+1):
    ref_model.load_state_dict(model.state_dict()) # update ref model every epoch like in the GRPO paper
    epoch_start_time = time.time()
    for i, batch in enumerate(tqdm(train_loader)): 
        old_model.load_state_dict(model.state_dict()) # update the old model before the update steps like in the paper

        start_time = time.time()
        # get a batch of prompts and answers
        prompts, right_answers = batch

        # decode prompts just to check
        decoded_prompts = [tokenizer.decode(prompt) for prompt in prompts["input_ids"]]
        # generate samples for each prompt
        with torch.no_grad():  
            outputs = model.generate(
                prompts["input_ids"].to(device), 
                max_new_tokens=200,
                num_return_sequences=num_samples,
                temperature=temperature,
                do_sample=True,
                
                )

        # outputs.shape = (prompt_length + answers_length + 1, batch_size * num_samples)
        prompt_length = prompts["input_ids"].shape[1]
        text_outputs = tokenizer.batch_decode(outputs[:, prompt_length:])
        questions = tokenizer.batch_decode(prompts["input_ids"])

        # print(f"text_outputs 0 {text_outputs[0]}")
        for prompt in questions:
            print(f"prompt: {prompt}")
            break
        for text_output in text_outputs:
            print(f"text_output: {text_output}")
            break
        

        # compute old log probabilities for ratio and ref for KL divergence penalty
        with torch.no_grad():
            ref_log_probs = compute_log_probs(ref_model, outputs, prompt_length).detach()
            old_log_probs = compute_log_probs(old_model, outputs, prompt_length).detach()


        right_answers = [answer for answer in right_answers for _ in range(num_samples)]
        rewards = compute_calculator_rewards(text_outputs, right_answers).to(device)
        reward_hist.append(rewards.mean().item())
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
            print(f"elapsed: {elapsed} mean rewards: {rewards.mean().item()}")
            # plot rewards
            plt.plot(reward_hist)
            plt.title(f"mean reward for gsm8k digits")
            plt.savefig(f"figures/gsm8k_rewards_epoch_{epoch}.png")


    test_accuracy = evaluate(use_calculator=use_calculator)
    accuracies.append(test_accuracy)
    plt.plot(accuracies)
    plt.title(f"test accuracy for gsm8k digits")
    plt.savefig(f"figures/gsm8k_accuracy_epoch_{epoch}.png")
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