import time
import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM
from rewards import compute_log_probs, compute_calculator_rewards, compute_ppo_loss, calculate_grpo_advantages, compute_rewards
from data import get_dataset
from peft import PeftModelForCausalLM
import matplotlib.pyplot as plt
import re
import datasets
from tools.calculator import calculator
import torch.nn.functional as F
from tqdm import tqdm


use_calculator = False
mode = "calculator" if use_calculator else "no_calculator"
num_digits = 3
temperature = 0.8
num_samples = 24 if torch.cuda.is_available() else 4
mu = 2
learning_rate = 1e-4
epochs = 10
batch_size = 2
log_interval = 5
test_interval = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "HuggingFaceTB/SmolLM2-360m-Instruct"
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





def evaluate(model, use_calculator=False):
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
    questions = []
    answers = []
    all_right_answers = []
    corrects = []
    
    for batch in tqdm(test_loader):
        prompts, right_answers = batch
        
        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                prompts["input_ids"].to(device),
                max_new_tokens=40,
                do_sample=False  # Use greedy decoding for evaluation
            )
        
        # Calculate prompt length to extract only the generated part
        prompt_length = prompts["input_ids"].shape[1]
        
        # Extract text outputs
        text_outputs = tokenizer.batch_decode(outputs[:, prompt_length:])
        answers.extend(text_outputs)
        questions.extend([tokenizer.decode(prompt) for prompt in prompts["input_ids"]])
        all_right_answers.extend(right_answers)

        
        # Check each output
        for output, target in zip(text_outputs, right_answers):
            # Try regular pattern first
            extracted = re.findall(pattern, output)
            corrects.append(False)
            
            # If not found, try use_calculator pattern
            try:
                # Use the use_calculator function to evaluate the expression
                if use_calculator:
                    result = float(calculator(extracted[0]))
                else:
                    result = float(eval(extracted[0]))
                if abs(result - target) <= tolerance:
                    correct += 1
                    corrects[-1] = True
                else:
                    print(f"wrong results output: {output} target: {target} result: {result}")
            except:
                print(f"exception output: {output} target: {target}")
                pass
                    
            total += 1
        # save predictions and results
    with open(f"{mode}_num_digits_{num_digits}_predictions.txt", "w") as f:
        print(f"saving to {mode}_num_digits_{num_digits}_predictions.txt")
        print(f"length: {len(questions)}")
        print(f"length: {len(answers)}")
        print(f"length: {len(all_right_answers)}")
        print(f"length: {len(corrects)}")

        for question, answer, right_answer, correct in zip(questions, answers, all_right_answers, corrects):
            f.write(f"question: {question}\n")
            f.write(f"answer: {answer}\n")
            f.write(f"right answer: {right_answer}\n")
            f.write(f"correct: {correct}\n")
            f.write("\n")
            

    accuracy = correct / total if total > 0 else 0
    return accuracy


def calculator_collate_fn(batch):
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
                {"role": "user", "content": x["prompt"]},
            ],
            tokenize=False)
        for x in batch
        ]
    # targets = [float(x["right_answer"].split("####")[-1].replace(",", "")) for x in batch]
    targets = [float(x["right_answer"]) for x in batch]
    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    # targets = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    return inputs, targets

def no_calculator_collate_fn(batch):
    inputs = [
        tokenizer.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False)
        for x in batch
        ]
    targets = [x["right_answer"] for x in batch]
    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    # targets = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    return inputs, targets






train_set = get_dataset(size=2000, digits=num_digits, use_calculator=use_calculator)
test_set = get_dataset(size=200, digits=num_digits, use_calculator=use_calculator)

collate_fn = calculator_collate_fn if use_calculator else no_calculator_collate_fn

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, collate_fn=collate_fn)



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_test_accuracy = None
test_accuracy = evaluate(model, use_calculator=use_calculator)
print('-' * 89)
print('| initialisation | test accuracy {:5.2f}'.format(test_accuracy))
print('-' * 89)

# switch eval for train model (enables dropout)
model.train()
# enable gradient checkpointing

ref_model = deepcopy(model) # reference model for KL divergence penalty
old_model = deepcopy(model) # old model for PPO ratio

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

reward_hist = [] 
accuracies = []
running_means = []


accuracies.append(test_accuracy)


for epoch in range(1, epochs+1):
    ref_model.load_state_dict(model.state_dict()) # update ref model every epoch like in the GRPO paper
    epoch_start_time = time.time()
    model.train()
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
                max_new_tokens=40,
                num_return_sequences=num_samples,
                temperature=temperature,
                do_sample=True,
                
                )

        # outputs.shape = (prompt_length + answers_length + 1, batch_size * num_samples)
        prompt_length = prompts["input_ids"].shape[1]
        text_outputs = tokenizer.batch_decode(outputs[:, prompt_length:])
        questions = tokenizer.batch_decode(prompts["input_ids"])

        
        right_answers = [answer for answer in right_answers for _ in range(num_samples)]
        if use_calculator:
            rewards = compute_calculator_rewards(text_outputs, right_answers).to(device)
        else:
            rewards = compute_rewards(text_outputs, right_answers).to(device)

        best_completion = text_outputs[rewards.argmax().item()]
        best_question = questions[rewards.argmax().item()//num_samples]
        best_right_answer = right_answers[rewards.argmax().item()]
        print(f"best question: {best_question}")
        print(f"best completion: {best_completion}")
        print(f"best right answer: {best_right_answer}")

        reward_hist.append(rewards.mean().item())
        window_size = min(20, len(reward_hist))
        running_mean = sum(reward_hist[-window_size:]) / window_size
        running_means.append(running_mean)
        print(f"rewards 0: {rewards}")
        

        if i % log_interval == 0:
            plt.plot(reward_hist, label="rewards", alpha=0.5)
            plt.title(f"mean rewards for {num_digits} digits with {mode}")

            plt.plot(running_means, label="running mean rewards")
            plt.legend()

            plt.savefig(f"figures/{mode}_{num_digits}_num_digits_running_mean_rewards_epoch_{epoch}.png")
            plt.close()

        if rewards.std() == 0:
            continue



        # compute advantages
        advantages = calculate_grpo_advantages(rewards, num_samples=num_samples)

        # compute loss
        responses = outputs[:, prompt_length:]

        # compute old log probabilities for ratio and ref for KL divergence penalty
        with torch.no_grad():
            ref_log_probs = compute_log_probs(ref_model, outputs, prompt_length).detach()
            old_log_probs = compute_log_probs(old_model, outputs, prompt_length).detach()

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
            # plot this one in semi transparent
            
        if i % test_interval == 0 and i > 0:

            model.eval()
            model.disable_input_require_grads()
            model.gradient_checkpointing_disable()  # Disable gradient checkpointing
            test_accuracy = evaluate(model, use_calculator=use_calculator)
            print(f"test accuracy: {test_accuracy}")
            accuracies.append(test_accuracy)
            plt.close()
            plt.plot(accuracies)
            plt.title(f"test accuracy for {num_digits} digits with {mode}")
            plt.savefig(f"figures/{mode}_{num_digits}_num_digits_accuracy_epoch_{epoch}.png")
 
            # Save the model if the test accuracy is the best we've seen so far.
            if not best_test_accuracy or test_accuracy > best_test_accuracy:
                print(f"saving model with test accuracy: {test_accuracy}")
                with open(f"{mode}_num_digits{num_digits}_vanilla_GRPO.pt", 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_test_accuracy = test_accuracy
            
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            model.train()

    if best_test_accuracy is not None and best_test_accuracy > 0.99:
        print(f"achieved near perfect accuracy after {epoch} epochs")
        break
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | test accuracy {:5.2f}'.format(epoch, (time.time() - epoch_start_time), accuracies[-1]))
    print(f"all accuracies: {accuracies}")
    print('-' * 89)
