# RL-agents
using Reinforcement Learning to teach agents 


to set-up the environment use uv (if you do not have it install it: curl -LsSf https://astral.sh/uv/install.sh | sh ) and do:


uv sync
source .venv/bin/activate

to train the model on digits do
python3 digit_train.py

to train the model on gsm8k do
python3 gsm8k train.py 
