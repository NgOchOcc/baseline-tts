export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=microsoft/Phi-4-mini-instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=10014
export WORKER_BASE_PORT=10081
export LLM_BASE_PORT=10082
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"


export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=10114
export WORKER_BASE_PORT=10181
export LLM_BASE_PORT=10182
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"


export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=microsoft/Phi-4-mini-instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=10314
export WORKER_BASE_PORT=10381
export LLM_BASE_PORT=10382
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"


export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=microsoft/Phi-4-mini-instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=10414
export WORKER_BASE_PORT=10481
export LLM_BASE_PORT=10482
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/roberta



export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=11414
export WORKER_BASE_PORT=11481
export LLM_BASE_PORT=11482
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/roberta



export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=12414
export WORKER_BASE_PORT=12481
export LLM_BASE_PORT=12482
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/roberta



export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=13414
export WORKER_BASE_PORT=13481
export LLM_BASE_PORT=13482
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/tts


export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=14414
export WORKER_BASE_PORT=14481
export LLM_BASE_PORT=14482
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/roberta
cd /lustre/scratch/client/movian/research/users/ngoclt69/workspace/baseline-tts/src


export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=15414
export WORKER_BASE_PORT=15481
export LLM_BASE_PORT=15482
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/roberta
cd /lustre/scratch/client/movian/research/users/ngoclt69/workspace/baseline-tts/src



export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=23414
export WORKER_BASE_PORT=23481
export LLM_BASE_PORT=23482
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/tts


### Conda Env
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/roberta


### Fastchat
python -m fastchat.serve.controller --port $CONTROLLER_PORT --host $HOST_ADDR


### Reward Model
CUDA_VISIBLE_DEVICES=0 python -m reason.llm_service.workers.reward_model_worker \
  --model-path Qwen/Qwen2.5-Math-PRM-7B \
  --controller-address http://$HOST_ADDR:$CONTROLLER_PORT \
  --host $HOST_ADDR \
  --port $WORKER_BASE_PORT \
  --worker-address http://$HOST_ADDR:$WORKER_BASE_PORT


### Policy Model
CUDA_VISIBLE_DEVICES=1 python -m reason.llm_service.workers.vllm_worker \
  --max_model_length 8192 \
  --gpu_memory_utilization 0.7 \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --controller-address http://$HOST_ADDR:$CONTROLLER_PORT \
  --host $HOST_ADDR \
  --port $LLM_BASE_PORT \
  --worker-address http://$HOST_ADDR:$LLM_BASE_PORT


### Test API
curl http://$HOST_ADDR:$CONTROLLER_PORT/receive_heart_beat

curl http://$HOST_ADDR:$LLM_BASE_PORT/worker_generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 50,
    "top_k": 0.7,
}'