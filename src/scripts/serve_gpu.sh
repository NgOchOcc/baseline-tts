export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=10014
export WORKER_BASE_PORT=10081
export LLM_BASE_PORT=10082
export no_proxy="127.0.0.0/8,10.0.0.0/8,0.0.0.0,harbor.vinai-systems.com,gitlab.vinai.io,gitlab.movian.ai"


### Conda Env
conda activate /lustre/scratch/client/movian/research/users/anhnd81/.conda/envs/roberta


### Fastchat
python -m fastchat.serve.controller --port 10014 --host 127.0.0.1


### Reward Model
CUDA_VISIBLE_DEVICES=0 python -m reason.llm_service.workers.reward_model_worker \
  --model-path Qwen/Qwen2.5-Math-PRM-7B \
  --controller-address http://127.0.0.1:10014 \
  --host 127.0.0.1 \
  --port 10081 \
  --worker-address http://127.0.0.1:10081


### Policy Model
CUDA_VISIBLE_DEVICES=1 python -m reason.llm_service.workers.vllm_worker \
  --max_model_length 8192 \
  --gpu_memory_utilization 0.7 \
  --swap_space 16 \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --controller-address http://127.0.0.1:10014 \
  --host 127.0.0.1 \
  --port 10082 \
  --worker-address http://127.0.0.1:10082


### Test API
curl http://127.0.0.1:10014/receive_heart_beat

curl http://127.0.0.1:10082/worker_generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 50,
    "top_k": 0.7,
}'