export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=0.0.0.0
export CONTROLLER_PORT=10014
export WORKER_BASE_PORT=10081

tmux new -n controller

conda activate cot
python -m fastchat.serve.controller --port 10014 --host 0.0.0.0


tmux new -n reward

conda activate cot
CUDA_VISIBLE_DEVICES=0 python -m reason.llm_service.workers.reward_model_worker \
  --model-path Qwen/Qwen2.5-Math-PRM-7B \
  --controller-address http://0.0.0.0:10014 \
  --host 0.0.0.0 \
  --port 10081 \
  --worker-address http://0.0.0.0:10081


tmux new -n policy
conda activate cot
CUDA_VISIBLE_DEVICES=1 python -m reason.llm_service.workers.vllm_worker \
  --max_model_length 8192 \
  --gpu_memory_utilization 0.7 \
  --swap_space 16 \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --controller-address http://0.0.0.0:10014 \
  --host 0.0.0.0 \
  --port 10082 \
  --worker-address http://0.0.0.0:10082