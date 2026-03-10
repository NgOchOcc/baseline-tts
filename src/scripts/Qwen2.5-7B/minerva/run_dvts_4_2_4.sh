#!/bin/bash
export VALUE_MODEL_PATH=Qwen/Qwen2.5-Math-PRM-7B
export POLICY_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export LOGDIR=logs/cot_math
export HOST_ADDR=127.0.0.1
export CONTROLLER_PORT=10014
export WORKER_BASE_PORT=10081
export PYTHONPATH=/lustre/scratch/client/movian/research/users/ngoclt69/workspace/baselines_tts/src

save_dir=${PYTHONPATH}/output
LOGDIR=${PYTHONPATH}/logs_fastchat
controller_addr=http://$HOST_ADDR:$CONTROLLER_PORT




LM=Qwen/Qwen2.5-Math-PRM-7B
RM=Qwen/Qwen2.5-7B-Instruct
task_names="MINERVA"
method=beam_search
temperature=0.7
max_new_tokens=2048
tree_max_depth=40
tree_max_width=4
num_sequence=2
question_parallel_num=4
batch_size=500
max_time=3
double_line_break=1
local=0
num_worker=1
seed=0

python reason/evaluation/evaluate.py \
    --LM $POLICY_MODEL_PATH \
    --RM $VALUE_MODEL_PATH \
    --task_name $task_names \
    --temperature $temperature \
    --max_new_tokens $max_new_tokens \
    --num_sequence $num_sequence \
    --tree_max_width $tree_max_width \
    --tree_max_depth $tree_max_depth \
    --save_dir $save_dir \
    --method $method \
    --num_worker $num_worker \
    --controller_addr $controller_addr \
    --add_step_prompt \
    --question_parallel_num $question_parallel_num \
    --double_line_break $double_line_break \
    --batch_size $batch_size \
    --max_time $max_time \
    --local $local \
    --seed $seed
