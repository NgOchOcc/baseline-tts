#!/bin/bash

LM=Qwen/Qwen2.5-Math-PRM-7B
RM=Qwen/Qwen2.5-7B-Instruct
task_names="AIME24 AMC23 MINERVA MATH"
method=beam_search
temperature=0.7
max_new_tokens=2048
tree_max_depth=40
tree_max_width=8
num_sequence=2
question_parallel_num=0
batch_size=500
max_time=3
double_line_break=1
local=0


for task in $task_names; do
    python reason/evaluation/evaluate.py \
        --LM $POLICY_MODEL_PATH \
        --RM $VALUE_MODEL_PATH \
        --task_name $task_name \
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
        --local $local
done
