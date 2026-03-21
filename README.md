
### Installation

Clone the repository:

```bash
git clone https://github.com/RyanLiu112/compute-optimal-tts.git
cd compute-optimal-tts/src
```

Create a new conda environment and install the dependencies:

```bash
conda create -n tts python=3.10
conda activate tts
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install "ray[default]==2.38.0"
pip install "fschat[model_worker,webui]"
pip install sympy==1.12
cd envs//latex2sympy
pip install -e .
```

Install `tmux` for serving policy models and PRMs:

```bash
sudo apt-get update
sudo apt-get install tmux
```

#### Policy Models

```bash
cd src
export VALUE_MODEL_PATH=path/to/RM  # dummy for CoT
export POLICY_MODEL_PATH=path/to/LM && export LOGDIR=path/to/logdir
export HOST_ADDR=0.0.0.0 && export CONTROLLER_PORT=10014 && export WORKER_BASE_PORT=10081
```

Run the corresponding script:

```bash
# 1 gpu
bash scripts/serve_gpu1.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

# 2 gpus (32B policy model + 1.5B-8B PRM)
bash scripts/serve_gpu2.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

# 3 gpus (72B policy model + 1.5B-8B PRM)
bash scripts/serve_gpu3_1-2.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

# 3 gpus (0.5B-32B policy model + 72B PRM)
bash scripts/serve_gpu3_2-1.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT

# 4 gpus (72B policy model + 72B PRM)
bash scripts/serve_gpu4.sh $POLICY_MODEL_PATH $VALUE_MODEL_PATH $HOST_ADDR $CONTROLLER_PORT $WORKER_BASE_PORT
```

#### Step 2: Run TTS methods

We provide the following commands for different TTS methods.

##### CoT

```bash
cd src
bash scripts/run.sh --method cot --LM $POLICY_MODEL_PATH --RM dummy --width 1 --num_seq 1
```

##### Best-of-N (BoN)

> [!NOTE]
> **Configuring batch size for BoN and DVTS**:
> For instance, when running BoN on MATH-500, it processes 500 problems with each executing 256 times (determined by `num_q`). To enhance the compute efficiency, it is recommended to distribute the problems across multiple GPUs by adjusting the `batch size` (bs). For example, set bs to 500 for 256 GPUs or 16000 for 8 GPUs.

```bash
cd src
bash scripts/run.sh --method best_of_n --LM $POLICY_MODEL_PATH --RM $VALUE_MODEL_PATH --width 1 --num_seq 1 --num_q 256 --bs batch_size
```

##### Beam Search

```bash
cd src
bash scripts/run.sh --method beam_search --LM $POLICY_MODEL_PATH --RM $VALUE_MODEL_PATH --width 4 --num_seq 1
```

##### DVTS

```bash
cd src
bash scripts/run.sh --method beam_search --LM $POLICY_MODEL_PATH --RM $VALUE_MODEL_PATH --width 4 --num_seq 1 --num_q 64 --bs batch_size
```

#### Step 3: Post process the results

For BoN and DVTS, no average result is computed by default. To compute the average, aggregate the `majority_vote` values from all jsonl files after processing all problems `num_q` times.