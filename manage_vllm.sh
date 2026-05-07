#!/bin/bash

# Configuration
GEN_MODEL="Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER="./adapters"
RL_ADAPTER="./adapters_rl"

# Stability Overrides
export VLLM_USE_V1=0  # Use stable V0 engine
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

wait_for_port() {
    local port=$1
    local name=$2
    echo "Waiting for $name on port $port to be ready (3-5 minutes)..."
    while ! curl -s http://localhost:$port/v1/models > /dev/null; do
        if ! pgrep -f "port $port" > /dev/null; then
            echo "ERROR: Server for $name on port $port died. Check ${name}_server.log"
            return 1
        fi
        sleep 5
    done
    echo "SUCCESS: $name is READY."
    return 0
}

start() {
    echo "Stopping any existing servers..."
    pkill -9 -f vllm
    sleep 5

    echo "Starting consolidated vLLM Servers..."

    # 1. Consolidated Generator (Port 8000)
    # This server handles Vanilla, SFT, and RL versions using LoRA modules
    echo "Launching Consolidated Generator (Base, SFT, RL)..."
    python -m vllm.entrypoints.openai.api_server \
        --model $GEN_MODEL \
        --enable-lora \
        --lora-modules sft-lora=$SFT_ADAPTER rl-lora=$RL_ADAPTER \
        --port 8000 \
        --gpu-memory-utilization 0.40 \
        --max-model-len 2048 \
        --enforce-eager \
        --distributed-executor-backend uniproc > gen_server.log 2>&1 &
    wait_for_port 8000 "Generator" || exit 1

    # 2. Judge (Port 8001)
    echo "Launching Judge..."
    python -m vllm.entrypoints.openai.api_server \
        --model $JUDGE_MODEL \
        --port 8001 \
        --gpu-memory-utilization 0.40 \
        --max-model-len 2048 \
        --enforce-eager \
        --distributed-executor-backend uniproc > judge_server.log 2>&1 &
    wait_for_port 8001 "Judge" || exit 1

    echo "------------------------------------------------"
    echo "All vLLM servers are UP and READY."
    echo "Consolidated Mode: Port 8000 handles all 3B variants."
    echo "------------------------------------------------"
}

stop() {
    echo "Stopping all vLLM servers..."
    pkill -9 -f vllm
    echo "Done."
}

case "$1" in
    start) start ;;
    stop) stop ;;
    *) echo "Usage: $0 {start|stop}" ;;
esac
