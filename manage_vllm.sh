#!/bin/bash

# Configuration
GEN_MODEL="Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SFT_ADAPTER="./adapters"
RL_ADAPTER="./adapters_rl"

# Function to wait for a port to be ready
wait_for_port() {
    local port=$1
    local name=$2
    echo "Waiting for $name on port $port to be ready (this can take 3-5 minutes)..."
    while ! curl -s http://localhost:$port/v1/models > /dev/null; do
        # Check if the process actually died while waiting
        if ! pgrep -f "port $port" > /dev/null; then
            echo "ERROR: Server for $name on port $port died during startup. Check ${name}_server.log"
            return 1
        fi
        sleep 5
    done
    echo "SUCCESS: $name is READY."
    return 0
}

start() {
    echo "Starting vLLM Servers in STAGGERED MODE to prevent system crash..."

    # 1. Generator (Port 8000)
    echo "Launching Generator..."
    python -m vllm.entrypoints.openai.api_server \
        --model $GEN_MODEL \
        --port 8000 \
        --gpu-memory-utilization 0.15 \
        --max-model-len 2048 \
        --enforce-eager > gen_server.log 2>&1 &
    wait_for_port 8000 "Generator" || exit 1

    # 2. Judge (Port 8001)
    echo "Launching Judge..."
    python -m vllm.entrypoints.openai.api_server \
        --model $JUDGE_MODEL \
        --port 8001 \
        --gpu-memory-utilization 0.30 \
        --max-model-len 2048 \
        --enforce-eager > judge_server.log 2>&1 &
    wait_for_port 8001 "Judge" || exit 1

    # 3. SFT Baseline (Port 8002)
    echo "Launching SFT..."
    python -m vllm.entrypoints.openai.api_server \
        --model $GEN_MODEL \
        --enable-lora \
        --lora-modules sft-lora=$SFT_ADAPTER \
        --port 8002 \
        --gpu-memory-utilization 0.15 \
        --max-model-len 1024 \
        --enforce-eager > sft_server.log 2>&1 &
    wait_for_port 8002 "SFT" || exit 1

    # 4. RL Baseline (Port 8003)
    echo "Launching RL..."
    python -m vllm.entrypoints.openai.api_server \
        --model $GEN_MODEL \
        --enable-lora \
        --lora-modules rl-lora=$RL_ADAPTER \
        --port 8003 \
        --gpu-memory-utilization 0.15 \
        --max-model-len 1024 \
        --enforce-eager > rl_server.log 2>&1 &
    wait_for_port 8003 "RL" || exit 1

    echo "------------------------------------------------"
    echo "All 4 vLLM servers are UP and READY."
    echo "You can now run the evaluation script."
    echo "------------------------------------------------"
}

check() {
    echo "Checking vLLM Endpoints..."
    for port in 8000 8001 8002 8003; do
        printf "Port %d: " $port
        res=$(curl -s http://localhost:$port/v1/models | grep -o '"id":"[^"]*"' | head -n 1)
        if [ -z "$res" ]; then
            echo "FAILED (Not Responding)"
        else
            echo "OK ($res)"
        fi
    done
}

stop() {
    echo "Stopping all vLLM servers..."
    pkill -f "vllm.entrypoints.openai.api_server"
    echo "Done."
}

case "$1" in
    start) start ;;
    check) check ;;
    stop) stop ;;
    *) echo "Usage: $0 {start|check|stop}" ;;
esac
