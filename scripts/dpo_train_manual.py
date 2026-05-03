
import json
import os
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.lora import LoRALinear
from tqdm import tqdm

def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    ref_log_ratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_log_ratios - ref_log_ratios
    losses = -nn.log_sigmoid(beta * logits)
    return losses.mean(), logits

def load_dpo_data(path, tokenizer):
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            # Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
            prompt_str = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            
            p_ids = tokenizer.encode(prompt_str)
            c_ids = tokenizer.encode(item['chosen']) + [tokenizer.eos_token_id]
            r_ids = tokenizer.encode(item['rejected']) + [tokenizer.eos_token_id]
            
            data.append({
                "prompt_len": len(p_ids),
                "chosen_ids": mx.array(p_ids + c_ids)[None, :],
                "rejected_ids": mx.array(p_ids + r_ids)[None, :]
            })
    return data

def get_log_probs(logits, labels):
    # Standard cross entropy returns -log(p)
    # We want log(p), so we take negative cross entropy
    # But for stability, we use log_softmax and gather
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    log_probs_selected = mx.take_along_axis(log_probs, labels[:, :, None], axis=-1).squeeze(-1)
    return log_probs_selected

def main():
    model_path = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    sft_adapter_path = "adapters"
    adapter_path = "adapters_rl"
    os.makedirs(adapter_path, exist_ok=True)

    print(f"Loading model with SFT adapters from {sft_adapter_path}...")
    model, tokenizer = load(model_path, adapter_path=sft_adapter_path)
    
    # The model already has LoRA layers from the SFT adapter.
    # We just need to ensure the optimizer targets those parameters.
    
    # Load dataset
    train_data = load_dpo_data("data/dpo/train.jsonl", tokenizer)
    print(f"Loaded {len(train_data)} DPO pairs.")

    optimizer = optim.Adam(learning_rate=2e-7)
    beta = 0.1

    def compute_loss(model, item, ref_chosen_logps, ref_rejected_logps):
        # Ensure scale is set correctly
        for _, m in model.named_modules():
            if isinstance(m, LoRALinear):
                m.scale = 2.0

        # Chosen Forward
        chosen_logits = model(item["chosen_ids"])[:, :-1, :]
        c_lp = get_log_probs(chosen_logits, item["chosen_ids"][:, 1:])
        c_lp = c_lp[:, item["prompt_len"]-1:].sum()

        # Rejected Forward
        rejected_logits = model(item["rejected_ids"])[:, :-1, :]
        r_lp = get_log_probs(rejected_logits, item["rejected_ids"][:, 1:])
        r_lp = r_lp[:, item["prompt_len"]-1:].sum()
        
        loss, _ = dpo_loss(c_lp, r_lp, ref_chosen_logps, ref_rejected_logps, beta)
        return loss

    loss_value_and_grad = nn.value_and_grad(model, compute_loss)

    print("Starting Manual DPO Training...")
    for epoch in range(10):
        total_loss = 0
        for item in tqdm(train_data, desc=f"Epoch {epoch+1}"):
            # 1. Get Reference Log-Probs (Disable LoRA)
            for _, m in model.named_modules():
                if isinstance(m, LoRALinear):
                    m.scale = 0.0
            
            # Ref Chosen
            logits = model(item["chosen_ids"])[:, :-1, :]
            ref_c_lp = get_log_probs(logits, item["chosen_ids"][:, 1:])
            ref_c_lp = mx.stop_gradient(ref_c_lp[:, item["prompt_len"]-1:].sum())
            
            # Ref Rejected
            logits = model(item["rejected_ids"])[:, :-1, :]
            ref_r_lp = get_log_probs(logits, item["rejected_ids"][:, 1:])
            ref_r_lp = mx.stop_gradient(ref_r_lp[:, item["prompt_len"]-1:].sum())

            # 2. Compute Loss and Grad
            loss, grads = loss_value_and_grad(model, item, ref_c_lp, ref_r_lp)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            if not mx.isnan(loss):
                total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(train_data):.4f}")

    # Save adapters
    # mlx-lm models are saved as safetensors
    model.save_weights(os.path.join(adapter_path, "adapters.safetensors"))
    
    # We also need a config file for mlx_lm.load to work
    with open(os.path.join(adapter_path, "adapter_config.json"), "w") as f:
        json.dump({
            "model": model_path,
            "fine_tune_type": "lora",
            "num_layers": 16,
            "lora_parameters": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "scale": 2.0
            }
        }, f)
    
    print(f"Training complete. Adapters saved to {adapter_path}")

    # Save adapters
    model.save_weights(os.path.join(adapter_path, "adapters.safetensors"))
    print(f"Training complete. Adapters saved to {adapter_path}")

if __name__ == "__main__":
    main()
