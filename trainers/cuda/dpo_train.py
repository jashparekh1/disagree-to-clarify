import json
import os
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer


def load_dpo_data(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            records.append({
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })
    return Dataset.from_list(records)


def main():
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "adapters_rl"
    os.makedirs(adapter_path, exist_ok=True)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dpo_data("data/dpo/train.jsonl")
    eval_dataset = load_dpo_data("data/dpo/valid.jsonl") if Path("data/dpo/valid.jsonl").exists() else None
    print(f"Train pairs: {len(train_dataset)}" + (f" | Eval pairs: {len(eval_dataset)}" if eval_dataset else ""))

    training_args = DPOConfig(
        output_dir=adapter_path,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        beta=0.1,
        max_length=512,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Done. Adapters saved to {adapter_path}/")


if __name__ == "__main__":
    main()
