import json
import os
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_sft_data(path: str, tokenizer) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            records.append({"text": text})
    return Dataset.from_list(records)


def main():
    base_model = "Qwen/Qwen3-1.7B-Instruct"
    adapter_path = "adapters"
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

    train_dataset = load_sft_data("data/sft/train.jsonl", tokenizer)
    eval_dataset = load_sft_data("data/sft/valid.jsonl", tokenizer) if Path("data/sft/valid.jsonl").exists() else None
    print(f"Train samples: {len(train_dataset)}" + (f" | Eval samples: {len(eval_dataset)}" if eval_dataset else ""))

    training_args = SFTConfig(
        output_dir=adapter_path,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    print("Starting SFT training...")
    trainer.train()
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Done. Adapters saved to {adapter_path}/")


if __name__ == "__main__":
    main()
