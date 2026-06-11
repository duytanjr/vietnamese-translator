"""
Train Translation Model
========================
Fine-tune Qwen 3 (0.6B) for English-to-Vietnamese translation
using SFT (Supervised Fine-Tuning) via TRL's SFTTrainer.

Requirements: GPU (run on Kaggle or Google Colab)

Usage:
    python src/train.py
    python src/train.py --config configs/config.yaml --output checkpoints/
    python src/train.py --data data/processed_data/translation_data.parquet
"""

import os
import argparse
import yaml
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_as_chat(example: dict) -> dict:
    """Convert a translation pair to chat messages format for SFTTrainer."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"translate English to Vietnamese:\n\n{example['input_text']}",
            },
            {
                "role": "assistant",
                "content": example["target_text"],
            },
        ]
    }


def load_dataset(data_path: str) -> tuple[Dataset, Dataset]:
    """Load parquet data and return train/eval HuggingFace Datasets."""
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} translation pairs from {data_path}")

    ds = Dataset.from_pandas(df[["input_text", "target_text"]])
    processed_ds = ds.map(format_as_chat, remove_columns=["input_text", "target_text"])

    split = processed_ds.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")
    print(f"\nSample: {train_dataset[0]['messages']}\n")

    return train_dataset, eval_dataset


def quick_test(model, tokenizer, device: str) -> None:
    """Run a few sample translations after training."""
    test_sentences = [
        "Hello, how are you?",
        "The weather is beautiful today.",
        "I am learning artificial intelligence.",
    ]
    model.eval()
    print("\n--- Quick Test ---")
    for sentence in test_sentences:
        messages = [
            {"role": "user", "content": f"translate English to Vietnamese:\n\n{sentence}"}
        ]
        text_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id
            )

        response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        translation = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        print(f"EN: {sentence}")
        print(f"VI: {translation}")
        print("---")


def train(config_path: str, data_path: str | None, output_dir: str) -> None:
    config = load_config(config_path)

    model_name = config["model"]["pretrained_model_name"]
    epochs = config["training"]["num_train_epochs"]
    batch_size = config["training"]["per_device_train_batch_size"]
    grad_acc = config["training"]["gradient_accumulation_steps"]
    lr = config["training"]["learning_rate"]
    max_length = config["training"]["max_length"]

    if data_path is None:
        data_path = config["data"]["processed_data_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 50)
    print("  Vietnamese AI Translator — Training")
    print("=" * 50)
    print(f"Device     : {device}")
    print(f"Model      : {model_name}")
    print(f"Epochs     : {epochs}")
    print(f"Batch size : {batch_size} (grad_acc={grad_acc}, effective={batch_size * grad_acc})")
    print(f"LR         : {lr}")
    print(f"Max length : {max_length}")
    print(f"Output dir : {output_dir}")
    print()

    # --- Dataset ---
    train_dataset, eval_dataset = load_dataset(data_path)

    # --- Model ---
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    print(f"Parameters : {model.num_parameters() / 1e6:.0f}M\n")

    # --- Training args ---
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        optim="adamw_torch_fused",
        learning_rate=lr,
        warmup_ratio=0,
        lr_scheduler_type="constant",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        max_length=max_length,
        bf16=(device == "cuda"),
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # --- Save metrics ---
    log_history = trainer.state.log_history
    epochs_list, train_losses, eval_losses = [], [], []

    for log in log_history:
        if "loss" in log and "eval_loss" not in log:
            epochs_list.append(int(log["epoch"]))
            train_losses.append(log["loss"])
        if "eval_loss" in log:
            eval_losses.append(log["eval_loss"])

    metrics_df = pd.DataFrame({
        "Epoch": epochs_list,
        "Train Loss": train_losses,
        "Eval Loss": eval_losses[: len(epochs_list)],
    })

    metrics_path = os.path.join(output_dir, "training_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print("\n--- Training Metrics ---")
    print(metrics_df.to_string(index=False))
    print(f"\nCheckpoints saved to : {output_dir}")
    print(f"Metrics saved to     : {metrics_path}")

    quick_test(model, tokenizer, device)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model for EN->VI translation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data parquet (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/",
        help="Directory to save checkpoints",
    )
    args = parser.parse_args()
    train(args.config, args.data, args.output)


if __name__ == "__main__":
    main()
