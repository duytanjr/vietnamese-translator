"""
Prepare Training Data
======================
Merges original and synthetic translation data into a single
shuffled training dataset.

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --config configs/config.yaml
"""

import os
import argparse
import yaml
import pandas as pd


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_original_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).dropna()
    df = df.rename(columns={"English": "input_text", "Vietnamese": "target_text"})
    return df[["input_text", "target_text"]]


def load_synthetic_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path).dropna()
    df = df.rename(columns={"english": "input_text", "vietnamese": "target_text"})
    return df[["input_text", "target_text"]]


def prepare_data(config_path: str) -> None:
    config = load_config(config_path)

    original_path = config["data"]["train_translation_path"]
    synthetic_path = config["data"]["synthetic_translation_path"]
    output_path = config["data"]["processed_data_path"]

    # --- Load ---
    print(f"Loading original data  : {original_path}")
    original_df = load_original_data(original_path)
    print(f"  Rows: {len(original_df)}")

    print(f"Loading synthetic data : {synthetic_path}")
    synthetic_df = load_synthetic_data(synthetic_path)
    print(f"  Rows: {len(synthetic_df)}")

    # --- Merge ---
    merged_df = pd.concat([original_df, synthetic_df], ignore_index=True)

    # --- Shuffle ---
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

    total = len(merged_df)
    print(f"\nMerged & shuffled dataset: {total} rows")
    print(f"  Original  : {len(original_df):>6}  ({len(original_df)/total*100:.1f}%)")
    print(f"  Synthetic : {len(synthetic_df):>6}  ({len(synthetic_df)/total*100:.1f}%)")

    # --- Stats ---
    print(f"\nDataset statistics:")
    print(f"  Avg input length  : {merged_df['input_text'].str.len().mean():.0f} chars")
    print(f"  Avg target length : {merged_df['target_text'].str.len().mean():.0f} chars")

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    prepare_data(args.config)


if __name__ == "__main__":
    main()
