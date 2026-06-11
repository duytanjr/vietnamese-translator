"""
Generate Synthetic Translation Data
=====================================
Generates English-Vietnamese translation pairs using Gemini API
to augment the training dataset.

Usage:
    python src/generate_synthetic_data.py
    python src/generate_synthetic_data.py --config configs/config.yaml
"""

import os
import argparse
import yaml
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(samples_per_call: int) -> str:
    return f"""You are a professional English-Vietnamese translator.

Generate exactly {samples_per_call} English-Vietnamese translation pairs.

Requirements:
1. English sentences should be diverse (daily conversation, news, technology, education, etc.)
2. Sentences can be short (5 words) or long (20+ words)
3. Vietnamese translations must be natural and accurate

Output format:
- One pair per line
- English and Vietnamese separated by ||
- No numbering or bullet points

Example:
The weather is beautiful today || Thời tiết hôm nay thật đẹp
I need to finish this report by Friday || Tôi cần hoàn thành báo cáo này trước thứ Sáu
"""


def generate_pairs(prompt: str, model_name: str, api_key: str) -> list[str]:
    """Call Gemini API to generate translation pairs."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip().splitlines()


def parse_and_clean(raw_lines: list[str]) -> pd.DataFrame:
    """Parse raw lines into a DataFrame of (english, vietnamese) pairs."""
    df = pd.DataFrame({"raw": raw_lines})
    split_data = df["raw"].str.split("||", expand=True, n=1)
    split_data = split_data.dropna()
    split_data.columns = ["english", "vietnamese"]
    split_data["english"] = split_data["english"].str.strip()
    split_data["vietnamese"] = split_data["vietnamese"].str.strip()
    # Remove empty or whitespace-only rows
    split_data = split_data[
        (split_data["english"] != "") & (split_data["vietnamese"] != "")
    ]
    return split_data.reset_index(drop=True)


def generate_synthetic_data(config_path: str) -> None:
    config = load_config(config_path)

    model_name = config["model"]["synthetic_model"]
    api_key = config["key"]["gemini_api_key"]
    num_calls = config["synthetic"]["num_calls"]
    samples_per_call = config["synthetic"]["samples_per_call"]
    output_path = config["data"]["synthetic_translation_path"]

    print(f"Model     : {model_name}")
    print(f"Calls     : {num_calls}")
    print(f"Per call  : {samples_per_call}")
    print(f"Expected  : ~{num_calls * samples_per_call} pairs")
    print(f"Output    : {output_path}")
    print()

    prompt = build_prompt(samples_per_call)
    all_lines = []

    for i in tqdm(range(num_calls), desc="Generating"):
        try:
            lines = generate_pairs(prompt, model_name, api_key)
            all_lines.extend(lines)
            print(f"  Call {i+1}/{num_calls}: {len(lines)} lines")
        except Exception as e:
            print(f"  Call {i+1} failed: {e}")
            continue

    print(f"\nTotal raw lines  : {len(all_lines)}")

    result_df = parse_and_clean(all_lines)
    print(f"Valid pairs      : {len(result_df)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_parquet(output_path, index=False)
    print(f"Saved to         : {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic EN-VI translation data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()
    generate_synthetic_data(args.config)


if __name__ == "__main__":
    main()
