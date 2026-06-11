"""
Evaluate Translation Model
===========================
Evaluates the fine-tuned model using:
  - BLEU (SacreBLEU)
  - ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
  - LLM-as-Judge (Gemini, 1-10 scale)

Usage:
    python src/evaluate.py --checkpoint checkpoints/checkpoint-XXXX
    python src/evaluate.py --checkpoint checkpoints/checkpoint-XXXX --no-llm
"""

import argparse
import json
import re
import time
import yaml
import torch
import pandas as pd
import evaluate as hf_evaluate
import google.generativeai as genai
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Constants ────────────────────────────────────────────────────────────────

LLM_JUDGE_PROMPT = """You are an expert English-Vietnamese translator. Score the following translation.

- Source (English): "{src}"
- Machine Translation (Vietnamese): "{hyp}"
- Reference Translation: "{ref}"

Score the translation quality from 1-10 (integer) based on accuracy and naturalness.

Return ONLY JSON: {{"score": 8}}
"""

LLM_JUDGE_SAMPLES = 50   # number of samples to judge (full set can be slow)
LLM_RATE_LIMIT_SEC = 4   # seconds between Gemini API calls


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return model, tokenizer


def translate(text: str, model, tokenizer, device: str, max_new_tokens: int = 100) -> str:
    """Translate a single English sentence to Vietnamese."""
    messages = [
        {"role": "user", "content": f"translate English to Vietnamese:\n\n{text}"}
    ]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        return_attention_mask=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    result = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Remove <think>...</think> if model uses thinking mode
    result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
    return result


def run_inference(eval_df: pd.DataFrame, model, tokenizer, device: str) -> tuple[list, list]:
    """Run translation inference on the full eval dataset."""
    predictions, references = [], []
    for i in tqdm(range(len(eval_df)), desc="Translating"):
        prediction = translate(eval_df.iloc[i]["English"], model, tokenizer, device)
        predictions.append(prediction)
        references.append(eval_df.iloc[i]["Vietnamese"])
    return predictions, references


def compute_auto_metrics(predictions: list, references: list) -> dict:
    """Compute BLEU and ROUGE scores."""
    bleu_metric = hf_evaluate.load("sacrebleu")
    rouge_metric = hf_evaluate.load("rouge")

    bleu = bleu_metric.compute(
        predictions=predictions,
        references=[[r] for r in references],
    )
    rouge = rouge_metric.compute(predictions=predictions, references=references)

    return {
        "bleu": round(bleu["score"], 2),
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
    }


def compute_llm_score(
    eval_df: pd.DataFrame,
    predictions: list,
    references: list,
    gemini_api_key: str,
    gemini_model: str,
    n_samples: int = LLM_JUDGE_SAMPLES,
) -> float:
    """Use Gemini as LLM-judge to score translation quality (1-10)."""
    genai.configure(api_key=gemini_api_key)
    judge = genai.GenerativeModel(gemini_model)

    scores = []
    for i in tqdm(range(min(n_samples, len(predictions))), desc="LLM Judging"):
        prompt = LLM_JUDGE_PROMPT.format(
            src=eval_df.iloc[i]["English"],
            hyp=predictions[i],
            ref=references[i],
        )
        try:
            response = judge.generate_content(prompt)
            json_str = re.sub(r"```json|```", "", response.text).strip()
            score = json.loads(json_str).get("score", -1)
            scores.append(score)
        except Exception as e:
            print(f"  Judge call {i} failed: {e}")
            scores.append(-1)
        time.sleep(LLM_RATE_LIMIT_SEC)

    valid = [s for s in scores if s > 0]
    return round(sum(valid) / len(valid), 2) if valid else 0.0


def print_results(metrics: dict, llm_score: float | None) -> None:
    print("\n" + "=" * 50)
    print("       EVALUATION RESULTS")
    print("=" * 50)
    print(f"  BLEU Score  : {metrics['bleu']}")
    print(f"  ROUGE-1     : {metrics['rouge1']}")
    print(f"  ROUGE-2     : {metrics['rouge2']}")
    print(f"  ROUGE-L     : {metrics['rougeL']}")
    if llm_score is not None:
        print(f"  LLM Judge   : {llm_score} / 10")
    print("=" * 50)


def print_samples(eval_df: pd.DataFrame, predictions: list, references: list, n: int = 10) -> None:
    print("\n--- Sample Translations ---")
    for i in range(min(n, len(predictions))):
        print(f"EN  : {eval_df.iloc[i]['English']}")
        print(f"REF : {references[i]}")
        print(f"OUT : {predictions[i]}")
        print("---")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: str,
    config_path: str,
    eval_data_path: str | None,
    use_llm_judge: bool,
) -> None:
    config = load_config(config_path)

    if eval_data_path is None:
        eval_data_path = config["data"]["valid_translation_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 50)
    print("  Vietnamese AI Translator — Evaluation")
    print("=" * 50)
    print(f"Device     : {device}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Eval data  : {eval_data_path}")
    print()

    # Load
    print("Loading model...")
    model, tokenizer = load_model(checkpoint_path, device)

    eval_df = pd.read_parquet(eval_data_path)
    print(f"Evaluation samples: {len(eval_df)}\n")

    # Inference
    predictions, references = run_inference(eval_df, model, tokenizer, device)

    # Auto metrics
    print("\nComputing BLEU and ROUGE...")
    metrics = compute_auto_metrics(predictions, references)

    # LLM judge
    llm_score = None
    if use_llm_judge:
        print(f"\nRunning LLM judge on {LLM_JUDGE_SAMPLES} samples...")
        llm_score = compute_llm_score(
            eval_df,
            predictions,
            references,
            gemini_api_key=config["key"]["gemini_api_key"],
            gemini_model=config["model"]["synthetic_model"],
        )

    # Report
    print_results(metrics, llm_score)
    print_samples(eval_df, predictions, references)


def main():
    parser = argparse.ArgumentParser(description="Evaluate EN->VI translation model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to validation parquet (overrides config)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM-as-Judge evaluation (faster, no Gemini API needed)",
    )
    args = parser.parse_args()
    evaluate(args.checkpoint, args.config, args.eval_data, not args.no_llm)


if __name__ == "__main__":
    main()
