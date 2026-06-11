# 🌐 Vietnamese AI Translator

A fine-tuned **Qwen 3 (0.6B)** model for English-to-Vietnamese translation, deployed as a web application.

## 🎯 Overview

This project fine-tunes a small language model (0.6B parameters) specifically for English-to-Vietnamese translation. The training data is augmented with synthetic translation pairs generated via Gemini API to improve translation quality and diversity.

**Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/duytanjr/vietnamese-translator)

**Model:** [duytanjr/vietnamese-translator](https://huggingface.co/duytanjr/vietnamese-translator)

## 📊 Results

| Metric | Score |
|--------|-------|
| BLEU (SacreBLEU) | **62.99** |
| ROUGE-1 | 0.8402 |
| ROUGE-2 | 0.7387 |
| ROUGE-L | 0.8263 |
| LLM Judge (Gemini) | N/A |

## 🏗️ Architecture

```
English Text → Chat Template → Qwen 3 (0.6B, fine-tuned) → Vietnamese Translation
```

- **Base Model:** Qwen/Qwen3-0.6B
- **Training Method:** SFT (Supervised Fine-Tuning) using TRL
- **Training Data:** ~60K original + ~10K synthetic translation pairs
- **Training Time:** ~6 hours on Kaggle GPU (T4)

## 📂 Project Structure

```
vietnamese-translator/
├── app/
│   └── app.py                  # Gradio web application
├── configs/
│   └── config.yaml             # Training & model configuration
├── data/
│   ├── original_data/          # Original EN-VI translation pairs
│   ├── synthetic_data/         # Generated synthetic data
│   └── processed_data/         # Merged training data
├── src/
│   ├── generate_synthetic_data.py
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Run the demo locally

```bash
pip install -r requirements.txt
python app/app.py
```

### Train from scratch

1. **Generate synthetic data** (optional, requires Gemini API key):
   ```bash
   python src/generate_synthetic_data.py --config configs/config.yaml
   ```

2. **Prepare training data:**
   ```bash
   python src/prepare_data.py --config configs/config.yaml
   ```

3. **Train the model** (requires GPU, run on Kaggle/Colab):
   ```bash
   python src/train.py --config configs/config.yaml --output checkpoints/
   ```

4. **Evaluate:**
   ```bash
   python src/evaluate.py --checkpoint duytanjr/vietnamese-translator
   ```

## 🛠️ Technologies

- **Python 3.11**
- **PyTorch** - Deep learning framework
- **HuggingFace Transformers** - Model loading and tokenization
- **TRL (SFTTrainer)** - Supervised fine-tuning
- **Gradio** - Web UI for demo
- **Google Gemini API** - Synthetic data generation & LLM evaluation
- **SacreBLEU / ROUGE** - Translation evaluation metrics

## 📝 Training Details

- **Epochs:** 3
- **Batch Size:** 4 (with gradient accumulation = 4, effective batch = 16)
- **Learning Rate:** 2e-5 (constant scheduler)
- **Max Sequence Length:** 512 tokens
- **Precision:** bfloat16

## 👤 Author

Nguyen Duy Tan
