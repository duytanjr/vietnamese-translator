"""
Vietnamese AI Translator - Gradio Web App
==========================================
A fine-tuned Qwen 3 (0.6B) model for English-to-Vietnamese translation.
Deploy this on HuggingFace Spaces or run locally.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# === Configuration ===
MODEL_PATH = "duytanjr/vietnamese-translator"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Load Model ===
print(f"Loading model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
model.eval()
print("Model loaded!")


def translate_en_to_vi(english_text: str) -> tuple[str, str]:
    """
    Translate English text to Vietnamese.
    Returns: (translation, latency_info)
    """
    if not english_text.strip():
        return "", ""

    start_time = time.time()

    # Format as chat message
    messages = [
        {"role": "user", "content": f"translate English to Vietnamese:\n\n{english_text}"}
    ]

    # Tokenize
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text_input, return_tensors="pt", padding=True,
        truncation=True, return_attention_mask=True
    ).to(DEVICE)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,  # Deterministic output
        )

    # Decode only the generated part
    response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    translation = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # Remove <think> tags if present (Qwen thinking mode)
    if "<think>" in translation:
        import re
        translation = re.sub(r"<think>.*?</think>", "", translation, flags=re.DOTALL).strip()

    elapsed = time.time() - start_time
    latency_info = f"⏱️ {elapsed:.1f}s"

    return translation, latency_info


# === Gradio Interface ===
with gr.Blocks(
    title="Vietnamese AI Translator",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # 🌐 Vietnamese AI Translator (EN → VI)

        Fine-tuned **Qwen 3 (0.6B)** for English-to-Vietnamese translation.
        Trained on 60K+ translation pairs with synthetic data augmentation.
        """
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="English Text",
                placeholder="Enter English text to translate...",
                lines=5,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Vietnamese Translation",
                lines=5,
                interactive=False,
            )
            latency_text = gr.Textbox(
                label="Latency",
                interactive=False,
                max_lines=1,
            )

    # Examples
    gr.Examples(
        examples=[
            "Hello, how are you today?",
            "Artificial intelligence is transforming the world.",
            "The cat is sleeping on the couch.",
            "I need to finish this project by next week.",
            "Vietnam is a beautiful country with rich culture.",
        ],
        inputs=input_text,
    )

    # Event handlers
    submit_btn.click(
        fn=translate_en_to_vi,
        inputs=input_text,
        outputs=[output_text, latency_text],
    )
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[input_text, output_text, latency_text],
    )

if __name__ == "__main__":
    demo.launch(share=False)
