import argparse
import os
from typing import List, Optional

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_dir: str, hf_token: Optional[str] = None, base_model: str = "meta-llama/Llama-3.2-1B"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base, model_dir, is_trainable=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Inference for llama3.2_1b_hh_dpo (LoRA) checkpoint.")
    parser.add_argument(
        "--model_dir",
        default="models/llama3.2_1b_hh_dpo",
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--base_model",
        default="meta-llama/Llama-3.2-1B",
        help="Base model name/path to load before applying the adapter.",
    )
    parser.add_argument("--csv_path", help="CSV file containing prompts.")
    parser.add_argument("--csv_column", default="x", help="Column name holding prompts.")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of random prompts to sample from CSV.")
    parser.add_argument("--seed", type=int, help="Random seed for sampling.")
    parser.add_argument(
        "--input_file",
        help="Path to a text file containing the full prompt block.",
    )
    parser.add_argument(
        "--input_text",
        help="Prompt text. If set, overrides --input_file.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output_csv", help="Optional path to save prompt/response pairs.")
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token if required.",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir, args.hf_token, args.base_model)

    prompts: List[str] = []
    if args.input_text:
        prompts = [args.input_text]
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            prompts = [f.read()]
    elif args.csv_path:
        df = pd.read_csv(args.csv_path)
        if args.csv_column not in df.columns:
            raise SystemExit(f"Column '{args.csv_column}' not found in {args.csv_path}.")
        series = df[args.csv_column].dropna().astype(str)
        prompts = series.sample(
            n=min(args.sample_size, len(series)),
            random_state=args.seed,
        ).tolist()
    else:
        raise SystemExit("Provide --input_text, --input_file, or --csv_path.")

    outputs = []
    for prompt in prompts:
        full = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        completion = full[len(prompt) :]
        outputs.append((prompt, completion))
        print("=== Prompt ===")
        print(prompt.strip())
        print("\n=== Completion ===")
        print(completion.strip())
        print("-" * 40)

    if args.output_csv:
        out_df = pd.DataFrame(outputs, columns=["prompt", "completion"])
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved {len(outputs)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
