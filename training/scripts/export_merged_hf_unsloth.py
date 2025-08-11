# training/scripts/export_merged_hf_unsloth.py

# 使用 Unsloth / PEFT 將 LoRA 權重 merge_and_unload，
# 寫出 HF 結構（config.json / tokenizer.* / model.safetensors）到 final-hf。

import argparse
import os
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="google/gemma-3n-4b", help="Base model name")
    parser.add_argument("--lora", type=str, default="./training/outputs/gemma3n-4b-title-lora", help="Path to LoRA model")
    parser.add_argument("--out", type=str, default="./training/outputs/final-hf", help="Output directory for merged HF model")
    args = parser.parse_args()

    # Load the base model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.base,
        max_seq_length = 2048, # This should match the training script's max_seq_len
        load_in_4bit = True,
        dtype = None,
    )

    # Load the LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # This should match the training script's lora_r
        lora_alpha = 32, # This should match the training script's lora_alpha
        lora_dropout = 0.05, # This should match the training script's lora_dropout
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model.load_adapter(args.lora)

    # Merge LoRA weights and save
    model.save_pretrained_merged(args.out, tokenizer, save_method = "safetensors")
    print(f"Merged model saved to {args.out}")

if __name__ == "__main__":
    main()
