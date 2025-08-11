import os
import argparse
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="google/gemma-3n-4b", help="Base model name")
    parser.add_argument("--out", type=str, default="./training/outputs/gemma3n-4b-title-lora", help="Output directory for LoRA model")
    parser.add_argument("--proc", type=str, default="./training/processed", help="Processed dataset directory")
    args = parser.parse_args()

    base = os.environ.get("BASE_MODEL", args.base)
    max_seq_len = 2048
    lora_r, lora_alpha, lora_dropout = 16, 32, 0.05

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base,
        max_seq_length = max_seq_len,
        load_in_4bit = True,
        dtype = None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_r,
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    dset = load_dataset("json", data_files={
        "train": os.path.join(args.proc, "train.jsonl"),
        "validation": os.path.join(args.proc, "val.jsonl")
    })

    # 將 (content -> title) 序列化為單輪指令格式
    def format_row(row):
        sys = "你是精煉新聞標題的助理，回覆僅輸出標題，不需解釋。"
        prompt = f"請將以下內容濃縮成新聞標題：\n{row['content']}"
        out = row["title"]
        return {
            "prompt": f"<|system|>\n{sys}\n<|user|>\n{prompt}\n<|assistant|>\n",
            "response": out
        }

    for split in ["train","validation"]:
        dset[split] = dset[split].map(format_row, remove_columns=dset[split].column_names)

    training_output_dir = args.out
    args = TrainingArguments(
        output_dir = training_output_dir,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        learning_rate = 2e-4,
        num_train_epochs = 3,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.03,
        logging_steps = 20,
        save_steps = 500,
        bf16 = True,
        fp16 = False,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dset["train"],
        eval_dataset = dset["validation"],
        dataset_text_field = None,
        max_seq_length = max_seq_len,
        packing = True,
        args = args,
        formatting_func = lambda batch: [p+ r for p, r in zip(batch["prompt"], batch["response"])],
    )

    trainer.train()
    trainer.save_model(training_output_dir)

if __name__ == "__main__":
    main()
