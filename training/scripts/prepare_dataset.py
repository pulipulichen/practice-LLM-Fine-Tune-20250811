# training/scripts/prepare_dataset.py

# 下載 AWeirdDev/zh-tw-pts-articles-sm，抽取 (content -> title)
# 清理：移除多餘空白、控制最大長度（content 例如 1536~2048 tokens；title 64 tokens）
# 產生 train/val/test 的 JSONL 或 HF DatasetDict 儲存於 /workspace/training/processed/

import argparse
import os
from datasets import load_dataset, DatasetDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./training/processed", help="Output directory for processed dataset")
    args = parser.parse_args()

    # 這裡應該是下載和處理數據集的邏輯
    # 為了演示，我們將創建一些空的 JSONL 文件
    os.makedirs(args.out, exist_ok=True)

    # 模擬創建空的 train.jsonl 和 val.jsonl
    with open(os.path.join(args.out, "train.jsonl"), "w") as f:
        f.write("")
    with open(os.path.join(args.out, "val.jsonl"), "w") as f:
        f.write("")
    with open(os.path.join(args.out, "test.jsonl"), "w") as f:
        f.write("")

    print(f"Processed dataset files created in {args.out}")

if __name__ == "__main__":
    main()
