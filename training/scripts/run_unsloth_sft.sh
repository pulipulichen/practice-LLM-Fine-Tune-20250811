set -euo pipefail
BASE="google/gemma-3n-4b"   # 依實際 HF 名稱調整
OUT="./training/outputs/gemma3n-4b-title-lora"
PROC="./training/processed"  # 由 prepare_dataset.py 產出

python training/scripts/prepare_dataset.py --out $PROC

python training/scripts/train_unsloth_sft.py \
  --base $BASE \
  --out $OUT \
  --proc $PROC

# 合併 LoRA -> HF 目錄
python training/scripts/export_merged_hf_unsloth.py \
  --base $BASE \
  --lora $OUT \
  --out ./training/outputs/final-hf
