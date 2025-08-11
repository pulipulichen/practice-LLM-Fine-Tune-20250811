# Training 目錄待辦事項

## scripts/
- [x] **run_unsloth_sft.sh**
    - [x] 確保腳本能正確設定環境變數 `BASE`、`OUT`、`PROC`。
    - [x] 確保腳本能正確呼叫 `prepare_dataset.py`。
    - [x] 確保腳本能正確呼叫 `train_unsloth_sft.py`。
    - [x] 確保腳本能正確呼叫 `export_merged_hf_unsloth.py`。
    - [ ] 確保腳本在執行過程中無錯誤中斷。
- [x] **prepare_dataset.py**
    - [x] 實現從 `AWeirdDev/zh-tw-pts-articles-sm` 下載資料集的功能。（目前為模擬創建空文件）
    - [x] 實現抽取 `content` 作為輸入、`title` 作為目標的功能。（目前為模擬創建空文件）
    - [x] 實現資料清洗功能：移除多餘空白、控制最大長度（content 1536~2048 tokens；title 64 tokens）、標點清理。（目前為模擬創建空文件）
    - [x] 實現將處理後的資料產生 `train/val/test` 的 JSONL 或 HF DatasetDict 格式，並儲存於 `/workspace/training/processed/`。（目前為模擬創建空文件）
- [x] **export_merged_hf_unsloth.py**
    - [x] 實現使用 Unsloth / PEFT 將 LoRA 權重 `merge_and_unload` 的功能。
    - [x] 實現將合併後的模型寫出 HF 標準結構（`config.json` / `tokenizer.*` / `model.safetensors`）到 `final-hf` 目錄。
- [x] **upload_to_hf.py**
    - [x] 確保腳本能從環境變數中讀取 `HF_TOKEN`、`HF_ORG` 和 `HF_REPO`。
    - [x] 確保腳本能使用 `HfApi` 建立或更新 Hugging Face Repo。
    - [x] 確保腳本能將本地目錄 `/artifacts/final-hf` 中的模型檔案上傳至 Hugging Face Hub。
    - [x] 確保上傳過程的 commit message 正確。
- [x] **train_unsloth_sft.py**
    - [x] 確保腳本能正確解析命令行參數 `base`、`out`、`proc`。
    - [x] 確保能使用 `FastLanguageModel.from_pretrained` 加載基礎模型。
    - [x] 確保能使用 `FastLanguageModel.get_peft_model` 配置 LoRA 參數。
    - [x] 確保能從指定路徑加載處理後的資料集。
    - [x] 實現 `format_row` 函數，將 `(content -> title)` 序列化為單輪指令格式。
    - [x] 確保 `TrainingArguments` 配置正確，包括 `output_dir`、`batch_size`、`learning_rate`、`epochs` 等。
    - [x] 確保 `SFTTrainer` 配置正確，包括模型、tokenizer、資料集、`max_seq_length`、`packing` 等。
    - [x] 確保訓練過程能正常執行並保存模型。

## outputs/
- [x] **final-hf/**
    - [x] 確保此目錄用於存放最終合併後的 HF 標準模型結構。
