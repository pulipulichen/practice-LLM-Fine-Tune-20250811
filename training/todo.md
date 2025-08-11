# Training 目錄待辦事項

## scripts/
- [ ] **run_unsloth_sft.sh**
    - [ ] 確保腳本能正確設定環境變數 `BASE`、`OUT`、`PROC`。
    - [ ] 確保腳本能正確呼叫 `prepare_dataset.py`。
    - [ ] 確保腳本能正確呼叫 `train_unsloth_sft.py`。
    - [ ] 確保腳本能正確呼叫 `export_merged_hf_unsloth.py`。
    - [ ] 確保腳本在執行過程中無錯誤中斷。
- [ ] **prepare_dataset.py**
    - [ ] 實現從 `AWeirdDev/zh-tw-pts-articles-sm` 下載資料集的功能。
    - [ ] 實現抽取 `content` 作為輸入、`title` 作為目標的功能。
    - [ ] 實現資料清洗功能：移除多餘空白、控制最大長度（content 1536~2048 tokens；title 64 tokens）、標點清理。
    - [ ] 實現將處理後的資料產生 `train/val/test` 的 JSONL 或 HF DatasetDict 格式，並儲存於 `/workspace/training/processed/`。
- [ ] **export_merged_hf_unsloth.py**
    - [ ] 實現使用 Unsloth / PEFT 將 LoRA 權重 `merge_and_unload` 的功能。
    - [ ] 實現將合併後的模型寫出 HF 標準結構（`config.json` / `tokenizer.*` / `model.safetensors`）到 `final-hf` 目錄。
- [ ] **upload_to_hf.py**
    - [ ] 確保腳本能從環境變數中讀取 `HF_TOKEN`、`HF_ORG` 和 `HF_REPO`。
    - [ ] 確保腳本能使用 `HfApi` 建立或更新 Hugging Face Repo。
    - [ ] 確保腳本能將本地目錄 `/artifacts/final-hf` 中的模型檔案上傳至 Hugging Face Hub。
    - [ ] 確保上傳過程的 commit message 正確。
- [ ] **train_unsloth_sft.py**
    - [ ] 確保腳本能正確解析命令行參數 `base`、`out`、`proc`。
    - [ ] 確保能使用 `FastLanguageModel.from_pretrained` 加載基礎模型。
    - [ ] 確保能使用 `FastLanguageModel.get_peft_model` 配置 LoRA 參數。
    - [ ] 確保能從指定路徑加載處理後的資料集。
    - [ ] 實現 `format_row` 函數，將 `(content -> title)` 序列化為單輪指令格式。
    - [ ] 確保 `TrainingArguments` 配置正確，包括 `output_dir`、`batch_size`、`learning_rate`、`epochs` 等。
    - [ ] 確保 `SFTTrainer` 配置正確，包括模型、tokenizer、資料集、`max_seq_length`、`packing` 等。
    - [ ] 確保訓練過程能正常執行並保存模型。

## outputs/
- [ ] **final-hf/**
    - [ ] 確保此目錄用於存放最終合併後的 HF 標準模型結構。
