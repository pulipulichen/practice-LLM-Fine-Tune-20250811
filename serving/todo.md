# Serving 目錄待辦事項

## ollama/
- [x] **Modelfile.template**
    - [x] 確保模板中的 `FROM hf://{{HF_ORG}}/{{HF_REPO}}` 或 `FROM gemma3n:4b` 設定正確。
    - [x] 確保 `PARAMETER temperature` 和 `num_ctx` 設定符合需求。
    - [x] 確保 `TEMPLATE` 定義的 Prompt 格式與訓練時的格式一致。
    - [x] 確保 `SYSTEM` 訊息定義正確。
- [ ] **Modelfile**
    - [ ] 確保此檔案能由 `Modelfile.template` 替換變數後自動生成。
- [x] **prompt_templates/**
    - [x] 確保此目錄包含推論所需的任何額外 Prompt 模板（如果有的話）。
