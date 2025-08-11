# Docker 目錄待辦事項

## compose.ollama.yml
- [ ] 確保 `trainer` 服務的 `build`、`environment`、`volumes`、`deploy` 和 `command` 設定正確。
- [ ] 確保 `tools` 服務的 `build`、`environment`、`volumes` 和 `command` 設定正確，特別是 `envsubst` 和 `upload_to_hf.py` 的執行順序。
- [ ] 確保 `ollama` 服務的 `image`、`runtime`、`environment`、`deploy`、`volumes`、`ports` 和 `healthcheck` 設定正確。
- [ ] 確保 `ollama` 服務的 `volumes` 中 `../serving/ollama:/models:rw` 設定正確，以便 Ollama 服務可以讀取 Modelfile。
- [ ] 確保 `volumes` 區塊中的 `ollama` 命名卷已定義。

## Dockerfile.train.unsloth
- [ ] 確保基礎映像檔 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04` 正確。
- [ ] 確保必要的套件 `python3-pip` 和 `git` 已安裝。
- [ ] 確保 PyTorch 及其相關套件已正確安裝，並指定 CUDA 版本。
- [ ] 確保 `unsloth`、`transformers`、`datasets`、`peft`、`bitsandbytes`、`accelerate`、`sentencepiece`、`evaluate`、`rouge-score` 和 `huggingface_hub` 已安裝。
- [ ] 確保工作目錄設定為 `/workspace`。

## Dockerfile.tools
- [ ] 確保基礎映像檔 `python:3.11-slim` 正確。
- [ ] 確保 `gettext-base` 已安裝以提供 `envsubst`。
- [ ] 確保 `huggingface_hub==0.24.*` 已安裝。
- [ ] 確保工作目錄設定為 `/tools`。
- [ ] 確保 `training/scripts/upload_to_hf.py` 已複製到 `/tools/upload_to_hf.py`。

## .env.example
- [ ] 確保 `HF_TOKEN`、`HF_ORG` 和 `HF_REPO` 的範例值已提供。
