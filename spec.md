# 目標

在 **Ubuntu 24.04 + NVIDIA GPU** 上，改用 **Ollama** 作為推論服務；訓練完成後，能**一鍵上傳**微調結果到 **Hugging Face Hub**。

> **相容性更新**：Ollama 已官方提供 `gemma3:1b`（可直接 `FROM gemma3:1b`）。若需載入你在 Hugging Face 上傳的微調權重，也可改用 `FROM hf://<org>/<repo>` 指向相同模型版本。

---

## 專案結構

```
project/
├─ docker/
│  ├─ compose.ollama.yml         # docker-compose 版本
│  ├─ Dockerfile.train
│  ├─ Dockerfile.tools            # hf 上傳工具
│  └─ .env.example
├─ serving/
│  └─ ollama/
│     ├─ Modelfile                # 指向 HF repo 或本地路徑
│     └─ prompt_templates/
├─ training/
│  ├─ scripts/
│  │  ├─ run_sft.sh
│  │  ├─ export_merged_hf.py      # 合併 LoRA → HF 目錄
│  │  └─ upload_to_hf.py          # 上傳至 HF Hub
│  └─ outputs/
│     └─ final-hf/                # 匯出後 HF 目錄
└─ docs/
   └─ spec.md
```

---

## Docker Compose（Ollama 版）

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    runtime: nvidia
    environment:
      - OLLAMA_NUM_PARALLEL=2
      - OLLAMA_KEEP_ALIVE=30m
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    volumes:
      - ollama:/root/.ollama
      - ../serving/ollama:/models:ro
    ports:
      - "11434:11434"
    healthcheck:
      test: ["CMD", "ollama", "--version"]
      interval: 30s
      timeout: 5s
      retries: 3

  trainer:
    build:
      context: ..
      dockerfile: docker/Dockerfile.train
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_HOME=/opt/hf
      - WANDB_DISABLED=true
    volumes:
      - ../datasets:/workspace/datasets:ro
      - ../training:/workspace/training
      - ../prompts:/workspace/prompts:ro
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    profiles: ["train"]
    command: ["bash","/workspace/training/scripts/run_sft.sh"]

  tools:
    build:
      context: ..
      dockerfile: docker/Dockerfile.tools
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - HF_ORG=${HF_ORG}
      - HF_REPO=${HF_REPO}
    volumes:
      - ../training/outputs/final-hf:/artifacts/final-hf:ro
      - ../training/scripts:/tools/scripts:ro
    profiles: ["tools"]

volumes:
  ollama:
```

### `.env.example`

```
HF_TOKEN=hf_xxx_your_token_here
HF_ORG=your-hf-org-or-username
HF_REPO=civserv-1b-tw
```

---

## 訓練與匯出

`docker/Dockerfile.train`

```Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir transformers datasets peft bitsandbytes accelerate trl sentencepiece huggingface_hub
WORKDIR /workspace
```

`training/scripts/run_sft.sh`

```bash
set -euo pipefail
DATA=./datasets/curated/train.jsonl
VAL=./datasets/curated/val.jsonl
OUT=./training/outputs/civserv-1b-lora
BASE=gemma3-1b

python -m accelerate.commands.launch \
  training/scripts/sft.py \
  --model_name_or_path $BASE \
  --train_file $DATA \
  --validation_file $VAL \
  --output_dir $OUT \
  --bf16 \
  --max_seq_length 4096 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --logging_steps 20 \
  --save_steps 500 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --target_modules attn,mlp \
  --report_to none

python training/scripts/export_merged_hf.py \
  --base $BASE \
  --lora $OUT \
  --out ./training/outputs/final-hf
```

---

## 上傳到 Hugging Face Hub

`docker/Dockerfile.tools`

```Dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir huggingface_hub==0.24.*
WORKDIR /tools
COPY training/scripts/upload_to_hf.py /tools/upload_to_hf.py
CMD ["python","/tools/upload_to_hf.py"]
```

`training/scripts/upload_to_hf.py`

```python
import os
from huggingface_hub import HfApi, create_repo, upload_folder

token = os.environ["HF_TOKEN"]
org = os.environ["HF_ORG"]
repo = os.environ["HF_REPO"]
local_dir = "/artifacts/final-hf"

api = HfApi(token=token)
repo_id = f"{org}/{repo}"
create_repo(repo_id, token=token, repo_type="model", exist_ok=True, private=True)

upload_folder(
    repo_id=repo_id,
    folder_path=local_dir,
    path_in_repo="",
    repo_type="model",
    commit_message="Upload merged LoRA -> HF format"
)
print(f"Uploaded {local_dir} to https://huggingface.co/{repo_id}")
```

---

## Ollama 服務與 Modelfile

`serving/ollama/Modelfile`

```
FROM hf://{{HF_ORG}}/{{HF_REPO}}
PARAMETER temperature 0.6
template system "{{ .System }}"
template user "{{ .Prompt }}"
SYSTEM "$(cat prompts/system/tw_civil_servant.txt)"
```

---

## 驗收清單（Checklist）

### 資料與環境準備

* [ ] `.env` 中已正確設定 `HF_TOKEN`、`HF_ORG`、`HF_REPO`
* [ ] 已建立 `datasets/` 並放置經過清理與切分的訓練/驗證資料集
* [ ] GPU 驅動與 NVIDIA Container Toolkit 已安裝並可在容器內使用 `nvidia-smi`

### 訓練與模型匯出

* [ ] 成功執行 `docker compose --profile train up trainer`，訓練過程無錯誤中斷
* [ ] LoRA 權重已產生並存於 `/training/outputs/<model>-lora`
* [ ] 成功執行 `export_merged_hf.py` 並在 `/training/outputs/final-hf/` 生成 HF 標準結構
* [ ] 匯出的模型可在本地使用 `transformers` 成功加載並推論

### Hugging Face 上傳

* [ ] 成功執行 `docker compose --profile tools up tools` 上傳至 Hugging Face
* [ ] HF 模型頁面可正常顯示並下載（config、tokenizer、model 權重完整）

### Ollama 部署與推論

* [ ] `Modelfile` 已設定正確（FROM hf://<ORG>/<REPO> 或官方 base）
* [ ] 成功執行 `ollama create <model-name> -f Modelfile` 並無錯誤
* [ ] API `/api/generate` 可正確回覆結果
* [ ] 推論延遲（p50 ≤ 500ms，p95 ≤ 1s）符合需求

### 文件與可重現性

* [ ] 所有指令、設定檔、腳本已整理並記錄在 spec.md / README
* [ ] 專案資料夾結構完整且無冗餘暫存檔
* [ ] 從原始資料至推論部署全流程可重現
