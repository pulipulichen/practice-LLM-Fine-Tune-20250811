# 專案名稱

Gemma 3n (4B) 以 Unsloth 微調的「內容→標題」生成模型，並以 Ollama 提供推論服務

## 需求變更（摘要）

* **模型**：改用 **Gemma 3n (4B)**。
* **訓練框架**：改用 **Unsloth**（QLoRA/LoRA，高效率微調）。
* **資料集**：`AWeirdDev/zh-tw-pts-articles-sm`（取 `content` 作輸入、`title` 作目標）。
* **任務**：將文章 `content` 摘要為 `title`（標題生成）。
* **參考作法**：`Simon-Liu/gemma2-9b-zhtw-news-title-generation-finetune`。
* **程序順序**：1) 訓練 → 2) 推論部署（Ollama）。
* **上傳**：訓練完成後，一鍵上傳到 **Hugging Face Hub**（HF Repo）。

---

## 架構總覽

* **OS/GPU**：Ubuntu 24.04 + NVIDIA（CUDA 12.x），NVIDIA Container Toolkit。
* **訓練**：Unsloth + QLoRA，從 HF 下載 `gemma-3n-4b`（或等效命名）作為 base。
* **資料處理**：將 dataset 中的 `content` 清洗為輸入、`title` 為標註目標；限制長度與標點清理。
* **評測**：ROUGE-1/2/L、BLEU-1 以及標題長度/可讀性檢查。
* **匯出**：合併 LoRA 權重並輸出 HF 標準結構（safetensors），推送至 HF。
* **推論**：Ollama 以 `FROM hf://<ORG>/<REPO>` 讀取你上傳的最終模型；提供 REST 介面。

### 目錄結構

```
project/
├─ docker/
│  ├─ compose.ollama.yml
│  ├─ Dockerfile.train.unsloth
│  ├─ Dockerfile.tools
│  └─ .env.example
├─ training/
│  ├─ scripts/
│  │  ├─ run_unsloth_sft.sh
│  │  ├─ prepare_dataset.py
│  │  ├─ export_merged_hf_unsloth.py
│  │  └─ upload_to_hf.py
│  └─ outputs/
│     └─ final-hf/
├─ serving/
│  └─ ollama/
│     ├─ Modelfile.template
│     ├─ Modelfile  # 由 Modelfile.template 替換變數後生成
│     └─ prompt_templates/
└─ docs/
   └─ spec.md
```

---

## Docker Compose（訓練→部署）

`docker/compose.ollama.yml`

```yaml
services:
  trainer:
    build:
      context: ..
      dockerfile: docker/Dockerfile.train.unsloth
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_HOME=/opt/hf
      - HF_TOKEN=${HF_TOKEN}
      - WANDB_DISABLED=true
    volumes:
      - ../training:/workspace/training
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    command: ["bash","/workspace/training/scripts/run_unsloth_sft.sh"]
    profiles: ["train"]

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
      - ../serving/ollama:/serving_ollama:rw # 新增：讓 tools 服務可以寫入 Modelfile
    command: ["bash", "-c", "envsubst < /serving_ollama/Modelfile.template > /serving_ollama/Modelfile && python /tools/upload_to_hf.py"] # 修改：先執行 envsubst 再上傳
    profiles: ["tools"]

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
      - ../serving/ollama:/models:rw
    ports: ["11434:11434"]
    healthcheck:
      test: ["CMD", "ollama", "--version"]
      interval: 30s
      timeout: 5s
      retries: 3
    profiles: ["serve"]

volumes:
  ollama:
```

### `.env.example`

```
HF_TOKEN=hf_xxx_your_token_here
HF_ORG=your-hf-org-or-username
HF_REPO=gemma3n-4b-zh-titlegen
```

---

## 訓練（Unsloth + QLoRA）

`docker/Dockerfile.train.unsloth`

```Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu24.04
RUN apt-get update && apt-get install -y python3-pip git && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir unsloth transformers datasets peft bitsandbytes accelerate sentencepiece evaluate rouge-score huggingface_hub
WORKDIR /workspace
```

`training/scripts/prepare_dataset.py`（摘要）

```python
# 下載 AWeirdDev/zh-tw-pts-articles-sm，抽取 (content -> title)
# 清理：移除多餘空白、控制最大長度（content 例如 1536~2048 tokens；title 64 tokens）
# 產生 train/val/test 的 JSONL 或 HF DatasetDict 儲存於 /workspace/training/processed/
```

`training/scripts/run_unsloth_sft.sh`

```bash
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
```

`training/scripts/export_merged_hf_unsloth.py`（摘要）

```python
# 使用 Unsloth / PEFT 將 LoRA 權重 merge_and_unload，
# 寫出 HF 結構（config.json / tokenizer.* / model.safetensors）到 final-hf。
```

`training/scripts/train_unsloth_sft.py`（摘要）

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
import os
import argparse

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
```

### 評測（摘錄）

* 計算 ROUGE-1/2/L、BLEU-1；
* 長度約束：標題長度（字數或 tokens）區間，如 8–30 字；
* 字詞品質：移除表情符號與多餘標點，避免「摘要式贅詞」。

---

## 上傳至 Hugging Face

`docker/Dockerfile.tools`

```Dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y gettext-base && rm -rf /var/lib/apt/lists/* # 新增：安裝 gettext-base 以提供 envsubst
RUN pip install --no-cache-dir huggingface_hub==0.24.*
WORKDIR /tools
COPY training/scripts/upload_to_hf.py /tools/upload_to_hf.py
# CMD ["python","/tools/upload_to_hf.py"] # 已由 docker-compose.yml 中的 command 覆寫
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
    commit_message="Upload merged LoRA -> HF format (Gemma 3n 4B title generation)"
)
print(f"Uploaded {local_dir} to https://huggingface.co/{repo_id}")
```

**執行順序**

```bash
# 1) 訓練
docker compose -f docker/compose.ollama.yml --profile train up --build trainer

# 2) 上傳 HF
docker compose -f docker/compose.ollama.yml --profile tools up --build tools
```

---

## 推論部署（Ollama）

`serving/ollama/Modelfile.template`

```
# 若使用你上傳到 HF 的模型
FROM hf://{{HF_ORG}}/{{HF_REPO}}
PARAMETER temperature 0.3
PARAMETER num_ctx 4096
TEMPLATE """
{{ if .System }}<|system|>
{{ .System }}
{{ end }}
<|user|>
請將以下內容濃縮成一則新聞標題（僅輸出標題）：
{{ .Prompt }}
<|assistant|>
"""
SYSTEM "你是精煉新聞標題的助理，回覆僅輸出標題，不需解釋。"
```

**建立與測試**

```bash
# 3) 部署（可在訓練完成後進行）
sudo docker compose -f docker/compose.ollama.yml --profile serve up -d ollama

# 4) 在主機上匯入模型
# 透過 envsubst 替換 Modelfile 中的變數，並建立模型
sudo docker exec -it $(docker ps -qf name=ollama) bash -lc "ollama create zh-titlegen -f /models/Modelfile"

# 5) 呼叫推論
curl http://localhost:11434/api/generate -d '{
  "model": "zh-titlegen",
  "prompt": "【內文】台北市政府今日宣布...（此處放 content）"
}'
```

> 若暫時以官方 base：可將 `FROM hf://...` 改為 `FROM gemma3n:4b`，但需注意你的微調權重尚未整合；推論品質會不同。

---

## 參考設定與最佳化

* **Unsloth 建議**：`load_in_4bit=True`、梯度累積 8–16、R=16、Alpha=32、Dropout=0.05；視 VRAM 調整。
* **長度控制**：推論時限制最大輸出 tokens，例如 `max_tokens=48`，避免超長標題。
* **資料增強**：

  * 過長內文先做句子層級抽取（Lead-3 / TextRank）再餵入模型；
  * 移除括號內贅字與來源尾註；
  * 標題平衡：過短/過長樣本再採樣。
* **評測門檻（建議）**：ROUGE-L ≥ 0.30、BLEU-1 ≥ 0.45、無解釋率（僅輸出標題）≥ 0.98。

---

## 驗收清單（Checklist）

### 資料與環境準備

* [ ] `.env` 中已正確設定 `HF_TOKEN`、`HF_ORG`、`HF_REPO`
* [ ] **已確認 `google/gemma-3n-4b` 或等效模型名稱在 Hugging Face Hub 上為有效且可存取**
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

---

## 備註

* 參考 `Simon-Liu/gemma2-9b-zhtw-news-title-generation-finetune` 之資料格式與 Prompt 風格，但已調整為 Gemma 3n 4B 與 Unsloth 管線。
* 若未能直接取得 `google/gemma-3n-4b` 名稱，請改用實際 HF 模型 repo 名稱或本地鏡像路徑。
