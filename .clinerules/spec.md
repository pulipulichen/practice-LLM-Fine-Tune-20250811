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

---

## Docker Compose（訓練→部署）

`docker/compose.ollama.yml`
```

### `.env.example`

### `.env.`

請複製 `.env.example` 為 `.env` 並填入你的 Hugging Face Token、組織/使用者名稱與模型儲存庫名稱。

```bash
cp .env.example .env
```

---

## 訓練（Unsloth + QLoRA）

`docker/Dockerfile.train.unsloth`

`training/scripts/prepare_dataset.py`

`training/scripts/run_unsloth_sft.sh`

`training/scripts/export_merged_hf_unsloth.py`

`training/scripts/train_unsloth_sft.py`

### 評測（摘錄）

* 計算 ROUGE-1/2/L、BLEU-1；
* 長度約束：標題長度（字數或 tokens）區間，如 8–30 字；
* 字詞品質：移除表情符號與多餘標點，避免「摘要式贅詞」。

---

## 上傳至 Hugging Face

`docker/Dockerfile.tools`

`training/scripts/upload_to_hf.py`

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
* [ ] 已新增 `.gitignore` 規則，排除 `.env`
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
