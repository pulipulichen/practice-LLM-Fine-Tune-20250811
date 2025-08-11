# 專案總覽與進度

## 專案名稱
- [ ] Gemma 3n (4B) 以 Unsloth 微調的「內容→標題」生成模型，並以 Ollama 提供推論服務

## 需求變更（摘要）
- [ ] 模型：改用 Gemma 3n (4B)
- [ ] 訓練框架：改用 Unsloth（QLoRA/LoRA，高效率微調）
- [ ] 資料集：AWeirdDev/zh-tw-pts-articles-sm（取 content 作輸入、title 作目標）
- [ ] 任務：將文章 content 摘要為 title（標題生成）
- [ ] 參考作法：Simon-Liu/gemma2-9b-zhtw-news-title-generation-finetune
- [ ] 程序順序：1) 訓練 → 2) 推論部署（Ollama）
- [ ] 上傳：訓練完成後，一鍵上傳到 Hugging Face Hub（HF Repo）

## 架構總覽
- [ ] OS/GPU：Ubuntu 24.04 + NVIDIA（CUDA 12.x），NVIDIA Container Toolkit
- [ ] 訓練：Unsloth + QLoRA，從 HF 下載 gemma-3n-4b（或等效命名）作為 base
- [ ] 資料處理：將 dataset 中的 content 清洗為輸入、title 為標註目標；限制長度與標點清理
- [ ] 評測：ROUGE-1/2/L、BLEU-1 以及標題長度/可讀性檢查
- [ ] 匯出：合併 LoRA 權重並輸出 HF 標準結構（safetensors），推送至 HF
- [ ] 推論：Ollama 以 FROM hf://<ORG>/<REPO> 讀取你上傳的最終模型；提供 REST 介面

## Docker Compose（訓練→部署）
- [ ] 確保 docker/compose.ollama.yml 配置正確
- [ ] 確保 .env.example 提供正確的環境變數範例

## 訓練（Unsloth + QLoRA）
- [ ] 確保 docker/Dockerfile.train.unsloth 配置正確
- [ ] 確保 training/scripts/prepare_dataset.py 實現資料處理邏輯
- [ ] 確保 training/scripts/run_unsloth_sft.sh 執行訓練流程
- [ ] 確保 training/scripts/export_merged_hf_unsloth.py 實現模型匯出邏輯
- [ ] 確保 training/scripts/train_unsloth_sft.py 實現模型訓練邏輯

## 上傳至 Hugging Face
- [ ] 確保 docker/Dockerfile.tools 配置正確
- [ ] 確保 training/scripts/upload_to_hf.py 實現模型上傳邏輯

## 推論部署（Ollama）
- [ ] 確保 serving/ollama/Modelfile.template 配置正確
- [ ] 確保 Modelfile 能由 Modelfile.template 替換變數後生成
- [ ] 確保 prompt_templates/ 包含推論所需的 prompt 模板

## 參考設定與最佳化
- [ ] 考慮 Unsloth 建議的參數設定
- [ ] 實施長度控制以避免超長標題
- [ ] 考慮資料增強策略
- [ ] 設定評測門檻

## 驗收清單
- [ ] 資料與環境準備：
    - [ ] .env 中已正確設定 HF_TOKEN、HF_ORG、HF_REPO
    - [ ] 已確認 google/gemma-3n-4b 或等效模型名稱在 Hugging Face Hub 上為有效且可存取
    - [ ] 已建立 datasets/ 並放置經過清理與切分的訓練/驗證資料集
    - [ ] GPU 驅動與 NVIDIA Container Toolkit 已安裝並可在容器內使用 nvidia-smi
- [ ] 訓練與模型匯出：
    - [ ] 成功執行 docker compose --profile train up trainer，訓練過程無錯誤中斷
    - [ ] LoRA 權重已產生並存於 /training/outputs/<model>-lora
    - [ ] 成功執行 export_merged_hf.py 並在 /training/outputs/final-hf/ 生成 HF 標準結構
    - [ ] 匯出的模型可在本地使用 transformers 成功加載並推論
- [ ] Hugging Face 上傳：
    - [ ] 成功執行 docker compose --profile tools up tools 上傳至 Hugging Face
    - [ ] HF 模型頁面可正常顯示並下載（config、tokenizer、model 權重完整）
- [ ] Ollama 部署與推論：
    - [ ] Modelfile 已設定正確（FROM hf://<ORG>/<REPO> 或官方 base）
    - [ ] 成功執行 ollama create <model-name> -f Modelfile 並無錯誤
    - [ ] API /api/generate 可正確回覆結果
    - [ ] 推論延遲（p50 ≤ 500ms，p95 ≤ 1s）符合需求
- [ ] 文件與可重現性：
    - [ ] 所有指令、設定檔、腳本已整理並記錄在 spec.md / README
    - [ ] 專案資料夾結構完整且無冗餘暫存檔
    - [ ] 從原始資料至推論部署全流程可重現
