#!/bin/bash
set -euo pipefail

echo "--- Deploying Ollama service ---"
# 3) 部署 Ollama 服務
# 啟動 docker-compose 中定義的 ollama 服務，並在背景執行 (-d)
sudo docker compose -f docker/compose.ollama.yml --profile serve up -d ollama

echo "--- Waiting for Ollama service to be healthy (max 120 seconds) ---"
# 等待 Ollama 服務啟動並健康檢查通過
# 這裡使用一個簡單的循環來檢查服務狀態
for i in $(seq 1 24); do
  if sudo docker ps -qf name=ollama | grep -q .; then
    echo "Ollama container is running. Checking health..."
    HEALTH_STATUS=$(sudo docker inspect --format='{{.State.Health.Status}}' $(sudo docker ps -qf name=ollama))
    if [ "$HEALTH_STATUS" == "healthy" ]; then
      echo "Ollama service is healthy."
      break
    fi
  fi
  echo "Waiting for Ollama service... ($i/24)"
  sleep 5
  if [ $i -eq 24 ]; then
    echo "Ollama service did not become healthy in time. Please check logs."
    exit 1
  fi
done

echo "--- Importing model into Ollama ---"
# 4) 在主機上匯入模型
# 執行 ollama create 命令，從 /models/Modelfile 建立模型
# Modelfile 應已由 'tools' profile 執行時的 envsubst 生成
sudo docker exec -it $(sudo docker ps -qf name=ollama) bash -lc "ollama create zh-titlegen -f /models/Modelfile"

echo "--- Calling inference API ---"
# 5) 呼叫推論 API
# 替換 '【內文】台北市政府今日宣布...（此處放 content）' 為實際文章內容進行測試
curl http://localhost:11434/api/generate -d '{
  "model": "zh-titlegen",
  "prompt": "【內文】台北市政府今日宣布，為提升市民生活品質，將於明年啟動一系列智慧城市建設計畫，包括智慧交通系統升級、公共設施物聯網化以及數位服務平台整合。預計這些措施將大幅改善城市運作效率，並提供更便捷的市民服務。市府強調，所有計畫將廣泛徵詢市民意見，確保符合實際需求。"
}'

echo "Inference script finished."
