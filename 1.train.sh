#!/bin/bash
set -euo pipefail

echo "Starting training process..."
docker compose -f docker/compose.ollama.yml --profile train up --build trainer

echo "Training process completed."
