#!/usr/bin/env bash
# 启动影剧漫 Agentic RAG Web 服务
# 用法: bash scripts/start_server.sh [port]   （默认 8080）
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${1:-${MEDIA_RAG_PORT:-8080}}"
echo "==> 启动 Web 服务: http://127.0.0.1:${PORT}"
exec .venv/bin/uvicorn app.server:app --host 127.0.0.1 --port "$PORT"
