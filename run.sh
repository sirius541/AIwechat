#!/bin/bash
set -e
cd "$(dirname "$0")"

PORT=${PORT:-8860}
# Use nougat env which has fastapi/uvicorn/httpx
PYTHON=/data/wangyuxuan/anaconda3/envs/nougat/bin/python

echo "======================================"
echo "  AI 微信 - 多模型群聊"
echo "  http://localhost:${PORT}"
echo "======================================"

# Install zhipuai if missing
$PYTHON -c "import zhipuai" 2>/dev/null || {
  echo "[*] 安装 zhipuai..."
  /data/wangyuxuan/anaconda3/envs/nougat/bin/pip install zhipuai -q
}

echo "[*] 启动服务..."
PORT=$PORT $PYTHON app.py
