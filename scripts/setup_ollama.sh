#!/bin/bash
# Kéo các model nhỏ cần dùng về Ollama local (chỉnh tên model theo config/config.yaml)
set -e
ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull phi4-mini
echo "Đã tải xong model."
