#!/usr/bin/env bash
# Install llama-server.service for the user who runs this (via sudo), using this repo checkout.
# Usage: sudo -E mi50/systemd/install-service.sh   Env: MODELS (default ~/models), HOST (default 0.0.0.0), PORT (8080)
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
U=${SUDO_USER:?run via sudo from your normal user}
H=$(getent passwd "$U" | cut -d: -f6)
REPO=$(cd "$(dirname "$0")/../.." && pwd)
MODELS=${MODELS:-$H/models}
[ -x "$REPO/build-mi50/bin/llama-server" ] || { echo "build first: mi50/build.sh"; exit 1; }
[ -f "$MODELS/Qwen3.8-27B-Q8_0.gguf" ] || { echo "models missing in $MODELS: mi50/download-models.sh"; exit 1; }
sed -e "s#@USER@#$U#g" -e "s#@REPO@#$REPO#g" -e "s#@MODELS@#$MODELS#g" \
    -e "s#@HOST@#${HOST:-0.0.0.0}#g" -e "s#@PORT@#${PORT:-8080}#g" \
    "$(dirname "$0")/llama-server.service.in" > /etc/systemd/system/llama-server.service
systemctl daemon-reload
systemctl enable --now llama-server
echo "llama-server.service installed and started (model load takes 1-2 min): curl -s 127.0.0.1:${PORT:-8080}/health"
echo "Port ${PORT:-8080} is plain HTTP with no auth: firewall it to your LAN (e.g. sudo ufw allow from <your LAN subnet> to any port ${PORT:-8080} proto tcp)."
