# Extras around the server (optional)

The reference box runs a few small services next to `llama-server`. None of them is needed for the speeds in this
repo, and their code is not part of it. This page lists what they are and how each is set up, so the whole stack can
be rebuilt. Replace `<LAN-IP>` and `<your LAN subnet>` with your own values. All ports are firewalled to the LAN.

## Overview

| Service | What it does | Listens on | Depends on |
|---|---|---|---|
| `llama-server` | the model (this repo) | `:8080` plain HTTP | - |
| TLS front for the web UI | HTTPS for browsers (microphone access needs HTTPS) | `:8443` -> `:8080` | `socat`, a certificate |
| Claude Code proxy | lets Claude Code talk to the local model | `127.0.0.1:8082` -> `:8080` | Python 3 |
| whisper.cpp server | speech to text for the web UI microphone | `127.0.0.1:8091`, TLS front on `:8081` | whisper.cpp |
| SearXNG | private web search backend | `127.0.0.1:8888` | Docker |
| Search tools MCP server | web search, page fetch and a few small tools for the model | stdio for Claude Code; HTTPS + SSE on `:8444` for the web UI | SearXNG, Python 3 |
| Memory MCP server | long-term notes for the web UI | `127.0.0.1:9510`, TLS front on `:8445` | `mcp-proxy`, `mempalace` |
| Monitoring | `glances` web view | `:61208` | - |

## 1. TLS certificate and fronts

Browsers only allow the microphone and cross-port tool calls over HTTPS, so every browser-facing port gets a TLS
front. One self-signed certificate serves all of them:

```bash
mkdir -p ~/tls && cd ~/tls
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout llama.key -out llama.crt \
  -subj "/CN=llama-box" -addext "subjectAltName=IP:<LAN-IP>,IP:127.0.0.1,DNS:localhost"
chmod 600 llama.key
```

Each front is one `socat` unit, for example for the web UI:

```ini
[Unit]
After=llama-server.service
PartOf=llama-server.service
[Service]
ExecStart=/usr/bin/socat OPENSSL-LISTEN:8443,reuseaddr,fork,cert=%h/tls/llama.crt,key=%h/tls/llama.key,verify=0 TCP:127.0.0.1:8080
Restart=always
[Install]
WantedBy=multi-user.target
```

- Never commit `llama.key`. Make a new certificate on a new box.
- The browser must accept the certificate once per port.
- `PartOf=` restarts a front together with its backend, but does not start a stopped one: after
  `systemctl stop llama-server`, start the fronts again by hand.

## 2. Claude Code with the local model

Claude Code sends `system` entries inside `messages[]`, which the Qwen3.8 chat template rejects. A small HTTP proxy
on `127.0.0.1:8082` moves those entries into the top-level `system` field and forwards everything else unchanged to
`:8080`. It also strips regex `pattern` fields that crash the server's JSON-schema converter (look-ahead patterns such
as `(?!...)`).

- Run it as a systemd unit with `After=` and `PartOf=llama-server.service`.
- Start Claude Code through a small wrapper script:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082   # the proxy
export ANTHROPIC_AUTH_TOKEN=local-llama           # any value; llama.cpp ignores it
export ANTHROPIC_MODEL=Qwen3.8-27B                # must match the server's --alias
export ANTHROPIC_SMALL_FAST_MODEL=Qwen3.8-27B     # keep the background model local too
exec claude --model Qwen3.8-27B "$@"
```

## 3. Voice input (whisper.cpp)

```bash
git clone https://github.com/ggerganov/whisper.cpp ~/whisper.cpp && cd ~/whisper.cpp
git checkout v1.9.2
cmake -B build -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_SERVER=ON
cmake --build build -j"$(nproc)" --target whisper-server
bash models/download-ggml-model.sh base.en
```

- Unit: `whisper-server -m models/ggml-base.en.bin --host 127.0.0.1 --port 8091 -t 8 --convert`, plus a TLS front
  on `:8081`. `--convert` needs `ffmpeg` (the browser records webm/ogg, whisper wants wav).
- A CPU build is enough: `base.en` transcribes a sentence in well under a second.
- The stock llama.cpp web UI has no microphone button. The reference box serves a patched copy of the web UI with
  `--path <dir>` (a record button that posts the audio to `https://<LAN-IP>:8081/inference` and puts the text into
  the prompt box). Keep `--path` when you change the server binary, or the button disappears.

## 4. Web search for the model (SearXNG + MCP)

```yaml
# ~/searxng/docker-compose.yml
services:
  searxng:
    image: searxng/searxng   # pin the image digest you tested
    restart: unless-stopped
    ports: ["127.0.0.1:8888:8080"]
    volumes: [searxng_config:/etc/searxng]
volumes:
  searxng_config:
```

- Enable the JSON output format once, then restart the container:
  `docker exec searxng sed -i '/^  formats:/a\    - json' /etc/searxng/settings.yml && docker restart searxng`
- The MCP server is a small Python program with the tools in one shared file and two entry points:
  - **stdio** for Claude Code: `claude mcp add -s user searxng -- python3 ~/searxng-mcp/searxng_mcp.py`
  - **HTTPS + SSE** on `:8444` for the web UI. Set its public base URL to `https://<LAN-IP>:8444` and allow the web
    UI's origin (`https://<LAN-IP>:8443`) for CORS.
- Tools on the reference box: search, fetch URL, weather, time, currency, Wikipedia summary, save file (files are
  served from an outbox folder, cleaned by a daily timer).
- In the web UI, add the server under MCP settings with the `:8444` URL.

## 5. Memory for the web UI (MemPalace MCP)

```bash
uv tool install mcp-proxy==0.12.0 --with 'mcp<2'    # 0.12.0 breaks with mcp 2.x
uv tool install mempalace==3.10.0
```

- Unit: `mcp-proxy --host 127.0.0.1 --port 9510 --allow-origin https://<LAN-IP>:8443 -- mempalace-light-mcp`,
  plus a TLS front on `:8445`. The light server exposes only a few tools, which keeps the model's context small.
- Run the unit sandboxed: `ProtectSystem=strict`, `ProtectHome=tmpfs`, and `BindPaths=` for only its own folders
  (`~/.mempalace`, `~/.config/mempalace`, `~/.cache/chroma`, its working folder). The tool can index whole folders
  if asked, and should not be able to reach the rest of the home directory.

## 6. Firewall

```bash
LAN="<your LAN subnet>"
sudo ufw default deny incoming && sudo ufw default allow outgoing
for p in 22 8080 8443 8081 8444 8445 61208; do sudo ufw allow from "$LAN" to any port $p proto tcp; done
sudo ufw enable
```

## Start order and checks

```bash
sudo systemctl start gpu-fan llama-server
sudo systemctl start llama-tls claude-local-proxy whisper-server whisper-tls searxng-mcp searxng-mcp-tls \
  mempalace-mcp mempalace-mcp-tls
curl -s 127.0.0.1:8080/health          # model
curl -sk https://127.0.0.1:8443/health # TLS front
curl -s 127.0.0.1:8888/search?q=test\&format=json | head -c 200   # SearXNG
```

Unit names above are the ones used on the reference box; pick your own.
