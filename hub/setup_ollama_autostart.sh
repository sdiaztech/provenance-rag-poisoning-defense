#!/usr/bin/env bash
# Run once in a JupyterHub terminal. From then on, Ollama starts by itself every time your JupyterHub server starts,
# listening on 0.0.0.0:11434 so your Mac can reach it at https://<hub>/user/<username>/proxy/11434.
# It stops when the server stops. Log: ~/ollama.log
set -euo pipefail
mkdir -p ~/.jupyter
CONFIG=~/.jupyter/jupyter_server_config.py
if grep -q "Ollama autostart" "$CONFIG" 2>/dev/null; then
  echo "Ollama autostart is already set up in $CONFIG"
else
  cat >> "$CONFIG" <<'EOF'

# --- Ollama autostart (provenance-rag-poisoning-defense) ---
def _start_ollama():
    import os, shutil, socket, subprocess
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", 11434)) == 0:
            return  # already running
    binary = shutil.which("ollama") or os.path.expanduser("~/ollama/bin/ollama")
    if not os.path.exists(binary):
        print("Ollama autostart: ollama not found")
        return
    log = open(os.path.expanduser("~/ollama.log"), "a")
    subprocess.Popen([binary, "serve"], env=dict(os.environ, OLLAMA_HOST="0.0.0.0:11434"),
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

try:
    _start_ollama()
except Exception as error:  # never block the server from starting
    print(f"Ollama autostart skipped: {error}")
# --- end Ollama autostart ---
EOF
  echo "Ollama autostart added to $CONFIG. It takes effect the next time your server starts."
fi
