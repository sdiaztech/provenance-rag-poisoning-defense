"""Connection to an Ollama server, on this computer or on a JupyterHub server.

To use Ollama on JupyterHub, set OLLAMA_URL to its proxy address, e.g. https://<hub>/user/<username>/proxy/11434,
and put your JupyterHub API token in JUPYTERHUB_TOKEN (main.py reads both from the git-ignored .env file). The token
is sent only as a request header and is never printed or saved.
"""
from __future__ import annotations

import os
import time
from collections.abc import Sequence

import requests

RETRY_STATUS = {408, 429, 500, 502, 503, 504}  # brief hub or proxy hiccups (408: the proxy timed out): try again
RETRY_WAITS = (10, 20, 30, 30, 30)  # seconds between attempts
KEEP_ALIVE = "30m"  # keep models loaded between requests and runs


class OllamaUnavailable(RuntimeError):
    """Ollama kept failing after every retry."""


class OllamaClient:
    """The Ollama server that writes poisoned passages, embeds text and answers questions."""

    def __init__(self, url: str | None = None, token: str | None = None) -> None:
        self.url = (url or os.environ.get("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        token = token if token is not None else os.environ.get("JUPYTERHUB_TOKEN")
        self.headers = {"Authorization": f"token {token}"} if token else {}

    @property
    def is_local(self) -> bool:
        return any(host in self.url for host in ("://localhost", "://127.0.0.1"))

    def problem(self, *models: str) -> str | None:
        """Why these models can't be used right now, or None if they can."""
        try:
            response = requests.get(f"{self.url}/api/tags", headers=self.headers, timeout=10, allow_redirects=False)
        except requests.RequestException:
            if self.is_local:
                return f"Ollama is not running at {self.url}. Open the Ollama app (or run: ollama serve) and try again."
            return (f"Can't reach {self.url}. Check that you're on the VPN, your JupyterHub server is running, and "
                    f"Ollama is running there.")
        if response.is_redirect or response.status_code in (401, 403):
            if not self.headers:
                return f"{self.url} needs your JupyterHub token. Set it first: export JUPYTERHUB_TOKEN=<your token>"
            return (f"{self.url} refused the request (HTTP {response.status_code}). Check that your token is current "
                    f"and that Ollama on the hub was started with OLLAMA_HOST=0.0.0.0:11434.")
        if response.status_code == 404:
            return f"Nothing answered at {self.url}. Check the address; on JupyterHub it ends in /proxy/11434."
        try:
            response.raise_for_status()
            names = {m.get("name", "") for m in response.json().get("models", [])}
        except (requests.RequestException, ValueError):
            return f"{self.url} answered, but not like Ollama (HTTP {response.status_code}). Check the address."
        missing = [m for m in models if m not in names and f"{m}:latest" not in names]
        if len(missing) == 1:
            return f"The model {missing[0]} is not installed there. Run: ollama pull {missing[0]}"
        if missing:
            return (f"The models {', '.join(missing)} are not installed there. Run: "
                    + " && ".join(f"ollama pull {m}" for m in missing))
        return None

    def generate(self, model: str, prompt: str, *, seed: int = 0, temperature: float = 0.0,
                 max_tokens: int | None = None) -> str:
        options = {"temperature": temperature, "seed": seed}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return self._post("/api/generate", {"model": model, "prompt": prompt, "stream": False, "options": options,
                                            "keep_alive": KEEP_ALIVE})["response"]

    def warm_up(self, *models: str, embedding: Sequence[str] = ()) -> None:
        """Loads models before the real requests. The first request after a model was unloaded can outlast the
        hub proxy's time limit; the load keeps going on the server, so retrying until it answers is enough."""
        for model in models:
            self.generate(model, "Reply with OK.", max_tokens=1)
        for model in embedding:
            self.embed(model, ["warm up"])

    def _post(self, path: str, payload: dict) -> dict:
        for attempt, wait in enumerate((*RETRY_WAITS, None)):
            try:
                response = requests.post(f"{self.url}{path}", headers=self.headers, timeout=300, json=payload)
                if response.status_code not in RETRY_STATUS:
                    response.raise_for_status()
                    return response.json()
                problem = f"HTTP {response.status_code}"
            except (requests.ConnectionError, requests.Timeout) as error:
                problem = type(error).__name__
            if wait is None:
                break
            time.sleep(wait)
        minutes = max(1, round(sum(RETRY_WAITS) / 60))
        raise OllamaUnavailable(
            f"{self.url}{path} kept failing ({problem}) for about {minutes} minutes. If the model is still loading, "
            f"run again in a minute; answers saved so far are kept. If it keeps happening, check that your hub server "
            f"has a GPU (run nvidia-smi in a hub terminal) and that Ollama is running there.")

    def embed(self, model: str, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """One vector per text, in order. Texts longer than the model's context are truncated, as in
        sentence-transformers."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors += self._post("/api/embed", {"model": model, "input": list(texts[start:start + batch_size]),
                                                 "truncate": True, "keep_alive": KEEP_ALIVE})["embeddings"]
        return vectors
