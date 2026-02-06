#!/usr/bin/env python3
import argparse
import gc
import json
import os
import socket
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HTTP LLM provider server with dynamic endpoint/model loading.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--server-info-file", default="logs/llm_provider_server.json")
    parser.add_argument("--registry-file", default="logs/llm_provider_endpoints.json")
    parser.add_argument("--max-endpoints", type=int, default=4)
    return parser.parse_args()


def resolve_dtype(dtype: str, has_cuda: bool) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return torch.float16 if has_cuda else torch.float32


class EndpointRuntime:
    def __init__(
        self,
        endpoint_id: str,
        model: str,
        dtype: str,
        trust_remote_code: bool,
        defaults: dict,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.model_id = model
        self.dtype_name = dtype
        self.trust_remote_code = trust_remote_code
        self.defaults = defaults
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = threading.Lock()

        torch_dtype = resolve_dtype(dtype, self.device == "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def infer(self, payload: dict) -> dict:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("missing_prompt")

        use_chat = bool(payload.get("chat", self.defaults["chat"]))
        system_prompt = str(payload.get("system_prompt", self.defaults["system_prompt"]))
        max_new_tokens = int(payload.get("max_new_tokens", self.defaults["max_new_tokens"]))
        temperature = float(payload.get("temperature", self.defaults["temperature"]))
        top_p = float(payload.get("top_p", self.defaults["top_p"]))

        if use_chat:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt_text = f"User: {prompt}\nAssistant:"
        else:
            prompt_text = prompt

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with self.lock, torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return {
            "endpoint_id": self.endpoint_id,
            "model": self.model_id,
            "device": self.device,
            "response": response,
            "timestamp_utc": utc_now(),
        }

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ProviderState:
    def __init__(self, registry_file: str, max_endpoints: int) -> None:
        self.registry_file = registry_file
        self.max_endpoints = max_endpoints
        self.endpoints: dict[str, EndpointRuntime] = {}
        self.meta: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.persist()

    def persist(self) -> None:
        os.makedirs(os.path.dirname(self.registry_file) or ".", exist_ok=True)
        endpoints_list = sorted(self.meta.values(), key=lambda x: x["endpoint_id"])
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump({"endpoints": endpoints_list, "updated_at_utc": utc_now()}, f, ensure_ascii=True, indent=2)

    def list(self) -> list[dict]:
        with self.lock:
            return sorted(self.meta.values(), key=lambda x: x["endpoint_id"])

    def load(self, payload: dict) -> dict:
        endpoint_id = str(payload.get("endpoint_id", "")).strip()
        model = str(payload.get("model", "")).strip()
        if not endpoint_id:
            raise ValueError("missing_endpoint_id")
        if not model:
            raise ValueError("missing_model")

        dtype = str(payload.get("dtype", "auto"))
        trust_remote_code = bool(payload.get("trust_remote_code", False))
        defaults = {
            "chat": bool(payload.get("chat", True)),
            "system_prompt": str(payload.get("system_prompt", "You are a concise and helpful assistant.")),
            "max_new_tokens": int(payload.get("max_new_tokens", 128)),
            "temperature": float(payload.get("temperature", 0.2)),
            "top_p": float(payload.get("top_p", 0.9)),
        }

        with self.lock:
            if endpoint_id in self.endpoints:
                raise ValueError("endpoint_already_exists")
            if len(self.endpoints) >= self.max_endpoints:
                raise ValueError("max_endpoints_reached")

        runtime = EndpointRuntime(
            endpoint_id=endpoint_id,
            model=model,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            defaults=defaults,
        )

        with self.lock:
            self.endpoints[endpoint_id] = runtime
            self.meta[endpoint_id] = {
                "endpoint_id": endpoint_id,
                "model": model,
                "dtype": dtype,
                "trust_remote_code": trust_remote_code,
                "device": runtime.device,
                "defaults": defaults,
                "loaded_at_utc": utc_now(),
            }
            self.persist()
            return self.meta[endpoint_id]

    def update(self, payload: dict) -> dict:
        endpoint_id = str(payload.get("endpoint_id", "")).strip()
        if not endpoint_id:
            raise ValueError("missing_endpoint_id")

        with self.lock:
            if endpoint_id not in self.endpoints:
                raise ValueError("endpoint_not_found")
            runtime = self.endpoints[endpoint_id]
            meta = self.meta[endpoint_id]

            if "chat" in payload:
                runtime.defaults["chat"] = bool(payload["chat"])
            if "system_prompt" in payload:
                runtime.defaults["system_prompt"] = str(payload["system_prompt"])
            if "max_new_tokens" in payload:
                runtime.defaults["max_new_tokens"] = int(payload["max_new_tokens"])
            if "temperature" in payload:
                runtime.defaults["temperature"] = float(payload["temperature"])
            if "top_p" in payload:
                runtime.defaults["top_p"] = float(payload["top_p"])

            meta["defaults"] = dict(runtime.defaults)
            meta["updated_at_utc"] = utc_now()
            self.persist()
            return meta

    def unload(self, endpoint_id: str) -> None:
        with self.lock:
            if endpoint_id not in self.endpoints:
                raise ValueError("endpoint_not_found")
            runtime = self.endpoints.pop(endpoint_id)
            self.meta.pop(endpoint_id, None)
            self.persist()
        runtime.unload()

    def infer(self, endpoint_id: str, payload: dict) -> dict:
        with self.lock:
            runtime = self.endpoints.get(endpoint_id)
        if runtime is None:
            raise ValueError("endpoint_not_found")
        return runtime.infer(payload)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    content_len = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_len) if content_len > 0 else b"{}"
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    args = parse_args()
    state = ProviderState(args.registry_file, args.max_endpoints)

    server_info = {
        "host": socket.gethostname(),
        "port": args.port,
        "base_url": f"http://{socket.gethostname()}:{args.port}",
        "health_url": f"http://{socket.gethostname()}:{args.port}/health",
        "registry_url": f"http://{socket.gethostname()}:{args.port}/endpoints",
        "registry_file": args.registry_file,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "started_at_utc": utc_now(),
    }
    os.makedirs(os.path.dirname(args.server_info_file) or ".", exist_ok=True)
    with open(args.server_info_file, "w", encoding="utf-8") as f:
        json.dump(server_info, f, ensure_ascii=True, indent=2)
    print(json.dumps(server_info, ensure_ascii=True, indent=2), flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(200, {"ok": True, "timestamp_utc": utc_now()})
                return
            if parsed.path == "/endpoints":
                self._send_json(200, {"ok": True, "endpoints": state.list()})
                return
            self._send_json(404, {"ok": False, "error": "not_found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = read_json_body(self)
            except Exception as exc:
                self._send_json(400, {"ok": False, "error": f"invalid_json: {exc}"})
                return

            try:
                if parsed.path == "/endpoints/load":
                    ep = state.load(payload)
                    self._send_json(200, {"ok": True, "endpoint": ep})
                    return
                if parsed.path == "/endpoints/update":
                    ep = state.update(payload)
                    self._send_json(200, {"ok": True, "endpoint": ep})
                    return
                if parsed.path == "/infer":
                    endpoint_id = str(payload.get("endpoint_id", "")).strip()
                    if not endpoint_id:
                        raise ValueError("missing_endpoint_id")
                    result = state.infer(endpoint_id, payload)
                    self._send_json(200, {"ok": True, **result})
                    return
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"internal_error: {exc}"})
                return

            self._send_json(404, {"ok": False, "error": "not_found"})

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/endpoints":
                self._send_json(404, {"ok": False, "error": "not_found"})
                return
            query = parse_qs(parsed.query)
            endpoint_id = str(query.get("endpoint_id", [""])[0]).strip()
            if not endpoint_id:
                self._send_json(400, {"ok": False, "error": "missing_endpoint_id"})
                return
            try:
                state.unload(endpoint_id)
                self._send_json(200, {"ok": True, "endpoint_id": endpoint_id})
            except ValueError as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *args_list) -> None:
            return

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print("Provider server is ready.", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
