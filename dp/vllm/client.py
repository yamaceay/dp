#!/usr/bin/env python3
import argparse
import json
import urllib.parse
import urllib.request


def load_base_url(server_info_file: str, url: str) -> str:
    if url:
        return url.rstrip("/")
    with open(server_info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    return str(info["base_url"]).rstrip("/")


def http_json(method: str, url: str, payload: dict | None = None) -> dict:
    body = None
    headers = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url=url, data=body, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def health(*, server_info_file: str = "logs/vllm_server.json", url: str = "") -> dict:
    base = load_base_url(server_info_file, url)
    return http_json("GET", f"{base}/health")


def list_endpoints(*, server_info_file: str = "logs/vllm_server.json", url: str = "") -> dict:
    base = load_base_url(server_info_file, url)
    return http_json("GET", f"{base}/endpoints")


def load_model(
    *,
    endpoint_id: str,
    model: str,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    chat: bool = False,
    system_prompt: str = "You are a concise and helpful assistant.",
    max_new_tokens: int = 128,
    temperature: float = 0.2,
    top_p: float = 0.9,
    server_info_file: str = "logs/vllm_server.json",
    url: str = "",
) -> dict:
    base = load_base_url(server_info_file, url)
    payload = {
        "endpoint_id": endpoint_id,
        "model": model,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "chat": chat,
        "system_prompt": system_prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    return http_json("POST", f"{base}/endpoints/load", payload)


def update_endpoint(
    *,
    endpoint_id: str,
    chat: bool | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    server_info_file: str = "logs/vllm_server.json",
    url: str = "",
) -> dict:
    base = load_base_url(server_info_file, url)
    payload = {"endpoint_id": endpoint_id}
    if chat is not None:
        payload["chat"] = bool(chat)
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if max_new_tokens is not None:
        payload["max_new_tokens"] = int(max_new_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    return http_json("POST", f"{base}/endpoints/update", payload)


def unload_model(
    *,
    endpoint_id: str,
    server_info_file: str = "logs/vllm_server.json",
    url: str = "",
) -> dict:
    base = load_base_url(server_info_file, url)
    q = urllib.parse.urlencode({"endpoint_id": endpoint_id})
    return http_json("DELETE", f"{base}/endpoints?{q}")


def infer(
    *,
    endpoint_id: str,
    prompt: str,
    chat: bool | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    server_info_file: str = "logs/vllm_server.json",
    url: str = "",
) -> dict:
    base = load_base_url(server_info_file, url)
    payload = {
        "endpoint_id": endpoint_id,
        "prompt": prompt,
    }
    if chat is not None:
        payload["chat"] = bool(chat)
    if system_prompt is not None:
        payload["system_prompt"] = system_prompt
    if max_new_tokens is not None:
        payload["max_new_tokens"] = int(max_new_tokens)
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    return http_json("POST", f"{base}/infer", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Client for vllm/server.py")
    parser.add_argument("--server-info-file", default="logs/vllm_server.json")
    parser.add_argument("--url", default="", help="Base URL override, e.g. http://serv-9223:18080")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")
    sub.add_parser("list-endpoints")

    load = sub.add_parser("load-model")
    load.add_argument("--endpoint-id", required=True)
    load.add_argument("--model", required=True)
    load.add_argument("--dtype", default="auto")
    load.add_argument("--trust-remote-code", action="store_true")
    load.add_argument("--chat", action="store_true")
    load.add_argument("--system-prompt", default="You are a concise and helpful assistant.")
    load.add_argument("--max-new-tokens", type=int, default=128)
    load.add_argument("--temperature", type=float, default=0.2)
    load.add_argument("--top-p", type=float, default=0.9)

    update = sub.add_parser("update-endpoint")
    update.add_argument("--endpoint-id", required=True)
    update.add_argument("--chat", action="store_true")
    update.add_argument("--system-prompt", default="")
    update.add_argument("--max-new-tokens", type=int, default=-1)
    update.add_argument("--temperature", type=float, default=-1.0)
    update.add_argument("--top-p", type=float, default=-1.0)

    unload = sub.add_parser("unload-model")
    unload.add_argument("--endpoint-id", required=True)

    infer_args = sub.add_parser("infer")
    infer_args.add_argument("--endpoint-id", required=True)
    infer_args.add_argument("--prompt", required=True)
    infer_args.add_argument("--chat", action="store_true")
    infer_args.add_argument("--system-prompt", default="")
    infer_args.add_argument("--max-new-tokens", type=int, default=-1)
    infer_args.add_argument("--temperature", type=float, default=-1.0)
    infer_args.add_argument("--top-p", type=float, default=-1.0)

    args = parser.parse_args()

    if args.cmd == "health":
        print(json.dumps(health(server_info_file=args.server_info_file, url=args.url), ensure_ascii=True, indent=2))
        return
    if args.cmd == "list-endpoints":
        print(
            json.dumps(
                list_endpoints(server_info_file=args.server_info_file, url=args.url),
                ensure_ascii=True,
                indent=2,
            )
        )
        return
    if args.cmd == "load-model":
        print(
            json.dumps(
                load_model(
                    endpoint_id=args.endpoint_id,
                    model=args.model,
                    dtype=args.dtype,
                    trust_remote_code=args.trust_remote_code,
                    chat=args.chat,
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    server_info_file=args.server_info_file,
                    url=args.url,
                ),
                ensure_ascii=True,
                indent=2,
            )
        )
        return
    if args.cmd == "update-endpoint":
        print(
            json.dumps(
                update_endpoint(
                    endpoint_id=args.endpoint_id,
                    chat=True if args.chat else None,
                    system_prompt=args.system_prompt if args.system_prompt else None,
                    max_new_tokens=args.max_new_tokens if args.max_new_tokens >= 0 else None,
                    temperature=args.temperature if args.temperature >= 0 else None,
                    top_p=args.top_p if args.top_p >= 0 else None,
                    server_info_file=args.server_info_file,
                    url=args.url,
                ),
                ensure_ascii=True,
                indent=2,
            )
        )
        return
    if args.cmd == "unload-model":
        print(
            json.dumps(
                unload_model(
                    endpoint_id=args.endpoint_id,
                    server_info_file=args.server_info_file,
                    url=args.url,
                ),
                ensure_ascii=True,
                indent=2,
            )
        )
        return
    if args.cmd == "infer":
        print(
            json.dumps(
                infer(
                    endpoint_id=args.endpoint_id,
                    prompt=args.prompt,
                    chat=True if args.chat else None,
                    system_prompt=args.system_prompt if args.system_prompt else None,
                    max_new_tokens=args.max_new_tokens if args.max_new_tokens >= 0 else None,
                    temperature=args.temperature if args.temperature >= 0 else None,
                    top_p=args.top_p if args.top_p >= 0 else None,
                    server_info_file=args.server_info_file,
                    url=args.url,
                ),
                ensure_ascii=True,
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
