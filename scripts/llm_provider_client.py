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


def main() -> None:
    parser = argparse.ArgumentParser(description="Client for llm_provider_server.py")
    parser.add_argument("--server-info-file", default="logs/llm_provider_server.json")
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

    infer = sub.add_parser("infer")
    infer.add_argument("--endpoint-id", required=True)
    infer.add_argument("--prompt", required=True)
    infer.add_argument("--chat", action="store_true")
    infer.add_argument("--system-prompt", default="")
    infer.add_argument("--max-new-tokens", type=int, default=-1)
    infer.add_argument("--temperature", type=float, default=-1.0)
    infer.add_argument("--top-p", type=float, default=-1.0)

    args = parser.parse_args()
    base = load_base_url(args.server_info_file, args.url)

    if args.cmd == "health":
        print(json.dumps(http_json("GET", f"{base}/health"), ensure_ascii=True, indent=2))
        return
    if args.cmd == "list-endpoints":
        print(json.dumps(http_json("GET", f"{base}/endpoints"), ensure_ascii=True, indent=2))
        return
    if args.cmd == "load-model":
        payload = {
            "endpoint_id": args.endpoint_id,
            "model": args.model,
            "dtype": args.dtype,
            "trust_remote_code": args.trust_remote_code,
            "chat": args.chat,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        print(json.dumps(http_json("POST", f"{base}/endpoints/load", payload), ensure_ascii=True, indent=2))
        return
    if args.cmd == "update-endpoint":
        payload = {"endpoint_id": args.endpoint_id}
        if args.chat:
            payload["chat"] = True
        if args.system_prompt:
            payload["system_prompt"] = args.system_prompt
        if args.max_new_tokens >= 0:
            payload["max_new_tokens"] = args.max_new_tokens
        if args.temperature >= 0:
            payload["temperature"] = args.temperature
        if args.top_p >= 0:
            payload["top_p"] = args.top_p
        print(json.dumps(http_json("POST", f"{base}/endpoints/update", payload), ensure_ascii=True, indent=2))
        return
    if args.cmd == "unload-model":
        q = urllib.parse.urlencode({"endpoint_id": args.endpoint_id})
        print(json.dumps(http_json("DELETE", f"{base}/endpoints?{q}"), ensure_ascii=True, indent=2))
        return
    if args.cmd == "infer":
        payload = {
            "endpoint_id": args.endpoint_id,
            "prompt": args.prompt,
        }
        if args.chat:
            payload["chat"] = True
        if args.system_prompt:
            payload["system_prompt"] = args.system_prompt
        if args.max_new_tokens >= 0:
            payload["max_new_tokens"] = args.max_new_tokens
        if args.temperature >= 0:
            payload["temperature"] = args.temperature
        if args.top_p >= 0:
            payload["top_p"] = args.top_p
        print(json.dumps(http_json("POST", f"{base}/infer", payload), ensure_ascii=True, indent=2))
        return


if __name__ == "__main__":
    main()
