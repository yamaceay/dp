## vLLM Inference Server and Client

Initialize a new vLLM server on the cluster. This will submit a job to the cluster, which will start the server and write its info (e.g. IP address and port) to a file.

```sh
scripts/submit_for_vllm.sh
```

You can then use the client (e.g. `vllm/client.py`) to interact with the server. For example, you can load a model and run inference:

```bash
python vllm/client.py load-model --endpoint-id=qwen_small --model=Qwen/Qwen2.5-0.5B-Instruct --dtype=float16 --chat
```

The above command will load the Qwen2.5-0.5B-Instruct model with float16 precision and chat settings, and create an endpoint with ID "qwen_small". The server will return the endpoint information, which includes the model, device, and default settings:

```json
{
  "ok": true,
  "endpoint": {
    "endpoint_id": "qwen_small",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "dtype": "float16",
    "trust_remote_code": false,
    "device": "cuda",
    "defaults": {
      "chat": true,
      "system_prompt": "You are a concise and helpful assistant.",
      "max_new_tokens": 128,
      "temperature": 0.2,
      "top_p": 0.9
    },
    "loaded_at_utc": "2026-02-06T15:10:04.366913+00:00"
  }
}
```

You can then use the endpoint to run inference with a prompt:

```sh
python vllm/client.py infer --endpoint-id=qwen_small --prompt="USER RESPONDS TO A QUESTION: well ain't nothin' like gridiron football now, is it? Been tailgatin' all my life, even met my sweetheart in one o' them tailgate parties, ha! Funny thing is, back in our dating days we were on old school flip phones, wasn't no IG or TikTok to document those day - while sometimes heart dreams up a wild river walk tailgate party, hashtagging #TailgateLife and all, haha. As to my team, well, I've been loyal to my boys in silver and black, ever since they were the cardiac crew and till now as the Raider Nation. Catch the wave, y'all!\n\nCan you guess about the user's characteristics?" --chat
```

With the above command, the server will run inference on the prompt and return a response:

```json
{
  "ok": true,
  "endpoint_id": "qwen_small",
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "device": "cuda",
  "response": "The user appears to be an avid fan of American football, particularly gridiron football. They mention attending tailgate parties with their girlfriend and have fond memories from past events. The user also expresses loyalty to their team, specifically mentioning their quarterback (silver) and running back (black). Additionally, they seem to enjoy documenting these moments through social media platforms such as Instagram and Twitter.",
  "timestamp_utc": "2026-02-06T15:19:05.174109+00:00"
}
```