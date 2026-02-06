from vllm.client import load_model, infer

load_model(
    endpoint_id="llama8b",
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="float16",
    chat=True,
    server_info_file="logs/vllm_server.json",
)

resp = infer(
    endpoint_id="llama8b",
    prompt="USER RESPONDS TO A QUESTION: well ain't nothin' like gridiron football now, is it? Been tailgatin' all my life, even met my sweetheart in one o' them tailgate parties, ha! Funny thing is, back in our dating days we were on old school flip phones, wasn't no IG or TikTok to document those day - while sometimes heart dreams up a wild river walk tailgate party, hashtagging #TailgateLife and all, haha. As to my team, well, I've been loyal to my boys in silver and black, ever since they were the cardiac crew and till now as the Raider Nation. Catch the wave, y'all!\n\nCan you guess about the user's characteristics?",
    chat=True,
    max_new_tokens=128,
    server_info_file="logs/vllm_server.json",
)
print(resp["response"])
