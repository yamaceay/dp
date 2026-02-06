from vllm.client import load_model, infer

# load_model(
#     endpoint_id="qwen_small",
#     model="Qwen/Qwen2.5-0.5B-Instruct",
#     dtype="float16",
#     chat=True,
#     server_info_file="logs/vllm_server.json",
# )

import json
with open("data/reddit/reddit.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

all_predictions = []
for i, item in enumerate(data):
    if i % 10 == 0:
        print(f"Processing {i}/{len(data)}")
        
    question = item["question_asked"]
    answer = item["response"]
    feature = item["feature"]
    hardness = item["hardness"]
    label = item["personality"][item["feature"]]
    if feature == "income_level":
        possible_values = ["high", "middle", "low"]
    elif feature in {"birth_city_country", "city_country"}:
        possible_values = ["Europe", "Americas", "Asia", "Africa", "Oceania", "Other"]
    elif feature == "age":
        possible_values = ["18-29", "30-44", "45-59", "60+"]
    elif feature == "education":
        possible_values = ["secondary", "studying", "bachelor", "master", "doctorate", "other"]
    elif feature == "occupation":
        possible_values = ["tech", "design", "education", "healthcare", "business_finance", "service", "culture", "other"]
    elif feature == "relationship_status":
        possible_values = ["single", "in a relationship", "married", "divorced", "widowed", "other"]
    elif feature == "sex":
        possible_values = ["male", "female"]
    else:
        possible_values = []

    prompt = (
        f"You are provided with a Reddit user's answer to a question about their {feature}. "
        f"Select the most appropriate value for {feature} from the following options: {possible_values}. "
        "Return only the value, in quotes, without any extra text or formatting.\n"
        f'<value>\n'
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Feature: {feature}\n"
        "Your response:"
    )
    resp = infer(
        endpoint_id="qwen_small",
        prompt=prompt,
        chat=True,
        max_new_tokens=128,
        server_info_file="logs/vllm_server.json",
    )
    print(resp["response"], label)
    all_predictions.append((resp["response"], label, hardness))

print(all_predictions)