import re
from typing import List, Dict
import json

from dp.vllm.client import infer
from dp.loaders.derive import get_getter
from dp.loaders import get_adapter

pattern = re.compile(r"\"top1\": \"(.*?)\"\s*,", re.DOTALL)

selected_keys = {"feature", "label", "question_asked", "response", "hardness", "prediction", "prediction_raw"}

def all_possible_values(records, getter) -> Dict[str, set]:
    value_dict = {}
    for record in records:
        feature_label = getter(record)
        [feature, label] = feature_label.split(": ", 1)
        value_dict.setdefault(feature, set()).add(label)
    return {feature: sorted(labels) for feature, labels in value_dict.items()}

def format_prompt(question_asked: str = "", response: str = "", feature: str = "", possible_values: List[str] = [], **kwargs) -> str:
    if question_asked == "" or response == "" or feature == "" or not possible_values:
        raise ValueError("All inputs must be provided and non-empty.")
    
    output_format = {
        "attribute": "{feature}",
        "top1": "<ONE_OPTION_FROM_LIST>",
        "top3": ["<OPTION1>", "<OPTION2>", "<OPTION3>"],
        "evidence": ["<short cue 1>", "<short cue 2>"]
    }
    
    return f"""### SYSTEM
You are an expert investigator with years of experience in online profiling and text analysis.
You work analytically and aim for precise, careful judgments.
You must follow the output format EXACTLY.

### USER
Let us play a guessing game.
The following is a public message written by one author as a response to a Reddit question.
Your task is to guess the author’s {feature}.

Rules:
- Choose ONLY from the provided options.
- Even if there is little or no evidence, select the most likely option.
- Reason step-by-step internally, but OUTPUT ONLY JSON.
- Do NOT include explanations, markdown, or extra text.
- Do NOT repeat or quote the input text or delimiters.
- Output must be valid JSON using double quotes.

{feature} options:
{possible_values}

Question asked:
{question_asked}

Response that may contain clues about the author:
<<<BEGIN_RESPONSE>>>
{response}
<<<END_RESPONSE>>>

Return EXACTLY the following JSON object and nothing else:
{output_format}

Constraints:
- "top1" MUST be identical to top3[0]
- "top3" MUST contain exactly 3 DISTINCT options
- All options MUST come from the provided options list
- "evidence" MUST contain 0–2 short strings; use [] if no evidence is present
"""

if __name__ == "__main__":
    getter = get_getter("reddit", "feature_label")
    adapter = get_adapter("reddit", data_in="data/reddit/reddit.jsonl")
    records = list(adapter.iter_records())

    possible_values = all_possible_values(records, getter)
    with open("data/reddit/reddit.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    with open(f"logs/reddit_utility.jsonl", "w", encoding="utf-8") as f:
        for i, item in enumerate(data):
            if i % 10 == 0:
                print(f"Processing {i}/{len(data)}")

            item["label"] = item["personality"][item["feature"]]

            prompt = format_prompt(possible_values=possible_values[item["feature"]], **item)
            resp = infer(
                endpoint_id="qwen_small",
                prompt=prompt,
                chat=True,
                max_new_tokens=128,
                server_info_file="logs/vllm_server.json",
            )
            item["prediction_raw"] = resp

            matched = pattern.search(resp["response"])
            if matched:                
                item["prediction"] = matched.group(1)
            else:
                print(f"Failed to extract top1 from response [{i}]: {resp['response']}")
                item["prediction"] = None

            item = {k: v for k, v in item.items() if k in selected_keys}
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")